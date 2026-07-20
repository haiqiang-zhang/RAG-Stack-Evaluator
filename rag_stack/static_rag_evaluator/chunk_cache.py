"""Lazy on-demand cache for chunked-corpus artifacts.

When the optimizer sweeps over `(chunk_method, chunk_size, chunk_overlap)`,
each unique tuple needs a distinct chunked corpus + recomputed token stats.
This module provides a deterministic, hash-keyed cache:

    <project>/data/chunks/<chunk_hash>/
        corpus.parquet     # chunked text (one row per chunk)
        token_stats.json   # avg_chunk_tokens etc. — recomputed per chunk
        meta.json          # source params (for debugging / auditing)

The hash collapses (chunk_method, chunk_size, chunk_overlap, raw_corpus_id)
into a stable 16-char hex string. Embeddings + FAISS indexes are NOT cached
here — the static evaluator already caches them per-vectordb path; the
chunk_hash is injected into that path so different chunks → different
index directories.

Backward compatibility: when the project has no `raw_corpus.parquet` and
no `corpus:` block in YAML, ``get_or_build`` returns the existing
``corpus.parquet`` with ``chunk_hash="none"``. This keeps pre-chunking
projects working without migration.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from typing import Any, Callable, Dict, Optional

import pandas as pd

logger = logging.getLogger("RAG-Stack")

NO_OP_HASH = "none"


def _raw_corpus_signature(raw_corpus_path: str) -> str:
    """Cheap-but-stable identifier for the raw corpus.

    Uses (size, mtime_ns) rather than full SHA256 — corpora can be GB-sized
    and we only need to detect "the underlying file changed", not assert
    cryptographic uniqueness. Document mtime+size limitation in the plan.
    """
    st = os.stat(raw_corpus_path)
    return f"{st.st_size}_{st.st_mtime_ns}"


def compute_chunk_hash(
    chunk_method: str,
    chunk_size: int,
    chunk_overlap: int,
    raw_corpus_signature: str,
) -> str:
    """Deterministic 16-char hash for a (chunk params, raw corpus) tuple."""
    blob = f"{chunk_method}|{chunk_size}|{chunk_overlap}|{raw_corpus_signature}"
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _cache_dir(project_dir: str, chunk_hash: str) -> str:
    return os.path.join(project_dir, "data", "chunks", chunk_hash)


# doc_id scheme marker (07-05): doc_ids became CONTENT-DERIVED (sha1+position,
# see langchain_chunk.py) so globally shared faiss indexes stay aligned with
# any re-chunk. Cache entries produced under the old uuid4 scheme carry no
# marker and are treated as incomplete → transparently rebuilt (deterministic
# ids make the rebuild converge with the new index namespace). Nothing is
# deleted — live runs on other machines keep their old artifacts.
DOC_ID_SCHEME = "chunkhash-pos-v2"


def _is_complete(cache_dir: str) -> bool:
    """Complete iff all artifacts exist AND the doc_id scheme matches."""
    names = ("corpus.parquet", "token_stats.json", "meta.json")
    if not all(os.path.isfile(os.path.join(cache_dir, n)) for n in names):
        return False
    marker = os.path.join(cache_dir, "id_scheme.txt")
    try:
        with open(marker) as f:
            return f.read().strip() == DOC_ID_SCHEME
    except OSError:
        return False


def _run_chunker(
    raw_df: pd.DataFrame,
    chunk_method: str,
    chunk_size: int,
    chunk_overlap: int,
) -> pd.DataFrame:
    """Dispatch to langchain_chunk / llama_index_chunk based on method.

    The raw DataFrame must have at least a ``texts`` column (long-document
    text). Path / page columns are passed through to the chunked output if
    present.

    Returns a DataFrame with the standard chunked-corpus columns:
    ``doc_id``, ``contents``, ``path``, ``start_end_idx``, ``metadata``.
    """
    from rag_stack.static_rag_evaluator.chunk._registry import chunk_modules
    from rag_stack.static_rag_evaluator.chunk import (
        langchain_chunk, llama_index_chunk,
    )

    if chunk_method not in chunk_modules:
        raise ValueError(
            f"Unknown chunk_method '{chunk_method}'. "
            f"Available: {sorted(chunk_modules.keys())}"
        )

    # Decide the underlying library (LangChain vs LlamaIndex) by which
    # registry entry the method came from. The registry stores the splitter
    # class; LangChain splitters subclass langchain TextSplitter, LlamaIndex
    # ones subclass NodeParser. Match by import path.
    splitter_cls = chunk_modules[chunk_method]
    is_langchain = splitter_cls.__module__.startswith("langchain")

    fn = langchain_chunk if is_langchain else llama_index_chunk
    return fn(
        parsed_result=raw_df,
        chunk_method=chunk_method,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def get_or_build(
    project_dir: str,
    raw_corpus_path: Optional[str],
    fallback_corpus_path: str,
    chunk_params: Dict[str, Any],
    base_token_stats=None,
    *,
    token_stats_factory: Optional[Callable[[pd.DataFrame], Any]] = None,
) -> Dict[str, Any]:
    """Return cached chunked-corpus artifacts, building them on first use.

    Args:
        project_dir: Project root. Cache lives under ``data/chunks/``.
        raw_corpus_path: Path to the raw long-doc parquet, or ``None`` if
            the project has no raw corpus (legacy / no-op path).
        fallback_corpus_path: Path to the existing pre-chunked corpus.parquet.
            Used when ``raw_corpus_path`` is None or chunking is a no-op.
        chunk_params: Dict with keys ``component`` / ``chunk_size`` /
            ``chunk_overlap``. May be empty (no `corpus:` block in YAML).
        base_token_stats: A live ``TokenStats`` instance used to recompute
            ``avg_chunk_tokens`` for each new chunked corpus. Other stats
            (query/output/prompt-template) are preserved. May be omitted only
            when ``token_stats_factory`` is supplied.
        token_stats_factory: Initialization-only callback that constructs the
            first live ``TokenStats`` directly from the canonical chunked
            frame. This lets a fresh project publish its default chunk through
            this same cache before a base ``TokenStats`` exists, instead of
            running a separate throw-away initialization chunk.

    Returns:
        Dict with keys:
            ``chunk_hash``       — for FAISS path keying
            ``corpus_path``      — absolute path to the chunked parquet
            ``token_stats``      — TokenStats instance for cost model
    """
    from rag_stack.cost_model.token_stats import TokenStats

    chunk_method = chunk_params.get("component")
    chunk_size = chunk_params.get("chunk_size")
    chunk_overlap = chunk_params.get("chunk_overlap")

    # --- No-op path: no chunk params, or no raw corpus → reuse fallback ---
    no_chunking_requested = not chunk_method or chunk_size is None
    if no_chunking_requested or raw_corpus_path is None:
        if not no_chunking_requested and raw_corpus_path is None:
            logger.warning(
                "Chunk params are sweepable but the project has no "
                "raw_corpus.parquet — falling back to the pre-chunked "
                "corpus.parquet. Quality scores will not actually vary "
                "with chunk_size. Provide a raw_corpus.parquet to enable "
                "chunking sweeps."
            )
        return {
            "chunk_hash": NO_OP_HASH,
            "corpus_path": fallback_corpus_path,
            "token_stats": base_token_stats,
        }

    # --- Cache-keyed build path -----------------------------------------
    if chunk_overlap is None:
        chunk_overlap = 0  # default for methods that don't expose it

    raw_sig = _raw_corpus_signature(raw_corpus_path)
    chunk_hash = compute_chunk_hash(
        str(chunk_method), int(chunk_size), int(chunk_overlap), raw_sig,
    )
    cache_dir = _cache_dir(project_dir, chunk_hash)

    if _is_complete(cache_dir):
        logger.info(
            f"chunk-cache hit: {chunk_hash} "
            f"({chunk_method}, size={chunk_size}, overlap={chunk_overlap})"
        )
        corpus_path = os.path.join(cache_dir, "corpus.parquet")
        with open(os.path.join(cache_dir, "token_stats.json")) as f:
            stats_dict = json.load(f)
        # Wrap stats in a TokenStats with tokenizer reattached so downstream
        # could call with_chunked_corpus again if needed.
        ts = TokenStats.from_dict(stats_dict)
        if base_token_stats is not None and hasattr(base_token_stats, "_tokenizer"):
            ts._tokenizer = base_token_stats._tokenizer
        # Backfill compressor elem stats for caches built before they
        # existed: the cached dict predates the fields, but the chunked
        # corpus (their only input besides the BERT tokenizer) is right
        # here. Rewrite the cache json so the cost is paid once per cache.
        comp_tok = getattr(base_token_stats, "_compressor_tokenizer", None)
        if (
            comp_tok is not None
            and stats_dict.get("avg_compressor_elems_per_chunk") is None
        ):
            chunked_df = pd.read_parquet(corpus_path)
            epc, ept = TokenStats._compute_compressor_elem_stats(
                chunked_df, comp_tok,
            )
            ts.avg_compressor_elems_per_chunk = epc
            ts.avg_compressor_elem_tokens = ept
            ts._compressor_tokenizer = comp_tok
            if epc is not None:
                stats_dict["avg_compressor_elems_per_chunk"] = epc
                stats_dict["avg_compressor_elem_tokens"] = ept
                stats_path = os.path.join(cache_dir, "token_stats.json")
                tmp_path = stats_path + ".tmp"
                with open(tmp_path, "w") as f:
                    json.dump(stats_dict, f, indent=1)
                os.replace(tmp_path, stats_path)
        elif comp_tok is not None:
            ts._compressor_tokenizer = comp_tok
        return {
            "chunk_hash": chunk_hash,
            "corpus_path": corpus_path,
            "token_stats": ts,
        }

    # Miss — build artifacts inside a temp dir, then atomic-rename.
    logger.info(
        f"chunk-cache miss: building {chunk_hash} "
        f"({chunk_method}, size={chunk_size}, overlap={chunk_overlap})"
    )
    raw_df = pd.read_parquet(raw_corpus_path)
    # Canonical raw-corpus schema is {doc_id, contents}. Reverse-alias legacy
    # raw corpora that still use "texts" so the chunker (which reads "contents")
    # works on both.
    if "contents" not in raw_df.columns and "texts" in raw_df.columns:
        raw_df = raw_df.rename(columns={"texts": "contents"})
    chunked_df = _run_chunker(
        raw_df,
        str(chunk_method),
        int(chunk_size),
        int(chunk_overlap),
    )

    # CANONICAL doc_id (07-05, single global mechanism): ``{chunk_hash}:{i}``.
    # Purely positional + content-addressed via the chunk hash — any re-chunk
    # of the same (corpus, params) reproduces the ids, and every EXISTING
    # shared faiss index converts by rewriting its meta.json from its own
    # directory name alone (no source parquet needed).
    chunked_df = chunked_df.copy()
    chunked_df["doc_id"] = [f"{chunk_hash}:{i}" for i in range(len(chunked_df))]
    if token_stats_factory is not None:
        new_stats = token_stats_factory(chunked_df)
    elif base_token_stats is not None:
        new_stats = base_token_stats.with_chunked_corpus(chunked_df)
    else:
        raise ValueError(
            "chunk-cache miss requires base_token_stats or token_stats_factory"
        )

    parent = os.path.dirname(cache_dir)
    os.makedirs(parent, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent) as tmp:
        chunked_df.to_parquet(os.path.join(tmp, "corpus.parquet"), index=False)
        with open(os.path.join(tmp, "id_scheme.txt"), "w") as f:
            f.write(DOC_ID_SCHEME)
        with open(os.path.join(tmp, "token_stats.json"), "w") as f:
            json.dump(new_stats.to_dict(), f, indent=2)
        with open(os.path.join(tmp, "meta.json"), "w") as f:
            json.dump({
                "chunk_hash": chunk_hash,
                "chunk_method": chunk_method,
                "chunk_size": int(chunk_size),
                "chunk_overlap": int(chunk_overlap),
                "raw_corpus_path": os.path.abspath(raw_corpus_path),
                "raw_corpus_signature": raw_sig,
                "n_chunks": int(len(chunked_df)),
            }, f, indent=2)
        # Atomic move into place
        # A STALE entry may occupy the target (07-05: doc_id scheme bump makes
        # old-scheme caches "incomplete" while their files still exist —
        # os.replace on a non-empty dir raises ENOTEMPTY). Clear it first.
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)
        os.replace(tmp, cache_dir)
        # Recreate the temp dir so the context manager's __exit__ doesn't
        # crash trying to delete it.
        os.makedirs(tmp, exist_ok=True)

    return {
        "chunk_hash": chunk_hash,
        "corpus_path": os.path.join(cache_dir, "corpus.parquet"),
        "token_stats": new_stats,
    }
