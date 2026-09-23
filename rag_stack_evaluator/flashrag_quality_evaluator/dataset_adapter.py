"""Parquet → FlashRAG JSONL + FAISS index materialization with content-hash caching.

FlashRAG cannot read rag-stack parquet schemas and cannot auto-build retrieval
indexes. This adapter bridges the gap:

  * :func:`materialize_qa_jsonl` — write rag-stack QA parquet to FlashRAG JSONL
    (``qid`` → ``id``, ``query`` → ``question``, ``generation_gt`` →
    ``golden_answers``; ``retrieval_gt`` stashed in ``metadata`` so rag-stack
    can still compute retrieval-side metrics post-hoc).
  * :func:`materialize_corpus_jsonl` — corpus parquet → FlashRAG corpus JSONL
    (``doc_id`` → ``id``, ``contents`` → ``contents``; other columns dropped).
  * :func:`ensure_index` — pre-build the FAISS index that FlashRAG's retriever
    will load (FlashRAG itself raises if the path is missing; see
    ``DenseRetriever.load_index`` in the FlashRAG fork). Caches by
    ``(corpus_hash, retrieval_method, faiss_type)`` so the optimizer's many
    trials only pay the build cost once per unique combination.

All caches live under ``<project_dir>/_flashrag/`` so they're cleaned up with
the project, and keyed by content hashes so re-running with the same data
hits the cache.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("RAG-Stack")


# ---------------------------------------------------------------------------
# Hashing helpers — cache keys for JSONL materialization and index building.
# ---------------------------------------------------------------------------


def _stable_cell_repr(value: Any) -> str:
    """Deterministic string form for an object-column cell.

    Parquet object columns can hold containers — ``ndarray`` (e.g. the
    dragonball corpus ``start_end_idx``), ``list``, ``dict`` (``metadata``)
    — which ``pd.util.hash_pandas_object`` cannot factorize (it raises
    ``TypeError: unhashable type``). JSON with sorted keys gives a stable
    representation; ``default=str`` covers anything non-JSON-native.
    """
    if isinstance(value, np.ndarray):
        value = value.tolist()
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return repr(value)


def _df_content_hash(df: pd.DataFrame, sort_by: str) -> str:
    """Stable content hash of a DataFrame, sorted by ``sort_by``.

    We sort to make the hash insensitive to row order (relevant when the
    same logical dataset comes in via different load paths). Object columns
    are serialized cell-by-cell first so container cells (ndarray / list /
    dict) hash instead of raising. The hash function is SHA-1 truncated to
    16 hex chars — long enough that collisions are negligible for the cache
    use case.
    """
    sorted_df = df.sort_values(sort_by).reset_index(drop=True)
    safe = sorted_df.copy()
    for col in safe.columns:
        if safe[col].dtype == object:
            safe[col] = safe[col].map(_stable_cell_repr)
    blob = pd.util.hash_pandas_object(safe, index=False).values.tobytes()
    return hashlib.sha1(blob).hexdigest()[:16]


def corpus_hash(corpus_df: pd.DataFrame) -> str:
    """Stable content hash for a corpus DataFrame (sorted by ``doc_id``)."""
    return _df_content_hash(corpus_df, sort_by="doc_id")


def qa_hash(qa_df: pd.DataFrame) -> str:
    """Stable content hash for a QA DataFrame (sorted by ``qid``)."""
    return _df_content_hash(qa_df, sort_by="qid")


# ---------------------------------------------------------------------------
# Parquet → JSONL materialization.
# ---------------------------------------------------------------------------


def materialize_qa_jsonl(qa_df: pd.DataFrame, out_path: str) -> None:
    """Write a rag-stack QA parquet to FlashRAG JSONL format.

    Column mapping:

      * ``qid`` → ``id``
      * ``query`` → ``question``
      * ``generation_gt`` (list[str]) → ``golden_answers`` (list[str])

    FlashRAG evaluates retrieval by answer-containment (golden_answers in the
    retrieved passage text), so no passage-id retrieval GT is carried — the
    deprecated ``retrieval_gt`` is intentionally dropped.

    Skips writing if ``out_path`` already exists — caller is responsible
    for cache invalidation when the QA content changes (typically via
    :func:`qa_hash` in the path).
    """
    if os.path.isfile(out_path):
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for _, row in qa_df.iterrows():
            golden = row["generation_gt"]
            if hasattr(golden, "tolist"):
                golden = golden.tolist()
            entry = {
                "id": str(row["qid"]),
                "question": str(row["query"]),
                "golden_answers": list(golden) if golden is not None else [],
                "metadata": {},
            }
            f.write(json.dumps(entry) + "\n")
    logger.info(f"Materialized FlashRAG QA JSONL → {out_path}")


def materialize_corpus_jsonl(corpus_df: pd.DataFrame, out_path: str) -> None:
    """Write a rag-stack corpus parquet to FlashRAG corpus JSONL format.

    Column mapping:

      * ``doc_id`` → ``id``
      * ``contents`` → ``contents``

    Other columns (``path``, ``start_end_idx``, ``metadata``) are dropped.
    Row order is preserved — FlashRAG's FAISS index aligns row N → corpus
    row N, so changing the order after building an index would silently
    corrupt retrieval. Caller MUST regenerate the index whenever this
    JSONL is regenerated (use :func:`corpus_hash` in the cache key).
    """
    if os.path.isfile(out_path):
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for _, row in corpus_df.iterrows():
            entry = {
                "id": str(row["doc_id"]),
                "contents": str(row["contents"]),
            }
            f.write(json.dumps(entry) + "\n")
    logger.info(f"Materialized FlashRAG corpus JSONL → {out_path}")


# ---------------------------------------------------------------------------
# FAISS index building (delegates to flashrag.retriever.index_builder).
# ---------------------------------------------------------------------------


def ensure_index(
    *,
    corpus_jsonl_path: str,
    retrieval_method: str,
    faiss_type: str,
    model_path: str,
    out_dir: str,
    batch_size: int = 512,
    use_fp16: bool = False,
    pooling_method: Optional[str] = None,
    max_length: int = 512,
) -> str:
    """Return the FAISS index path, building it if missing.

    FlashRAG's :class:`flashrag.retriever.index_builder.Index_Builder` runs
    the encoder over the corpus and writes a single ``.index`` file under
    ``out_dir`` named ``<retrieval_method>_<faiss_type>.index``. We check
    for the file first to keep optimizer reruns fast (the GPU encode is
    the slow step; subsequent FAISS-only variants reuse it via FlashRAG's
    ``save_embedding`` / ``embedding_path``).

    Args:
        corpus_jsonl_path: FlashRAG-format corpus JSONL.
        retrieval_method: short alias (``e5`` / ``bge`` / ``bm25`` / ...).
        faiss_type: FAISS factory string (``Flat`` / ``IVF1024,Flat`` /
            ``HNSW32`` / etc.). Passed directly to ``faiss.index_factory``.
        model_path: HuggingFace path or alias for the encoder.
        out_dir: cache directory under ``<project_dir>/_flashrag/indexes/<corpus_hash>/``.
        batch_size, use_fp16, pooling_method: forwarded to ``Index_Builder``.
        max_length: encoder max sequence length per passage (required by
            ``Index_Builder`` since the FlashRAG fork update; 512 = e5 limit).

    Returns:
        Path to the ``.index`` file (built or cached).
    """
    os.makedirs(out_dir, exist_ok=True)
    index_path = os.path.join(out_dir, f"{retrieval_method}_{faiss_type}.index")

    if os.path.isfile(index_path):
        logger.info(f"FlashRAG index cache hit: {index_path}")
        return index_path

    # Cache miss — build via FlashRAG's Index_Builder.
    from flashrag.retriever.index_builder import Index_Builder

    logger.info(
        f"FlashRAG index cache miss — building "
        f"{retrieval_method}_{faiss_type} into {out_dir}"
    )
    Index_Builder(
        retrieval_method=retrieval_method,
        model_path=model_path,
        corpus_path=corpus_jsonl_path,
        save_dir=out_dir,
        max_length=max_length,
        faiss_type=faiss_type,
        batch_size=batch_size,
        use_fp16=use_fp16,
        pooling_method=pooling_method,
        save_embedding=True,
    ).build_index()

    if not os.path.isfile(index_path):
        raise RuntimeError(
            f"Index_Builder did not produce expected file at {index_path}. "
            f"Check FlashRAG logs for build failure."
        )
    return index_path


def ensure_bm25_index(
    *,
    corpus_jsonl_path: str,
    out_dir: str,
    bm25_backend: str = "bm25s",
) -> str:
    """Return the BM25 index dir, building it (bm25s) if missing.

    FlashRAG's :class:`Index_Builder` writes the BM25 index under
    ``<out_dir>/bm25`` — that directory is what ``BM25Retriever`` loads. Cached
    by presence (the corpus hash is already baked into ``out_dir``, so a hit
    means this exact corpus was indexed before). Used by A-RAG's Keyword tool.
    """
    bm25_dir = os.path.join(out_dir, "bm25")
    if os.path.isdir(bm25_dir) and os.listdir(bm25_dir):
        logger.info(f"FlashRAG BM25 index cache hit: {bm25_dir}")
        return bm25_dir

    os.makedirs(out_dir, exist_ok=True)
    from flashrag.retriever.index_builder import Index_Builder

    logger.info(f"FlashRAG BM25 index cache miss — building into {bm25_dir}")
    Index_Builder(
        retrieval_method="bm25",
        model_path="bm25",  # unused for bm25 (no encoder), but required positionally
        corpus_path=corpus_jsonl_path,
        save_dir=out_dir,
        max_length=512,
        batch_size=512,
        use_fp16=False,
        bm25_backend=bm25_backend,
    ).build_index()

    if not (os.path.isdir(bm25_dir) and os.listdir(bm25_dir)):
        raise RuntimeError(
            f"BM25 Index_Builder did not produce {bm25_dir}. Check FlashRAG logs."
        )
    return bm25_dir
