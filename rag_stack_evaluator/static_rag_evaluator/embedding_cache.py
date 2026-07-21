"""Global (rag_stack-wide) corpus-embedding cache.

The corpus embedding (``embedding_model.encode`` over every chunk) is the dominant
non-generation cost of a measured/static eval (~40-60s for a few-k-doc corpus) and
depends ONLY on ``(corpus contents, embedding model)`` — NOT on the FAISS index
params, the generator, the placement, or the project. With a fixed chunker and a
handful of embedding models there are only a few distinct embedding sets per
dataset, yet every case re-encodes them because the vectordb is rebuilt under a
fresh per-case ``${PROJECT_DIR}``.

This cache lifts the encoded VECTORS to a USER-LEVEL shared directory keyed by a
human-readable ``{dataset_name}__{embedding_id}__{content_sig}`` so any project /
case / run reuses them. Only the vectors are cached — the (cheap, param-varying)
FAISS index build stays per-case.

Design notes:
- Key is readable so ``ls`` tells you what's cached; ``content_sig`` (a hash of the
  exact ordered chunk contents) guarantees correctness — change the corpus or
  chunking and the sig changes → a new entry, the old one is never mis-served.
- Retrieval is automatic: the caller always holds the corpus it is about to encode,
  so it re-derives the same key and hits. You never look up by a bare hash.
- The cache must NEVER break evaluation: every I/O path is guarded; on ANY error we
  fall back to encoding live.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Callable, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Cache location now lives in the shared cache_paths module (one root for
# embeddings + faiss indexes, single RAG_STACK_CACHE_DIR override). The
# resolved path is byte-identical to the historical <repo>/.cache/rag_stack/
# embeddings, so the existing vector cache is preserved. RAG_STACK_EMBED_CACHE
# still overrides embeddings specifically (handled inside embeddings_dir).
from rag_stack_evaluator.static_rag_evaluator.cache_paths import embeddings_dir


def cache_dir() -> str:
    """Shared encoded-vector cache dir (see cache_paths.embeddings_dir)."""
    return embeddings_dir()


def _disabled() -> bool:
    return os.environ.get("RAG_STACK_EMBED_CACHE_DISABLE", "").strip().lower() in (
        "1", "true", "yes",
    )


def _sanitize(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "-._") else "-" for c in str(s))[:80]


def content_sig(contents: List[str], normalize: bool) -> str:
    """Order-sensitive hash of the exact chunk contents + normalize flag → 16 hex."""
    h = hashlib.sha1()
    h.update(f"n={int(bool(normalize))};count={len(contents)};".encode("utf-8"))
    for c in contents:
        # length-prefixed to avoid "ab"+"c" colliding with "a"+"bc"
        b = c.encode("utf-8", "ignore")
        h.update(len(b).to_bytes(8, "little"))
        h.update(b)
    return h.hexdigest()[:16]


def _entry_dir(dataset_name: Optional[str], embedding_id: str, sig: str) -> str:
    key = f"{_sanitize(dataset_name or 'corpus')}__{_sanitize(embedding_id or 'embedding')}__{sig}"
    return os.path.join(cache_dir(), key)


def get_or_encode(
    contents: List[str],
    encode_fn: Callable[[], "np.ndarray"],
    *,
    dataset_name: Optional[str] = None,
    embedding_id: Optional[str] = None,
    normalize: bool = True,
) -> "np.ndarray":
    """Return embeddings for ``contents``, from the global cache if present.

    On a miss, calls ``encode_fn()`` (the live encode), stores the result, and
    returns it. Any cache error is swallowed and falls back to ``encode_fn()`` —
    the cache is a pure speedup, never a correctness/availability dependency.
    """
    if _disabled() or not contents:
        return encode_fn()

    try:
        sig = content_sig(contents, normalize)
        d = _entry_dir(dataset_name, embedding_id or "embedding", sig)
        vpath = os.path.join(d, "vectors.npy")
        if os.path.exists(vpath):
            vecs = np.load(vpath)
            if vecs.shape[0] == len(contents):
                logger.info(
                    "[embed-cache] HIT %s (%d vecs, dim=%d) — skipping encode",
                    os.path.basename(d), vecs.shape[0],
                    vecs.shape[1] if vecs.ndim > 1 else -1,
                )
                return vecs
            logger.warning(
                "[embed-cache] stale entry %s (rows %d != %d) — re-encoding",
                os.path.basename(d), vecs.shape[0], len(contents),
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[embed-cache] read failed (%s) — encoding live", exc)
        return encode_fn()

    # Miss → encode live, then store (store failure is non-fatal).
    vecs = encode_fn()
    try:
        vecs = np.asarray(vecs)
        os.makedirs(d, exist_ok=True)
        # NB np.save appends ".npy" unless the name already ends in it — so the tmp
        # name MUST end in ".npy" or os.replace would chase a non-existent file.
        tmp = vpath + f".tmp.{os.getpid()}.npy"
        np.save(tmp, vecs)
        os.replace(tmp, vpath)
        meta = {
            "dataset_name": dataset_name,
            "embedding_id": embedding_id,
            "normalize": bool(normalize),
            "n_vectors": int(vecs.shape[0]),
            "dim": int(vecs.shape[1]) if vecs.ndim > 1 else None,
            "content_sig": sig,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(os.path.join(d, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
        logger.info(
            "[embed-cache] STORED %s (%d vecs) at %s",
            os.path.basename(d), int(vecs.shape[0]), cache_dir(),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[embed-cache] store failed (%s) — continuing", exc)
    return vecs
