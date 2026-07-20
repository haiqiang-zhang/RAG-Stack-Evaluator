"""Single source of truth for RAG-Stack's shared, content-addressed on-disk
caches.

Everything lands under ONE root so a single env override relocates them all
together. These are PROJECT DATA (encoded vectors, faiss indexes) keyed by a
content hash and shared across every run / seed — never per-project, never in
the user home (a msmarco warm once quietly dropped 14G+ into ~/.cache before
the vectors were pinned to the repo volume, 07-04).

Root: ``<repo>/.cache/rag_stack``  (override: ``RAG_STACK_CACHE_DIR``).

Sub-caches:
- ``embeddings_dir()`` — encoded vectors, keyed by (corpus, embedding model).
  Legacy override ``RAG_STACK_EMBED_CACHE`` still wins for back-compat.
- ``faiss_index_root()`` — built faiss indexes, keyed by
  ``<chunk_hash>/<index_params>`` (see StaticRagEvaluator._resolve_vectordb_paths).
  Global so every project/seed reuses one built index instead of rebuilding a
  project-local copy.
"""
import os

# <repo> is three levels up: this file is rag_stack/static_rag_evaluator/
# cache_paths.py, so dirname×3 → repo root. MUST match embedding_cache.py's
# historical derivation byte-for-byte or the existing 39G vector cache is lost.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def cache_root() -> str:
    """Shared cache root ``<repo>/.cache/rag_stack`` (override
    ``RAG_STACK_CACHE_DIR`` relocates ALL sub-caches together)."""
    return os.environ.get("RAG_STACK_CACHE_DIR") or os.path.join(
        _REPO_ROOT, ".cache", "rag_stack",
    )


def embeddings_dir() -> str:
    """Encoded-vector cache dir. ``RAG_STACK_EMBED_CACHE`` (legacy, embeddings-
    only) still takes precedence over the shared root."""
    return os.environ.get("RAG_STACK_EMBED_CACHE") or os.path.join(
        cache_root(), "embeddings",
    )


def faiss_index_root() -> str:
    """Shared faiss index root ``<cache_root>/faiss``. The per-index
    ``<chunk_hash>/<index_params>`` subtree is appended downstream."""
    return os.path.join(cache_root(), "faiss")
