"""Shared cache root + faiss index global-ization.

One root (``<repo>/.cache/rag_stack``, override ``RAG_STACK_CACHE_DIR``) holds
both the encoded-vector cache and the built faiss indexes. The embedding path
must stay byte-identical to its historical location (the existing multi-GB
vector cache must survive), and faiss indexes must resolve UNDER the shared
root (was project-local → rebuilt per run) while non-faiss stores keep their
configured path.

Run: python -m pytest tests/test_cache_paths.py -q
"""
import os
import warnings

warnings.filterwarnings("ignore")

from rag_stack.static_rag_evaluator import cache_paths
from rag_stack.static_rag_evaluator import embedding_cache
from rag_stack.static_rag_evaluator.static_rag_evaluator import (
    StaticRAGEvaluatorQualityOnly as _E,
)

_REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(cache_paths.__file__)))
)
_CACHE_ROOT = os.environ.get("RAG_STACK_CACHE_DIR") or os.path.join(
    _REPO, ".cache", "rag_stack",
)


def test_embedding_path_byte_identical_to_historical():
    """The 39G vector cache lives at <repo>/.cache/rag_stack/embeddings — the
    refactor must resolve to exactly that, or the cache is silently orphaned."""
    expect = os.path.join(_CACHE_ROOT, "embeddings")
    assert cache_paths.embeddings_dir() == expect
    assert embedding_cache.cache_dir() == expect  # the public accessor agrees


def test_shared_root_and_faiss_subdir():
    assert cache_paths.cache_root() == _CACHE_ROOT
    assert cache_paths.faiss_index_root() == os.path.join(
        _CACHE_ROOT, "faiss",
    )


def test_env_override_relocates_all_subcaches_together():
    prev = os.environ.get("RAG_STACK_CACHE_DIR")
    try:
        os.environ["RAG_STACK_CACHE_DIR"] = "/tmp/xcache"
        # embeddings honors its own legacy override first; clear it here
        prev_emb = os.environ.pop("RAG_STACK_EMBED_CACHE", None)
        assert cache_paths.cache_root() == "/tmp/xcache"
        assert cache_paths.faiss_index_root() == "/tmp/xcache/faiss"
        assert cache_paths.embeddings_dir() == "/tmp/xcache/embeddings"
    finally:
        if prev is None:
            os.environ.pop("RAG_STACK_CACHE_DIR", None)
        else:
            os.environ["RAG_STACK_CACHE_DIR"] = prev
        if prev_emb is not None:
            os.environ["RAG_STACK_EMBED_CACHE"] = prev_emb


def test_faiss_indexes_redirect_to_global_root():
    cfgs = [
        {"name": "i", "db_type": "faiss_ivf", "path": "/proj/data/faiss",
         "M": 32, "nbits": 8, "nlist": 1024, "embedding_model": "BAAI/bge-small"},
        {"name": "h", "db_type": "faiss_hnsw", "path": "/proj/data/faiss",
         "M": 16, "embedding_model": "mpnet"},
    ]
    out = _E._resolve_vectordb_paths(cfgs, chunk_hash="abc123")
    root = cache_paths.faiss_index_root()
    for o in out:
        assert o["path"].startswith(root), o
        assert "/abc123/" in o["path"] + "/"       # chunk_hash segment kept
    assert "bge-small" in out[0]["path"]           # index-param signature kept


def test_non_faiss_store_keeps_project_local_path():
    out = _E._resolve_vectordb_paths(
        [{"name": "c", "db_type": "chroma", "path": "/proj/data/chroma"}],
        chunk_hash="abc123",
    )
    assert out[0]["path"] == "/proj/data/chroma/abc123"


def test_redirect_subtree_matches_old_project_local_layout():
    """The <chunk_hash>/<params> subtree is IDENTICAL to the old project-local
    one (only the root changed) — one global namespace; legacy uuid4-id
    indexes were converted IN PLACE (meta.json rewrite) by the one-off
    merge_faiss_to_global_cache migration (script since removed)."""
    cfg = {"name": "i", "db_type": "faiss_ivf", "path": "/proj/data/faiss",
           "M": 32, "nbits": 8, "nlist": 1024, "embedding_model": "mpnet"}
    new = _E._resolve_vectordb_paths([dict(cfg)], chunk_hash="h9")[0]["path"]
    subtree = os.path.relpath(new, cache_paths.faiss_index_root())
    assert subtree == os.path.join("h9", "M32_embedding_modelmpnet_nbits8_nlist1024")


def test_faiss_resolves_without_a_config_path():
    """The decoupling: faiss indexes resolve under the global root even when
    the config declares NO ``path`` — the field is no longer load-bearing for
    faiss (it used to gate the whole resolution). Non-faiss still need one."""
    out = _E._resolve_vectordb_paths(
        [{"name": "i", "db_type": "faiss_ivf", "M": 32, "nbits": 8,
          "nlist": 1024, "embedding_model": "mpnet"}],   # NO "path"
        chunk_hash="h9",
    )
    assert out[0]["path"] == os.path.join(
        cache_paths.faiss_index_root(), "h9",
        "M32_embedding_modelmpnet_nbits8_nlist1024")

    # a non-faiss store WITHOUT a path is left unresolved (nothing to key on)
    out2 = _E._resolve_vectordb_paths(
        [{"name": "c", "db_type": "chroma"}], chunk_hash="h9")
    assert "path" not in out2[0]
