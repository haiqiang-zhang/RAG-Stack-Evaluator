"""Shared-cache routing and explicit override contracts, isolated from user caches."""
import os
from pathlib import Path

import pytest

from rag_stack_evaluator.static_rag_evaluator import cache_paths
from rag_stack_evaluator.static_rag_evaluator import embedding_cache
from rag_stack_evaluator.static_rag_evaluator.static_rag_evaluator import (
    StaticRAGEvaluatorQualityOnly as _E,
)

@pytest.fixture(autouse=True)
def isolated_cache_root(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_STACK_CACHE_DIR", str(tmp_path / "shared-cache"))
    monkeypatch.delenv("RAG_STACK_EMBED_CACHE", raising=False)


def test_shared_root_routes_embedding_and_index_caches(tmp_path):
    root = tmp_path / "shared-cache"
    assert cache_paths.cache_root() == str(root)
    assert cache_paths.faiss_index_root() == str(root / "faiss")
    assert cache_paths.embeddings_dir() == str(root / "embeddings")
    assert embedding_cache.cache_dir() == str(root / "embeddings")


def test_legacy_embedding_override_does_not_relocate_indexes(tmp_path, monkeypatch):
    legacy = tmp_path / "legacy-embeddings"
    monkeypatch.setenv("RAG_STACK_EMBED_CACHE", str(legacy))
    assert cache_paths.embeddings_dir() == str(legacy)
    assert embedding_cache.cache_dir() == str(legacy)
    assert cache_paths.faiss_index_root() == str(tmp_path / "shared-cache" / "faiss")


def test_unconfigured_cache_root_follows_evaluator_installation(monkeypatch):
    monkeypatch.delenv("RAG_STACK_CACHE_DIR")
    installation = Path(cache_paths.__file__).resolve().parents[2]
    assert cache_paths.cache_root() == str(installation / ".cache" / "rag_stack")


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
