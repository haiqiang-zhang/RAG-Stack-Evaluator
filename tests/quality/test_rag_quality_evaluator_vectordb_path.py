"""Tests for StaticRAGEvaluatorQualityOnly vectordb path resolution and index caching."""

from rag_stack_evaluator.static_rag_evaluator.static_rag_evaluator import StaticRAGEvaluatorQualityOnly


def test_resolve_vectordb_paths_hashing():
    """Verify _resolve_vectordb_paths appends param-signature subdirectory."""
    configs_a = [{"name": "idx", "db_type": "faiss_ivf", "path": "/data/faiss",
                  "N": 1000000, "embedding_dim": 768, "nlist": 1024, "M": 32, "nbits": 8}]
    configs_b = [{"name": "idx", "db_type": "faiss_ivf", "path": "/data/faiss",
                  "N": 1000000, "embedding_dim": 768, "nlist": 2048, "M": 16, "nbits": 4}]

    resolved_a = StaticRAGEvaluatorQualityOnly._resolve_vectordb_paths(configs_a)
    resolved_b = StaticRAGEvaluatorQualityOnly._resolve_vectordb_paths(configs_b)
    resolved_a2 = StaticRAGEvaluatorQualityOnly._resolve_vectordb_paths(configs_a)

    path_a = resolved_a[0]["path"]
    path_b = resolved_b[0]["path"]
    path_a2 = resolved_a2[0]["path"]

    # Different params → different paths
    assert path_a != path_b
    # Same params → same path (deterministic)
    assert path_a == path_a2
    # Subdirectory is appended under the SHARED global faiss root (faiss
    # indexes are content-addressed and reused across runs, not project-local)
    from rag_stack_evaluator.static_rag_evaluator.cache_paths import faiss_index_root
    assert path_a.startswith(faiss_index_root() + "/")
    assert "M32" in path_a and "nlist1024" in path_a
    assert "M16" in path_b and "nlist2048" in path_b
    # N is in META_KEYS so should NOT appear in path
    assert "N1000000" not in path_a
    # embedding_dim IS in the path (not in META_KEYS)
    assert "embedding_dim768" in path_a

    print(f"path_a: {path_a}")
    print(f"path_b: {path_b}")


def test_resolve_vectordb_paths_embedding_model_is_index_param():
    """embedding_model is an index-structure param (different models → different
    vectors → different dirs), and the chunk_hash segment ("none" by default) is
    always inserted — even for a metadata-only store like Chroma."""
    configs = [{"name": "chroma", "db_type": "chroma", "path": "/data/chroma",
                "collection_name": "test", "embedding_model": "mpnet"}]
    resolved = StaticRAGEvaluatorQualityOnly._resolve_vectordb_paths(configs)

    # chunk_hash segment is always inserted; embedding_model enters the signature.
    assert resolved[0]["path"] == "/data/chroma/none/embedding_modelmpnet"
    print(f"Chroma path: {resolved[0]['path']}")


def test_resolve_vectordb_paths_only_metadata_keys():
    """A config with ONLY metadata keys (no index-structure params at all) gets
    just the chunk_hash segment appended — no param signature."""
    configs = [{"name": "chroma", "db_type": "chroma", "path": "/data/chroma",
                "collection_name": "test"}]
    resolved = StaticRAGEvaluatorQualityOnly._resolve_vectordb_paths(configs)

    # No non-metadata params → base path + chunk_hash segment only.
    assert resolved[0]["path"] == "/data/chroma/none"
    print(f"Metadata-only path: {resolved[0]['path']}")


def test_resolve_vectordb_paths_embedding_dtype_bytes_is_metadata():
    """Communication-accounting metadata must not split byte-identical indexes."""
    for db_type, build_params in (
        ("faiss_ivf", {"index_type": "flat", "nlist": 1024}),
        ("faiss_hnsw", {"M": 32, "ef_construction": 200}),
    ):
        base = {
            "name": "idx",
            "db_type": db_type,
            "embedding_model": "huggingface_all_mpnet_base_v2",
            "embedding_dim": 768,
            **build_params,
        }
        without_dtype = StaticRAGEvaluatorQualityOnly._resolve_vectordb_paths(
            [base], chunk_hash="corpus-hash",
        )[0]["path"]
        with_dtype = StaticRAGEvaluatorQualityOnly._resolve_vectordb_paths(
            [{**base, "embedding_dtype_bytes": 4}], chunk_hash="corpus-hash",
        )[0]["path"]

        assert with_dtype == without_dtype
        assert "embedding_dtype_bytes" not in with_dtype


def test_resolve_vectordb_paths_does_not_mutate_input():
    """Verify _resolve_vectordb_paths does not mutate the input configs."""
    configs = [{"name": "idx", "db_type": "faiss_ivf", "path": "/data/faiss",
                "nlist": 1024, "M": 16}]
    original_path = configs[0]["path"]
    StaticRAGEvaluatorQualityOnly._resolve_vectordb_paths(configs)
    assert configs[0]["path"] == original_path


if __name__ == "__main__":
    print("\n--- test_resolve_vectordb_paths_hashing ---")
    test_resolve_vectordb_paths_hashing()
    print("\n--- test_resolve_vectordb_paths_embedding_model_is_index_param ---")
    test_resolve_vectordb_paths_embedding_model_is_index_param()
    print("\n--- test_resolve_vectordb_paths_only_metadata_keys ---")
    test_resolve_vectordb_paths_only_metadata_keys()
    print("\n--- test_resolve_vectordb_paths_does_not_mutate_input ---")
    test_resolve_vectordb_paths_does_not_mutate_input()
    print("\nAll tests passed!")
