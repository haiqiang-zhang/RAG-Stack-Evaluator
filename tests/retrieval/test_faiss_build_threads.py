"""faiss index BUILD runs on cpu_count-2 threads, then restores to 1.

Build (IVF kmeans/PQ, HNSW graph) is CPU-bound and outside the cost model's
search-latency measurement, so it is parallelised for wall-clock; search must
stay on its single-thread calibration anchor, so the build thread count must
never leak past the build.

Run: python -m pytest tests/retrieval/test_faiss_build_threads.py -q
"""
import os

import faiss
import numpy as np
import pytest


from rag_stack_evaluator.static_rag_evaluator.vectordb._faiss_threads import (
    build_num_threads,
    faiss_build_threads,
)


def _expected() -> int:
    return max(1, (os.cpu_count() or 3) - 2)


@pytest.mark.parametrize("cpu_count, expected", [(None, 1), (1, 1), (2, 1), (4, 2), (16, 14)])
def test_build_num_threads_is_cpu_minus_two(monkeypatch, cpu_count, expected):
    monkeypatch.setattr(os, "cpu_count", lambda: cpu_count)
    assert build_num_threads() == expected


def test_context_sets_then_restores_to_one():
    faiss.omp_set_num_threads(1)
    with faiss_build_threads() as n:
        assert n == _expected()
        assert faiss.omp_get_max_threads() == _expected()
    assert faiss.omp_get_max_threads() == 1


def test_restores_on_exception():
    faiss.omp_set_num_threads(1)
    with pytest.raises(RuntimeError):
        with faiss_build_threads():
            assert faiss.omp_get_max_threads() == _expected()
            raise RuntimeError("boom")
    assert faiss.omp_get_max_threads() == 1  # restored despite the raise


def test_restores_on_early_return():
    faiss.omp_set_num_threads(1)

    def build_and_return():
        with faiss_build_threads():
            return  # early return inside the context (the add_embedding shape)

    build_and_return()
    assert faiss.omp_get_max_threads() == 1


def test_real_ivf_build_leaves_threads_at_one(tmp_path):
    """End-to-end on the actual build path: after add_embedding the process
    is back on 1 thread so a subsequent unspecified query stays on-anchor."""

    from rag_stack_evaluator.static_rag_evaluator.vectordb.faiss_ivf import FaissIVF

    faiss.omp_set_num_threads(1)
    store = FaissIVF(
        embedding_model="mock", similarity_metric="l2", collection_name="t",
        path=str(tmp_path), embedding_dim=64, index_type="pq",
        nlist=4, M=8, nbits=4, nprobe=4, faiss_indexing_thread=1,
    )
    rng = np.random.default_rng(0)
    embs = rng.standard_normal((2000, 64)).astype("float32").tolist()
    store.add_embedding([str(i) for i in range(2000)], embs)
    assert faiss.omp_get_max_threads() == 1


def test_explicit_indexing_thread_overrides_default():
    from rag_stack_evaluator.static_rag_evaluator.vectordb._faiss_threads import build_num_threads
    assert build_num_threads(8) == 8              # explicit wins
    assert build_num_threads(None) == _expected() # None → cpu-2
    assert build_num_threads(0) == _expected()    # non-positive → default
    assert build_num_threads("x") == _expected()  # unparseable → default


def test_context_honors_explicit_then_restores():
    faiss.omp_set_num_threads(1)
    with faiss_build_threads(6) as n:
        assert n == 6 and faiss.omp_get_max_threads() == 6
    assert faiss.omp_get_max_threads() == 1


def test_ivf_store_threads_indexing_knob_through_build(tmp_path):
    """Constructor stores faiss_indexing_thread (via **kwargs) and add_embedding
    threads it into the build; META so it never enters the cache path."""
    from rag_stack_evaluator.static_rag_evaluator.vectordb.faiss_ivf import FaissIVF

    faiss.omp_set_num_threads(1)
    store = FaissIVF(
        embedding_model="mock", similarity_metric="l2", collection_name="t",
        path=str(tmp_path), embedding_dim=64, index_type="pq",
        nlist=4, M=8, nbits=4, nprobe=4, faiss_indexing_thread=5,
    )
    assert store._faiss_indexing_thread == 5
    rng = np.random.default_rng(0)
    store.add_embedding([str(i) for i in range(2000)],
                        rng.standard_normal((2000, 64)).astype("float32").tolist())
    assert faiss.omp_get_max_threads() == 1       # restored after build


def test_indexing_thread_is_meta_not_in_index_path():
    from rag_stack_evaluator.static_rag_evaluator.static_rag_evaluator import (
        StaticRAGEvaluatorQualityOnly as _E,
    )
    out = _E._resolve_vectordb_paths(
        [{"name": "i", "db_type": "faiss_ivf", "path": "/p/faiss",
          "M": 32, "nbits": 8, "nlist": 1024, "embedding_model": "mpnet",
          "faiss_indexing_thread": 12}],
        chunk_hash="h",
    )
    assert "faiss_indexing_thread" not in out[0]["path"]
    assert "indexing" not in out[0]["path"]
