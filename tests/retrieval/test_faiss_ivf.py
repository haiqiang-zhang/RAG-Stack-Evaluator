"""Deterministic CPU coverage of the immutable FAISS IVF-PQ store."""

import asyncio

import faiss
import numpy as np
import pytest

from rag_stack_evaluator.static_rag_evaluator.vectordb.faiss_ivf import FaissIVF


@pytest.fixture
def store_factory(tmp_path):
    def create(metric="l2"):
        return FaissIVF(
            embedding_model="mock", similarity_metric=metric,
            collection_name="test", path=str(tmp_path / metric),
            embedding_dim=64, index_type="pq", nlist=4, M=8,
            nbits=4, nprobe=4, faiss_indexing_thread=1,
        )
    return create


@pytest.fixture
def embeddings():
    # PQ requires at least 16 * 39 vectors; keep training local and bounded.
    return np.random.default_rng(7).standard_normal((640, 64)).astype(np.float32)


def test_index_roundtrip_and_immutable_store(store_factory, embeddings):
    store = store_factory()
    identifiers = [f"doc-{i}" for i in range(len(embeddings))]
    store.add_embedding(identifiers, embeddings.tolist())
    assert store.index.ntotal == len(identifiers)

    store.add_embedding(identifiers[:10], embeddings[:10].tolist())
    assert store.index.ntotal == len(identifiers)
    _, neighbors = store.index.search(embeddings[:1], 5)
    assert neighbors[0][0] == 0
    assert asyncio.run(store.is_exist(["doc-0", "missing", "doc-5"])) == [True, False, True]
    fetched = asyncio.run(store.fetch(["doc-0", "doc-1", "missing"]))
    assert len(fetched[0]) == len(fetched[1]) == 64
    assert fetched[2] == []
    with pytest.raises(NotImplementedError):
        asyncio.run(store.delete(["doc-0"]))

    restored = store_factory()
    assert restored.index.ntotal == len(identifiers)
    assert asyncio.run(restored.is_exist(["doc-0", "doc-639", "missing"])) == [True, True, False]
    assert len(asyncio.run(restored.fetch(["doc-2"]))[0]) == 64
    assert faiss.omp_get_max_threads() == 1


@pytest.mark.parametrize("metric", ["cosine", "ip"])
def test_similarity_metric_supports_index_and_search(store_factory, embeddings, metric):
    store = store_factory(metric)
    store.add_embedding([str(i) for i in range(len(embeddings))], embeddings.tolist())
    assert store.index.ntotal == len(embeddings)
    assert store.index.metric_type == faiss.METRIC_INNER_PRODUCT
    query = embeddings[:1].copy()
    if metric == "cosine":
        faiss.normalize_L2(query)
    scores, neighbors = store.index.search(query, 3)
    assert neighbors.shape == scores.shape == (1, 3)
    assert np.isfinite(scores).all()
    assert (neighbors >= 0).all()
