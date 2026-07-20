"""CPU-only contracts for measured semantic-retrieval encoder DP."""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from rag_stack.static_rag_evaluator import LazyInit
from rag_stack.static_rag_evaluator.embedding.data_parallel import (
    DataParallelEmbedding,
)
from rag_stack.static_rag_evaluator.nodes.semanticretrieval.vectordb import (
    VectorDB,
)


class _FakeEmbedding:
    def __init__(self, device: str, barrier: threading.Barrier | None = None):
        self.device = device
        self.barrier = barrier
        self.embed_batch_size = None
        self.calls = []

    def get_text_embedding_batch(self, texts, *args, **kwargs):
        del args, kwargs
        self.calls.append((threading.get_ident(), list(texts)))
        if self.barrier is not None:
            # A sequential dispatcher times out here, so passing the test also
            # proves both replica forwards overlap rather than merely shard.
            self.barrier.wait(timeout=1.0)
        return [(self.device, text) for text in texts]


def test_embedding_dp_builds_independent_replica_per_device(monkeypatch):
    from rag_stack.static_rag_evaluator.embedding import base as embedding_base

    class FakeHF:
        def __init__(self, *, device=None, marker=None):
            self.device = device
            self.marker = marker

    marker = object()
    monkeypatch.setattr(embedding_base, "_HF_EMBEDDING_CLS", FakeHF)
    monkeypatch.setitem(
        embedding_base.embedding_models,
        "fake_hf",
        LazyInit(FakeHF, marker=marker),
    )
    first = FakeHF(device="cuda:0", marker=marker)

    replicas = embedding_base.build_huggingface_embedding_replicas(
        "fake_hf",
        ["cuda:0", "cuda:1", "cuda:2"],
        first_replica=first,
    )

    assert replicas[0] is first
    assert [replica.device for replica in replicas] == [
        "cuda:0", "cuda:1", "cuda:2",
    ]
    assert len({id(replica) for replica in replicas}) == 3


def test_embedding_dp_splits_concurrently_and_preserves_flattened_order():
    barrier = threading.Barrier(2)
    replicas = [
        _FakeEmbedding("cuda:0", barrier),
        _FakeEmbedding("cuda:1", barrier),
    ]
    embedding = DataParallelEmbedding(replicas, ["cuda:0", "cuda:1"])
    embedding.embed_batch_size = 3

    result = embedding.get_text_embedding_batch(["a", "b", "c", "d", "e"])

    assert result == [
        ("cuda:0", "a"),
        ("cuda:0", "b"),
        ("cuda:0", "c"),
        ("cuda:1", "d"),
        ("cuda:1", "e"),
    ]
    assert [replica.embed_batch_size for replica in replicas] == [3, 3]
    assert replicas[0].calls[0][1] == ["a", "b", "c"]
    assert replicas[1].calls[0][1] == ["d", "e"]
    assert replicas[0].calls[0][0] != replicas[1].calls[0][0]


class _FakeVectorStore:
    def __init__(self, replica_count: int):
        replicas = [_FakeEmbedding(f"cuda:{index}") for index in range(replica_count)]
        self.embedding = (
            replicas[0]
            if replica_count == 1
            else DataParallelEmbedding(
                replicas, [f"cuda:{index}" for index in range(replica_count)]
            )
        )
        self.query_calls = []
        self.encoded = []

    async def query(self, queries, top_k, **kwargs):
        del top_k, kwargs
        self.query_calls.append(list(queries))
        self.encoded.append(self.embedding.get_text_embedding_batch(queries))
        return (
            [[f"doc:{query}"] for query in queries],
            [[1.0] for _ in queries],
        )


def _vector_node(replica_count: int) -> VectorDB:
    node = object.__new__(VectorDB)
    node.vector_store = _FakeVectorStore(replica_count)
    node.embedding_model = node.vector_store.embedding
    node._embedding_replica_count = replica_count
    return node


def test_dp_mqe_flattens_in_order_then_runs_one_vector_search():
    node = _vector_node(replica_count=2)
    ids, scores = node._pure(
        [["q0-a", "q0-b"], ["q1-a", "q1-b"]],
        top_k=2,
        embedding_batch=1,
    )

    assert node.vector_store.query_calls == [["q0-a", "q0-b", "q1-a", "q1-b"]]
    assert node.vector_store.encoded == [[
        ("cuda:0", "q0-a"),
        ("cuda:0", "q0-b"),
        ("cuda:1", "q1-a"),
        ("cuda:1", "q1-b"),
    ]]
    assert ids == [
        ["doc:q0-a", "doc:q0-b"],
        ["doc:q1-a", "doc:q1-b"],
    ]
    assert len(scores) == 2 and all(len(row) == 2 for row in scores)


def test_single_replica_keeps_legacy_query_chunk_boundaries():
    node = _vector_node(replica_count=1)
    node._pure(
        [["q0-a", "q0-b"], ["q1-a", "q1-b"]],
        top_k=2,
        embedding_batch=1,
    )

    assert node.vector_store.query_calls == [
        ["q0-a"], ["q0-b"], ["q1-a"], ["q1-b"],
    ]


def test_faiss_timing_wraps_one_parallel_encode_and_one_search():
    from rag_stack.static_rag_evaluator.vectordb.base import (
        get_retrieval_timings,
        reset_retrieval_timings,
    )
    from rag_stack.static_rag_evaluator.vectordb.faiss_hnsw import FaissHNSW

    barrier = threading.Barrier(2)

    class VectorEmbedding(_FakeEmbedding):
        def get_text_embedding_batch(self, texts, *args, **kwargs):
            values = super().get_text_embedding_batch(texts, *args, **kwargs)
            time.sleep(0.005)
            return [[float(text), float(text) + 1.0] for _, text in values]

    class FakeIndex:
        ntotal = 4

        def __init__(self):
            self.hnsw = SimpleNamespace(efSearch=0)
            self.search_calls = 0

        def search(self, vectors, top_k):
            self.search_calls += 1
            rows = len(vectors)
            return (
                np.zeros((rows, top_k), dtype=np.float32),
                np.zeros((rows, top_k), dtype=np.int64),
            )

    store = object.__new__(FaissHNSW)
    store.embedding = DataParallelEmbedding(
        [VectorEmbedding("cuda:0", barrier), VectorEmbedding("cuda:1", barrier)],
        ["cuda:0", "cuda:1"],
    )
    store.similarity_metric = "l2"
    store.embedding_dim = 2
    store.index = FakeIndex()
    store._idx_to_id = {0: "doc-0"}

    reset_retrieval_timings()
    ids, _ = asyncio.run(store.query(["0", "1", "2", "3"], top_k=1))
    timings = get_retrieval_timings()

    assert ids == [["doc-0"]] * 4
    assert store.index.search_calls == 1
    assert timings["encode_s"] > 0.0
    assert timings["encode_active_s"] >= timings["encode_s"]
    assert timings["search_s"] >= 0.0
    assert timings["vectorsearch_active_s"] >= timings["search_s"]


def test_semantic_retrieval_aggregate_cap_is_per_replica_batch_times_width():
    from rag_stack.static_rag_evaluator.measured import serving_runtime as sr

    system_config = {
        "batch_size_request": 1,
        "measured_load_concurrency": 2,
        "measured_warmup_queries": 0,
        "measured_queries": 1,
        "batching": {"dynamic_timeout_s": 0.01},
        "layout": {
            "performance_stages": {
                "semantic_retrieval_encode": {
                    "kind": "gpu",
                    "engine": "semantic_retrieval",
                    "devices": ["cuda:0", "cuda:1"],
                    "num_chips": 2,
                },
            },
            "engines": {
                "semantic_retrieval": {
                    "devices": ["cuda:0", "cuda:1"],
                    "num_chips": 2,
                },
                "generator": {
                    "pd_serving": "collocated_pd",
                    "devices": ["cuda:0"],
                    "num_chips": 1,
                    "tp": 1,
                    "pp": 1,
                },
            },
            "resource_groups": [],
        },
    }
    stage = {
        "stage": "semantic_retrieval",
        "node": SimpleNamespace(stage="semantic_retrieval"),
        "params": {},
        "instance": object(),
    }
    runtime = sr.MeasuredServingRuntime(
        owner=SimpleNamespace(),
        stages=[stage],
        qa_data=pd.DataFrame({"query": ["q"]}),
        node_lines={},
        system_config=system_config,
        config={"pipeline_runtime": {"rag_dataflow": "sequential"}},
    )

    async def scenario():
        services = runtime._build_services()
        try:
            assert len(services) == 1
            assert services[0].batch_size == 2
        finally:
            await asyncio.gather(*(service.close() for service in services))

    asyncio.run(scenario())
