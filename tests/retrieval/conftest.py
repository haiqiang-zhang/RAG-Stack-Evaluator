"""Shared fake dense retriever and tracing context for hybrid contracts."""

import asyncio
from types import SimpleNamespace

import pandas as pd
import pytest

from rag_stack.evaluation import _recording


@pytest.fixture
def vector_node(monkeypatch):
    from rag_stack_evaluator.static_rag_evaluator.nodes.semanticretrieval import vectordb

    rows = {
        "single": (["a", "a", "b"], [3.0, 2.0, 1.0]),
        "left": (["x", "x", "y"], [3.0, 2.0, 1.0]),
        "right": (["y", "z"], [4.0, 1.0]),
    }

    async def query(*, queries, top_k, **_kwargs):
        return ([list(rows[q][0]) for q in queries],
                [list(rows[q][1]) for q in queries])

    node = object.__new__(vectordb.VectorDB)
    node.vector_store = SimpleNamespace(embedding=SimpleNamespace(), query=query)
    node._embedding_replica_count = 1
    node._contents_by_doc_id = {key: "passage " + key for key in "abxyz"}
    node.corpus_df = pd.DataFrame({
        "doc_id": list(node._contents_by_doc_id),
        "contents": list(node._contents_by_doc_id.values()),
    })
    node.cast_to_run = lambda frame: frame["queries"].tolist()
    loops = []

    def event_loop():
        loop = asyncio.new_event_loop()
        loops.append(loop)
        return loop

    monkeypatch.setattr(vectordb, "get_event_loop", event_loop)
    yield node
    for loop in loops:
        loop.close()


@pytest.fixture
def recorder(monkeypatch):
    # Tokenization is unrelated to fusion profiles; no models are loaded.
    monkeypatch.setattr(_recording, "count_tokens_batch", lambda values, _model=None: [1] * len(values))
    value = _recording.TraceRecorder()
    token = _recording._current_recorder.set(value)
    yield value
    _recording._current_recorder.reset(token)
