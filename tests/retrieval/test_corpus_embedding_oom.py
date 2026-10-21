"""Corpus ingestion preserves content and only retries CUDA allocation failures."""

import weakref
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from rag_stack_evaluator.static_rag_evaluator.nodes.semanticretrieval import vectordb


def test_oom_retries_preserve_inputs_options_and_release_failed_forward(monkeypatch):
    contents = ["long passage", "short", "middle passage"]
    calls, released, failed_allocations = [], [], []

    class Allocation:
        pass

    def encode(texts, **options):
        calls.append((list(texts), dict(options)))
        if options["batch_size"] > 2:
            allocation = Allocation()
            failed_allocations.append(weakref.ref(allocation))
            raise torch.cuda.OutOfMemoryError("simulated CUDA allocation failure")
        return np.array([[len(text), index] for index, text in enumerate(texts)], dtype=np.float32)

    def empty_cache():
        released.append(all(reference() is None for reference in failed_allocations))

    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    result = vectordb._encode_corpus_with_oom_backoff(
        SimpleNamespace(encode=encode), contents, batch_size=9, normalize=True,
    )
    assert [options["batch_size"] for _, options in calls] == [9, 4, 2]
    assert all(texts == contents for texts, _ in calls)
    assert all(options["normalize_embeddings"] is True for _, options in calls)
    assert all(options["show_progress_bar"] is True for _, options in calls)
    np.testing.assert_array_equal(result, [[12, 0], [5, 1], [14, 2]])
    assert released == [True, True]


def test_successful_original_batch_is_unchanged(monkeypatch):
    sentinel = object()
    calls = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: pytest.fail("Unexpected CUDA cleanup"))

    def encode(texts, **options):
        calls.append((texts, options))
        return sentinel

    contents = ["unchanged"]
    result = vectordb._encode_corpus_with_oom_backoff(
        SimpleNamespace(encode=encode), contents, batch_size=100, normalize=False,
    )
    assert result is sentinel
    assert calls == [(contents, {"batch_size": 100, "normalize_embeddings": False, "show_progress_bar": True})]


def test_non_oom_failure_is_not_retried(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: pytest.fail("Unexpected CUDA cleanup"))

    def encode(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("unrelated model error")

    with pytest.raises(RuntimeError, match="unrelated model error"):
        vectordb._encode_corpus_with_oom_backoff(
            SimpleNamespace(encode=encode), ["text"], batch_size=8, normalize=True,
        )
    assert calls == [1]


def test_oom_at_batch_one_propagates_without_infinite_retry(monkeypatch):
    batches = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    def encode(*args, **kwargs):
        batches.append(kwargs["batch_size"])
        raise torch.cuda.OutOfMemoryError("even one passage does not fit")

    with pytest.raises(torch.cuda.OutOfMemoryError, match="even one passage"):
        vectordb._encode_corpus_with_oom_backoff(
            SimpleNamespace(encode=encode), ["text"], batch_size=3, normalize=True,
        )
    assert batches == [3, 1]


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_invalid_batch_is_rejected(batch_size):
    with pytest.raises(ValueError, match="positive integer"):
        vectordb._encode_corpus_with_oom_backoff(
            None, ["text"], batch_size=batch_size, normalize=True,
        )


def test_ingestion_caches_and_indexes_only_complete_ordered_vectors(monkeypatch):
    writes, cached, batches = [], [], []

    def encode(texts, **kwargs):
        batches.append(kwargs["batch_size"])
        if kwargs["batch_size"] > 2:
            raise torch.cuda.OutOfMemoryError("too large")
        return np.array([[float(text)] for text in texts])

    def get_or_encode(contents, encode_fn, **kwargs):
        vectors = encode_fn()
        cached.append((contents, vectors.copy(), kwargs))
        return vectors

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(vectordb.embedding_cache, "get_or_encode", get_or_encode)
    store = SimpleNamespace(
        embedding_batch=4,
        embedding=SimpleNamespace(_model=SimpleNamespace(encode=encode), normalize=True),
        add_embedding=lambda ids, vectors: writes.append((list(ids), vectors.copy())),
    )
    corpus = pd.DataFrame({"doc_id": ["c", "a", "b"], "contents": ["3", "1", "2"]})
    vectordb.vectordb_ingest_huggingface(store, corpus, dataset_name="unit", embedding_id="fake")
    assert batches == [4, 2]
    assert len(cached) == len(writes) == 1
    assert cached[0][0] == ["3", "1", "2"]
    assert writes[0][0] == ["c", "a", "b"]
    np.testing.assert_array_equal(writes[0][1], [[3], [1], [2]])
