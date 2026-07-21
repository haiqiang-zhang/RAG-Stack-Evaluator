import sys
import threading
from types import SimpleNamespace

import pytest

from rag_stack_evaluator.static_rag_evaluator import recording


@pytest.fixture
def isolated_tokenizer_cache():
    """Keep fake tokenizer objects from leaking into another CPU test."""
    recording._tokenizer_for.cache_clear()
    recording._load_local_tokenizer.cache_clear()
    yield
    recording._tokenizer_for.cache_clear()
    recording._load_local_tokenizer.cache_clear()


def _record(recorder, qid, stage="generator"):
    recorder.record(
        qid,
        stage,
        input_tokens=1,
        output_tokens=1,
    )


def test_trace_tokenizer_prefers_real_model_and_is_strictly_local(
    monkeypatch,
    isolated_tokenizer_cache,
):
    calls = []
    tokenizer = object()

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            calls.append((name, kwargs))
            if kwargs != {"local_files_only": True}:
                raise AssertionError("trace tokenizer attempted a network-capable load")
            if name != "local/real-model":
                raise OSError(name)
            return tokenizer

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
    )
    from rag_stack import model_map

    monkeypatch.setattr(
        model_map,
        "resolve_tokenizer_name",
        lambda model_id: (
            "local/real-model" if model_id == "model-alias" else model_id
        ),
    )

    assert recording._tokenizer_for("model-alias") is tokenizer
    assert calls == [("local/real-model", {"local_files_only": True})]


def test_unknown_trace_model_never_networks_and_reuses_local_fallback(
    monkeypatch,
    isolated_tokenizer_cache,
):
    calls = []
    fallback = object()

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            calls.append((name, kwargs))
            if kwargs != {"local_files_only": True}:
                raise AssertionError("trace tokenizer attempted a network-capable load")
            if name == "gpt2":
                return fallback
            raise OSError(name)

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
    )

    assert recording._tokenizer_for("faiss_ivf_index") is fallback
    assert recording._tokenizer_for("another_component_id") is fallback
    assert calls == [
        ("faiss_ivf_index", {"local_files_only": True}),
        ("gpt2", {"local_files_only": True}),
        ("another_component_id", {"local_files_only": True}),
    ]


def test_quality_only_default_keeps_first_seen_order():
    recorder = recording.TraceRecorder()
    _record(recorder, "q2")
    _record(recorder, "q1")
    _record(recorder, "q2", "passage_reranker")

    assert recorder.qids == ("q2", "q1")
    assert [
        [entry["stage"] for entry in query_trace]
        for query_trace in recorder.to_trace_v1()
    ] == [["generator", "passage_reranker"], ["generator"]]
    performance_fields = {
        "qps", "latency_s", "service_s", "queue_wait_s", "batch_size",
        "request_send_ts", "first_token_ts", "last_token_ts",
    }
    assert all(
        not (set(call) & performance_fields)
        for query_trace in recorder.to_trace_v1()
        for call in query_trace
    )


def test_quality_only_evaluate_forwards_pipeline_envelope_by_identity():
    """evaluate() must return the SAME envelope object _run_pipeline built
    (and already fired through on_trace_ready) — the controller's judge/CM
    overlap matches by identity."""
    from rag_stack.rag_ir import make_quality_trace_envelope
    from rag_stack_evaluator.static_rag_evaluator.static_rag_evaluator import (
        StaticRAGEvaluatorQualityOnly,
    )

    envelope = make_quality_trace_envelope(
        [[{
            "stage": "generator",
            "input_tokens": 3,
            "output_tokens": 2,
            "input_bytes": 17,
            "output_bytes": 9,
            "step_idx": 0,
        }]],
        question_ids=["0"],
    )
    evaluator = object.__new__(StaticRAGEvaluatorQualityOnly)
    evaluator._run_pipeline = lambda *args, **kwargs: SimpleNamespace(
        quality={"exact_match": 0.5},
        trace=envelope,
    )

    result = evaluator.evaluate({})

    assert result == {"exact_match": 0.5, "__execution_dag__": envelope}
    assert result["__execution_dag__"] is envelope
    assert "performance" not in result
    assert "measured_gt_admissibility" not in result


def test_publish_quality_trace_fires_hook_with_the_shipped_envelope():
    """The on_trace_ready hook argument IS the returned envelope object, the
    envelope validates against the frozen v2 contract, question ids are the
    recorder's permanent qids (dataset row positions), and bytes ride along."""
    from rag_stack.rag_ir import validate_quality_trace_envelope
    from rag_stack_evaluator.static_rag_evaluator.static_rag_evaluator import (
        _publish_quality_trace,
    )

    trace = [
        [
            {"stage": "semantic_retrieval_encode", "input_tokens": 5,
             "output_tokens": 0, "input_bytes": 21, "output_bytes": 0,
             "step_idx": 0, "model_id": "mpnet"},
            {"stage": "generator", "input_tokens": 50, "output_tokens": 7,
             "input_bytes": 200, "output_bytes": 30, "step_idx": 1,
             "model_id": "qwen"},
        ],
        [
            {"stage": "generator", "input_tokens": 40, "output_tokens": 6,
             "input_bytes": 160, "output_bytes": 25, "step_idx": 0,
             "model_id": "qwen"},
        ],
    ]
    hook_calls = []
    envelope = _publish_quality_trace(trace, (0, 1), 2, hook_calls.append)

    assert hook_calls and hook_calls[0] is envelope
    validate_quality_trace_envelope(envelope)
    assert [q["question_id"] for q in envelope["queries"]] == ["0", "1"]
    assert envelope["queries"][0]["calls"][0]["input_bytes"] == 21

    # Cardinality is the producer's responsibility: a partial trace raises.
    with pytest.raises(RuntimeError, match="does not cover every dataset"):
        _publish_quality_trace(trace[:1], (0,), 2, None)

    # No recorded calls → None (the caller falls back to legacy CM payloads).
    assert _publish_quality_trace(None, (), 2, None) is None
    assert _publish_quality_trace([], (), 0, None) is None


def test_select_qids_is_ordered_atomic_and_frozen():
    recorder = recording.TraceRecorder()
    for qid in ("warmup", "complete-b", "complete-a", "post-window"):
        _record(recorder, qid)

    with pytest.raises(ValueError, match="duplicate"):
        recorder.select_qids(["complete-a", "complete-a"])
    assert recorder.qids == ("warmup", "complete-b", "complete-a", "post-window")

    with pytest.raises(KeyError, match="never recorded"):
        recorder.select_qids(["missing"])
    assert recorder.qids == ("warmup", "complete-b", "complete-a", "post-window")

    selected = recorder.select_qids(["complete-a", "complete-b"])
    assert selected == ("complete-a", "complete-b")
    assert recorder.qids == selected
    _record(recorder, "late-worker")
    assert recorder.qids == selected


def test_select_serializes_with_concurrent_late_writer():
    recorder = recording.TraceRecorder()
    _record(recorder, "complete")
    writer_ready = threading.Event()

    def late_write():
        writer_ready.set()
        _record(recorder, "late-worker")

    # Force the writer to queue on the recorder lock. select_qids re-enters
    # that lock, freezes and projects, then the writer observes frozen state.
    with recorder._lock:
        writer = threading.Thread(target=late_write)
        writer.start()
        assert writer_ready.wait(timeout=1.0)
        recorder.select_qids(["complete"])
    writer.join(timeout=1.0)

    assert not writer.is_alive()
    assert recorder.qids == ("complete",)


def test_discard_qids_preserves_survivor_order_and_fails_closed():
    recorder = recording.TraceRecorder()
    for qid in ("q0", "q1", "q2", "q3"):
        _record(recorder, qid)

    with pytest.raises(KeyError, match="never recorded"):
        recorder.discard_qids(["q1", "missing"])
    assert recorder.qids == ("q0", "q1", "q2", "q3")

    assert recorder.discard_qids(["q3", "q1"]) == ("q1", "q3")
    assert recorder.qids == ("q0", "q2")


def test_discard_qids_does_not_rebuild_retained_population():
    recorder = recording.TraceRecorder()
    for index in range(2_000):
        _record(recorder, f"q{index}")

    backing = recorder._by_qid
    for index in range(1_500):
        assert recorder.discard_qids([f"q{index}"]) == (f"q{index}",)
        # Regression guard for the former O(retained qids) dict rebuild on
        # every measured completion.
        assert recorder._by_qid is backing

    assert len(recorder) == 500
    assert recorder.qids[:2] == ("q1500", "q1501")


def test_projected_readout_defers_tokenization_and_prunes_peers_untokenized(monkeypatch):
    """Deferred token counting still happens ONLY at readout (after the
    measured services closed) — and with the padding machinery deleted,
    unselected closed-loop peers are pruned first and never tokenized."""
    token_counts = {"winner text": 7, "peer text": 11}
    counted_batches = []

    def count_batch(texts, model_id=None):
        counted_batches.append(list(texts))
        return [token_counts[text] for text in texts]

    monkeypatch.setattr(
        recording,
        "count_tokens_batch",
        count_batch,
    )
    recorder = recording.TraceRecorder()
    recorder.record_batch(
        "passage_reranker",
        qids=["selected", "unselected"],
        input_tokens=[0, 0],
        pending_in_texts=["winner text", "peer text"],
    )

    recorder.select_qids(["selected"])
    assert counted_batches == []  # selection runs before measured services close
    trace = recorder.to_trace_v1()

    assert counted_batches == [["winner text"]]
    assert trace[0][0]["input_tokens"] == 7


def test_deferred_trace_finalization_uses_bounded_output_identical_batches(
    monkeypatch,
):
    """A large retained ReAct cohort must not become one unbounded batch."""
    limit = recording._TRACE_FINALIZE_BATCH_SIZE
    n_queries = 2 * limit + 7
    token_batches = []
    byte_batches = []

    def count_tokens(texts, model_id=None):
        assert model_id == "test-model"
        assert 0 < len(texts) <= limit
        token_batches.append(list(texts))
        return [len(text) + 100 for text in texts]

    def count_bytes(texts):
        assert 0 < len(texts) <= limit
        byte_batches.append(list(texts))
        return [len(text.encode("utf-8")) for text in texts]

    monkeypatch.setattr(recording, "count_tokens_batch", count_tokens)
    monkeypatch.setattr(recording, "count_bytes_batch", count_bytes)

    qids = [f"q{index}" for index in range(n_queries)]
    inputs = [("input", str(index)) for index in range(n_queries)]
    outputs = [f"output-{index}" for index in range(n_queries)]
    recorder = recording.TraceRecorder()
    recorder.record_batch(
        "generator",
        qids=qids,
        input_tokens=[0] * n_queries,
        output_tokens=[0] * n_queries,
        model_id="test-model",
        pending_in_texts=inputs,
        pending_out_texts=outputs,
        pending_out_needs_tokens=True,
    )

    trace = recorder.to_trace_v1()

    assert len(token_batches) > 1
    assert len(byte_batches) > 1
    assert sum(map(len, token_batches)) == 2 * n_queries
    assert sum(map(len, byte_batches)) == 2 * n_queries
    for index, query_trace in enumerate(trace):
        call = query_trace[0]
        input_text = f"input {index}"
        output_text = f"output-{index}"
        assert call["input_tokens"] == len(input_text) + 100
        assert call["output_tokens"] == len(output_text) + 100
        assert call["input_bytes"] == len(input_text.encode("utf-8"))
        assert call["output_bytes"] == len(output_text.encode("utf-8"))
        assert not any(key.startswith("_pend_") for key in call)


def test_recorder_entries_carry_no_padding_bookkeeping():
    """The r26 padding machinery is deleted: entries never grow
    ``_pad_group``/``padded_pair_tokens`` (there is no consumer left — CM
    padding is internal, from chunk quantiles + the shared reranker policy)
    and the recorder rejects the retired keyword arguments outright."""
    recorder = recording.TraceRecorder()
    recorder.record_batch(
        "passage_reranker",
        qids=["q0", "q1"],
        input_tokens=[3, 4],
        model_id="monot5",
    )
    retired = {
        "_pad_group", "_pad_sem", "_pad_fwd", "_pend_pairs",
        "padded_pair_tokens", "pad_group", "pad_semantics",
        "pad_forward_batch",
    }
    for query_trace in recorder.to_trace_v1():
        for call in query_trace:
            assert not (set(call) & retired)

    with pytest.raises(TypeError):
        recorder.record("q2", "passage_reranker", input_tokens=1, pad_group="g")
    with pytest.raises(TypeError):
        recording.record_io(
            "passage_reranker", ["q2"], ["text"], pad_semantics="call",
        )
