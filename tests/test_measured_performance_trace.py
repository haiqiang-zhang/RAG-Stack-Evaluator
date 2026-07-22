import hashlib
import json

import pytest

from rag_stack_evaluator.static_rag_evaluator import recording
from rag_stack_evaluator.static_rag_evaluator.measured.performance_trace import (
    PERFORMANCE_TRACE_PRIORITY_SEED,
    make_performance_trace_envelope,
    performance_trace_calls,
    validate_performance_trace_envelope,
)
from rag_stack_evaluator.static_rag_evaluator.measured.serving_runtime import (
    _DeterministicTraceReservoir,
)


def _calls(output_tokens: int) -> list[dict]:
    return [
        {
            "stage": "semantic_retrieval_encode",
            "input_tokens": 5,
            "output_tokens": 0,
            "input_bytes": 20,
            "output_bytes": 0,
            "step_idx": 0,
            "model_id": "embed",
        },
        {
            "stage": "semantic_retrieval_vectorsearch",
            "input_tokens": 5,
            "output_tokens": 0,
            "input_bytes": 20,
            "output_bytes": 40,
            "step_idx": 1,
            "model_id": None,
        },
        {
            "stage": "generator",
            "input_tokens": 50,
            "output_tokens": output_tokens,
            "input_bytes": 200,
            "output_bytes": output_tokens * 4,
            "step_idx": 2,
            "model_id": "generator",
        },
    ]


def test_performance_trace_is_independent_call_only_contract():
    envelope = make_performance_trace_envelope(
        [_calls(7), _calls(101)],
        invocation_ids=["measured-10", "warmup-3"],
        source_question_ids=["dataset-4", "dataset-4"],
        capacity=2,
        population_queries=17,
    )

    validate_performance_trace_envelope(envelope)
    assert set(envelope) == {"trace_kind", "sampling", "queries"}
    assert envelope["trace_kind"] == "measurement_phase_completion_sample"
    assert envelope["sampling"]["population_scope"] == (
        "measurement_phase_completion_window"
    )
    assert [
        query["calls"][-1]["output_tokens"] for query in envelope["queries"]
    ] == [7, 101]
    assert [
        query["source_question_id"] for query in envelope["queries"]
    ] == ["dataset-4", "dataset-4"]
    forbidden = {
        "qps", "latency_s", "service_s", "queue_wait_s",
        "request_send_ts", "first_token_ts", "last_token_ts",
    }
    assert not (forbidden & set(envelope))
    assert all(
        not (forbidden & set(call))
        for query in envelope["queries"]
        for call in query["calls"]
    )

    leaked = dict(envelope)
    leaked["qps"] = 123.0
    with pytest.raises(ValueError, match="unsupported fields"):
        validate_performance_trace_envelope(leaked)

    leaked_call = json.loads(json.dumps(envelope))
    leaked_call["queries"][0]["calls"][0]["latency_s"] = 0.5
    with pytest.raises(ValueError, match="unsupported fields"):
        validate_performance_trace_envelope(leaked_call)

    producer_leak = _calls(7)
    producer_leak[0]["qps"] = 123.0
    with pytest.raises(ValueError, match="unsupported fields"):
        make_performance_trace_envelope(
            [producer_leak],
            invocation_ids=["measured-10"],
            capacity=1,
            population_queries=1,
        )


def test_performance_trace_rejects_unknown_envelope_field():
    envelope = make_performance_trace_envelope(
        [_calls(7)],
        invocation_ids=["measured-10"],
        capacity=1,
        population_queries=1,
    )
    envelope["obsolete_header"] = 1

    with pytest.raises(
        ValueError,
        match="unsupported fields",
    ):
        validate_performance_trace_envelope(envelope)


def test_performance_trace_requires_complete_bottom_k_cardinality():
    with pytest.raises(ValueError, match="complete bottom-k cohort"):
        make_performance_trace_envelope(
            [_calls(7)],
            invocation_ids=["measured-1"],
            capacity=2,
            population_queries=9,
        )

    envelope = make_performance_trace_envelope(
        [_calls(7), _calls(8)],
        invocation_ids=["measured-1", "measured-2"],
        capacity=2,
        population_queries=9,
    )
    envelope["queries"].pop()
    envelope["sampling"]["sample_queries"] = 1
    with pytest.raises(ValueError, match="complete bottom-k cohort"):
        validate_performance_trace_envelope(envelope)


def test_record_io_defers_all_text_work_and_keeps_exact_snapshot(monkeypatch):
    recorder = recording.TraceRecorder()
    input_parts = ["alpha", None, "β", 7]
    output_text = "résumé"
    joined = []
    byte_batches = []
    token_batches = []
    real_join = recording._join_text

    def tracked_join(item):
        result = real_join(item)
        joined.append(result)
        return result

    def tracked_bytes(texts):
        values = list(texts)
        byte_batches.append(values)
        return [len(text.encode("utf-8")) for text in values]

    def tracked_tokens(texts, model_id=None):
        values = list(texts)
        token_batches.append((values, model_id))
        return [len(text) + 10 for text in values]

    monkeypatch.setattr(recording, "_join_text", tracked_join)
    monkeypatch.setattr(recording, "count_bytes_batch", tracked_bytes)
    monkeypatch.setattr(recording, "count_tokens_batch", tracked_tokens)
    recording.set_current_recorder(recorder)
    try:
        recording.record_io(
            "generator",
            ["measured-1"],
            [input_parts],
            out_texts=[output_text],
            out_token_ids=[[11, 12, 13]],
            model_id="generator-model",
        )
    finally:
        recording.clear_current_recorder()

    # Caller mutations after recording cannot change the deferred snapshot.
    input_parts[0] = "mutated"
    input_parts.append("late")
    assert joined == []
    assert byte_batches == []
    assert token_batches == []

    trace = recorder.trace_for_qids(["measured-1"])
    call = trace[0][0]
    assert joined == ["alpha β 7", output_text]
    # Token-counted and byte-only texts are finalized in independently
    # bounded batches, while preserving their exact snapshots.
    assert byte_batches == [["alpha β 7"], [output_text]]
    # Exact model output ids are never re-tokenized; only the input is.
    assert token_batches == [(["alpha β 7"], "generator-model")]
    assert call["input_tokens"] == len("alpha β 7") + 10
    assert call["output_tokens"] == 3
    assert call["input_bytes"] == len("alpha β 7".encode("utf-8"))
    assert call["output_bytes"] == len(output_text.encode("utf-8"))
    assert not any(key.startswith("_pend_") for key in call)


def test_bottom_k_reservoir_is_bounded_deterministic_and_order_independent():
    capacity = 7
    qids = [f"measured-{index}" for index in range(100)]
    first = _DeterministicTraceReservoir(capacity)
    second = _DeterministicTraceReservoir(capacity)
    for qid in qids:
        first.offer(qid)
    for qid in reversed(qids):
        second.offer(qid)

    expected = tuple(sorted(
        qids,
        key=lambda qid: (
            hashlib.sha256(
                f"{PERFORMANCE_TRACE_PRIORITY_SEED}\0{qid}".encode("utf-8")
            ).digest(),
            qid,
        ),
    )[:capacity])
    assert first.qids == expected
    assert second.qids == expected
    assert len(first.qids) == capacity


def test_recorder_reads_performance_cohort_without_changing_quality_projection():
    recorder = recording.TraceRecorder()
    for index, qid in enumerate(("quality-0", "perf-a", "quality-1", "perf-b")):
        for call in _calls(index + 1):
            recorder.record(
                qid,
                call["stage"],
                input_tokens=call["input_tokens"],
                output_tokens=call["output_tokens"],
                input_bytes=call["input_bytes"],
                output_bytes=call["output_bytes"],
                model_id=call["model_id"],
            )

    recorder.select_qids(["quality-0", "quality-1"])
    performance = recorder.trace_for_qids(["perf-b", "perf-a"])
    quality = recorder.to_trace_v1()

    assert [calls[-1]["output_tokens"] for calls in performance] == [4, 2]
    assert [calls[-1]["output_tokens"] for calls in quality] == [1, 3]
    assert recorder.qids == ("quality-0", "quality-1")


def test_performance_trace_calls_returns_full_stage_sequences():
    envelope = make_performance_trace_envelope(
        [_calls(9)],
        invocation_ids=["measured-1"],
        source_question_ids=["dataset-3"],
        capacity=1,
        population_queries=1,
    )
    calls, invocation_ids, source_question_ids = performance_trace_calls(envelope)
    assert invocation_ids == ["measured-1"]
    assert source_question_ids == ["dataset-3"]
    assert [call["stage"] for call in calls[0]] == [
        "semantic_retrieval_encode",
        "semantic_retrieval_vectorsearch",
        "generator",
    ]
