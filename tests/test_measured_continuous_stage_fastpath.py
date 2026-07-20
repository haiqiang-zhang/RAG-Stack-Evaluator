"""CPU-only contracts for measured continuous-stage DataFrame fast paths."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pandas as pd
from pandas.testing import assert_frame_equal


class _Owner:
    """Use the production merge for inputs that are ineligible for fast path."""

    @staticmethod
    def _merge_service_node_result(node, previous, result):
        from rag_stack.static_rag_evaluator.measured.evaluator import MeasuredEvaluator

        return MeasuredEvaluator._merge_service_node_result(node, previous, result)


def _stage(name: str) -> dict:
    return {
        "stage": name,
        "node": SimpleNamespace(stage=name, strategy={}),
        "params": {"model": "test-model"},
        "instance": object(),
    }


def _state(sr, frame: pd.DataFrame, *, record_trace: bool):
    return sr.RequestState(
        seq=0,
        idx=0,
        qid="q0",
        df=frame,
        is_measured=True,
        record_trace=record_trace,
    )


def _capture_record_io(monkeypatch):
    from rag_stack.static_rag_evaluator import recording

    calls = []

    def capture(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(recording, "record_io", capture)
    return calls


def test_llm_fast_path_matches_reference_merge_and_trace(monkeypatch):
    """The one-row, collision-free serving frame is equivalent to the old merge."""
    from rag_stack.static_rag_evaluator.measured import serving_runtime as sr
    from rag_stack.static_rag_evaluator.measured.evaluator import MeasuredEvaluator

    previous = pd.DataFrame(
        {
            "query": ["question"],
            "prompts": ["rendered prompt"],
            "__qid__": ["q0"],
        }
    ).set_axis([41])
    previous_before = previous.copy(deep=True)
    stage = _stage("generator")
    service = sr.LLMContinuousStageService(
        _Owner(), stage, max_inflight=4
    )
    gen_perf = {"n_output_tokens": 3}

    async def fake_generate_text(_state, prompt, sampling_params=None):
        assert prompt == "rendered prompt"
        assert sampling_params is None
        return "answer", gen_perf, 0.125, 0.25, 2

    service.generate_text = fake_generate_text
    trace_calls = _capture_record_io(monkeypatch)
    original_record_generate = sr._record_generate
    recorder_result_columns = []

    def inspect_record_generate(previous_arg, result_arg, params_arg):
        recorder_result_columns.append(result_arg.columns.tolist())
        original_record_generate(previous_arg, result_arg, params_arg)

    monkeypatch.setattr(sr, "_record_generate", inspect_record_generate)

    actual = asyncio.run(
        service.run(_state(sr, previous, record_trace=True), previous)
    )
    node_result = pd.DataFrame(
        {
            "generated_texts": ["answer"],
            "generated_tokens": [[0, 0, 0]],
            "generated_log_probs": [[0.0, 0.0, 0.0]],
        }
    )
    expected = MeasuredEvaluator._merge_service_node_result(
        stage["node"], previous, node_result
    )

    assert_frame_equal(actual.df, expected, check_exact=True)
    assert_frame_equal(previous, previous_before, check_exact=True)
    assert actual.gen_perf is gen_perf
    assert actual.queue_wait_s == 0.125
    assert actual.service_s == 0.25
    assert actual.elapsed_s == actual.queue_wait_s + actual.service_s
    assert actual.batch_size == 2

    # Trace keeps the old output-fragment contract rather than exposing all
    # upstream columns from the shallow merged frame.
    assert recorder_result_columns == [node_result.columns.tolist()]
    assert len(trace_calls) == 1
    fast_path_trace = trace_calls.pop()
    original_record_generate(previous, node_result, stage["params"])
    assert trace_calls == [fast_path_trace]


def test_query_expansion_fast_path_matches_reference_merge_and_trace(monkeypatch):
    """Query expansion is equivalent when the upstream frame has no queries column."""
    from rag_stack.static_rag_evaluator.measured import serving_runtime as sr
    from rag_stack.static_rag_evaluator.measured.evaluator import MeasuredEvaluator

    previous = pd.DataFrame(
        {
            "query": ["question"],
            "generation_gt": [["answer"]],
            "__qid__": ["q0"],
        }
    ).set_axis([73])
    previous_before = previous.copy(deep=True)
    stage = _stage("query_expansion")
    service = sr.QueryExpansionContinuousStageService(
        _Owner(), stage, max_inflight=4
    )
    gen_perf = {"n_output_tokens": 2}
    expanded = ["question", "alternative one", "alternative two"]

    monkeypatch.setattr(
        sr,
        "_query_expansion_prompt_and_parser",
        lambda _instance, _params, _previous: (
            "expansion prompt",
            lambda text: expanded if text == "raw expansion" else [],
        ),
    )

    async def fake_generate_text(_state, prompt, sampling_params=None):
        assert prompt == "expansion prompt"
        assert sampling_params is None
        return "raw expansion", gen_perf, 0.05, 0.1, 3

    service.generate_text = fake_generate_text
    trace_calls = _capture_record_io(monkeypatch)

    actual = asyncio.run(
        service.run(_state(sr, previous, record_trace=True), previous)
    )
    node_result = pd.DataFrame({"queries": [expanded]})
    expected = MeasuredEvaluator._merge_service_node_result(
        stage["node"], previous, node_result
    )

    assert_frame_equal(actual.df, expected, check_exact=True)
    assert_frame_equal(previous, previous_before, check_exact=True)
    assert actual.gen_perf is gen_perf
    assert actual.queue_wait_s == 0.05
    assert actual.service_s == 0.1
    assert actual.elapsed_s == actual.queue_wait_s + actual.service_s
    assert actual.batch_size == 3

    assert len(trace_calls) == 1
    fast_path_trace = trace_calls.pop()
    sr._record_query_expansion(
        previous,
        "expansion prompt",
        "raw expansion",
        gen_perf,
        stage["params"],
    )
    assert trace_calls == [fast_path_trace]


def test_query_expansion_existing_queries_falls_back_to_reference_merge(
    monkeypatch,
):
    """An upstream queries column must retain the legacy preserve-old behavior."""
    from rag_stack.static_rag_evaluator.measured import serving_runtime as sr
    from rag_stack.static_rag_evaluator.measured.evaluator import MeasuredEvaluator

    previous = pd.DataFrame(
        {
            "query": ["question"],
            "queries": [["upstream query"]],
            "__qid__": ["q0"],
        }
    )
    stage = _stage("query_expansion")
    service = sr.QueryExpansionContinuousStageService(
        _Owner(), stage, max_inflight=1
    )
    expanded = ["new expansion"]

    monkeypatch.setattr(
        sr,
        "_query_expansion_prompt_and_parser",
        lambda _instance, _params, _previous: (
            "expansion prompt",
            lambda _text: expanded,
        ),
    )

    async def fake_generate_text(_state, _prompt, sampling_params=None):
        assert sampling_params is None
        return "raw expansion", {"n_output_tokens": 2}, 0.0, 0.1, 1

    service.generate_text = fake_generate_text
    actual = asyncio.run(
        service.run(_state(sr, previous, record_trace=False), previous)
    ).df
    reference = MeasuredEvaluator._merge_service_node_result(
        stage["node"], previous, pd.DataFrame({"queries": [expanded]})
    )

    # Legacy query-expansion merge drops the overlapping result column. The
    # optimized service must bypass assignment for this collision case.
    assert reference["queries"].iloc[0] == ["upstream query"]
    assert_frame_equal(actual, reference, check_exact=True)


def test_llm_existing_output_column_falls_back_to_reference_merge():
    """Generator overlap keeps the generic merge's duplicate-column semantics."""
    from rag_stack.static_rag_evaluator.measured import serving_runtime as sr
    from rag_stack.static_rag_evaluator.measured.evaluator import MeasuredEvaluator

    previous = pd.DataFrame(
        {
            "query": ["question"],
            "prompts": ["rendered prompt"],
            "generated_texts": ["upstream answer"],
        },
        index=[19],
    )
    stage = _stage("generator")
    service = sr.LLMContinuousStageService(
        _Owner(), stage, max_inflight=1
    )
    gen_perf = {"n_output_tokens": 2}

    async def fake_generate_text(_state, _prompt, sampling_params=None):
        assert sampling_params is None
        return "new answer", gen_perf, 0.0, 0.1, 1

    service.generate_text = fake_generate_text
    actual = asyncio.run(
        service.run(_state(sr, previous, record_trace=False), previous)
    ).df
    node_result = pd.DataFrame(
        {
            "generated_texts": ["new answer"],
            "generated_tokens": [[0, 0]],
            "generated_log_probs": [[0.0, 0.0]],
        }
    )
    reference = MeasuredEvaluator._merge_service_node_result(
        stage["node"], previous, node_result
    )

    assert_frame_equal(actual, reference, check_exact=True)
    assert actual.columns.tolist().count("generated_texts") == 2
