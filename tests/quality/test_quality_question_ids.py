"""Quality publication uses dataset IDs without changing internal row IDs."""

from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest

from rag_stack.evaluator_contract import normalize_backend_result
from rag_stack_evaluator.backends import _EngineAdapter
from rag_stack_evaluator.static_rag_evaluator.static_rag_evaluator import (
    StaticRAGEvaluatorQualityOnly, _publish_quality_trace,
)


def calls(tokens):
    return [{"stage": "generator", "input_tokens": tokens, "output_tokens": 2,
             "step_idx": 0, "model_id": "fixture"}]


def test_static_publication_maps_internal_ids_before_callback_and_preserves_identity():
    raw_trace = [calls(19), calls(7)]
    original = deepcopy(raw_trace)
    recorder_ids = (7, 2)
    dataset_mapping = {2: "question-alpha", 7: "question-beta"}
    callbacks = []
    engine = object.__new__(StaticRAGEvaluatorQualityOnly)

    def run_pipeline(config, run_dir=None, on_trace_ready=None):
        envelope = _publish_quality_trace(
            raw_trace, recorder_ids, 2, on_trace_ready,
            question_id_map=dataset_mapping,
        )
        return SimpleNamespace(quality={"quality": 0.75}, trace=envelope)

    engine._run_pipeline = run_pipeline
    config = {"node_lines": [{"node_line_name": "main", "nodes": [
        {"stage": "generator", "modules": [{"component": "vllm", "model": "fixture"}]},
    ]}]}
    result = _EngineAdapter(engine).evaluate(config, on_trace_ready=callbacks.append)
    assert callbacks[0] is result.trace
    assert result.trace["provenance"] == {
        "producer": "rag-stack-evaluator", "trace_origin": "observed",
    }
    assert [q["question_id"] for q in callbacks[0]["queries"]] == [
        "question-beta", "question-alpha",
    ]
    assert [q["calls"][0]["input_tokens"] for q in result.trace["queries"]] == [19, 7]
    assert raw_trace == original
    assert recorder_ids == (7, 2)
    assert dataset_mapping == {2: "question-alpha", 7: "question-beta"}
    payload = normalize_backend_result(
        result, backend="static_gt", pipeline_config=config,
        question_ids=["question-alpha", "question-beta"],
    )
    assert payload["__execution_dag__"] is callbacks[0]


def test_reconstructed_trace_is_distinguished_from_observed_publication(monkeypatch):
    from rag_stack_evaluator.static_rag_evaluator import static_rag_evaluator as module

    engine = object.__new__(StaticRAGEvaluatorQualityOnly)
    run = SimpleNamespace(
        quality={}, trace=None, node_lines={}, token_stats={},
        previous_result=pd.DataFrame({"qid": ["question-alpha"]}),
    )
    monkeypatch.setattr(engine, "_run_pipeline", lambda *a, **kw: run)
    monkeypatch.setattr(module, "build_static_execution_dag", lambda *a, **kw: {
        "trace_v1": [calls(7)],
    })
    result = engine.evaluate({"_runtime": {"rag_ir_mode": "dynamic"}},
                             on_trace_ready=lambda _: pytest.fail("Fallback is not observed"))
    assert result["__execution_dag__"]["provenance"]["trace_origin"] == "reconstructed"


@pytest.mark.parametrize("mapping", [
    {2: "alpha", 99: "beta"},
    {2: "alpha", 7: "beta", 99: "extra"},
])
def test_static_invalid_mapping_fails_before_callback(mapping):
    with pytest.raises(RuntimeError, match="dataset mapping"):
        _publish_quality_trace(
            [calls(1), calls(2)], (7, 2), 2,
            lambda trace: pytest.fail("published invalid cohort"),
            question_id_map=mapping,
        )


def test_static_duplicate_public_ids_fail_before_callback():
    with pytest.raises(ValueError, match="duplicate"):
        _publish_quality_trace(
            [calls(1), calls(2)], (7, 2), 2,
            lambda trace: pytest.fail("published invalid cohort"),
            question_id_map={7: "same", 2: "same"},
        )
