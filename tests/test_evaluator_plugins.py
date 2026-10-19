"""CPU-only public contract and external backend discovery tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from rag_stack_evaluator.api import (
    EvaluationResult, Evaluator, normalize_evaluation_result, publish_trace,
)
from rag_stack_evaluator import factory
from rag_stack_evaluator.factory import BackendSpec, EvaluatorContext


class ExternalEvaluator:
    def __init__(self, context):
        self.context = context

    def evaluate(self, config, run_dir=None, metrics_override=None, on_trace_ready=None):
        trace = make_trace()
        publish_trace(trace, on_trace_ready=on_trace_ready)
        return EvaluationResult({"quality": 0.75}, trace)


def make_trace():
    return {"queries": [{"question_id": "q1", "calls": [{
        "stage": "generator", "input_tokens": 8, "output_tokens": 3,
        "step_idx": 0, "model_id": "test-model",
    }]}]}


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    monkeypatch.setattr(factory, "_BACKENDS", dict(factory._BACKENDS))
    monkeypatch.setattr(factory, "_entry_points", lambda: ())


def spec(name="external", **kwargs):
    return BackendSpec(name=name, factory=ExternalEvaluator, metrics=("quality",), **kwargs)


def test_structural_backend_receives_context_and_can_publish_trace():
    factory.register_backend(spec())
    manager = object()
    options = {"judge": "cpu-fixture"}
    evaluator = factory.create_quality_evaluator(
        dataset_manager=manager, project_dir="project", backend="external",
        backend_options=options,
    )
    assert isinstance(evaluator, Evaluator)
    assert evaluator.context.dataset_manager is manager
    assert evaluator.context.project_dir == "project"
    assert evaluator.context.options == options
    assert evaluator.context.options is not options
    observed = []
    result = normalize_evaluation_result(
        evaluator.evaluate({}, on_trace_ready=observed.append), require_trace=True,
    )
    assert result["quality"] == 0.75
    assert result["__execution_dag__"] is observed[0]
    assert factory.supported_metrics("external") == ("quality",)
    assert factory.supported_pipeline_modes("external") == ("sequential",)
    assert not factory.get_backend_spec("external").supports_measured


def test_builtin_metadata_and_lazy_capabilities():
    assert factory.get_backend_spec("static_gt").pipeline_style == "node_lines"
    assert factory.get_backend_spec("static_gt").supports_measured
    assert factory.get_backend_spec("flashrag").pipeline_style == "methods"
    assert "react" in factory.supported_pipeline_modes("flashrag")
    assert "rouge_l" in factory.supported_metrics("flashrag")
    called = []
    factory.register_backend(replace(spec(), metrics=lambda: called.append(True) or ("quality",)))
    assert not called
    assert factory.supported_metrics("external") == ("quality",)
    assert called == [True]


def test_discovery_does_not_import_unselected_entry_points(monkeypatch):
    loaded = []
    entries = (
        SimpleNamespace(name="external", load=lambda: loaded.append("external") or spec()),
        SimpleNamespace(name="unused", load=lambda: pytest.fail("loaded unused plugin")),
    )
    monkeypatch.setattr(factory, "_entry_points", lambda: entries)
    assert factory.available_backends() == ("external", "flashrag", "static_gt", "unused")
    assert not loaded
    assert factory.get_backend_spec("external").name == "external"
    assert factory.get_backend_spec("external").name == "external"
    assert loaded == ["external"]


def test_duplicate_names_cannot_replace_builtin_or_registered_backend():
    with pytest.raises(ValueError, match="already registered"):
        factory.register_backend(spec("static_gt"))
    factory.register_backend(spec())
    with pytest.raises(ValueError, match="already registered"):
        factory.register_backend(spec())


def test_duplicate_installed_names_and_invalid_entry_points_fail(monkeypatch):
    entry = SimpleNamespace(name="external", load=lambda: spec())
    monkeypatch.setattr(factory, "_entry_points", lambda: (entry, entry))
    with pytest.raises(ValueError, match="Multiple installed"):
        factory.get_backend_spec("external")
    with pytest.raises(ValueError, match="already registered or installed"):
        factory.register_backend(spec())
    monkeypatch.setattr(factory, "_entry_points", lambda: (
        SimpleNamespace(name="external", load=lambda: ExternalEvaluator),
    ))
    with pytest.raises(TypeError, match="must resolve to a BackendSpec"):
        factory.get_backend_spec("external")
    monkeypatch.setattr(factory, "_entry_points", lambda: (
        SimpleNamespace(name="external", load=lambda: spec("different")),
    ))
    with pytest.raises(ValueError, match="declares backend name"):
        factory.get_backend_spec("external")


def test_unknown_backend_and_retired_alias_are_actionable():
    with pytest.raises(ValueError, match="Supported: flashrag, static_gt"):
        factory.get_backend_spec("unknown")
    with pytest.raises(ValueError, match="renamed to 'static_gt'"):
        factory.get_backend_spec("static")


def test_factory_rejects_missing_evaluate_and_ambiguous_options():
    factory.register_backend(replace(spec(), factory=lambda context: object()))
    with pytest.raises(TypeError, match="evaluate method"):
        factory.create_quality_evaluator(backend="external")
    with pytest.raises(ValueError, match="not both"):
        factory.create_quality_evaluator(backend="external", backend_options={}, flashrag_options={})
    with pytest.raises(ValueError, match="only configure"):
        factory.create_quality_evaluator(backend="external", flashrag_options={})


def test_legacy_flashrag_options_are_preserved_without_loading_runtime(monkeypatch):
    monkeypatch.setitem(factory._BACKENDS, "flashrag", replace(
        factory.get_backend_spec("flashrag"), factory=ExternalEvaluator,
    ))
    evaluator = factory.create_quality_evaluator(backend="flashrag", flashrag_options={"seed": 5})
    assert evaluator.context.options == {"seed": 5}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, None, "0.8"])
def test_strict_results_reject_invalid_scores(value):
    with pytest.raises(ValueError, match="finite number"):
        normalize_evaluation_result(EvaluationResult({"quality": value}))
    with pytest.raises(ValueError, match="finite number"):
        normalize_evaluation_result({"quality": value})


def test_strict_result_requires_trace_when_used_for_cost_model():
    assert normalize_evaluation_result(EvaluationResult({"quality": 1})) == {"quality": 1.0}
    with pytest.raises(ValueError, match="requires a canonical"):
        normalize_evaluation_result(EvaluationResult({"quality": 1}), require_trace=True)
    with pytest.raises(TypeError, match="envelope must be a dict"):
        normalize_evaluation_result(EvaluationResult({"quality": 1}, []))


def test_legacy_flat_results_retain_private_metadata_and_static_fanout():
    fanout = {"retrieval": {"input_tokens": 8}}
    result = {"quality": None, "__fp_hit__": "donor", "__execution_dag__": fanout}
    normalized = normalize_evaluation_result(result, allow_legacy=True)
    assert normalized == result
    assert normalized["__execution_dag__"] is fanout
    with pytest.raises(ValueError, match="unsupported fields"):
        normalize_evaluation_result(result, require_trace=True, allow_legacy=True)
    with pytest.raises(ValueError, match="Unsupported evaluator result metadata"):
        normalize_evaluation_result(result)


def test_typed_results_never_enable_legacy_score_validation():
    with pytest.raises(ValueError, match="finite number"):
        normalize_evaluation_result(EvaluationResult({"quality": None}), allow_legacy=True)


def test_workflow_published_before_callback_and_preserved_with_trace_identity():
    trace = make_trace()
    trace["provenance"] = None
    workflow = {"dataflow_type": "sequential", "stages": ["generator_prefill", "generator_decode"]}
    observed = []
    published = publish_trace(trace, workflow=workflow, on_trace_ready=observed.append)
    assert published is trace is observed[0]
    assert observed[0]["provenance"]["workflow_schema"]["dataflow_type"] == "sequential"
    result = normalize_evaluation_result(EvaluationResult({"quality": 0.75}, trace, workflow))
    assert result["__execution_dag__"] is trace
    assert result["__workflow_schema__"] == trace["provenance"]["workflow_schema"]
    replay = normalize_evaluation_result({"quality": 0.75, "__execution_dag__": trace})
    assert replay["__workflow_schema__"] == result["__workflow_schema__"]
    with pytest.raises(ValueError, match="conflicts"):
        normalize_evaluation_result(EvaluationResult(
            {"quality": 0.75}, trace, dict(workflow, dataflow_type="agentic"),
        ))


def test_trace_callback_failure_is_advisory_but_bad_trace_is_rejected(caplog):
    def broken_callback(trace):
        raise RuntimeError("CM fixture failure")

    trace = make_trace()
    assert publish_trace(trace, on_trace_ready=broken_callback) is trace
    assert "continuing quality evaluation" in caplog.text
    trace["queries"][0]["calls"][-1]["stage"] = "semantic_retrieval_encode"
    with pytest.raises(ValueError, match="incomplete"):
        publish_trace(trace, on_trace_ready=lambda trace: pytest.fail("invalid trace published"))


def test_public_contract_and_discovery_do_not_import_host_or_models():
    code = r'''
import sys
class RejectRuntimeImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"rag_stack", "torch", "vllm", "flashrag", "transformers"}:
            raise AssertionError("Unexpected runtime dependency: " + fullname)
sys.meta_path.insert(0, RejectRuntimeImports())
from rag_stack_evaluator.api import EvaluationResult, Evaluator, normalize_evaluation_result
from rag_stack_evaluator.base import BaseEvaluator
from rag_stack_evaluator.factory import available_backends, get_backend_spec
assert "static_gt" in available_backends()
assert get_backend_spec("static_gt").supports_measured
assert normalize_evaluation_result(EvaluationResult({"quality": 0.5})) == {"quality": 0.5}
'''
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
