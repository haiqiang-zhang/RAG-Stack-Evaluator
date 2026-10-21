"""CPU checks for evaluator-owned descriptors and legacy engine adapters.

Generic API/discovery tests live in the host repository that owns the contract.
"""

from types import SimpleNamespace
import sys

import pytest

from rag_stack.evaluation import api, registry
from rag_stack_evaluator import backends, factory


def make_trace():
    return {"queries": [{"question_id": "q1", "calls": [{
        "stage": "generator", "input_tokens": 8, "output_tokens": 3,
        "step_idx": 0,
    }]}]}


class Engine:
    def __init__(self, dataset=None, project_dir=None, *, dataset_manager=None, flashrag_options=None):
        self.dataset = dataset
        self.project_dir = project_dir
        self.dataset_manager = dataset_manager
        self.flashrag_options = flashrag_options
        self.payload = {"quality": 0.75, "missing": None, "__execution_dag__": make_trace()}
        self.call = None

    def evaluate(self, config, run_dir=None, metrics_override=None, on_trace_ready=None):
        self.call = (config, run_dir, metrics_override)
        if on_trace_ready is not None:
            on_trace_ready(self.payload["__execution_dag__"])
        return self.payload


@pytest.mark.parametrize("descriptor, module, symbol", [
    (backends.STATIC_GT, "rag_stack_evaluator.static_rag_evaluator", "StaticRAGEvaluatorQualityOnly"),
])
def test_all_descriptors_construct_the_same_typed_interface(monkeypatch, descriptor, module, symbol):
    monkeypatch.setitem(sys.modules, module, SimpleNamespace(**{symbol: Engine}))
    context = registry.EvaluatorContext(dataset=object(), dataset_manager=object(),
                                        project_dir="project", options={"seed": 5})
    evaluator = descriptor.factory(context)
    assert isinstance(evaluator, api.Evaluator)
    assert evaluator.dataset is context.dataset
    assert evaluator.dataset_manager is context.dataset_manager
    seen = []
    config = {"node_lines": [{"node_line_name": "main", "nodes": [
        {"stage": "generator", "modules": [{"component": "vllm", "model": "test"}]},
    ]}]}
    result = evaluator.evaluate(config, "run", ["quality"], seen.append)
    assert isinstance(result, api.EvaluationResult)
    assert result.metrics == {"quality": 0.75}
    assert result.trace is seen[0]
    assert evaluator.call == (config, "run", ["quality"])
    assert result.workflow is not None


def test_adapter_preserves_aggregate_and_fingerprint_metadata_separately():
    engine = Engine()
    aggregate = {"retrieval": {"input_tokens": 8}}
    engine.payload = {"quality": 0.75, "missing": None, "__execution_dag__": aggregate,
                      "__fp_hit__": "donor"}
    result = backends._EngineAdapter(engine).evaluate({})
    assert result.trace is None
    assert result.runtime_stats is aggregate
    assert result.metadata == {"__fp_hit__": "donor"}
    assert api.normalize_evaluation_result(result)["__fp_hit__"] == "donor"
    with pytest.raises(ValueError, match="requires a canonical"):
        api.normalize_evaluation_result(result, require_trace=True)


def test_api_and_base_compatibility_imports_have_host_type_identity():
    from rag_stack_evaluator import api as old_api
    from rag_stack_evaluator.base import BaseEvaluator

    assert old_api is api
    assert old_api.EvaluationResult is api.EvaluationResult
    assert BaseEvaluator is api.BaseEvaluator
    assert factory.BackendSpec is registry.BackendSpec


def test_compatibility_factory_delegates_to_host_registry(monkeypatch):
    monkeypatch.setattr(registry, "_BACKENDS", {})
    monkeypatch.setattr(registry, "_entry_points", lambda: ())
    registry.register_backend(registry.BackendSpec(name="flashrag", metrics=("quality",), factory=lambda context: Engine(
        dataset_manager=context.dataset_manager, flashrag_options=context.options,
    )))
    manager = object()
    instance = factory.create_quality_evaluator(
        dataset_manager=manager, backend="flashrag", flashrag_options={"seed": 7},
    )
    assert instance.dataset_manager is manager
    assert instance.flashrag_options == {"seed": 7}
    with pytest.raises(ValueError, match="not both"):
        factory.create_quality_evaluator(backend="flashrag", flashrag_options={}, backend_options={})


def test_selected_descriptor_identity_uses_the_host_registry(monkeypatch):
    monkeypatch.setattr(registry, "_BACKENDS", {})
    monkeypatch.setattr(registry, "_entry_points", lambda: (
        SimpleNamespace(name="static_gt", load=lambda: backends.STATIC_GT),
    ))
    assert registry.get_backend_spec("static_gt") is backends.STATIC_GT
    assert factory.get_backend_spec("static_gt") is registry.get_backend_spec("static_gt")


def test_dataset_update_writes_through_adapter_to_actual_engine(tmp_path):
    import pandas as pd
    from rag_stack_evaluator.static_rag_evaluator.dataset import DatasetEvalManager

    original = pd.DataFrame({"qid": ["original"], "query": ["Old question"]})
    replacement = pd.DataFrame({"qid": ["replacement"], "query": ["New question"]})
    engine = Engine()
    engine.qa_data = original
    evaluator = backends._EngineAdapter(engine)
    manager = object.__new__(DatasetEvalManager)
    manager.qa_data = original
    manager.project_dir = str(tmp_path)
    (tmp_path / "data").mkdir()

    manager.update_qa_data(replacement, evaluator=evaluator)

    assert engine.qa_data is replacement
    assert evaluator.qa_data is replacement
    assert manager.qa_data is replacement
    assert "qa_data" not in evaluator.__dict__
    assert pd.read_parquet(tmp_path / "data" / "qa.parquet")["qid"].tolist() == ["replacement"]
