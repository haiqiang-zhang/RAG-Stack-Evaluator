"""Backend descriptors contributed through standard RAG-Stack entry points.

The host owns discovery and contracts. This package owns construction and
backend-specific capabilities; no runtime is imported until selected.
"""

from collections.abc import Mapping

from rag_stack.evaluation.api import EvaluationResult, Evaluator
from rag_stack.evaluation.registry import BackendSpec, EvaluatorContext


class _EngineAdapter:
    """Translate backend-private legacy engine output to the shared contract."""

    def __init__(self, engine):
        self._engine = engine

    def __getattr__(self, name):
        return getattr(self._engine, name)

    @property
    def qa_data(self):
        return self._engine.qa_data

    @qa_data.setter
    def qa_data(self, value):
        # DatasetEvalManager.update_qa_data writes through this wrapper when
        # callers replace the evaluation cohort between runs.
        self._engine.qa_data = value

    def evaluate(self, config, run_dir=None, metrics_override=None, on_trace_ready=None):
        payload = self._engine.evaluate(
            config, run_dir=run_dir, metrics_override=metrics_override,
            on_trace_ready=on_trace_ready,
        )
        if isinstance(payload, EvaluationResult):
            return payload
        if not isinstance(payload, Mapping):
            raise TypeError("Backend engine must return a metric mapping")
        payload = dict(payload)
        execution = payload.pop("__execution_dag__", None)
        workflow = payload.pop("__workflow_schema__", None)
        metadata = {
            name: value for name, value in payload.items()
            if isinstance(name, str) and name.startswith("__")
        }
        metrics = {
            name: value for name, value in payload.items()
            if name not in metadata and value is not None
        }
        canonical = isinstance(execution, dict) and "queries" in execution
        return EvaluationResult(
            metrics=metrics,
            trace=execution if canonical else None,
            workflow=workflow,
            runtime_stats=None if canonical else execution,
            metadata=metadata,
        )


def _create_static(context: EvaluatorContext) -> Evaluator:
    from rag_stack_evaluator.static_rag_evaluator import StaticRAGEvaluatorQualityOnly

    return _EngineAdapter(StaticRAGEvaluatorQualityOnly(
        context.dataset, context.project_dir, dataset_manager=context.dataset_manager,
    ))



def _static_metrics() -> tuple[str, ...]:
    # Preserve the built-in registry's lazy discovery and optional dependencies.
    metrics = {"retrieval_token_recall", "retrieval_token_precision", "retrieval_token_f1"}
    try:
        from rag_stack_evaluator.static_rag_evaluator.evaluation.generation import (
            GENERATION_METRIC_FUNC_DICT,
        )
        metrics.update(GENERATION_METRIC_FUNC_DICT)
    except ImportError:
        pass
    try:
        from rag_stack_evaluator.static_rag_evaluator.evaluation.retrieval import (
            RETRIEVAL_METRIC_FUNC_DICT,
        )
        metrics.update(RETRIEVAL_METRIC_FUNC_DICT)
    except ImportError:
        pass
    return tuple(sorted(metrics))



STATIC_GT = BackendSpec(
        name="static_gt", factory=_create_static, metrics=_static_metrics,
        pipeline_modes=("react", "sequential"), supports_measured=True, supports_static_ir=True,
)


__all__ = ["STATIC_GT"]
