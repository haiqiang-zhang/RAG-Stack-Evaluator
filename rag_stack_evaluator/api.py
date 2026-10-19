"""Public evaluator result contract, without eager model/runtime imports.

External backends may implement :class:`Evaluator` structurally; inheritance
from ``BaseEvaluator`` is optional. RAG-Stack normalizes the result at its
boundary so existing flat-dictionary backends remain compatible.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import logging
from numbers import Real
from typing import Any, Callable, Protocol, TYPE_CHECKING, runtime_checkable

if TYPE_CHECKING:
    from rag_stack.rag_ir import QualityTraceEnvelope, WorkflowSchema


@dataclass(frozen=True)
class EvaluationResult:
    """Quality scores and the measured input/output shape of one quality run.

    ``trace`` contains serving component calls only, never judge calls. A
    backend using the host's resolved pipeline structure can omit ``workflow``.
    Otherwise supply a ``WorkflowSchema`` or its JSON representation. Workflows
    are also persisted in trace provenance so cached traces remain self-contained.
    """

    metrics: dict[str, float]
    trace: QualityTraceEnvelope | None = None
    workflow: WorkflowSchema | dict[str, Any] | None = None


@runtime_checkable
class Evaluator(Protocol):
    """Structural interface implemented by built-in and external backends."""

    def evaluate(
        self,
        config: dict,
        run_dir: str | None = None,
        metrics_override: Sequence[str | dict] | None = None,
        on_trace_ready: Callable[[dict], None] | None = None,
    ) -> EvaluationResult | dict:
        """Evaluate one resolved configuration.

        A backend may publish its complete serving trace before scoring through
        ``on_trace_ready``. The exact same trace object must be returned; callback
        failures are advisory and must not discard a valid quality result.
        """
        ...


def _workflow_dict(workflow: Any) -> dict:
    # Shared host types are imported only when a result is validated, never at
    # plugin discovery or evaluator-package import time.
    from rag_stack.rag_ir import WorkflowSchema

    if isinstance(workflow, WorkflowSchema):
        schema = workflow
    elif isinstance(workflow, dict):
        schema = WorkflowSchema.from_dict(workflow)
    else:
        raise TypeError("workflow must be a WorkflowSchema or its dictionary representation")
    if schema.dataflow_type not in {"sequential", "agentic"}:
        raise ValueError("workflow.dataflow_type must be 'sequential' or 'agentic'")
    if not isinstance(schema.stages, list) or not all(
        isinstance(stage, str) and stage for stage in schema.stages
    ):
        raise ValueError("workflow.stages must be a list of non-empty stage names")
    for name in ("has_query_expansion", "has_reranker", "has_encode", "has_compressor"):
        if not isinstance(getattr(schema, name), bool):
            raise ValueError(f"workflow.{name} must be a boolean")
    return schema.to_dict()


def normalize_evaluation_result(
    result: EvaluationResult | Mapping[str, Any],
    *,
    require_trace: bool = False,
    allow_legacy: bool = False,
) -> dict[str, Any]:
    """Validate and convert an evaluator result to the host's flat dictionary.

    Every supplied score must be finite; booleans and ``None`` are rejected.
    Empty metrics are permitted when publishing a trace before judging; the
    host separately requires the configured optimization objective after judging.
    ``require_trace`` additionally requires a canonical quality trace for cost
    modeling. ``allow_legacy`` is reserved for existing built-in flat dictionaries:
    it preserves their missing metrics, private metadata and static fanout data.
    Typed results always follow the strict contract.

    The trace object is never copied: early callback/CM overlap relies on its
    identity. An explicit workflow is serialized both as ``__workflow_schema__``
    and in ``trace['provenance']['workflow_schema']``; conflicting declarations
    fail rather than replacing the structure used by a cached trace.
    """
    typed = isinstance(result, EvaluationResult)
    legacy = allow_legacy and not typed
    if typed:
        if not isinstance(result.metrics, Mapping):
            raise TypeError("EvaluationResult.metrics must be a mapping")
        metrics = dict(result.metrics)
        trace = result.trace
        workflow = result.workflow
        output: dict[str, Any] = {}
    elif isinstance(result, Mapping):
        output = dict(result)
        trace = output.pop("__execution_dag__", None)
        workflow = output.pop("__workflow_schema__", None)
        metrics = {}
        for name, value in list(output.items()):
            if isinstance(name, str) and name.startswith("__"):
                if not legacy:
                    raise ValueError(f"Unsupported evaluator result metadata: {name!r}")
            else:
                metrics[name] = value
                del output[name]
    else:
        raise TypeError("Evaluator.evaluate must return EvaluationResult or a mapping")

    for name, value in metrics.items():
        if not isinstance(name, str) or not name.strip() or name.startswith("__"):
            raise ValueError(f"Invalid quality metric name: {name!r}")
        if legacy and value is None:
            output[name] = None
            continue
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"Quality metric {name!r} must be a finite number, got {value!r}")
        output[name] = float(value)

    if trace is None:
        if require_trace:
            raise ValueError("This evaluation requires a canonical quality execution trace")
    else:
        # Legacy built-ins also support an explicit static/fanout CM path. A
        # canonical-looking envelope is always validated, including in legacy mode.
        if not legacy or require_trace or (isinstance(trace, dict) and "queries" in trace):
            from rag_stack.rag_ir import validate_quality_trace_envelope

            validate_quality_trace_envelope(trace)
        output["__execution_dag__"] = trace

    embedded = None
    if isinstance(trace, dict) and isinstance(trace.get("provenance"), dict):
        embedded = trace["provenance"].get("workflow_schema")
    if workflow is not None or embedded is not None:
        schema = _workflow_dict(workflow if workflow is not None else embedded)
        if embedded is not None and _workflow_dict(embedded) != schema:
            raise ValueError("Result workflow conflicts with trace provenance workflow_schema")
        if isinstance(trace, dict):
            if trace.get("provenance") is None:
                trace["provenance"] = {}
            trace.setdefault("provenance", {})["workflow_schema"] = schema
        output["__workflow_schema__"] = schema
    return output


def publish_trace(
    trace: QualityTraceEnvelope,
    *,
    workflow: WorkflowSchema | dict[str, Any] | None = None,
    on_trace_ready: Callable[[dict], None] | None = None,
) -> QualityTraceEnvelope:
    """Validate and publish a complete serving trace before running judges.

    An explicit workflow is attached before the callback runs. Callback errors
    are logged and ignored: the host can retry cost modeling after scoring.
    The return value and callback argument are the original ``trace`` object.
    """
    normalize_evaluation_result(
        EvaluationResult(metrics={}, trace=trace, workflow=workflow), require_trace=True,
    )
    if on_trace_ready is not None:
        try:
            on_trace_ready(trace)
        except Exception:
            logging.getLogger(__name__).warning(
                "on_trace_ready callback failed; continuing quality evaluation",
                exc_info=True,
            )
    return trace


__all__ = ["EvaluationResult", "Evaluator", "normalize_evaluation_result", "publish_trace"]
