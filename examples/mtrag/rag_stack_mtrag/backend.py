"""Run an instrumented external pipeline, then MT-RAG's generation evaluator."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from importlib import import_module
import math
from pathlib import Path
from typing import Any, Callable, Sequence

from rag_stack_evaluator.api import EvaluationResult, publish_trace
from rag_stack_evaluator.factory import BackendSpec, EvaluatorContext
from rag_stack_evaluator.integrations.mtrag import load_tasks, score_with_mtrag
from rag_stack_evaluator.tracing import TraceSession


METRIC_KEYS = {
    "mt_rag_faithfulness": "RL_F",
    "mt_rag_radbench": "RB_llm",
    "mt_rag_aggregate": "RB_agg",
    "mt_rag_faithfulness_idk": "RL_F_idk_underspecified",
    "mt_rag_radbench_idk": "RB_llm_idk_underspecified",
    "mt_rag_aggregate_idk": "RB_agg_idk_underspecified",
}
DEFAULT_METRICS = tuple(METRIC_KEYS)[:3]


@dataclass(frozen=True)
class Prediction:
    """Actual serving outputs, using the existing static metric-input names."""

    qid: str
    generated_texts: str
    retrieved_contents: list[str]


@dataclass(frozen=True)
class PipelineRun:
    """The runner declares its workflow and returns one prediction per request."""

    predictions: Sequence[Prediction]
    workflow: Any


def import_callable(reference: str) -> Callable:
    """Load trusted extension code, explicitly configured as ``module:name``."""
    if not isinstance(reference, str) or reference.count(":") != 1:
        raise ValueError("pipeline must be a trusted 'module:callable' import path")
    module, name = reference.split(":")
    value = getattr(import_module(module), name)
    if not callable(value):
        raise ValueError(f"Pipeline is not callable: {reference!r}")
    return value


def metric_names(specifications: Sequence[str | dict]) -> tuple[str, ...]:
    if isinstance(specifications, (str, bytes)) or not isinstance(specifications, (list, tuple)):
        raise ValueError("metrics must be a list of metric names or metric_name mappings")
    names = []
    for item in specifications:
        if isinstance(item, dict):
            if set(item) - {"metric_name", "description"}:
                raise ValueError("MT-RAG example metrics do not accept per-metric parameters")
            item = item.get("metric_name")
        if not isinstance(item, str) or item not in METRIC_KEYS or item in names:
            raise ValueError(f"Unknown or duplicate MT-RAG metric: {item!r}")
        names.append(item)
    if not names:
        raise ValueError("Select at least one MT-RAG metric")
    return tuple(names)


class MTRAGEvaluator:
    """An Evaluator Protocol implementation, with no built-in pipeline nodes.

    The injected runner accepts ``(config, requests, recorder, *, options)``.
    Requests contain only task_id/input; benchmark references, contexts, and
    answerability labels are supplied to the scorer after serving completes.
    The runner must map every relevant resolved candidate knob to its own
    components. Silently ignoring search-space choices invalidates a study.
    """

    def __init__(self, context: EvaluatorContext, *, pipeline: Callable | None = None):
        self.context = context
        self.options = deepcopy(context.options)
        for field in ("tasks_path", "tasks_sha256", "scorer_command", "scorer_identity",
                      "upstream_revision", "pipeline_identity"):
            if not self.options.get(field):
                raise ValueError(f"Missing MT-RAG backend option: {field}")
        identity = self.options["pipeline_identity"]
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("pipeline_identity must identify the pipeline code and resources")
        self.pipeline = pipeline if pipeline is not None else import_callable(self.options.get("pipeline"))
        if not callable(self.pipeline):
            raise ValueError("pipeline must be callable")
        self.metrics = metric_names(self.options.get("metrics", list(DEFAULT_METRICS)))

    def _tasks(self) -> list[dict]:
        tasks = load_tasks(self.options["tasks_path"], self.options["tasks_sha256"])
        manager = self.context.dataset_manager
        if manager is None:
            if self.context.dataset is not None:
                raise ValueError("Supply dataset_manager to select host QA tasks, or omit dataset")
            return tasks
        # The host manager owns the active workload, including QA subsets.
        ids = manager.qa_data["qid"].tolist()
        if not ids or any(not isinstance(qid, str) or not qid for qid in ids) or len(set(ids)) != len(ids):
            raise ValueError("Active QA data must have unique nonempty string qids")
        indexed = {task["task_id"]: task for task in tasks}
        if set(ids) - indexed.keys():
            raise ValueError("Active QA data contains IDs absent from the MT-RAG sidecar")
        return [indexed[qid] for qid in ids]

    def evaluate(
        self, config: dict, run_dir: str | None = None,
        metrics_override: Sequence[str | dict] | None = None,
        on_trace_ready: Callable[[dict], None] | None = None,
    ) -> EvaluationResult:
        configured = (config.get("eval_backend_setting") or {}).get("metrics", self.metrics)
        names = metric_names(configured if metrics_override is None else metrics_override)
        runtime = config.get("pipeline_runtime") or {}
        if runtime.get("mode", "sequential") != "sequential" or runtime.get("rag_dataflow") not in (None, "", "sequential"):
            raise ValueError("The MT-RAG example backend declares sequential pipelines only")
        tasks = self._tasks()
        requests = [{"task_id": task["task_id"], "input": deepcopy(task["input"])}
                    for task in tasks]
        recorder = TraceSession()
        run = self.pipeline(
            deepcopy(config), requests, recorder,
            options=deepcopy(self.options.get("pipeline_options", {})),
        )
        if not isinstance(run, PipelineRun) or run.workflow is None:
            raise ValueError("External pipeline must return PipelineRun with its workflow")
        ids = [task["task_id"] for task in tasks]
        output_ids = [getattr(item, "qid", None) for item in run.predictions]
        if any(not isinstance(qid, str) for qid in output_ids) or len(output_ids) != len(ids) or set(output_ids) != set(ids):
            raise ValueError("External pipeline must return exactly one prediction per requested task")
        trace = recorder.finish(expected_question_ids=ids)
        trace.setdefault("provenance", {}).update(
            external_pipeline=self.options["pipeline_identity"],
            mtrag_upstream_revision=self.options["upstream_revision"],
        )
        # No judge call is inside a query/stage scope. Publish schema and trace
        # together so the host can overlap RAG-CM with subsequent judging.
        publish_trace(trace, workflow=run.workflow, on_trace_ready=on_trace_ready)
        audit = str(Path(run_dir) / "mtrag") if run_dir else self.options.get("audit_dir")
        per_task = score_with_mtrag(
            run.predictions, tasks_path=self.options["tasks_path"],
            tasks_sha256=self.options["tasks_sha256"],
            scorer_command=self.options["scorer_command"],
            scorer_identity=self.options["scorer_identity"],
            upstream_revision=self.options["upstream_revision"],
            metric_keys={name: METRIC_KEYS[name] for name in names},
            timeout_seconds=self.options.get("timeout_seconds", 3600), audit_dir=audit,
        )
        metrics = {name: math.fsum(values) / len(values) for name, values in per_task.items()}
        return EvaluationResult(metrics=metrics, trace=trace, workflow=run.workflow)


BACKEND = BackendSpec(
    name="mtrag", factory=MTRAGEvaluator, metrics=tuple(METRIC_KEYS),
    pipeline_modes=("sequential",), pipeline_style="node_lines", supports_measured=False,
)
