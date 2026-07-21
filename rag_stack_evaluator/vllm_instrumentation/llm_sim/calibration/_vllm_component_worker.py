"""Opt-in vLLM 0.18.1 worker for GenZ component GPU telemetry.

This module is imported only in vLLM worker processes selected explicitly via
``--worker-cls``.  Importing :mod:`component_telemetry` itself does not import
torch, vLLM or initialize CUDA.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import vllm
from vllm.v1.worker.gpu_worker import Worker

from rag_stack_evaluator.vllm_instrumentation.calibration.runtime_identity import (
    resolve_calibration_run_id,
    vllm_model_identity,
)

from .component_telemetry import (
    BOUNDARY_CONTRACT,
    BOUNDARY_CONTRACT_SHA256,
    COMPONENT_TELEMETRY_PATH_ENV,
    COMPONENT_TELEMETRY_SCHEMA,
    ComponentTelemetryError,
    append_component_record,
    extract_scheduler_cycle,
    make_cycle_id,
)


_SUPPORTED_VLLM_VERSION = "0.18.1"


@dataclass(slots=True)
class _PendingComponentRecord:
    """One record whose CUDA intervals are resolved only after generation."""

    payload: dict[str, Any]
    forward_events: tuple[Any, Any]
    logits_events: tuple[Any, Any]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _qualified_name(value: object) -> str:
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _source_hash(value: object, *, label: str) -> str:
    try:
        source = inspect.getsource(value)
    except (OSError, TypeError) as exc:
        raise ComponentTelemetryError(
            f"cannot freeze vLLM source contract for {label}"
        ) from exc
    return _sha256_text(source)


@contextmanager
def _temporary_callable(
    target: object, name: str, replacement: Callable[..., Any]
) -> Iterator[None]:
    """Patch one instance callable and restore its exact prior ownership."""

    namespace = getattr(target, "__dict__", None)
    had_instance_value = isinstance(namespace, dict) and name in namespace
    prior_instance_value = namespace.get(name) if had_instance_value else None
    setattr(target, name, replacement)
    try:
        yield
    finally:
        if had_instance_value:
            setattr(target, name, prior_instance_value)
        else:
            delattr(target, name)


class VllmComponentTelemetryWorker(Worker):
    """GPU worker that emits component-only timing under an explicit opt-in.

    The vLLM worker API is not a stable plugin contract, so this class pins the
    exact 0.18.1 source methods in every record.  Any unsupported runner,
    missing hook or incomplete communication boundary aborts the calibration
    instead of falling back to request wall time.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        path = os.environ.get(COMPONENT_TELEMETRY_PATH_ENV)
        if path is None or not path.strip():
            raise ComponentTelemetryError(
                f"{COMPONENT_TELEMETRY_PATH_ENV} must name a JSONL output file"
            )
        self._component_telemetry_path = Path(path).expanduser()
        try:
            self._component_telemetry_run_id = resolve_calibration_run_id()
        except ValueError as exc:
            raise ComponentTelemetryError(str(exc)) from exc
        self._component_cycle_sequence = 0
        self._component_work_sequence = 0
        self._component_source_contract: dict[str, object] | None = None
        self._component_pending_records: deque[_PendingComponentRecord] = deque()

        version = str(getattr(vllm, "__version__", ""))
        if version != _SUPPORTED_VLLM_VERSION:
            raise ComponentTelemetryError(
                "component telemetry was audited only for vLLM "
                f"{_SUPPORTED_VLLM_VERSION}; got {version!r}"
            )
        if self.use_v2_model_runner:
            raise ComponentTelemetryError(
                "vLLM V2 model runner has no audited component timing hook"
            )

        parallel = self.parallel_config
        tp = int(parallel.tensor_parallel_size)
        pp = int(parallel.pipeline_parallel_size)
        dp = int(parallel.data_parallel_size)
        world = int(parallel.world_size)
        if tp <= 0 or pp <= 0 or dp <= 0 or world <= 0:
            raise ComponentTelemetryError("vLLM parallel sizes must be positive")
        if pp != 1:
            raise ComponentTelemetryError(
                "PP>1 component telemetry is disabled: vLLM 0.18.1 PP recv/send "
                "sits outside the audited model-runner boundary"
            )
        if dp != 1 or world != tp:
            raise ComponentTelemetryError(
                "component telemetry currently requires DP=1 and world_size=TP"
            )
        if self.speculative_config is not None:
            raise ComponentTelemetryError(
                "speculative decoding has additional model execution outside the "
                "audited target-model boundary"
            )
        if getattr(self.model_config, "runner_type", None) != "generate":
            raise ComponentTelemetryError(
                "GenZ component telemetry requires a generate-model runner"
            )

    def _build_source_contract(self) -> dict[str, object]:
        if self._component_source_contract is not None:
            return self._component_source_contract
        runner = self.model_runner
        if runner is None:
            raise ComponentTelemetryError("vLLM model runner is not initialized")
        runner_type = type(runner)
        runner_execute = getattr(runner_type, "execute_model", None)
        model_forward = getattr(runner_type, "_model_forward", None)
        if not callable(runner_execute) or not callable(model_forward):
            raise ComponentTelemetryError(
                "vLLM model runner lacks the audited execute/_model_forward hooks"
            )
        model = getattr(runner, "model", None)
        # vLLM may wrap the concrete model in a compiled/CUDA-graph proxy whose
        # class does not declare ``compute_logits`` but whose instance delegates
        # the exact bound Qwen/model method.  Audit the callable that the runner
        # will actually invoke; inspecting only ``type(model)`` incorrectly
        # rejects that production layout.
        model_compute_logits = getattr(model, "compute_logits", None)
        if not callable(model_compute_logits):
            raise ComponentTelemetryError(
                "vLLM model class lacks an auditable compute_logits hook"
            )
        instrumentation_sha256 = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        contract: dict[str, object] = {
            "vllm_version": str(vllm.__version__),
            "torch_version": str(torch.__version__),
            "torch_cuda_version": None
            if torch.version.cuda is None
            else str(torch.version.cuda),
            "worker_class": _qualified_name(self),
            "model_runner_class": _qualified_name(runner),
            "vllm_worker_execute_model_sha256": _source_hash(
                Worker.execute_model, label="gpu_worker.Worker.execute_model"
            ),
            "vllm_model_runner_execute_model_sha256": _source_hash(
                runner_execute, label="GPUModelRunner.execute_model"
            ),
            "vllm_model_forward_sha256": _source_hash(
                model_forward, label="GPUModelRunner._model_forward"
            ),
            "vllm_model_compute_logits_sha256": _source_hash(
                model_compute_logits, label="model.compute_logits"
            ),
            "scheduler_cycle_extractor_sha256": _source_hash(
                extract_scheduler_cycle, label="extract_scheduler_cycle"
            ),
            "component_telemetry_flush_sha256": _source_hash(
                type(self).flush_component_telemetry,
                label="VllmComponentTelemetryWorker.flush_component_telemetry",
            ),
            "instrumentation_sha256": instrumentation_sha256,
            "boundary_contract_sha256": BOUNDARY_CONTRACT_SHA256,
        }
        self._component_source_contract = contract
        return contract

    def _model_identity(self) -> dict[str, object]:
        try:
            return vllm_model_identity(self.model_config)
        except ValueError as exc:
            raise ComponentTelemetryError(str(exc)) from exc

    def _record_cuda_callable(
        self,
        callable_: Callable[..., Any],
        intervals: list[tuple[Any, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if self.device is None or self.device.type != "cuda":
            raise ComponentTelemetryError(
                "component telemetry requires an initialized CUDA worker"
            )
        stream = torch.cuda.current_stream(device=self.device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        result = callable_(*args, **kwargs)
        end.record(stream)
        # Never synchronize or resolve an event in the scheduler cycle.  The
        # completed request is followed by an explicit control-plane flush.
        intervals.append((start, end))
        return result

    @staticmethod
    def _completed_interval_s(events: tuple[Any, Any]) -> float | None:
        start, end = events
        # Event.query() is the non-blocking CUDA completion test.  elapsed_time
        # is read only after the terminal event reports completion.
        if not bool(end.query()):
            return None
        duration_s = float(start.elapsed_time(end)) / 1000.0
        if not math.isfinite(duration_s) or duration_s <= 0.0:
            raise ComponentTelemetryError(
                f"CUDA event produced invalid component duration {duration_s!r}"
            )
        return duration_s

    def _resolve_pending_records(
        self,
    ) -> list[tuple[_PendingComponentRecord, float, float]]:
        resolved: list[tuple[_PendingComponentRecord, float, float]] = []
        for pending in self._component_pending_records:
            forward_s = self._completed_interval_s(pending.forward_events)
            logits_s = self._completed_interval_s(pending.logits_events)
            if forward_s is None or logits_s is None:
                sequence = pending.payload["work_sequence"]
                raise ComponentTelemetryError(
                    "component telemetry flush observed an unfinished CUDA "
                    f"interval at work_sequence={sequence}; refusing to wait"
                )
            resolved.append((pending, forward_s, logits_s))
        return resolved

    def flush_component_telemetry(self) -> dict[str, object]:
        """Resolve and publish one completed cohort without a CUDA wait.

        This method is a control-plane RPC invoked only after ``LLM.generate``
        has returned.  An empty queue is rejected so a duplicate/misordered
        flush cannot masquerade as a successfully observed cohort.
        """

        if not self._component_pending_records:
            raise ComponentTelemetryError(
                "component telemetry flush has no pending cohort records"
            )
        resolved = self._resolve_pending_records()
        sequences = [
            int(pending.payload["work_sequence"])
            for pending, _forward_s, _logits_s in resolved
        ]
        if any(
            current != previous + 1
            for previous, current in zip(sequences, sequences[1:])
        ):
            raise ComponentTelemetryError(
                "component telemetry pending work sequences are not contiguous"
            )

        for pending, forward_s, logits_s in resolved:
            record = dict(pending.payload)
            record["gpu_execution_s"] = forward_s + logits_s
            record["gpu_execution_breakdown_s"] = {
                "model_forward": forward_s,
                "lm_head_compute_logits": logits_s,
            }
            append_component_record(self._component_telemetry_path, record)
            removed = self._component_pending_records.popleft()
            if removed is not pending:
                raise ComponentTelemetryError(
                    "component telemetry pending queue order changed during flush"
                )
        if self._component_pending_records:
            raise ComponentTelemetryError(
                "component telemetry flush left pending cohort records"
            )
        return {
            "flushed_count": len(sequences),
            "work_sequences": sequences,
            "pending_count": 0,
        }

    def execute_model(self, scheduler_output: object) -> object:
        worker_execute_sequence = self._component_cycle_sequence
        self._component_cycle_sequence += 1
        extracted = extract_scheduler_cycle(scheduler_output)

        # vLLM runs synthetic prefill/decode during startup and an empty cleanup
        # cycle.  They establish kernels/cache state but are never observations.
        if extracted.warmup or not extracted.shape.has_work:
            return super().execute_model(scheduler_output)  # type: ignore[arg-type]

        work_sequence = self._component_work_sequence
        self._component_work_sequence += 1

        runner = self.model_runner
        if runner is None:
            raise ComponentTelemetryError("vLLM model runner is not initialized")
        model = getattr(runner, "model", None)
        original_forward = getattr(runner, "_model_forward", None)
        original_logits = getattr(model, "compute_logits", None)
        if not callable(original_forward) or not callable(original_logits):
            raise ComponentTelemetryError(
                "vLLM generate runner lacks model-forward or LM-head timing hooks"
            )

        forward_intervals: list[tuple[Any, Any]] = []
        logits_intervals: list[tuple[Any, Any]] = []
        forward_active = False

        def timed_forward(*args: Any, **kwargs: Any) -> Any:
            nonlocal forward_active
            if forward_active:
                raise ComponentTelemetryError("nested model forward is unsupported")
            forward_active = True
            try:
                return self._record_cuda_callable(
                    original_forward, forward_intervals, *args, **kwargs
                )
            finally:
                forward_active = False

        def timed_logits(*args: Any, **kwargs: Any) -> Any:
            if forward_active:
                raise ComponentTelemetryError(
                    "LM head executed inside model forward; timing intervals overlap"
                )
            return self._record_cuda_callable(
                original_logits, logits_intervals, *args, **kwargs
            )

        with (
            _temporary_callable(runner, "_model_forward", timed_forward),
            _temporary_callable(model, "compute_logits", timed_logits),
        ):
            output = super().execute_model(scheduler_output)  # type: ignore[arg-type]

        if len(forward_intervals) != 1 or len(logits_intervals) != 1:
            raise ComponentTelemetryError(
                "audited generate cycle requires exactly one model forward and "
                "one LM-head call; got "
                f"forward={len(forward_intervals)}, logits={len(logits_intervals)}"
            )

        source_contract = self._build_source_contract()
        source_contract_sha256 = hashlib.sha256(
            json.dumps(
                source_contract,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        phase_pure = extracted.shape.is_phase_pure
        reason = (
            "publishable"
            if phase_pure
            else "mixed_prefill_decode_not_component_identifiable"
        )
        parallel = self.parallel_config
        record = {
            "schema": COMPONENT_TELEMETRY_SCHEMA,
            "evidence_scope": "component_model_internal",
            "run_id": self._component_telemetry_run_id,
            "cycle_sequence": work_sequence,
            "work_sequence": work_sequence,
            "worker_execute_sequence": worker_execute_sequence,
            "cycle_id": make_cycle_id(
                self._component_telemetry_run_id,
                work_sequence,
                extracted.shape_digest,
            ),
            "shape_digest": extracted.shape_digest,
            "workload_fingerprint": extracted.workload_fingerprint,
            "shape": asdict(extracted.shape),
            "rank": int(self.rank),
            "local_rank": int(self.local_rank),
            "world_size": int(parallel.world_size),
            "tensor_parallel_size": int(parallel.tensor_parallel_size),
            "pipeline_parallel_size": int(parallel.pipeline_parallel_size),
            "publishable": phase_pure,
            "reason": reason,
            "boundary_contract": BOUNDARY_CONTRACT,
            "boundary_contract_sha256": BOUNDARY_CONTRACT_SHA256,
            "source_contract": source_contract,
            "source_contract_sha256": source_contract_sha256,
            "model_identity": self._model_identity(),
            "process_id": os.getpid(),
        }
        self._component_pending_records.append(_PendingComponentRecord(
            payload=record,
            forward_events=forward_intervals[0],
            logits_events=logits_intervals[0],
        ))
        return output

    def shutdown(self) -> None:
        try:
            if self._component_pending_records:
                # Normal calibration flushes after every cohort.  This path is
                # only a fail-closed last chance during orderly teardown and
                # uses the same non-blocking completion checks.
                self.flush_component_telemetry()
        finally:
            super().shutdown()


__all__ = ["VllmComponentTelemetryWorker"]
