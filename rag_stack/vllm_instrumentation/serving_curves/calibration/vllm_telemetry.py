"""Opt-in, per-scheduler-cycle telemetry for vLLM serving calibration.

This module deliberately has no import-time dependency on vLLM, torch, or
CUDA.  :mod:`vllm_server` installs :class:`VllmStageTelemetryStatLogger` as a
vLLM V1 stat-logger factory only when ``RAG_STACK_STAGE_TELEMETRY_PATH`` is
set.

The timestamp delivered with a scheduler stat is a frontend observation of a
completed EngineCore cycle.  Consequently, the interval between two records
is usable as active service only when the previous cycle left real backlog and
the current cycle reports real scheduled work.  The first record establishes
the clock origin and is never publishable.  Queue filling and idle intervals
remain in the JSONL audit trail but never receive an ``active_service_s``.

The logger itself delays the next observation.  Each interval therefore
subtracts the measured duration of the *previous* telemetry record, as well as
the current vLLM performance-stat calculation duration.  Both exclusions are
written explicitly; they are never silently folded into a fitted curve.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from rag_stack.vllm_instrumentation.calibration.runtime_identity import (
    CALIBRATION_RUN_ID_ENV,
    identity_sha256,
    resolve_calibration_run_id,
    scheduler_cycle_id,
    scheduler_shape_sha256,
    vllm_model_identity,
)

from .vllm_frontend_boundary import (
    consume_frontend_instrumentation_exclusion_s,
)


TELEMETRY_ENV = "RAG_STACK_STAGE_TELEMETRY_PATH"
TELEMETRY_SCHEMA = "rag_stack.vllm.scheduler_cycle_stage_service"

_APPEND_LOCK = threading.Lock()
_CONTEXT_FIELDS = (
    "num_prefill_requests",
    "prefill_num_tokens",
    "prefill_context_len",
    "prefill_token_context_product",
    "num_decode_requests",
    "decode_num_tokens",
    "decode_context_len",
    "decode_token_context_product",
)


@dataclass(frozen=True, slots=True)
class ScheduledPhaseShape:
    """Engine-observed work for one phase within one scheduler cycle."""

    scheduled_sequences: int
    scheduled_tokens: int
    scheduled_context_tokens: int
    token_context_product: int


@dataclass(frozen=True, slots=True)
class SchedulerCycleShape:
    """The exact prefill/decode work vLLM scheduled in one cycle."""

    terminal_performance_stage: Literal["prefill", "decode"] | None
    prefill: ScheduledPhaseShape
    decode: ScheduledPhaseShape

    @property
    def has_work(self) -> bool:
        return (
            self.prefill.scheduled_sequences > 0
            or self.decode.scheduled_sequences > 0
            or self.prefill.scheduled_tokens > 0
            or self.decode.scheduled_tokens > 0
        )


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer, got {value!r}")
    return value


def _nonnegative_finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return result


def _extract_cycle_shape(
    debug_stats: object,
) -> tuple[SchedulerCycleShape | None, str | None]:
    context = getattr(debug_stats, "context_breakdown", None)
    if not isinstance(context, Mapping):
        return None, "missing_context_breakdown"

    try:
        values = {
            field: _nonnegative_int(context[field], field=field)
            for field in _CONTEXT_FIELDS
        }
    except KeyError as exc:
        return None, f"missing_context_field:{exc.args[0]}"
    except ValueError as exc:
        return None, f"invalid_context_breakdown:{exc}"

    # vLLM exposes the request counts both directly and in the context
    # breakdown.  A disagreement means this is not the 0.18.1 contract we
    # calibrated against; fail closed instead of choosing one silently.
    for phase in ("prefill", "decode"):
        field = f"num_{phase}_requests"
        direct_value = getattr(debug_stats, field, None)
        if direct_value != values[field]:
            return None, f"debug_context_mismatch:{field}"

    prefill = ScheduledPhaseShape(
        scheduled_sequences=values["num_prefill_requests"],
        scheduled_tokens=values["prefill_num_tokens"],
        scheduled_context_tokens=values["prefill_context_len"],
        token_context_product=values["prefill_token_context_product"],
    )
    decode = ScheduledPhaseShape(
        scheduled_sequences=values["num_decode_requests"],
        scheduled_tokens=values["decode_num_tokens"],
        scheduled_context_tokens=values["decode_context_len"],
        token_context_product=values["decode_token_context_product"],
    )

    if prefill.scheduled_context_tokens < prefill.scheduled_tokens:
        return None, "invalid_context_breakdown:prefill_context_lt_tokens"
    if decode.scheduled_context_tokens < decode.scheduled_tokens:
        return None, "invalid_context_breakdown:decode_context_lt_tokens"

    # A mixed cycle has exactly one terminal contribution.  Decode is terminal
    # because it materializes the cycle's sampled-token/postprocess result.
    terminal: Literal["prefill", "decode"] | None
    if decode.scheduled_sequences or decode.scheduled_tokens:
        terminal = "decode"
    elif prefill.scheduled_sequences or prefill.scheduled_tokens:
        terminal = "prefill"
    else:
        terminal = None

    return SchedulerCycleShape(
        terminal_performance_stage=terminal,
        prefill=prefill,
        decode=decode,
    ), None


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    """Append one complete JSON object under thread and process locks."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")

    with _APPEND_LOCK:
        fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            # The wrapper is intentionally single-frontend, but an advisory
            # process lock also protects accidental concurrent writers and
            # per-engine logger instances sharing one audit file.
            fcntl.flock(fd, fcntl.LOCK_EX)
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("short write while appending vLLM telemetry")
                view = view[written:]
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)


class VllmStageTelemetryStatLogger:
    """Duck-typed vLLM V1 per-engine ``StatLoggerBase`` implementation.

    vLLM's stat-logger API is explicitly unstable.  Avoiding inheritance keeps
    this module importable in unit tests and tooling that do not install vLLM;
    vLLM's ``PerEngineStatLoggerAdapter`` accepts the same callable contract.
    """

    def __init__(
        self,
        vllm_config: object,
        engine_index: int = 0,
        *,
        output_path: str | os.PathLike[str] | None = None,
        run_id: str | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        try:
            self._model_identity = vllm_model_identity(vllm_config)
            self._model_identity_sha256 = identity_sha256(self._model_identity)
        except ValueError:
            # Keep the telemetry logger unit-testable with a duck-typed empty
            # config.  The trusted publication driver rejects records without
            # a complete identity, so this can never become fit evidence.
            self._model_identity = None
            self._model_identity_sha256 = None
        parallel_config = getattr(vllm_config, "parallel_config", None)
        try:
            self._tensor_parallel_size = _nonnegative_int(
                getattr(parallel_config, "tensor_parallel_size", None),
                field="tensor_parallel_size",
            )
            self._pipeline_parallel_size = _nonnegative_int(
                getattr(parallel_config, "pipeline_parallel_size", None),
                field="pipeline_parallel_size",
            )
            if self._tensor_parallel_size == 0 or self._pipeline_parallel_size == 0:
                raise ValueError("parallel sizes must be positive")
        except ValueError:
            self._tensor_parallel_size = None
            self._pipeline_parallel_size = None
        configured_path = output_path or os.environ.get(TELEMETRY_ENV)
        if configured_path is None or not str(configured_path).strip():
            raise RuntimeError(
                f"{TELEMETRY_ENV} must name an append-only JSONL output file"
            )

        self._path = Path(configured_path).expanduser()
        try:
            self._run_id = resolve_calibration_run_id(explicit=run_id)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        self._engine_index = _nonnegative_int(engine_index, field="engine_index")
        self._monotonic = monotonic
        self._sequence = 0
        self._work_sequence = 0
        self._previous_record_started_s: float | None = None
        self._previous_record_overhead_s = 0.0
        self._previous_running: int | None = None
        self._previous_waiting: int | None = None

    @property
    def output_path(self) -> Path:
        return self._path

    def record(
        self,
        scheduler_stats: object | None,
        iteration_stats: object | None,
        mm_cache_stats: object | None = None,
        engine_idx: int = 0,
    ) -> None:
        del iteration_stats, mm_cache_stats
        if scheduler_stats is None:
            # An output-only frontend callback is not a scheduler-cycle sample
            # and must not split the next real cycle interval.
            return

        if int(engine_idx) != self._engine_index:
            raise RuntimeError(
                "vLLM telemetry logger received a different engine index: "
                f"factory={self._engine_index}, record={engine_idx}"
            )

        record_started_s = float(self._monotonic())
        if not math.isfinite(record_started_s):
            raise RuntimeError("monotonic clock returned a non-finite value")

        running_error: str | None = None
        try:
            current_running = _nonnegative_int(
                getattr(scheduler_stats, "num_running_reqs", None),
                field="num_running_reqs",
            )
            current_waiting = _nonnegative_int(
                getattr(scheduler_stats, "num_waiting_reqs", None),
                field="num_waiting_reqs",
            )
        except ValueError as exc:
            current_running = 0
            current_waiting = 0
            running_error = f"invalid_backlog_stats:{exc}"

        perf_stats = getattr(scheduler_stats, "perf_stats", None)
        debug_stats = getattr(perf_stats, "debug_stats", None)
        shape: SchedulerCycleShape | None = None
        shape_error: str | None = None
        calc_duration_s: float | None = None
        if debug_stats is None:
            shape_error = "missing_perf_debug_stats"
        else:
            shape, shape_error = _extract_cycle_shape(debug_stats)
            try:
                calc_duration_s = _nonnegative_finite_float(
                    getattr(debug_stats, "calc_duration", None),
                    field="perf_debug_calc_duration",
                )
            except ValueError:
                shape_error = shape_error or "invalid_perf_debug_calc_duration"

        raw_interval_s: float | None
        if self._previous_record_started_s is None:
            raw_interval_s = None
        else:
            raw_interval_s = record_started_s - self._previous_record_started_s

        # Production-path coverage spans are written on the same API frontend
        # event loop.  Their values are evidence only and are never added to
        # stage service.  Remove the measured JSONL instrumentation itself from
        # the scheduler interval so enabling the proof cannot inflate a curve.
        frontend_telemetry_s = consume_frontend_instrumentation_exclusion_s()

        previous_backlog = (
            None
            if self._previous_running is None or self._previous_waiting is None
            else self._previous_running + self._previous_waiting
        )
        has_work = shape is not None and shape.has_work

        reason = "publishable"
        active_service_s: float | None = None
        if raw_interval_s is None:
            reason = "first_sample_no_interval"
        elif not math.isfinite(raw_interval_s) or raw_interval_s <= 0.0:
            reason = "non_positive_cycle_interval"
        elif running_error is not None:
            reason = running_error
        elif shape_error is not None:
            reason = shape_error
        elif previous_backlog is None:
            reason = "missing_previous_backlog_state"
        elif previous_backlog <= 0:
            reason = "previous_cycle_not_backlogged"
        elif not has_work:
            reason = "current_cycle_has_no_scheduled_work"
        elif calc_duration_s is None:
            reason = "missing_perf_debug_calc_duration"
        else:
            corrected = (
                raw_interval_s
                - self._previous_record_overhead_s
                - calc_duration_s
                - frontend_telemetry_s
            )
            if not math.isfinite(corrected) or corrected <= 0.0:
                reason = "instrumentation_exceeds_cycle_interval"
            else:
                active_service_s = corrected

        publishable = reason == "publishable" and active_service_s is not None
        work_sequence: int | None = None
        shape_digest: str | None = None
        cycle_id: str | None = None
        if shape is not None and shape.has_work:
            work_sequence = self._work_sequence
            self._work_sequence += 1
            shape_digest = scheduler_shape_sha256(shape)
            cycle_id = scheduler_cycle_id(
                self._run_id, work_sequence, shape_digest
            )
        total_exclusion_s = (
            None
            if calc_duration_s is None
            else self._previous_record_overhead_s + calc_duration_s
            + frontend_telemetry_s
        )

        instrumentation_exclusions = {
            "previous_telemetry_record_s": self._previous_record_overhead_s,
            "current_vllm_perf_debug_calc_s": calc_duration_s,
            "frontend_boundary_telemetry_s": frontend_telemetry_s,
            "total_s": total_exclusion_s,
        }
        record = {
            "schema": TELEMETRY_SCHEMA,
            "evidence_scope": "active_stage_service",
            "run_id": self._run_id,
            "sequence": self._sequence,
            "work_sequence": work_sequence,
            "cycle_id": cycle_id,
            "shape_digest": shape_digest,
            "engine": self._engine_index,
            "model_identity": self._model_identity,
            "model_identity_sha256": self._model_identity_sha256,
            "tensor_parallel_size": self._tensor_parallel_size,
            "pipeline_parallel_size": self._pipeline_parallel_size,
            "process_id": os.getpid(),
            "monotonic_timestamp_s": record_started_s,
            "raw_interval_s": raw_interval_s,
            "instrumentation_exclusions_s": instrumentation_exclusions,
            "active_service_s": active_service_s,
            "shape": None if shape is None else asdict(shape),
            "backlog_before_interval": None
            if previous_backlog is None
            else {
                "running_sequences": self._previous_running,
                "waiting_sequences": self._previous_waiting,
            },
            "backlog_after_cycle": {
                "running_sequences": current_running,
                "waiting_sequences": current_waiting,
            },
            "publishable": publishable,
            "reason": reason,
        }

        _append_jsonl(self._path, record)
        record_finished_s = float(self._monotonic())
        measured_overhead = record_finished_s - record_started_s
        self._previous_record_overhead_s = (
            measured_overhead
            if math.isfinite(measured_overhead) and measured_overhead >= 0.0
            else 0.0
        )
        self._previous_record_started_s = record_started_s
        self._previous_running = current_running
        self._previous_waiting = current_waiting
        self._sequence += 1

    def log_engine_initialized(self) -> None:
        pass

    def log(self) -> None:
        pass

    def record_sleep_state(self, is_awake: int, level: int) -> None:
        del is_awake, level


__all__ = [
    "CALIBRATION_RUN_ID_ENV",
    "ScheduledPhaseShape",
    "SchedulerCycleShape",
    "TELEMETRY_ENV",
    "TELEMETRY_SCHEMA",
    "VllmStageTelemetryStatLogger",
]
