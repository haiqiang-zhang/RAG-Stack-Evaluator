"""Schemas and validators for vLLM component-only calibration telemetry.

The records produced here are *not* serving-stage observations.  They cover
only the GPU work represented by the GenZ component model: the transformer
forward (including tensor-parallel collectives issued by the model) and the LM
head.  Scheduler, HTTP, queueing, host preprocessing, sampling and result
materialisation stay outside this boundary.

This module deliberately has no direct dependency on vLLM, torch or CUDA.  The
runtime worker is loaded lazily only when vLLM resolves the opt-in
``--worker-cls`` value.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import threading
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from rag_stack_evaluator.vllm_instrumentation.calibration.runtime_identity import (
    CALIBRATION_RUN_ID_ENV,
    LEGACY_COMPONENT_RUN_ID_ENV,
    scheduler_cycle_id,
    scheduler_shape_sha256,
    validate_calibration_run_id,
)


COMPONENT_TELEMETRY_PATH_ENV = "RAG_STACK_LLM_COMPONENT_TELEMETRY_PATH"
COMPONENT_TELEMETRY_RUN_ID_ENV = LEGACY_COMPONENT_RUN_ID_ENV
COMPONENT_TELEMETRY_SCHEMA = "rag_stack.vllm.genz_component_gpu_execution"
COMPONENT_WORKER_QUALNAME = (
    "rag_stack_evaluator.vllm_instrumentation.llm_sim.calibration.component_telemetry."
    "VllmComponentTelemetryWorker"
)

BOUNDARY_CONTRACT: dict[str, object] = {
    "name": "genz_component_gpu_execution",
    "included": [
        "input_embedding_and_transformer_forward_gpu",
        "tensor_parallel_collectives_issued_on_the_model_stream",
        "lm_head_compute_logits_gpu",
    ],
    "excluded": [
        "http_and_request_queueing",
        "scheduler_and_dynamic_batch_waiting",
        "host_input_preparation_and_dispatch",
        "kv_connector_and_cross_performance_stage_transfer",
        "sampling_and_host_bookkeeping",
        "serialization_and_result_materialization",
        "telemetry_write_and_event_synchronization",
    ],
    # In vLLM 0.18.1 PP receive/send is outside GPUModelRunner's stable model
    # call boundary.  The runtime worker therefore rejects PP>1 rather than
    # publishing an observation with missing component-internal communication.
    "pipeline_parallel": "unsupported_fail_closed",
    "rank_assembly": "maximum_rank_gpu_duration",
}


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


BOUNDARY_CONTRACT_SHA256 = hashlib.sha256(
    _canonical_json(BOUNDARY_CONTRACT)
).hexdigest()

_APPEND_LOCK = threading.Lock()


class ComponentTelemetryError(RuntimeError):
    """A telemetry record cannot be used as component calibration evidence."""


@dataclass(frozen=True, slots=True)
class ComponentPhaseShape:
    """Actual work for one LLM phase in one vLLM scheduler cycle."""

    scheduled_sequences: int
    scheduled_tokens: int
    scheduled_context_tokens: int
    token_context_product: int


@dataclass(frozen=True, slots=True)
class ComponentCycleShape:
    """Scheduler-observed prefill/decode work, independent of client load."""

    terminal_performance_stage: Literal["prefill", "decode"] | None
    prefill: ComponentPhaseShape
    decode: ComponentPhaseShape

    @property
    def has_work(self) -> bool:
        return bool(
            self.prefill.scheduled_sequences
            or self.decode.scheduled_sequences
            or self.prefill.scheduled_tokens
            or self.decode.scheduled_tokens
        )

    @property
    def is_phase_pure(self) -> bool:
        return not (
            self.prefill.scheduled_sequences and self.decode.scheduled_sequences
        )


@dataclass(frozen=True, slots=True)
class ExtractedCycle:
    """Strictly validated worker-side scheduler-cycle identity."""

    shape: ComponentCycleShape
    shape_digest: str
    workload_fingerprint: str
    warmup: bool


@dataclass(frozen=True, slots=True)
class ComponentCycleObservation:
    """One complete TP group's component service observation."""

    run_id: str
    cycle_id: str
    cycle_sequence: int
    shape: ComponentCycleShape
    component_service_s: float
    rank_service_s: tuple[float, ...]
    tensor_parallel_size: int
    model_identity_sha256: str
    source_contract_sha256: str
    boundary_contract_sha256: str


@dataclass(frozen=True, slots=True)
class JoinedStageComponentCycle:
    """Diagnostic join of distinct sensors for one scheduler cycle.

    A join proves cycle correspondence; it does not authorize fitting the
    component model and stage overhead in the same publication generation.
    """

    run_id: str
    cycle_id: str
    work_sequence: int
    shape: ComponentCycleShape
    component_service_s: float
    active_stage_service_s: float | None
    stage_publishable: bool
    stage_reason: str


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ComponentTelemetryError(
            f"{field} must be a non-negative integer, got {value!r}"
        )
    return value


def _positive_int(value: object, *, field: str) -> int:
    result = _nonnegative_int(value, field=field)
    if result == 0:
        raise ComponentTelemetryError(f"{field} must be positive")
    return result


def _finite_nonnegative_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComponentTelemetryError(
            f"{field} must be a finite non-negative number"
        )
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ComponentTelemetryError(
            f"{field} must be a finite non-negative number"
        )
    return result


def _request_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ComponentTelemetryError(f"{field} must be a non-empty string")
    return value


def _sequence_attr(obj: object, name: str) -> list[object]:
    value = getattr(obj, name, None)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ComponentTelemetryError(f"scheduler_output.{name} must be a sequence")
    return list(value)


def extract_scheduler_cycle(scheduler_output: object) -> ExtractedCycle:
    """Extract an exact component workload from vLLM's ``SchedulerOutput``.

    Request batch, scheduled sequences/tokens and accumulated context are
    derived from worker input.  HTTP client concurrency is neither read nor
    represented.  A cached request with zero output tokens remains prefill even
    when its final chunk contains one token; this avoids treating concurrency or
    token-count heuristics as phase semantics.
    """

    scheduled_raw = getattr(scheduler_output, "num_scheduled_tokens", None)
    if not isinstance(scheduled_raw, Mapping):
        raise ComponentTelemetryError(
            "scheduler_output.num_scheduled_tokens must be a mapping"
        )

    scheduled: dict[str, int] = {}
    for raw_req_id, raw_tokens in scheduled_raw.items():
        req_id = _request_id(raw_req_id, field="scheduled request id")
        if req_id in scheduled:
            raise ComponentTelemetryError(f"duplicate scheduled request {req_id!r}")
        scheduled[req_id] = _positive_int(
            raw_tokens, field=f"num_scheduled_tokens[{req_id!r}]"
        )

    total = _nonnegative_int(
        getattr(scheduler_output, "total_num_scheduled_tokens", None),
        field="total_num_scheduled_tokens",
    )
    if total != sum(scheduled.values()):
        raise ComponentTelemetryError(
            "total_num_scheduled_tokens does not equal the per-request sum"
        )

    new_requests = _sequence_attr(scheduler_output, "scheduled_new_reqs")
    new_state: dict[str, int] = {}
    for index, request in enumerate(new_requests):
        req_id = _request_id(
            getattr(request, "req_id", None),
            field=f"scheduled_new_reqs[{index}].req_id",
        )
        if req_id in new_state:
            raise ComponentTelemetryError(f"duplicate new request {req_id!r}")
        new_state[req_id] = _nonnegative_int(
            getattr(request, "num_computed_tokens", None),
            field=f"scheduled_new_reqs[{index}].num_computed_tokens",
        )

    cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
    if cached is None:
        raise ComponentTelemetryError("scheduler_output.scheduled_cached_reqs is missing")
    cached_ids = _sequence_attr(cached, "req_ids")
    cached_computed = _sequence_attr(cached, "num_computed_tokens")
    cached_outputs = _sequence_attr(cached, "num_output_tokens")
    if not (
        len(cached_ids) == len(cached_computed) == len(cached_outputs)
    ):
        raise ComponentTelemetryError(
            "cached request ids/computed/output-token arrays have different lengths"
        )

    cached_state: dict[str, tuple[int, int]] = {}
    for index, (raw_req_id, raw_computed, raw_outputs) in enumerate(
        zip(cached_ids, cached_computed, cached_outputs, strict=True)
    ):
        req_id = _request_id(raw_req_id, field=f"cached req_ids[{index}]")
        if req_id in cached_state:
            raise ComponentTelemetryError(f"duplicate cached request {req_id!r}")
        if req_id in new_state:
            raise ComponentTelemetryError(
                f"request {req_id!r} appears in both new and cached inputs"
            )
        cached_state[req_id] = (
            _nonnegative_int(
                raw_computed, field=f"cached num_computed_tokens[{index}]"
            ),
            _nonnegative_int(
                raw_outputs, field=f"cached num_output_tokens[{index}]"
            ),
        )

    phase_totals: dict[str, list[int]] = {
        "prefill": [0, 0, 0, 0],
        "decode": [0, 0, 0, 0],
    }
    canonical_items: list[dict[str, object]] = []
    warmup_flags: list[bool] = []
    for req_id, num_tokens in scheduled.items():
        if req_id in new_state:
            computed = new_state[req_id]
            phase = "prefill"
        elif req_id in cached_state:
            computed, output_tokens = cached_state[req_id]
            phase = "prefill" if output_tokens == 0 else "decode"
        else:
            raise ComponentTelemetryError(
                f"scheduled request {req_id!r} has no new/cached state"
            )

        context_len = computed + num_tokens
        bucket = phase_totals[phase]
        bucket[0] += 1
        bucket[1] += num_tokens
        bucket[2] += context_len
        bucket[3] += num_tokens * context_len
        canonical_items.append(
            {
                # Do not persist request text/identities in calibration data.
                "request_sha256": hashlib.sha256(req_id.encode("utf-8")).hexdigest(),
                "phase": phase,
                "scheduled_tokens": num_tokens,
                "context_tokens": context_len,
            }
        )
        warmup_flags.append(req_id.startswith("_warmup_") and req_id.endswith("_"))

    if any(warmup_flags) and not all(warmup_flags):
        raise ComponentTelemetryError(
            "synthetic warmup and production requests cannot share a cycle"
        )

    def _phase(name: str) -> ComponentPhaseShape:
        values = phase_totals[name]
        return ComponentPhaseShape(*values)

    prefill = _phase("prefill")
    decode = _phase("decode")
    terminal: Literal["prefill", "decode"] | None
    if decode.scheduled_sequences:
        terminal = "decode"
    elif prefill.scheduled_sequences:
        terminal = "prefill"
    else:
        terminal = None
    shape = ComponentCycleShape(
        terminal_performance_stage=terminal,
        prefill=prefill,
        decode=decode,
    )
    workload_payload = {
        "shape": asdict(shape),
        "items": sorted(canonical_items, key=lambda item: str(item["request_sha256"])),
    }
    return ExtractedCycle(
        shape=shape,
        # This aggregate digest is intentionally shared with the frontend stat
        # logger.  Per-request identities remain in a separate worker-only
        # fingerprint and are never required for a join.
        shape_digest=scheduler_shape_sha256(shape),
        workload_fingerprint=hashlib.sha256(
            _canonical_json(workload_payload)
        ).hexdigest(),
        warmup=bool(warmup_flags and all(warmup_flags)),
    )


def validate_run_id(value: object) -> str:
    try:
        return validate_calibration_run_id(value)
    except ValueError as exc:
        raise ComponentTelemetryError(str(exc)) from exc


def make_cycle_id(run_id: str, cycle_sequence: int, shape_digest: str) -> str:
    try:
        return scheduler_cycle_id(run_id, cycle_sequence, shape_digest)
    except ValueError as exc:
        raise ComponentTelemetryError(str(exc)) from exc


def append_component_record(
    path: str | os.PathLike[str], record: Mapping[str, Any]
) -> None:
    """Append one atomic JSONL record under thread and process locks."""

    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json(dict(record)) + b"\n"
    with _APPEND_LOCK:
        fd = os.open(output_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("short write while appending component telemetry")
                view = view[written:]
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)


def _shape_from_mapping(value: object) -> ComponentCycleShape:
    if not isinstance(value, Mapping):
        raise ComponentTelemetryError("shape must be an object")

    def _phase(name: str) -> ComponentPhaseShape:
        raw = value.get(name)
        if not isinstance(raw, Mapping):
            raise ComponentTelemetryError(f"shape.{name} must be an object")
        return ComponentPhaseShape(
            scheduled_sequences=_nonnegative_int(
                raw.get("scheduled_sequences"),
                field=f"shape.{name}.scheduled_sequences",
            ),
            scheduled_tokens=_nonnegative_int(
                raw.get("scheduled_tokens"), field=f"shape.{name}.scheduled_tokens"
            ),
            scheduled_context_tokens=_nonnegative_int(
                raw.get("scheduled_context_tokens"),
                field=f"shape.{name}.scheduled_context_tokens",
            ),
            token_context_product=_nonnegative_int(
                raw.get("token_context_product"),
                field=f"shape.{name}.token_context_product",
            ),
        )

    terminal = value.get("terminal_performance_stage")
    if terminal not in (None, "prefill", "decode"):
        raise ComponentTelemetryError("shape terminal stage is invalid")
    return ComponentCycleShape(
        terminal_performance_stage=terminal,
        prefill=_phase("prefill"),
        decode=_phase("decode"),
    )


def aggregate_component_cycle(
    records: Sequence[Mapping[str, Any]],
    *,
    require_calibration_eligible: bool = True,
) -> ComponentCycleObservation:
    """Validate and assemble one complete TP group, taking its critical rank.

    The function rejects partial, duplicated, mixed-phase, PP or cross-contract
    evidence.  It never fills a missing rank with zero.
    """

    if not records:
        raise ComponentTelemetryError("component cycle has no rank records")

    first = records[0]
    run_id = validate_run_id(first.get("run_id"))
    cycle_id = first.get("cycle_id")
    if not isinstance(cycle_id, str) or not cycle_id:
        raise ComponentTelemetryError("cycle_id must be a non-empty string")
    cycle_sequence = _nonnegative_int(
        first.get("cycle_sequence"), field="cycle_sequence"
    )
    work_sequence = _nonnegative_int(
        first.get("work_sequence"), field="work_sequence"
    )
    if work_sequence != cycle_sequence:
        raise ComponentTelemetryError(
            "cycle_sequence and work_sequence must identify the same cycle"
        )
    shape = _shape_from_mapping(first.get("shape"))
    if not shape.has_work:
        raise ComponentTelemetryError("component calibration cycle has no work")
    if require_calibration_eligible and not shape.is_phase_pure:
        raise ComponentTelemetryError(
            "mixed prefill/decode cycles cannot calibrate either GenZ phase"
        )
    source_hash = first.get("source_contract_sha256")
    boundary_hash = first.get("boundary_contract_sha256")
    if (
        not isinstance(source_hash, str)
        or len(source_hash) != 64
        or any(char not in "0123456789abcdef" for char in source_hash)
    ):
        raise ComponentTelemetryError("source_contract_sha256 is invalid")
    source_contract = first.get("source_contract")
    if not isinstance(source_contract, Mapping):
        raise ComponentTelemetryError("source_contract must be an object")
    if hashlib.sha256(_canonical_json(dict(source_contract))).hexdigest() != source_hash:
        raise ComponentTelemetryError("source contract content hash does not match")
    if boundary_hash != BOUNDARY_CONTRACT_SHA256:
        raise ComponentTelemetryError("component boundary contract does not match")
    if first.get("boundary_contract") != BOUNDARY_CONTRACT:
        raise ComponentTelemetryError("component boundary contract payload changed")
    shape_digest = first.get("shape_digest")
    if (
        not isinstance(shape_digest, str)
        or len(shape_digest) != 64
        or any(char not in "0123456789abcdef" for char in shape_digest)
    ):
        raise ComponentTelemetryError("shape_digest is invalid")
    if cycle_id != make_cycle_id(run_id, cycle_sequence, shape_digest):
        raise ComponentTelemetryError("cycle_id does not match its run/sequence/shape")
    if shape_digest != scheduler_shape_sha256(shape):
        raise ComponentTelemetryError("shape_digest does not match aggregate shape")
    workload_fingerprint = first.get("workload_fingerprint")
    if (
        not isinstance(workload_fingerprint, str)
        or len(workload_fingerprint) != 64
        or any(char not in "0123456789abcdef" for char in workload_fingerprint)
    ):
        raise ComponentTelemetryError("workload_fingerprint is invalid")
    model_identity = first.get("model_identity")
    if not isinstance(model_identity, Mapping) or not model_identity:
        raise ComponentTelemetryError("model_identity must be a non-empty object")
    model_identity_sha256 = hashlib.sha256(
        _canonical_json(dict(model_identity))
    ).hexdigest()

    world_size = _positive_int(first.get("world_size"), field="world_size")
    tensor_parallel_size = _positive_int(
        first.get("tensor_parallel_size"), field="tensor_parallel_size"
    )
    pipeline_parallel_size = _positive_int(
        first.get("pipeline_parallel_size"), field="pipeline_parallel_size"
    )
    if pipeline_parallel_size != 1:
        raise ComponentTelemetryError(
            "PP component telemetry is unsupported until PP communication is timed"
        )
    if world_size != tensor_parallel_size:
        raise ComponentTelemetryError(
            "component telemetry currently requires DP=1 and world_size=TP"
        )

    rank_times: dict[int, float] = {}
    reference_shape = asdict(shape)
    for record in records:
        if record.get("schema") != COMPONENT_TELEMETRY_SCHEMA:
            raise ComponentTelemetryError("unexpected component telemetry schema")
        if record.get("publishable") is not True:
            mixed_audit_record = (
                not require_calibration_eligible
                and not shape.is_phase_pure
                and record.get("reason")
                == "mixed_prefill_decode_not_component_identifiable"
            )
            if not mixed_audit_record:
                raise ComponentTelemetryError(
                    f"rank record is not publishable: {record.get('reason')!r}"
                )
        invariants = {
            "evidence_scope": "component_model_internal",
            "run_id": run_id,
            "cycle_id": cycle_id,
            "cycle_sequence": cycle_sequence,
            "work_sequence": work_sequence,
            "shape": reference_shape,
            "shape_digest": shape_digest,
            "workload_fingerprint": workload_fingerprint,
            "source_contract": source_contract,
            "source_contract_sha256": source_hash,
            "boundary_contract": BOUNDARY_CONTRACT,
            "boundary_contract_sha256": boundary_hash,
            "model_identity": model_identity,
            "world_size": world_size,
            "tensor_parallel_size": tensor_parallel_size,
            "pipeline_parallel_size": pipeline_parallel_size,
        }
        for field, expected in invariants.items():
            if record.get(field) != expected:
                raise ComponentTelemetryError(
                    f"rank records disagree on {field}: "
                    f"{record.get(field)!r} != {expected!r}"
                )
        rank = _nonnegative_int(record.get("rank"), field="rank")
        if rank in rank_times:
            raise ComponentTelemetryError(f"duplicate component telemetry rank {rank}")
        total_s = _finite_nonnegative_float(
            record.get("gpu_execution_s"), field="gpu_execution_s"
        )
        breakdown = record.get("gpu_execution_breakdown_s")
        if not isinstance(breakdown, Mapping):
            raise ComponentTelemetryError(
                "gpu_execution_breakdown_s must be an object"
            )
        forward_s = _finite_nonnegative_float(
            breakdown.get("model_forward"),
            field="gpu_execution_breakdown_s.model_forward",
        )
        logits_s = _finite_nonnegative_float(
            breakdown.get("lm_head_compute_logits"),
            field="gpu_execution_breakdown_s.lm_head_compute_logits",
        )
        if forward_s <= 0.0 or logits_s <= 0.0:
            raise ComponentTelemetryError(
                "both audited component CUDA intervals must be positive"
            )
        if not math.isclose(
            total_s, forward_s + logits_s, rel_tol=1e-12, abs_tol=1e-15
        ):
            raise ComponentTelemetryError(
                "gpu_execution_s does not equal the audited CUDA interval sum"
            )
        rank_times[rank] = total_s

    expected_ranks = set(range(world_size))
    actual_ranks = set(rank_times)
    if actual_ranks != expected_ranks:
        raise ComponentTelemetryError(
            "component telemetry rank set is incomplete: "
            f"expected={sorted(expected_ranks)}, actual={sorted(actual_ranks)}"
        )

    ordered = tuple(rank_times[rank] for rank in range(world_size))
    service = max(ordered)
    if service <= 0.0:
        raise ComponentTelemetryError("component GPU execution time must be positive")
    return ComponentCycleObservation(
        run_id=run_id,
        cycle_id=cycle_id,
        cycle_sequence=cycle_sequence,
        shape=shape,
        component_service_s=service,
        rank_service_s=ordered,
        tensor_parallel_size=tensor_parallel_size,
        model_identity_sha256=model_identity_sha256,
        source_contract_sha256=source_hash,
        boundary_contract_sha256=boundary_hash,
    )


def parse_component_observations(
    records: Sequence[Mapping[str, Any]],
) -> tuple[ComponentCycleObservation, ...]:
    """Parse one pure-phase run into the exact offline-fitter input.

    Output retains the phase shape, critical-rank ``component_service_s`` and
    the complete ordered TP rank durations.  Any missing/duplicate rank,
    mixed-phase cycle or sequence gap fails the whole parse.
    """

    if not records:
        raise ComponentTelemetryError("component telemetry run is empty")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        cycle_id = record.get("cycle_id")
        if not isinstance(cycle_id, str) or not cycle_id:
            raise ComponentTelemetryError("component cycle_id is invalid")
        grouped.setdefault(cycle_id, []).append(record)
    observations = [
        aggregate_component_cycle(group) for group in grouped.values()
    ]
    observations.sort(key=lambda item: item.cycle_sequence)
    sequences = [item.cycle_sequence for item in observations]
    if sequences != sorted(set(sequences)):
        raise ComponentTelemetryError(
            f"component work_sequence is not strictly increasing: {sequences}"
        )
    run_ids = {item.run_id for item in observations}
    if len(run_ids) > 1:
        raise ComponentTelemetryError(
            f"component records span multiple calibration runs: {sorted(run_ids)}"
        )
    return tuple(observations)


def join_stage_component_records(
    stage_records: Sequence[Mapping[str, Any]],
    component_records: Sequence[Mapping[str, Any]],
) -> tuple[JoinedStageComponentCycle, ...]:
    """Strictly join frontend and all-rank worker telemetry by cycle identity.

    Every real-work cycle must have exactly one frontend record and a complete
    TP rank set. Missing, duplicated, reordered or shape-mismatched evidence is
    rejected; no fuzzy timestamp, request batch or concurrency matching is
    attempted.
    """

    stage_by_cycle: dict[str, Mapping[str, Any]] = {}
    for record in stage_records:
        if record.get("schema") != "rag_stack.vllm.scheduler_cycle_stage_service":
            raise ComponentTelemetryError("unexpected stage telemetry schema")
        cycle_id = record.get("cycle_id")
        work_sequence = record.get("work_sequence")
        if cycle_id is None:
            if work_sequence is not None or record.get("shape_digest") is not None:
                raise ComponentTelemetryError(
                    "no-work stage record has a partial cycle identity"
                )
            continue
        if not isinstance(cycle_id, str) or not cycle_id:
            raise ComponentTelemetryError("stage cycle_id is invalid")
        if cycle_id in stage_by_cycle:
            raise ComponentTelemetryError(
                f"duplicate stage telemetry cycle {cycle_id!r}"
            )
        if record.get("evidence_scope") != "active_stage_service":
            raise ComponentTelemetryError("unexpected stage evidence scope")
        if record.get("engine") != 0:
            raise ComponentTelemetryError(
                "component join currently requires the sole DP engine index 0"
            )
        stage_by_cycle[cycle_id] = record

    component_by_cycle: dict[str, list[Mapping[str, Any]]] = {}
    for record in component_records:
        cycle_id = record.get("cycle_id")
        if not isinstance(cycle_id, str) or not cycle_id:
            raise ComponentTelemetryError("component cycle_id is invalid")
        component_by_cycle.setdefault(cycle_id, []).append(record)

    if set(stage_by_cycle) != set(component_by_cycle):
        missing_stage = sorted(set(component_by_cycle) - set(stage_by_cycle))
        missing_component = sorted(set(stage_by_cycle) - set(component_by_cycle))
        raise ComponentTelemetryError(
            "stage/component cycle sets differ: "
            f"missing_stage={missing_stage}, missing_component={missing_component}"
        )

    joined: list[JoinedStageComponentCycle] = []
    for cycle_id, stage in stage_by_cycle.items():
        component = aggregate_component_cycle(
            component_by_cycle[cycle_id], require_calibration_eligible=False
        )
        stage_work_sequence = _nonnegative_int(
            stage.get("work_sequence"), field="stage.work_sequence"
        )
        if stage.get("run_id") != component.run_id:
            raise ComponentTelemetryError("stage/component run_id mismatch")
        if stage_work_sequence != component.cycle_sequence:
            raise ComponentTelemetryError("stage/component work_sequence mismatch")
        if stage.get("shape_digest") != scheduler_shape_sha256(component.shape):
            raise ComponentTelemetryError("stage/component shape digest mismatch")
        if stage.get("shape") != asdict(component.shape):
            raise ComponentTelemetryError("stage/component aggregate shape mismatch")
        if stage.get("cycle_id") != component.cycle_id:
            raise ComponentTelemetryError("stage/component cycle_id mismatch")

        active_raw = stage.get("active_service_s")
        active_service_s: float | None
        if active_raw is None:
            active_service_s = None
        else:
            active_service_s = _finite_nonnegative_float(
                active_raw, field="stage.active_service_s"
            )
            if active_service_s <= 0.0:
                raise ComponentTelemetryError(
                    "publishable active-stage service must be positive"
                )
        stage_publishable = stage.get("publishable") is True
        if stage_publishable != (active_service_s is not None):
            raise ComponentTelemetryError(
                "stage publishable flag disagrees with active_service_s"
            )
        stage_reason = stage.get("reason")
        if not isinstance(stage_reason, str) or not stage_reason:
            raise ComponentTelemetryError("stage reason is invalid")
        joined.append(
            JoinedStageComponentCycle(
                run_id=component.run_id,
                cycle_id=component.cycle_id,
                work_sequence=component.cycle_sequence,
                shape=component.shape,
                component_service_s=component.component_service_s,
                active_stage_service_s=active_service_s,
                stage_publishable=stage_publishable,
                stage_reason=stage_reason,
            )
        )

    joined.sort(key=lambda item: item.work_sequence)
    sequences = [item.work_sequence for item in joined]
    if sequences != list(range(len(sequences))):
        raise ComponentTelemetryError(
            f"joined work_sequence is not contiguous from zero: {sequences}"
        )
    return tuple(joined)


def __getattr__(name: str) -> object:
    """Load the CUDA/vLLM worker only when the opt-in class is resolved."""

    if name == "VllmComponentTelemetryWorker":
        from ._vllm_component_worker import VllmComponentTelemetryWorker

        return VllmComponentTelemetryWorker
    raise AttributeError(name)


__all__ = [
    "BOUNDARY_CONTRACT",
    "BOUNDARY_CONTRACT_SHA256",
    "CALIBRATION_RUN_ID_ENV",
    "COMPONENT_TELEMETRY_PATH_ENV",
    "COMPONENT_TELEMETRY_RUN_ID_ENV",
    "COMPONENT_TELEMETRY_SCHEMA",
    "COMPONENT_WORKER_QUALNAME",
    "ComponentCycleObservation",
    "ComponentCycleShape",
    "ComponentPhaseShape",
    "ComponentTelemetryError",
    "ExtractedCycle",
    "JoinedStageComponentCycle",
    "aggregate_component_cycle",
    "append_component_record",
    "extract_scheduler_cycle",
    "join_stage_component_records",
    "make_cycle_id",
    "parse_component_observations",
    "validate_run_id",
]
