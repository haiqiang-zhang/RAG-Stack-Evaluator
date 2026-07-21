"""Opt-in production-path coverage telemetry for LLM stage calibration.

The scheduler-cycle stat logger measures the saturated active-service wall
interval.  It must not add independently timed frontend spans to that interval:
the OpenAI frontend and the AsyncLLM output handler share one event loop, so
doing so would count the same wall time twice.  The spans emitted here are
therefore *coverage evidence*.  They prove that the measured saturated window
actually exercised the production tokenizer/preprocessor, request dispatch,
result materialization, response construction and JSON serialization paths.

Every telemetry append is timed and exposed to the scheduler logger as an
instrumentation exclusion.  Queueing or dynamic tokenizer batch-wait time is
never fitted from the span durations; the durations are not curve inputs.

This module has no import-time dependency on vLLM, torch or CUDA.  The custom
server imports vLLM classes only when stage telemetry is explicitly enabled.
"""

from __future__ import annotations

import contextvars
import fcntl
import functools
import inspect
import json
import math
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from rag_stack_evaluator.vllm_instrumentation.calibration.runtime_identity import (
    resolve_calibration_run_id,
    validate_calibration_run_id,
)

TELEMETRY_ENV = "RAG_STACK_STAGE_TELEMETRY_PATH"
FRONTEND_BOUNDARY_SCHEMA = "rag_stack.vllm.frontend_stage_boundary_span"

FrontendSpanKind = Literal[
    "preprocess_tokenize",
    "request_dispatch_ipc",
    "output_materialization",
    "response_build_postprocess",
    "response_json_serialization",
]

_PREFILL_REQUIRED_KINDS = frozenset({
    "preprocess_tokenize",
    "request_dispatch_ipc",
    "output_materialization",
})
_DECODE_REQUIRED_KINDS = frozenset({
    "output_materialization",
    "response_build_postprocess",
    "response_json_serialization",
})
_ALL_KINDS = _PREFILL_REQUIRED_KINDS | _DECODE_REQUIRED_KINDS

_APPEND_LOCK = threading.Lock()
_EXCLUSION_LOCK = threading.Lock()
_FRONTEND_APPEND_EXCLUSION_S = 0.0
_REQUEST_CONTEXT: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "rag_stack_stage_frontend_request", default=None
)


class FrontendBoundaryCoverageError(RuntimeError):
    """The saturated window did not exercise the complete production path."""


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    with _APPEND_LOCK:
        fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("short write while appending frontend telemetry")
                view = view[written:]
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)


def _add_append_exclusion(seconds: float) -> None:
    if not math.isfinite(seconds) or seconds < 0.0:
        return
    global _FRONTEND_APPEND_EXCLUSION_S
    with _EXCLUSION_LOCK:
        _FRONTEND_APPEND_EXCLUSION_S += seconds


def consume_frontend_instrumentation_exclusion_s() -> float:
    """Atomically drain frontend telemetry work since the last cycle sample."""

    global _FRONTEND_APPEND_EXCLUSION_S
    with _EXCLUSION_LOCK:
        result = _FRONTEND_APPEND_EXCLUSION_S
        _FRONTEND_APPEND_EXCLUSION_S = 0.0
    return result


def _record_span(
    *,
    kind: FrontendSpanKind,
    started_s: float,
    finished_s: float,
    request_id: str | None,
) -> None:
    path_value = os.environ.get(TELEMETRY_ENV)
    if path_value is None:
        return
    if not path_value.strip():
        raise RuntimeError(f"{TELEMETRY_ENV} cannot be empty")
    duration_s = finished_s - started_s
    if (
        not math.isfinite(started_s)
        or not math.isfinite(finished_s)
        or not math.isfinite(duration_s)
        or duration_s < 0.0
    ):
        raise RuntimeError("frontend boundary clock produced an invalid span")
    run_id = resolve_calibration_run_id()
    record = {
        "schema": FRONTEND_BOUNDARY_SCHEMA,
        "evidence_scope": "active_stage_boundary_coverage_only",
        "run_id": run_id,
        "kind": kind,
        "request_id": request_id,
        "process_id": os.getpid(),
        "thread_id": threading.get_ident(),
        "started_monotonic_s": started_s,
        "finished_monotonic_s": finished_s,
        "active_span_s": duration_s,
        "fit_value": False,
        "reason": "coverage_only_do_not_add_to_scheduler_interval",
    }
    append_started_s = time.monotonic()
    _append_jsonl(Path(path_value).expanduser(), record)
    append_finished_s = time.monotonic()
    _add_append_exclusion(append_finished_s - append_started_s)


def _timed_sync_span(
    kind: FrontendSpanKind,
    original: Callable[..., Any],
    *args: Any,
    request_id: str | None = None,
    **kwargs: Any,
) -> Any:
    started_s = time.monotonic()
    try:
        return original(*args, **kwargs)
    finally:
        _record_span(
            kind=kind,
            started_s=started_s,
            finished_s=time.monotonic(),
            request_id=request_id,
        )


def _patch_once(target: type, name: str, replacement: Callable[..., Any]) -> None:
    marker = f"_rag_stack_stage_boundary_original_{name}"
    if hasattr(target, marker):
        return
    original = getattr(target, name, None)
    if not callable(original):
        raise RuntimeError(
            f"vLLM frontend boundary contract is missing {target.__name__}.{name}"
        )
    setattr(target, marker, original)
    setattr(target, name, replacement(original))


def install_frontend_boundary_telemetry(
    *,
    completion_cls: type | None = None,
    chat_cls: type | None = None,
    async_llm_cls: type | None = None,
    output_processor_cls: type | None = None,
    json_response_cls: type | None = None,
) -> bool:
    """Patch the exact vLLM 0.18.1 production OpenAI paths.

    The optional class arguments make the contract CPU-testable without
    importing vLLM.  Production passes no arguments and resolves the audited
    classes lazily.  If either endpoint class is injected, only explicitly
    supplied endpoint classes are patched; this keeps isolated CPU fixtures
    from importing vLLM.
    """

    path_value = os.environ.get(TELEMETRY_ENV)
    if path_value is None:
        return False
    if not path_value.strip():
        raise RuntimeError(f"{TELEMETRY_ENV} cannot be empty")

    endpoints_injected = completion_cls is not None or chat_cls is not None
    if not endpoints_injected:
        from vllm.entrypoints.openai.completion.serving import (
            OpenAIServingCompletion,
        )
        from vllm.entrypoints.openai.chat_completion.serving import (
            OpenAIServingChat,
        )

        completion_cls = OpenAIServingCompletion
        chat_cls = OpenAIServingChat
    if async_llm_cls is None:
        from vllm.v1.engine.async_llm import AsyncLLM

        async_llm_cls = AsyncLLM
    if output_processor_cls is None:
        from vllm.v1.engine.output_processor import OutputProcessor

        output_processor_cls = OutputProcessor
    if json_response_cls is None:
        from starlette.responses import JSONResponse

        json_response_cls = JSONResponse

    def request_wrapper(original):
        @functools.wraps(original)
        async def wrapped(self, request, raw_request=None):
            # The real vLLM request id does not exist until preprocessing has
            # completed and AsyncLLM.add_request is called.  Keep the context
            # explicitly empty until dispatch supplies that actual argument;
            # never manufacture a look-alike correlation id.
            token = _REQUEST_CONTEXT.set(None)
            try:
                result = await original(self, request, raw_request)
            except BaseException:
                _REQUEST_CONTEXT.reset(token)
                raise
            # Do not reset on success. JSONResponse.render runs after this
            # method returns, in the same request task. Task-local context is
            # discarded when the ASGI request completes.
            return result

        return wrapped

    def preprocess_wrapper(original):
        @functools.wraps(original)
        async def wrapped(self, request):
            started_s = time.monotonic()
            try:
                return await original(self, request)
            finally:
                _record_span(
                    kind="preprocess_tokenize",
                    started_s=started_s,
                    finished_s=time.monotonic(),
                    request_id=_REQUEST_CONTEXT.get(),
                )

        return wrapped

    def dispatch_wrapper(original):
        signature = inspect.signature(original)

        @functools.wraps(original)
        async def wrapped(self, *args, **kwargs):
            # This is the real request identity accepted by AsyncLLM, not a
            # frontend-generated surrogate. Bind against the audited vLLM
            # signature so positional and keyword calls share one exact rule;
            # never infer identity from an argument position.
            try:
                bound = signature.bind(self, *args, **kwargs)
            except TypeError as exc:
                raise RuntimeError(
                    "vLLM AsyncLLM.add_request call differs from its audited "
                    "signature"
                ) from exc
            request_id = bound.arguments.get("request_id")
            if not isinstance(request_id, str) or not request_id:
                raise RuntimeError(
                    "vLLM AsyncLLM.add_request lacks a real request_id"
                )
            _REQUEST_CONTEXT.set(request_id)
            started_s = time.monotonic()
            try:
                return await original(self, *args, **kwargs)
            finally:
                _record_span(
                    kind="request_dispatch_ipc",
                    started_s=started_s,
                    finished_s=time.monotonic(),
                    request_id=request_id,
                )

        return wrapped

    def response_build_wrapper(original):
        @functools.wraps(original)
        def wrapped(self, *args, **kwargs):
            return _timed_sync_span(
                "response_build_postprocess",
                original,
                self,
                *args,
                request_id=_REQUEST_CONTEXT.get(),
                **kwargs,
            )

        return wrapped

    def async_response_build_wrapper(original):
        @functools.wraps(original)
        async def wrapped(self, *args, **kwargs):
            started_s = time.monotonic()
            try:
                return await original(self, *args, **kwargs)
            finally:
                _record_span(
                    kind="response_build_postprocess",
                    started_s=started_s,
                    finished_s=time.monotonic(),
                    request_id=_REQUEST_CONTEXT.get(),
                )

        return wrapped

    def output_wrapper(original):
        @functools.wraps(original)
        def wrapped(self, *args, **kwargs):
            return _timed_sync_span(
                "output_materialization",
                original,
                self,
                *args,
                request_id=None,
                **kwargs,
            )

        return wrapped

    def json_wrapper(original):
        @functools.wraps(original)
        def wrapped(self, content):
            request_id = _REQUEST_CONTEXT.get()
            if request_id is None:
                return original(self, content)
            return _timed_sync_span(
                "response_json_serialization",
                original,
                self,
                content,
                request_id=request_id,
            )

        return wrapped

    if completion_cls is not None:
        _patch_once(completion_cls, "create_completion", request_wrapper)
        _patch_once(
            completion_cls, "render_completion_request", preprocess_wrapper
        )
        _patch_once(
            completion_cls,
            "request_output_to_completion_response",
            response_build_wrapper,
        )
    if chat_cls is not None:
        _patch_once(chat_cls, "create_chat_completion", request_wrapper)
        _patch_once(chat_cls, "render_chat_request", preprocess_wrapper)
        _patch_once(
            chat_cls,
            "chat_completion_full_generator",
            async_response_build_wrapper,
        )
    _patch_once(async_llm_cls, "add_request", dispatch_wrapper)
    _patch_once(output_processor_cls, "process_outputs", output_wrapper)
    _patch_once(json_response_cls, "render", json_wrapper)
    return True


def _finite_timestamp(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FrontendBoundaryCoverageError(f"{field} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise FrontendBoundaryCoverageError(f"{field} must be finite")
    return result


def frontend_boundary_coverage(
    records: Sequence[Mapping[str, object]],
    *,
    run_id: str,
    phase: Literal["prefill", "decode"],
    window_start_monotonic_s: float,
    window_end_monotonic_s: float,
) -> dict[str, int]:
    """Validate and count required spans overlapping one saturated window."""

    expected_run_id = validate_calibration_run_id(run_id)
    start = _finite_timestamp(
        window_start_monotonic_s, field="window_start_monotonic_s"
    )
    end = _finite_timestamp(
        window_end_monotonic_s, field="window_end_monotonic_s"
    )
    if end <= start:
        raise FrontendBoundaryCoverageError(
            "frontend coverage window must have positive duration"
        )
    if phase not in ("prefill", "decode"):
        raise ValueError("phase must be prefill or decode")

    counts = {kind: 0 for kind in sorted(_ALL_KINDS)}
    for record in records:
        if record.get("schema") != FRONTEND_BOUNDARY_SCHEMA:
            continue
        if record.get("evidence_scope") != "active_stage_boundary_coverage_only":
            raise FrontendBoundaryCoverageError(
                "unexpected frontend boundary evidence scope"
            )
        if validate_calibration_run_id(record.get("run_id")) != expected_run_id:
            raise FrontendBoundaryCoverageError(
                "frontend boundary run_id differs from scheduler telemetry"
            )
        kind = record.get("kind")
        if kind not in _ALL_KINDS:
            raise FrontendBoundaryCoverageError(
                f"unknown frontend boundary span kind {kind!r}"
            )
        request_id = record.get("request_id")
        if kind == "request_dispatch_ipc":
            if not isinstance(request_id, str) or not request_id:
                raise FrontendBoundaryCoverageError(
                    "request dispatch span lacks the real AsyncLLM request_id"
                )
        elif request_id is not None and (
            not isinstance(request_id, str) or not request_id
        ):
            raise FrontendBoundaryCoverageError(
                "frontend boundary request_id is invalid"
            )
        started = _finite_timestamp(
            record.get("started_monotonic_s"), field="span.started_monotonic_s"
        )
        finished = _finite_timestamp(
            record.get("finished_monotonic_s"), field="span.finished_monotonic_s"
        )
        duration = _finite_timestamp(
            record.get("active_span_s"), field="span.active_span_s"
        )
        if finished < started or duration < 0.0 or not math.isclose(
            duration, finished - started, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise FrontendBoundaryCoverageError(
                "frontend boundary span duration is inconsistent"
            )
        if record.get("fit_value") is not False or record.get("reason") != (
            "coverage_only_do_not_add_to_scheduler_interval"
        ):
            raise FrontendBoundaryCoverageError(
                "frontend span was not marked coverage-only"
            )
        if finished >= start and started <= end:
            counts[str(kind)] += 1

    required = (
        _PREFILL_REQUIRED_KINDS if phase == "prefill" else _DECODE_REQUIRED_KINDS
    )
    missing = sorted(kind for kind in required if counts[kind] == 0)
    if missing:
        raise FrontendBoundaryCoverageError(
            "saturated scheduler window lacks production boundary coverage: "
            + ", ".join(missing)
        )
    return counts


__all__ = [
    "FRONTEND_BOUNDARY_SCHEMA",
    "FrontendBoundaryCoverageError",
    "consume_frontend_instrumentation_exclusion_s",
    "frontend_boundary_coverage",
    "install_frontend_boundary_telemetry",
]
