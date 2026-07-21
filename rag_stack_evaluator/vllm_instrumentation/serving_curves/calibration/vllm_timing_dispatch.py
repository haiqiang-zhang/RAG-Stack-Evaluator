"""Opt-in, no-JSONL dispatch fence for production timing calibration.

The counter is incremented only after ``AsyncLLM.add_request`` has awaited the
EngineCore ADD send.  A subsequent ``is_paused`` utility acknowledgement uses
the same FIFO socket, so it proves all earlier requests were inserted into the
paused scheduler before timing starts.  No hook runs in an active cohort
interval other than the normal production path.
"""

from __future__ import annotations

import functools
import os
import threading
from typing import Any


TIMING_DISPATCH_ENV = "RAG_STACK_V1_TIMING_DISPATCH_COUNTER"
TIMING_DISPATCH_METRIC = "rag_stack_v1_timing_dispatch_count"

_LOCK = threading.Lock()
_COUNT = 0
_GAUGE: Any | None = None


def timing_dispatch_enabled() -> bool:
    return os.environ.get(TIMING_DISPATCH_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def install_timing_dispatch_counter(
    async_llm_cls: type,
    *,
    gauge_factory: Any | None = None,
) -> bool:
    if not timing_dispatch_enabled():
        return False
    marker = "_rag_stack_timing_dispatch_original_add_request"
    if hasattr(async_llm_cls, marker):
        return True
    original = getattr(async_llm_cls, "add_request", None)
    if not callable(original):
        raise RuntimeError("timing dispatch contract requires AsyncLLM.add_request")
    if gauge_factory is None:
        from prometheus_client import Gauge

        gauge_factory = Gauge
    global _GAUGE
    with _LOCK:
        if _GAUGE is None:
            _GAUGE = gauge_factory(
                TIMING_DISPATCH_METRIC,
                "Calibration requests dispatched to the paused EngineCore",
            )
            _GAUGE.set(_COUNT)

    @functools.wraps(original)
    async def wrapped(self, *args, **kwargs):
        result = await original(self, *args, **kwargs)
        global _COUNT
        with _LOCK:
            _COUNT += 1
            count = _COUNT
            assert _GAUGE is not None
            _GAUGE.set(count)
        return result

    setattr(async_llm_cls, marker, original)
    setattr(async_llm_cls, "add_request", wrapped)
    return True


__all__ = [
    "TIMING_DISPATCH_ENV",
    "TIMING_DISPATCH_METRIC",
    "install_timing_dispatch_counter",
    "timing_dispatch_enabled",
]
