from __future__ import annotations

import asyncio
import importlib

import pytest


class _Gauge:
    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description
        self.values: list[int] = []

    def set(self, value: int) -> None:
        self.values.append(int(value))


def _fresh_module(monkeypatch):
    from rag_stack_evaluator.vllm_instrumentation.serving_curves.calibration import (
        vllm_timing_dispatch,
    )

    monkeypatch.setenv(vllm_timing_dispatch.TIMING_DISPATCH_ENV, "1")
    return importlib.reload(vllm_timing_dispatch)


def test_counter_increments_only_after_async_add_returns(monkeypatch) -> None:
    module = _fresh_module(monkeypatch)
    release = asyncio.Event()

    class AsyncLLM:
        async def add_request(self, value):
            await release.wait()
            return value

    gauge = _Gauge("unused", "unused")
    assert module.install_timing_dispatch_counter(
        AsyncLLM,
        gauge_factory=lambda *_args: gauge,
    )

    async def exercise() -> None:
        task = asyncio.create_task(AsyncLLM().add_request("request"))
        await asyncio.sleep(0)
        assert gauge.values == [0]
        release.set()
        assert await task == "request"

    asyncio.run(exercise())
    assert gauge.values == [0, 1]


def test_counter_is_opt_in_and_requires_async_llm_add_request(monkeypatch) -> None:
    from rag_stack_evaluator.vllm_instrumentation.serving_curves.calibration import (
        vllm_timing_dispatch,
    )

    monkeypatch.delenv(vllm_timing_dispatch.TIMING_DISPATCH_ENV, raising=False)
    module = importlib.reload(vllm_timing_dispatch)
    assert not module.install_timing_dispatch_counter(object)

    monkeypatch.setenv(module.TIMING_DISPATCH_ENV, "1")
    with pytest.raises(RuntimeError, match="AsyncLLM.add_request"):
        module.install_timing_dispatch_counter(object)
