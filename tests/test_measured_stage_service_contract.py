"""Regression contracts for measured-mode CPU stage scheduling.

These tests are intentionally independent of model/GPU construction.  They
exercise only the asyncio stage-service machinery with tiny in-memory frames.
"""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import pandas as pd
import pytest


class _RuntimeOwner:
    @staticmethod
    def _generator_chip_count(_system_config):
        return 1

    @staticmethod
    def _add_deployment_metadata(_summary, _node_lines, _system_config):
        return None


def _runtime(*, stages=None, batch_size=4, measured_queries=1):
    from rag_stack.static_rag_evaluator.measured import serving_runtime as sr

    return sr.MeasuredServingRuntime(
        owner=_RuntimeOwner(),
        stages=list(stages or []),
        qa_data=pd.DataFrame({"query": ["q"]}),
        node_lines={},
        system_config={
            "batch_size_request": batch_size,
            "measured_load_concurrency": 1,
            "measured_warmup_queries": 0,
            "measured_queries": measured_queries,
            "batching": {"dynamic_timeout_s": 0.01},
            "layout": {
                "engines": {
                    "generator": {
                        "pd_serving": "collocated_pd",
                        "devices": ["cuda:0"],
                        "num_chips": 1,
                        "tp": 1,
                        "pp": 1,
                    }
                }
            },
        },
        config={"pipeline_runtime": {"rag_dataflow": "sequential"}},
    )


def test_prompt_maker_uses_dynamic_batch_service():
    """Vectorized prompt builders must not become one thread job per query."""
    from rag_stack.static_rag_evaluator.measured import serving_runtime as sr

    stage = {
        "stage": "prompt_maker",
        "node": SimpleNamespace(stage="prompt_maker"),
        "params": {},
        "instance": object(),
    }

    async def scenario():
        services = _runtime(stages=[stage], batch_size=8)._build_services()
        try:
            assert len(services) == 1
            assert isinstance(services[0], sr.BatchedPureStageService)
            assert services[0].batch_size == 8
            assert services[0].timeout_s == 0.01
        finally:
            await asyncio.gather(*(service.close() for service in services))

    asyncio.run(scenario())


def test_partial_batch_timeout_is_anchored_to_oldest_enqueue():
    """A request that aged while the worker was busy must flush immediately."""
    from rag_stack.static_rag_evaluator.measured import serving_runtime as sr

    async def scenario():
        # Construct only the queueing state so no background worker can consume
        # the synthetic aged item before _next_batch observes it.
        service = object.__new__(sr.BatchedPureStageService)
        service.batch_size = 2
        service.timeout_s = 0.1
        service._condition = asyncio.Condition()
        service._queue = [{"enqueue_s": time.perf_counter() - 1.0}]
        service._closed = False

        started = time.perf_counter()
        items = await service._next_batch()
        elapsed = time.perf_counter() - started

        assert len(items) == 1
        # Generous scheduler allowance while still rejecting another full
        # 100-ms timeout window.
        assert elapsed < 0.03

    asyncio.run(scenario())


def test_full_batch_flushes_before_oldest_timeout():
    """Filling an idle partial queue must wake it without waiting for timeout."""
    from rag_stack.static_rag_evaluator.measured import serving_runtime as sr

    async def scenario():
        # Exercise _next_batch directly so the assertion covers only the
        # condition/deadline state machine, not executor startup variance.
        service = object.__new__(sr.BatchedPureStageService)
        service.batch_size = 2
        service.timeout_s = 1.0
        service._condition = asyncio.Condition()
        service._queue = [
            {
                "enqueue_s": time.perf_counter(),
                "enqueue_loop_s": asyncio.get_running_loop().time(),
            }
        ]
        service._closed = False

        waiter = asyncio.create_task(service._next_batch())
        await asyncio.sleep(0)
        async with service._condition:
            service._queue.append(
                {
                    "enqueue_s": time.perf_counter(),
                    "enqueue_loop_s": asyncio.get_running_loop().time(),
                }
            )
            service._condition.notify_all()

        items = await asyncio.wait_for(waiter, timeout=0.1)
        assert len(items) == 2

    asyncio.run(scenario())


def test_cancelling_only_queued_request_keeps_open_worker_alive():
    """A transient empty queue must not be mistaken for service shutdown."""
    from rag_stack.static_rag_evaluator.measured import serving_runtime as sr

    async def scenario():
        # Reproduce run()'s queued-cancellation path without constructing an
        # executor: _next_batch has already captured the cancelled row's
        # deadline when the row disappears from the queue.
        service = object.__new__(sr.BatchedPureStageService)
        service.batch_size = 2
        service.timeout_s = 0.02
        service._condition = asyncio.Condition()
        service._queue = [
            {
                "enqueue_s": time.perf_counter(),
                "enqueue_loop_s": asyncio.get_running_loop().time(),
            }
        ]
        service._closed = False

        waiter = asyncio.create_task(service._next_batch())
        await asyncio.sleep(0)
        async with service._condition:
            service._queue.clear()

        try:
            await asyncio.sleep(0.05)
            # Returning [] makes _worker() exit even though the service is
            # still open; a later quality-fill request then has no consumer.
            assert not waiter.done()
        finally:
            async with service._condition:
                service._closed = True
                service._condition.notify_all()
            await waiter

    asyncio.run(scenario())


def test_close_cancels_queued_partial_batch_without_running_it():
    """Public close cancels queued work; it is not a partial-batch drain."""
    from rag_stack.static_rag_evaluator.measured import serving_runtime as sr

    class Owner:
        @staticmethod
        def _merge_service_node_result(_node, previous, result):
            return pd.concat(
                [previous.reset_index(drop=True), result.reset_index(drop=True)],
                axis=1,
            )

    class Instance:
        calls = 0

        @classmethod
        def pure(cls, previous, **_params):
            cls.calls += 1
            return pd.DataFrame({"value": list(range(len(previous)))})

    async def scenario():
        stage = {
            "stage": "prompt_maker",
            "node": SimpleNamespace(stage="prompt_maker"),
            "params": {},
            "instance": Instance(),
        }
        service = sr.BatchedPureStageService(
            Owner(), stage, batch_size=2, timeout_s=10.0
        )
        state = sr.RequestState(
            seq=0,
            idx=0,
            qid="q0",
            df=pd.DataFrame({"query": ["q"]}),
            is_measured=True,
        )
        request = asyncio.create_task(service.run(state, state.df))
        for _ in range(100):
            if service.backlog() == 1:
                break
            await asyncio.sleep(0)
        assert service.backlog() == 1

        await service.close()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert Instance.calls == 0

    asyncio.run(scenario())


def test_close_waits_for_an_already_started_batch():
    """Closing may cancel queued rows, but an executor-owned batch must finish."""
    from rag_stack.static_rag_evaluator.measured import serving_runtime as sr

    started = threading.Event()
    release = threading.Event()

    class Owner:
        @staticmethod
        def _merge_service_node_result(_node, previous, result):
            return pd.concat(
                [previous.reset_index(drop=True), result.reset_index(drop=True)],
                axis=1,
            )

    class Instance:
        @staticmethod
        def pure(previous, **_params):
            started.set()
            if not release.wait(timeout=1.0):
                raise TimeoutError("test did not release active batch")
            return pd.DataFrame({"value": list(range(len(previous)))})

    async def scenario():
        stage = {
            "stage": "prompt_maker",
            "node": SimpleNamespace(stage="prompt_maker"),
            "params": {},
            "instance": Instance(),
        }
        service = sr.BatchedPureStageService(
            Owner(), stage, batch_size=1, timeout_s=0.0
        )
        state = sr.RequestState(
            seq=0,
            idx=0,
            qid="q0",
            df=pd.DataFrame({"query": ["q"]}),
            is_measured=True,
        )
        request = asyncio.create_task(service.run(state, state.df))
        assert await asyncio.to_thread(started.wait, 1.0)

        closing = asyncio.create_task(service.close())
        await asyncio.sleep(0)
        assert not closing.done()
        release.set()

        result = await asyncio.wait_for(request, timeout=1.0)
        await asyncio.wait_for(closing, timeout=1.0)
        assert result.batch_size == 1

    asyncio.run(scenario())


def test_batched_service_time_covers_postprocess_until_result_release():
    """Stage service is worker-busy time, not only instance.pure() time."""
    from rag_stack.static_rag_evaluator.measured import serving_runtime as sr

    postprocess_s = 0.05

    class Owner:
        @staticmethod
        def _merge_service_node_result(_node, previous, result):
            time.sleep(postprocess_s)
            return pd.concat(
                [previous.reset_index(drop=True), result.reset_index(drop=True)],
                axis=1,
            )

    class Instance:
        @staticmethod
        def pure(previous, **_params):
            return pd.DataFrame({"value": list(range(len(previous)))})

    async def scenario():
        stage = {
            "stage": "prompt_maker",
            "node": SimpleNamespace(stage="prompt_maker"),
            "params": {},
            "instance": Instance(),
        }
        service = sr.BatchedPureStageService(
            Owner(), stage, batch_size=1, timeout_s=0.0
        )
        state = sr.RequestState(
            seq=0,
            idx=0,
            qid="q0",
            df=pd.DataFrame({"query": ["q"]}),
            is_measured=True,
        )
        try:
            result = await service.run(state, state.df)
        finally:
            await service.close()

        assert result.service_s >= postprocess_s * 0.8
        assert result.elapsed_s >= result.queue_wait_s + result.service_s

    asyncio.run(scenario())


def test_repeated_completions_copy_only_the_first_quality_winner(monkeypatch):
    """Repeated closed-loop completions must not maintain an unused latest copy."""
    original_copy = pd.DataFrame.copy
    completed_frame_copies = []

    def tracking_copy(frame, *args, **kwargs):
        if "__completed_seq__" in frame.columns:
            completed_frame_copies.append(int(frame["__completed_seq__"].iloc[0]))
        return original_copy(frame, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "copy", tracking_copy)
    runtime = _runtime(measured_queries=3)

    async def fake_request(state):
        state.df = pd.DataFrame(
            {
                "query": ["q"],
                "generated_texts": [f"answer-{state.seq}"],
                "__completed_seq__": [state.seq],
            }
        )

    result, summary = asyncio.run(
        runtime._run_closed_loop_saturated(fake_request)
    )

    assert len(result) == 1
    assert summary["measured_queries"] == 3
    # Copy/reset_index may internally copy the winner more than once. The
    # invariant is that no later duplicate completion is copied/retained.
    assert completed_frame_copies
    assert set(completed_frame_copies) == {0}
