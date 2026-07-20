"""CPU-only lifecycle contracts for measured batched-stage executors."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time
from types import SimpleNamespace

import pandas as pd
import pytest


class _Owner:
    @staticmethod
    def _merge_service_node_result(_node, previous, result):
        return pd.concat(
            [previous.reset_index(drop=True), result.reset_index(drop=True)],
            axis=1,
        )


def _state(sr, qid: str):
    frame = pd.DataFrame({"query": [qid], "__qid__": [qid]})
    return sr.RequestState(
        seq=0,
        idx=0,
        qid=qid,
        df=frame,
        is_measured=True,
    )


def _stage(name: str, instance):
    return {
        "stage": name,
        "node": SimpleNamespace(stage=name),
        "params": {},
        "instance": instance,
    }


def test_batched_services_use_distinct_private_executors():
    """Neither batched stage may submit work to the loop's default executor."""
    from rag_stack.static_rag_evaluator.measured import serving_runtime as sr

    class ExplodingDefaultExecutor(concurrent.futures.ThreadPoolExecutor):
        def submit(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("batched stage used the default executor")

    worker_threads: dict[str, threading.Thread] = {}

    class Instance:
        def __init__(self, name: str):
            self.name = name

        def pure(self, previous, **_params):
            worker_threads[self.name] = threading.current_thread()
            return pd.DataFrame({f"value_{self.name}": list(range(len(previous)))})

    async def scenario():
        loop = asyncio.get_running_loop()
        default_executor = ExplodingDefaultExecutor(
            max_workers=1,
            thread_name_prefix="forbidden-default",
        )
        loop.set_default_executor(default_executor)
        services = [
            sr.BatchedPureStageService(
                _Owner(), _stage(name, Instance(name)), batch_size=1, timeout_s=0.0
            )
            for name in ("stage_a", "stage_b")
        ]
        states = [_state(sr, service.name) for service in services]
        try:
            assert services[0]._executor is not services[1]._executor
            results = await asyncio.wait_for(
                asyncio.gather(
                    *(service.run(state, state.df)
                      for service, state in zip(services, states))
                ),
                timeout=1.0,
            )
            assert len(results) == 2
            assert set(worker_threads) == {"stage_a", "stage_b"}
            assert worker_threads["stage_a"] is not worker_threads["stage_b"]
            assert worker_threads["stage_a"].name.startswith("measured-stage_a")
            assert worker_threads["stage_b"].name.startswith("measured-stage_b")
        finally:
            await asyncio.gather(*(service.close() for service in services))
            default_executor.shutdown(wait=True, cancel_futures=True)

    asyncio.run(scenario())


def test_cancelled_close_waits_for_worker_then_shuts_down_executor():
    """Cancellation propagates only after in-flight work and executor teardown."""
    from rag_stack.static_rag_evaluator.measured import serving_runtime as sr

    started = threading.Event()
    release = threading.Event()
    timeline: list[tuple[str, float]] = []
    worker_thread: list[threading.Thread] = []

    class BlockingInstance:
        @staticmethod
        def pure(previous, **_params):
            worker_thread.append(threading.current_thread())
            timeline.append(("work_started", time.monotonic()))
            started.set()
            if not release.wait(timeout=2.0):
                raise TimeoutError("test did not release the stage worker")
            timeline.append(("work_done", time.monotonic()))
            return pd.DataFrame({"value": list(range(len(previous)))})

    async def event_is_set(event: threading.Event):
        while not event.is_set():
            await asyncio.sleep(0.001)

    async def scenario():
        service = sr.BatchedPureStageService(
            _Owner(),
            _stage("blocking_stage", BlockingInstance()),
            batch_size=1,
            timeout_s=0.0,
        )
        original_shutdown = service._shutdown_executor

        def tracked_shutdown():
            original_shutdown()
            timeline.append(("shutdown_done", time.monotonic()))

        service._shutdown_executor = tracked_shutdown
        state = _state(sr, "q0")
        run_task = asyncio.create_task(service.run(state, state.df))
        timer = threading.Timer(0.2, release.set)
        try:
            await asyncio.wait_for(event_is_set(started), timeout=1.0)
            close_task = asyncio.create_task(service.close())
            await asyncio.sleep(0)
            timer.start()
            close_task.cancel()

            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(close_task, timeout=1.5)
            timeline.append(("cancellation_propagated", time.monotonic()))

            result = await asyncio.wait_for(run_task, timeout=1.0)
            assert result.df["value"].tolist() == [0]
            assert service._executor_shutdown is True
            assert worker_thread and not worker_thread[0].is_alive()

            observed = {label: timestamp for label, timestamp in timeline}
            assert observed["work_done"] <= observed["shutdown_done"]
            assert observed["shutdown_done"] <= observed["cancellation_propagated"]
        finally:
            release.set()
            if timer.is_alive():
                timer.cancel()
            timer.join(timeout=1.0)
            if not run_task.done():
                await asyncio.gather(run_task, return_exceptions=True)
            if not service._executor_shutdown:
                await asyncio.gather(service.close(), return_exceptions=True)

    asyncio.run(scenario())
