"""Quality-path engine slot: adjacent-eval reuse without measured pollution.

07-04 user directive: reuse engines across quality evals (the per-eval load is
pure tax), but keep it OUT of the measured path (its ModelCache owns real
deployment lifetimes) — hence the extracted path-neutral slot. These tests pin
the slot contract with fake engines (no GPU): hit on same key, teardown+rebuild
on key change, keep_pids bookkeeping, and the reaper's spare list.
"""
import types

import pytest

from rag_stack_evaluator.static_rag_evaluator import engine_slot as es


class FakeEngine:
    def __init__(self, tag):
        self.tag = tag
        self.torn_down = False


def test_same_key_reuses_and_factory_called_once(monkeypatch):
    slot = es.InProcessEngineSlot("t")
    calls = []
    monkeypatch.setattr(es, "_live_engine_core_pids", lambda: set())
    e1 = slot.get(("m", ("a", "1")), lambda: calls.append(1) or FakeEngine("x"))
    e2 = slot.get(("m", ("a", "1")), lambda: calls.append(2) or FakeEngine("y"))
    assert e1 is e2 and calls == [1]


def test_key_change_tears_down_then_rebuilds(monkeypatch):
    slot = es.InProcessEngineSlot("t")
    monkeypatch.setattr(es, "_live_engine_core_pids", lambda: set())
    torn = []
    monkeypatch.setattr(es, "teardown_inprocess_engine", lambda e: torn.append(e.tag))
    a = slot.get(("m1", ()), lambda: FakeEngine("a"))
    b = slot.get(("m2", ()), lambda: FakeEngine("b"))
    assert torn == ["a"] and b.tag == "b" and slot.occupied


def test_child_pid_bookkeeping_and_clear(monkeypatch):
    slot = es.InProcessEngineSlot("t")
    seq = iter([set(), {111, 222}])          # before-build, after-build
    monkeypatch.setattr(es, "_live_engine_core_pids", lambda: next(seq))
    monkeypatch.setattr(es, "teardown_inprocess_engine", lambda e: None)
    killed = []
    import os
    monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append(pid))
    slot.get(("m", ()), lambda: FakeEngine("a"))
    assert slot.keep_pids() == {111, 222}
    slot.clear()
    assert sorted(killed) == [111, 222] and not slot.occupied
    assert slot.keep_pids() == set()


def test_reaper_spares_keep_pids(monkeypatch):
    """_force_kill_engine_core_orphans must skip pids in keep_pids."""
    from rag_stack_evaluator.static_rag_evaluator.nodes.generator import vllm as vmod

    class FakeProc:
        def __init__(self, pid, name):
            self.pid = pid
            self._name = name
            self.killed = []
        def name(self): return self._name
        def cmdline(self): return [self._name]
        def terminate(self): self.killed.append("term")
        def kill(self): self.killed.append("kill")
        def wait(self, timeout=None): return 0
        def is_running(self): return False

    keep = FakeProc(1, "VLLM::EngineCore")
    orphan = FakeProc(2, "VLLM::EngineCore")
    me = types.SimpleNamespace(children=lambda recursive: [keep, orphan])
    fake_psutil = types.SimpleNamespace(
        Process=lambda: me,
        NoSuchProcess=Exception, AccessDenied=Exception, TimeoutExpired=Exception,
        wait_procs=lambda procs, timeout=None: ([], [p for p in procs]),
    )
    import sys
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    vmod._force_kill_engine_core_orphans(timeout=0.01, keep_pids={1})
    assert keep.killed == []          # spared
    assert orphan.killed              # reaped


def test_resource_adapter_clears_quality_slot_then_resamples(monkeypatch):
    """A cached small main engine must not permanently block a larger aux.

    The first memory sample has no card large enough for 7B. Clearing this
    process's quality slot makes the second sample fit, so placement succeeds
    without waiting for the deadline or invoking the orphan reaper.
    """
    import logging
    import sys
    import time

    from rag_stack_evaluator.static_rag_evaluator.nodes.generator import vllm as vmod

    gib = 1024 ** 3

    class FakeSlot:
        occupied = True

        def __init__(self):
            self.clear_calls = 0

        def clear(self):
            self.clear_calls += 1
            self.occupied = False

        def keep_pids(self):
            return {123} if self.occupied else set()

    slot = FakeSlot()

    class FakeCuda:
        def __init__(self):
            self.memory_samples = 0

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 1

        @staticmethod
        def get_device_properties(_index):
            return types.SimpleNamespace(total_memory=24 * gib)

        def mem_get_info(self, _index):
            self.memory_samples += 1
            free_gib = 10 if slot.occupied else 23
            return free_gib * gib, 24 * gib

    fake_cuda = FakeCuda()
    fake_torch = types.SimpleNamespace(cuda=fake_cuda)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(es, "quality_engine_slot", lambda: slot)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    reaper_calls = []
    monkeypatch.setattr(
        vmod,
        "_force_kill_engine_core_orphans",
        lambda **_kwargs: reaper_calls.append(True),
    )

    kwargs = {"gpu_memory_utilization": 0.4, "tensor_parallel_size": 1}
    device = vmod._adapt_engine_resources(
        "Qwen/Qwen2.5-7B-Instruct",
        kwargs,
        None,
        logging.getLogger("test"),
        allow_quality_slot_reuse=False,
    )

    assert device == "0"
    assert slot.clear_calls == 1
    assert fake_cuda.memory_samples == 2
    assert reaper_calls == []


def test_resource_adapter_requires_headroom_before_engine_launch(monkeypatch):
    """A merely-equal free-memory fit must reclaim and resample first.

    vLLM allocates its KV cache only after worker startup and profiling.  The
    first sample models eval_0059 attempt 1: GPU 1 has enough free memory for
    the nominal reservation but less than the placement safety margin.  Once
    the different-key quality slot is released, GPU 3 is a safe fit and must
    be selected without invoking the orphan reaper.
    """
    import logging
    import sys
    import time

    from rag_stack_evaluator.static_rag_evaluator.nodes.generator import vllm as vmod

    gib = 1024 ** 3

    class FakeSlot:
        occupied = True

        def __init__(self):
            self.clear_calls = 0

        def clear(self):
            self.clear_calls += 1
            self.occupied = False

        def keep_pids(self):
            return {123} if self.occupied else set()

        def has_live_key(self, _key):
            return False

    slot = FakeSlot()

    class FakeCuda:
        def __init__(self):
            self.memory_samples = 0

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 4

        @staticmethod
        def get_device_properties(_index):
            return types.SimpleNamespace(total_memory=24 * gib)

        def mem_get_info(self, index):
            self.memory_samples += 1
            if slot.occupied:
                # 3B reserves ~11GiB.  GPU 1 is nominally feasible, but does
                # not have the additional 1GiB launch margin.
                free_gib = (8.0, 11.5, 7.0, 10.0)[index]
            else:
                free_gib = (8.0, 11.5, 7.0, 15.0)[index]
            return int(free_gib * gib), 24 * gib

    fake_cuda = FakeCuda()
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=fake_cuda))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(es, "quality_engine_slot", lambda: slot)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    reaper_calls = []
    monkeypatch.setattr(
        vmod,
        "_force_kill_engine_core_orphans",
        lambda **_kwargs: reaper_calls.append(True),
    )

    kwargs = {"gpu_memory_utilization": 0.4, "tensor_parallel_size": 1}
    device = vmod._adapt_engine_resources(
        "Qwen/Qwen2.5-3B-Instruct",
        kwargs,
        None,
        logging.getLogger("test"),
        allow_quality_slot_reuse=True,
    )

    assert device == "3"
    assert slot.clear_calls == 1
    assert fake_cuda.memory_samples == 8
    assert reaper_calls == []


def test_resource_adapter_deadline_logs_least_selected_tp_gpu(monkeypatch, caplog):
    """A TP launch must report the limiting selected GPU at its deadline."""
    import logging
    import sys
    import time

    from rag_stack_evaluator.static_rag_evaluator.nodes.generator import vllm as vmod

    gib = 1024 ** 3

    class EmptySlot:
        occupied = False

        @staticmethod
        def clear():
            raise AssertionError("an empty slot must not be cleared")

        @staticmethod
        def keep_pids():
            return set()

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 2

        @staticmethod
        def get_device_properties(_index):
            return types.SimpleNamespace(total_memory=24 * gib)

        @staticmethod
        def mem_get_info(index):
            # 14B adapts to TP2 with a 19GiB reservation per GPU and therefore
            # a 20GiB safe-launch threshold. GPU 0 is safe; GPU 1 is not.
            return int((22.0, 19.5)[index] * gib), 24 * gib

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=FakeCuda()))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(es, "quality_engine_slot", lambda: EmptySlot())
    now = [0.0]

    def fake_time():
        value = now[0]
        now[0] = 91.0
        return value

    monkeypatch.setattr(time, "time", fake_time)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    reaper_calls = []
    monkeypatch.setattr(
        vmod,
        "_force_kill_engine_core_orphans",
        lambda **_kwargs: reaper_calls.append(True),
    )

    with caplog.at_level(logging.WARNING, logger="RAG-Stack"):
        device = vmod._adapt_engine_resources(
            "Qwen/Qwen2.5-14B-Instruct",
            {"gpu_memory_utilization": 0.4, "tensor_parallel_size": 1},
            None,
            logging.getLogger("RAG-Stack"),
            allow_quality_slot_reuse=False,
        )

    assert device == "0,1"
    assert reaper_calls == [True]
    assert "least selected free 19.5GB < safe launch 20.0GB" in caplog.text


def test_resource_adapter_preserves_live_same_key_slot_on_single_gpu(monkeypatch):
    """A cache hit needs no free build capacity and must bypass placement."""
    import logging
    import sys
    import time

    from rag_stack_evaluator.static_rag_evaluator.nodes.generator import vllm as vmod

    gib = 1024 ** 3
    model = "Qwen/Qwen2.5-7B-Instruct"
    # 7B on a 24GB card adapts to (14GB weights + 5GB overhead) / 24GB.
    adapted_kwargs = {
        "gpu_memory_utilization": 0.792,
        "tensor_parallel_size": 1,
    }
    slot = es.InProcessEngineSlot("same-key")
    monkeypatch.setattr(es, "_live_engine_core_pids", lambda: set())
    engine = slot.get(
        vmod._quality_slot_key(model, adapted_kwargs),
        lambda: FakeEngine("cached-7b"),
    )
    monkeypatch.setattr(es, "quality_engine_slot", lambda: slot)

    clear_calls = []
    original_clear = slot.clear

    def tracked_clear():
        clear_calls.append(True)
        original_clear()

    monkeypatch.setattr(slot, "clear", tracked_clear)

    class FakeCuda:
        def __init__(self):
            self.memory_samples = 0

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 1

        @staticmethod
        def get_device_properties(_index):
            return types.SimpleNamespace(total_memory=24 * gib)

        def mem_get_info(self, _index):
            self.memory_samples += 1
            return 4 * gib, 24 * gib

    fake_cuda = FakeCuda()
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=fake_cuda))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    sleep_calls = []
    monkeypatch.setattr(time, "sleep", lambda seconds: sleep_calls.append(seconds))
    reaper_calls = []
    monkeypatch.setattr(
        vmod,
        "_force_kill_engine_core_orphans",
        lambda **_kwargs: reaper_calls.append(True),
    )

    kwargs = {"gpu_memory_utilization": 0.4, "tensor_parallel_size": 1}
    device = vmod._adapt_engine_resources(
        model,
        kwargs,
        None,
        logging.getLogger("test"),
        allow_quality_slot_reuse=True,
    )

    assert device is None
    assert kwargs == adapted_kwargs
    assert slot.has_live_key(vmod._quality_slot_key(model, kwargs))
    assert slot.get(vmod._quality_slot_key(model, kwargs), lambda: None) is engine
    assert clear_calls == []
    assert fake_cuda.memory_samples == 0
    assert sleep_calls == []
    assert reaper_calls == []


def test_failed_engine_factory_restores_cvd_before_next_selection(monkeypatch):
    """A failed first build cannot remap retry 2 through its stale CVD pin."""
    import os

    from rag_stack_evaluator.static_rag_evaluator.nodes.generator import vllm as vmod

    class BareVllm(vmod.Vllm):
        def __init__(self):
            self._cvd_prev = None
            self._cvd_set = None

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    first = BareVllm()
    first._maybe_pin_cvd("2")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"

    def fail_factory():
        raise RuntimeError("engine init failed")

    with pytest.raises(RuntimeError, match="engine init failed"):
        vmod._build_inprocess_engine(first, fail_factory)
    assert "CUDA_VISIBLE_DEVICES" not in os.environ

    # With the leak, this logical GPU 1 request would see prior=['2'] and be
    # clamped back to physical GPU 2. Synchronous restoration keeps it at 1.
    second = BareVllm()
    second._maybe_pin_cvd("1")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
    second._restore_cvd()
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
