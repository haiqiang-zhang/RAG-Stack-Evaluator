"""CPU-only contracts for transient vLLM startup-port failures."""

from collections import deque
import io

import pytest

import rag_stack_evaluator.static_rag_evaluator.measured.vllm_subprocess as subprocess_module
from rag_stack.controller import _eval_retry_decision
from rag_stack_evaluator.static_rag_evaluator.measured.vllm_deployment import (
    TrialInvalid,
    VllmDeploymentManager,
)
from rag_stack_evaluator.static_rag_evaluator.measured.vllm_subprocess import (
    RetryableVllmStartupError,
    VllmStartupKey,
    VllmSubprocess,
    _retryable_startup_port_failure,
    _start_output_tee,
    _wait_for_health,
)
from rag_stack.types import EvalType


class _ExitedProcess:
    returncode = 1

    def __init__(self, *lines):
        self._rag_stack_output_tail = deque(lines, maxlen=256)

    def poll(self):
        return self.returncode


def test_health_wait_classifies_exact_eaddrinuse_as_retryable():
    proc = _ExitedProcess(
        "torch.distributed.DistNetworkError: port: 43393, code: -98, "
        "name: EADDRINUSE, message: address already in use\n"
    )

    with pytest.raises(RetryableVllmStartupError, match="EADDRINUSE") as caught:
        _wait_for_health("http://127.0.0.1:1", timeout_s=0.01, proc=proc)

    assert _eval_retry_decision(caught.value) == (EvalType.INVALID, True)


def test_health_wait_does_not_reclassify_ordinary_startup_exit():
    proc = _ExitedProcess("EngineCore failed to start: invalid model config\n")

    with pytest.raises(RuntimeError, match="subprocess exited") as caught:
        _wait_for_health("http://127.0.0.1:1", timeout_s=0.01, proc=proc)

    assert not isinstance(caught.value, RetryableVllmStartupError)


def test_health_wait_drains_late_eaddrinuse_line_before_generic_exit(monkeypatch):
    proc = _ExitedProcess()

    def drain_tail(observed_proc, timeout_s):
        assert observed_proc is proc
        assert timeout_s == 0.5
        proc._rag_stack_output_tail.append(
            "DistNetworkError: name: EADDRINUSE, message: address already in use\n"
        )

    monkeypatch.setattr(subprocess_module, "_join_output_tee", drain_tail)

    with pytest.raises(RetryableVllmStartupError, match="EADDRINUSE"):
        _wait_for_health("http://127.0.0.1:1", timeout_s=0.01, proc=proc)


def test_startup_port_marker_survives_bounded_tail_eviction(monkeypatch):
    class _StreamProcess:
        def __init__(self):
            self.stdout = io.StringIO(
                "DistNetworkError: address already in use\n"
                + "".join(f"traceback line {index}\n" for index in range(20))
            )

    proc = _StreamProcess()
    tail = deque(maxlen=2)
    monkeypatch.setattr(subprocess_module.sys, "stderr", io.StringIO())

    thread = _start_output_tee(proc, "test", None, tail)
    assert thread is not None
    thread.join(timeout=1.0)

    assert list(tail) == ["traceback line 18\n", "traceback line 19\n"]
    assert "address already in use" in _retryable_startup_port_failure(proc)


def test_collocated_vllm_launch_captures_startup_output_tail(monkeypatch):
    launched = {}

    class _Process:
        pass

    def launch(cmd, *, env, label, capture_tail=False):
        launched.update(
            cmd=cmd,
            env=env,
            label=label,
            capture_tail=capture_tail,
        )
        return _Process()

    monkeypatch.setattr(subprocess_module, "_find_free_port", lambda: 43210)
    monkeypatch.setattr(
        subprocess_module, "configure_vllm_worker_env", lambda **_kwargs: None
    )
    monkeypatch.setattr(subprocess_module, "_popen_with_output_tee", launch)
    monkeypatch.setattr(
        subprocess_module, "_wait_for_health", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        subprocess_module,
        "_resolve_served_max_model_len",
        lambda *_args, **_kwargs: 4096,
    )
    monkeypatch.setattr(
        "rag_stack_evaluator.static_rag_evaluator.measured.gpu_mem.effective_util",
        lambda *_args, **_kwargs: 0.8,
    )

    server = VllmSubprocess(
        VllmStartupKey(
            model="Qwen/Test",
            device="cuda:0",
            gpu_memory_utilization=0.8,
        )
    )

    assert server.proc is not None
    assert launched["capture_tail"] is True
    assert launched["label"] == "vllm:cuda:0:43210"


class _FailingPdCache:
    def __init__(self, error):
        self.error = error

    def register_main_vllm_pd(self, _key):
        raise self.error


def _register_pd_with(error):
    manager = VllmDeploymentManager(["cuda:0", "cuda:1"])
    manager._register_engine(
        cache=_FailingPdCache(error),
        role="main",
        model="Qwen/Test",
        dtype="bfloat16",
        util=0.8,
        devices=["cuda:0", "cuda:1"],
        disagg=True,
        prefill_devices=["cuda:0"],
        decode_devices=["cuda:1"],
        knobs={
            "max_num_seqs": 8,
            "max_num_batched_tokens": None,
            "enable_prefix_caching": None,
            "max_model_len": -1,
            "kv_cache_dtype": "auto",
        },
        prefill_parallelism=(1, 1),
        decode_parallelism=(1, 1),
        pd_seq_caps=(8, 8),
    )


def test_deployment_preserves_retryable_startup_error():
    error = RetryableVllmStartupError("EADDRINUSE")
    with pytest.raises(RetryableVllmStartupError) as caught:
        _register_pd_with(error)
    assert caught.value is error


def test_deployment_keeps_ordinary_launch_failure_trial_invalid():
    with pytest.raises(TrialInvalid, match="PD pair launch failed"):
        _register_pd_with(RuntimeError("invalid model config"))


def test_deployment_keeps_oom_deterministic_and_nonretryable():
    with pytest.raises(TrialInvalid) as caught:
        _register_pd_with(RuntimeError("CUDA out of memory during model load"))

    assert _eval_retry_decision(caught.value) == (EvalType.OOM, False)
