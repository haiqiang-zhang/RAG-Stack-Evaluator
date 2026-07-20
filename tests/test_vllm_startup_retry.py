"""CPU-only contracts for transient vLLM startup-port failures."""

from collections import deque

import pytest

import rag_stack.static_rag_evaluator.measured.vllm_subprocess as subprocess_module
from rag_stack.controller import _eval_retry_decision
from rag_stack.static_rag_evaluator.measured.vllm_deployment import (
    TrialInvalid,
    VllmDeploymentManager,
)
from rag_stack.static_rag_evaluator.measured.vllm_subprocess import (
    RetryableVllmStartupError,
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
