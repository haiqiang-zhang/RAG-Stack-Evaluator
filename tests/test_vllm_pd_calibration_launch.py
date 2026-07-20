"""CPU-only launch contracts for direct P/D calibration telemetry."""

import json

import pytest

import rag_stack.static_rag_evaluator.measured.vllm_pd_pair as pd_module
from rag_stack.static_rag_evaluator.measured.vllm_pd_pair import (
    VllmPdCalibrationTelemetry,
    VllmPdPair,
    VllmPdPairKey,
    VllmPdRoleTelemetry,
)


def _unlaunched_pair(
    *,
    calibration_telemetry=None,
    enable_prefix_caching=None,
):
    pair = object.__new__(VllmPdPair)
    pair.key = VllmPdPairKey(
        model="Qwen/Test",
        prefill_device="cuda:0",
        decode_device="cuda:1",
        prefill_max_num_seqs=8,
        decode_max_num_seqs=8,
        enable_prefix_caching=enable_prefix_caching,
    )
    pair.calibration_telemetry = calibration_telemetry
    pair.prefill_port = 10_001
    pair.decode_port = 10_002
    pair.prefill_side_port = 61_001
    pair.decode_side_port = 61_002
    pair.prefill_proc = None
    pair.decode_proc = None
    return pair


def _capture_launches(monkeypatch):
    launches = []

    monkeypatch.setattr(
        pd_module,
        "configure_vllm_worker_env",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "rag_stack.static_rag_evaluator.measured.gpu_mem.effective_util",
        lambda _devices, requested, **_kwargs: float(requested),
    )

    def fake_popen(command, *, env, label):
        process = object()
        launches.append({
            "command": list(command),
            "env": dict(env),
            "label": label,
            "process": process,
        })
        return process

    monkeypatch.setattr(pd_module, "_popen_with_output_tee", fake_popen)
    for name in (
        "RAG_STACK_STAGE_TELEMETRY_PATH",
        "RAG_STACK_VLLM_CALIBRATION_RUN_ID",
        "VLLM_SERVER_DEV_MODE",
        "VLLM_DEBUG_MFU_METRICS",
        "RAG_STACK_PD_CALIBRATION_SAFE_PROMETHEUS",
    ):
        monkeypatch.delenv(name, raising=False)
    return launches


@pytest.mark.parametrize(
    ("path", "run_id", "message"),
    [
        ("", "run-p", "path must be non-empty"),
        ("p.jsonl", "", "run_id must be non-empty"),
    ],
)
def test_role_telemetry_rejects_empty_identity(path, run_id, message):
    with pytest.raises(ValueError, match=message):
        VllmPdRoleTelemetry(path=path, run_id=run_id)


def test_pd_telemetry_requires_distinct_role_paths_and_run_ids():
    prefill = VllmPdRoleTelemetry(path="p.jsonl", run_id="run-p")
    decode = VllmPdRoleTelemetry(path="d.jsonl", run_id="run-d")

    telemetry = VllmPdCalibrationTelemetry(prefill=prefill, decode=decode)
    assert telemetry.for_role("prefill") is prefill
    assert telemetry.for_role("decode") is decode
    with pytest.raises(ValueError, match="unknown PD telemetry role"):
        telemetry.for_role("generator")

    with pytest.raises(ValueError, match="paths must be distinct"):
        VllmPdCalibrationTelemetry(
            prefill=prefill,
            decode=VllmPdRoleTelemetry(
                path="p.jsonl",
                run_id="other-run",
            ),
        )
    with pytest.raises(ValueError, match="run_ids must be distinct"):
        VllmPdCalibrationTelemetry(
            prefill=prefill,
            decode=VllmPdRoleTelemetry(
                path="other.jsonl",
                run_id="run-p",
            ),
        )


def test_normal_pd_launch_keeps_measured_stats_and_environment_contract(
    monkeypatch,
):
    launches = _capture_launches(monkeypatch)
    pair = _unlaunched_pair()

    pair._launch_engine("prefill")

    assert len(launches) == 1
    launch = launches[0]
    command = launch["command"]
    assert command[:3] == [
        pd_module.sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
    ]
    assert "--disable-log-stats" in command
    assert "--enable-mfu-metrics" not in command
    assert "--no-enable-prefix-caching" not in command
    assert "RAG_STACK_STAGE_TELEMETRY_PATH" not in launch["env"]
    assert "RAG_STACK_VLLM_CALIBRATION_RUN_ID" not in launch["env"]
    assert "VLLM_SERVER_DEV_MODE" not in launch["env"]
    assert "VLLM_DEBUG_MFU_METRICS" not in launch["env"]
    assert pair.prefill_proc is launch["process"]


def test_calibration_launch_uses_role_telemetry_and_pd_safe_metrics(
    monkeypatch,
    tmp_path,
):
    launches = _capture_launches(monkeypatch)
    prefill_path = str((tmp_path / "prefill.jsonl").resolve())
    decode_path = str((tmp_path / "decode.jsonl").resolve())
    telemetry = VllmPdCalibrationTelemetry(
        prefill=VllmPdRoleTelemetry(prefill_path, "prefill-run"),
        decode=VllmPdRoleTelemetry(decode_path, "decode-run"),
    )
    pair = _unlaunched_pair(
        calibration_telemetry=telemetry,
        enable_prefix_caching=False,
    )

    pair._launch_engine("prefill")
    pair._launch_engine("decode")

    assert len(launches) == 2
    expected = [
        ("prefill", prefill_path, "prefill-run", "kv_producer"),
        ("decode", decode_path, "decode-run", "kv_consumer"),
    ]
    for launch, (role, path, run_id, kv_role) in zip(launches, expected):
        command = launch["command"]
        env = launch["env"]
        assert command[:3] == [
            pd_module.sys.executable,
            "-m",
            "rag_stack.vllm_instrumentation.serving_curves.calibration.vllm_server",
        ]
        assert "--enable-mfu-metrics" in command
        assert "--disable-log-stats" in command
        assert "--no-enable-prefix-caching" in command
        assert env["RAG_STACK_STAGE_TELEMETRY_PATH"] == path
        assert env["RAG_STACK_VLLM_CALIBRATION_RUN_ID"] == run_id
        assert env["VLLM_SERVER_DEV_MODE"] == "1"
        assert env["VLLM_DEBUG_MFU_METRICS"] == "1"
        assert env["RAG_STACK_PD_CALIBRATION_SAFE_PROMETHEUS"] == "1"
        assert env["CUDA_VISIBLE_DEVICES"] == (
            "0" if role == "prefill" else "1"
        )
        kv_index = command.index("--kv-transfer-config")
        assert json.loads(command[kv_index + 1]) == {
            "kv_connector": "NixlConnector",
            "kv_role": kv_role,
        }

    assert pair.prefill_proc is launches[0]["process"]
    assert pair.decode_proc is launches[1]["process"]
