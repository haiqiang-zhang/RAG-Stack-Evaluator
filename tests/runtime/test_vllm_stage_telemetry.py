from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import rag_stack_evaluator.vllm_instrumentation.serving_curves.calibration.vllm_server as server_module
from rag_stack_evaluator.vllm_instrumentation.serving_curves.calibration.vllm_server import (
    PD_SAFE_PROMETHEUS_ENV,
    install_telemetry_stat_logger,
)
from rag_stack_evaluator.vllm_instrumentation.serving_curves.calibration.vllm_telemetry import (
    TELEMETRY_ENV,
    TELEMETRY_SCHEMA,
    VllmStageTelemetryStatLogger,
)


class _Clock:
    def __init__(self, *values: float) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


def _stats(
    *,
    running: int,
    waiting: int,
    prefill_sequences: int = 0,
    prefill_tokens: int = 0,
    prefill_context: int = 0,
    prefill_token_context_product: int = 0,
    decode_sequences: int = 0,
    decode_tokens: int = 0,
    decode_context: int = 0,
    decode_token_context_product: int = 0,
    calc_duration: float = 0.0,
) -> SimpleNamespace:
    context = {
        "num_prefill_requests": prefill_sequences,
        "prefill_num_tokens": prefill_tokens,
        "prefill_context_len": prefill_context,
        "prefill_token_context_product": prefill_token_context_product,
        "num_decode_requests": decode_sequences,
        "decode_num_tokens": decode_tokens,
        "decode_context_len": decode_context,
        "decode_token_context_product": decode_token_context_product,
    }
    debug = SimpleNamespace(
        calc_duration=calc_duration,
        num_prefill_requests=prefill_sequences,
        num_decode_requests=decode_sequences,
        context_breakdown=context,
    )
    return SimpleNamespace(
        num_running_reqs=running,
        num_waiting_reqs=waiting,
        perf_stats=SimpleNamespace(debug_stats=debug),
    )


def _read(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_backlogged_mixed_cycle_is_publishable_and_terminal_decode(tmp_path):
    path = tmp_path / "cycles.jsonl"
    clock = _Clock(10.0, 10.01, 11.0, 11.02)
    logger = VllmStageTelemetryStatLogger(
        None,
        engine_index=3,
        output_path=path,
        run_id="stage-test",
        monotonic=clock,
    )
    mixed = _stats(
        running=5,
        waiting=2,
        prefill_sequences=2,
        prefill_tokens=12,
        prefill_context=20,
        prefill_token_context_product=130,
        decode_sequences=3,
        decode_tokens=3,
        decode_context=90,
        decode_token_context_product=90,
        calc_duration=0.05,
    )

    logger.record(mixed, None, engine_idx=3)
    logger.record(mixed, None, engine_idx=3)

    first, second = _read(path)
    assert first["schema"] == TELEMETRY_SCHEMA
    assert first["run_id"] == "stage-test"
    assert first["sequence"] == 0
    assert first["work_sequence"] == 0
    assert first["cycle_id"].startswith("stage-test:0:")
    assert len(first["shape_digest"]) == 64
    assert first["publishable"] is False
    assert first["reason"] == "first_sample_no_interval"
    assert first["active_service_s"] is None

    assert second["sequence"] == 1
    assert second["work_sequence"] == 1
    assert second["cycle_id"].startswith("stage-test:1:")
    assert second["shape_digest"] == first["shape_digest"]
    assert second["engine"] == 3
    assert second["raw_interval_s"] == pytest.approx(1.0)
    assert second["instrumentation_exclusions_s"] == pytest.approx(
            {
                "previous_telemetry_record_s": 0.01,
                "current_vllm_perf_debug_calc_s": 0.05,
                "frontend_boundary_telemetry_s": 0.0,
                "total_s": 0.06,
            }
    )
    assert second["active_service_s"] == pytest.approx(0.94)
    assert second["publishable"] is True
    assert second["reason"] == "publishable"
    assert second["shape"]["terminal_performance_stage"] == "decode"
    assert second["shape"]["prefill"] == {
        "scheduled_sequences": 2,
        "scheduled_tokens": 12,
        "scheduled_context_tokens": 20,
        "token_context_product": 130,
    }
    assert second["shape"]["decode"]["scheduled_sequences"] == 3
    assert second["backlog_before_interval"] == {
        "running_sequences": 5,
        "waiting_sequences": 2,
    }
    assert "client_concurrency" not in path.read_text()


def test_prefill_only_cycle_has_prefill_terminal(tmp_path):
    path = tmp_path / "cycles.jsonl"
    logger = VllmStageTelemetryStatLogger(
        None,
        output_path=path,
        run_id="stage-test",
        monotonic=_Clock(1.0, 1.001, 1.5, 1.502),
    )
    prefill = _stats(
        running=1,
        waiting=1,
        prefill_sequences=1,
        prefill_tokens=32,
        prefill_context=40,
        prefill_token_context_product=1280,
        calc_duration=0.004,
    )

    logger.record(prefill, None)
    logger.record(prefill, None)

    record = _read(path)[1]
    assert record["publishable"] is True
    assert record["shape"]["terminal_performance_stage"] == "prefill"
    assert record["shape"]["decode"]["scheduled_sequences"] == 0


def test_idle_fill_and_no_work_never_become_active_service(tmp_path):
    path = tmp_path / "cycles.jsonl"
    logger = VllmStageTelemetryStatLogger(
        None,
        output_path=path,
        run_id="stage-test",
        monotonic=_Clock(0.0, 0.001, 1.0, 1.001, 2.0, 2.001),
    )
    work_without_backlog = _stats(
        running=0,
        waiting=0,
        decode_sequences=1,
        decode_tokens=1,
        decode_context=10,
        decode_token_context_product=10,
        calc_duration=0.001,
    )
    work_leaving_backlog = _stats(
        running=1,
        waiting=0,
        decode_sequences=1,
        decode_tokens=1,
        decode_context=11,
        decode_token_context_product=11,
        calc_duration=0.001,
    )
    no_work = _stats(running=1, waiting=0, calc_duration=0.001)

    logger.record(work_without_backlog, None)
    logger.record(work_leaving_backlog, None)
    logger.record(no_work, None)

    _, fill, idle = _read(path)
    assert fill["reason"] == "previous_cycle_not_backlogged"
    assert fill["active_service_s"] is None
    assert fill["publishable"] is False
    assert idle["reason"] == "current_cycle_has_no_scheduled_work"
    assert idle["work_sequence"] is None
    assert idle["cycle_id"] is None
    assert idle["active_service_s"] is None
    assert idle["publishable"] is False


def test_missing_debug_stats_fail_closed_without_synthetic_zero(tmp_path):
    path = tmp_path / "cycles.jsonl"
    logger = VllmStageTelemetryStatLogger(
        None,
        output_path=path,
        run_id="stage-test",
        monotonic=_Clock(0.0, 0.001, 1.0, 1.001),
    )
    baseline = _stats(
        running=1,
        waiting=1,
        decode_sequences=1,
        decode_tokens=1,
        decode_context=10,
        decode_token_context_product=10,
    )
    missing = SimpleNamespace(
        num_running_reqs=1,
        num_waiting_reqs=1,
        perf_stats=None,
    )

    logger.record(baseline, None)
    logger.record(missing, None)

    record = _read(path)[1]
    assert record["reason"] == "missing_perf_debug_stats"
    assert record["shape"] is None
    assert record["active_service_s"] is None
    assert record["publishable"] is False
    assert record["instrumentation_exclusions_s"][
        "current_vllm_perf_debug_calc_s"
    ] is None


def test_scheduler_stats_none_does_not_split_cycle_interval(tmp_path):
    path = tmp_path / "cycles.jsonl"
    logger = VllmStageTelemetryStatLogger(
        None,
        output_path=path,
        run_id="stage-test",
        monotonic=_Clock(2.0, 2.01, 3.0, 3.01),
    )
    stats = _stats(
        running=1,
        waiting=0,
        decode_sequences=1,
        decode_tokens=1,
        decode_context=4,
        decode_token_context_product=4,
        calc_duration=0.01,
    )

    logger.record(stats, None)
    logger.record(None, None)
    logger.record(stats, None)

    rows = _read(path)
    assert len(rows) == 2
    assert rows[1]["raw_interval_s"] == pytest.approx(1.0)


def test_wrapper_patch_is_opt_in_and_replaces_async_llm_symbol(monkeypatch):
    fake_module = SimpleNamespace(load_stat_logger_plugin_factories=lambda: [])
    monkeypatch.delenv(TELEMETRY_ENV, raising=False)
    assert install_telemetry_stat_logger(fake_module) is False
    assert fake_module.load_stat_logger_plugin_factories() == []

    monkeypatch.setenv(TELEMETRY_ENV, "/tmp/stage-cycles.jsonl")
    assert install_telemetry_stat_logger(fake_module) is True
    assert fake_module.load_stat_logger_plugin_factories() == [
        VllmStageTelemetryStatLogger
    ]


def test_pd_safe_prometheus_keeps_scheduler_and_drops_iteration_stats(
    monkeypatch,
):
    class FakePrometheusStatLogger:
        def __init__(self, *_args, **_kwargs):
            self.calls = []

        def record(
            self,
            scheduler_stats,
            iteration_stats,
            mm_cache_stats=None,
            engine_idx=0,
        ):
            self.calls.append(
                (
                    scheduler_stats,
                    iteration_stats,
                    mm_cache_stats,
                    engine_idx,
                )
            )

    monkeypatch.setenv(TELEMETRY_ENV, "/tmp/pd-cycles.jsonl")
    monkeypatch.setenv(PD_SAFE_PROMETHEUS_ENV, "1")
    monkeypatch.setattr(
        server_module,
        "_prometheus_stat_logger_base",
        lambda: FakePrometheusStatLogger,
    )
    fake_module = SimpleNamespace(load_stat_logger_plugin_factories=lambda: [])

    assert install_telemetry_stat_logger(fake_module) is True
    factories = fake_module.load_stat_logger_plugin_factories()
    assert factories[0] is VllmStageTelemetryStatLogger
    assert len(factories) == 2
    assert issubclass(factories[1], FakePrometheusStatLogger)

    logger = factories[1](None, [0])
    scheduler = object()
    unsafe_pd_iteration = object()
    mm_cache = object()
    logger.record(
        scheduler,
        unsafe_pd_iteration,
        mm_cache_stats=mm_cache,
        engine_idx=7,
    )
    assert logger.calls == [(scheduler, None, mm_cache, 7)]
