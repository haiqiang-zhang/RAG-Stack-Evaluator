import asyncio
import time
from types import SimpleNamespace

import pandas as pd
import pytest

from rag_stack_evaluator.static_rag_evaluator.measured import serving_runtime as sr


class _Owner:
    def _generator_chip_count(self, system_config):
        return 1

    def _add_deployment_metadata(self, summary, node_lines, system_config):
        return None


def test_dataset_admission_order_is_exactly_balanced_per_epoch():
    n_rows = 100
    order = sr._BalancedDatasetAdmissionOrder(n_rows)
    admitted = [order.idx_for_seq(seq) for seq in range(3 * n_rows)]

    epochs = [
        admitted[start:start + n_rows]
        for start in range(0, len(admitted), n_rows)
    ]
    assert all(sorted(epoch) == list(range(n_rows)) for epoch in epochs)
    assert epochs[0] != epochs[1]
    assert epochs[1] != epochs[2]

    # A partial epoch can differ by only one occurrence per row. This retains
    # the old exact global balance without replaying the file order verbatim.
    prefix = admitted[:256]
    counts = [prefix.count(idx) for idx in range(n_rows)]
    assert min(counts) == 2
    assert max(counts) == 3
    assert counts.count(3) == 56


def test_dataset_admission_order_repeats_across_runtime_local_instances():
    first_runtime_order = sr._BalancedDatasetAdmissionOrder(17)
    second_runtime_order = sr._BalancedDatasetAdmissionOrder(17)

    first = [first_runtime_order.idx_for_seq(seq) for seq in range(85)]
    second = [second_runtime_order.idx_for_seq(seq) for seq in range(85)]

    assert first == second
    assert first_runtime_order._epoch == 4
    assert second_runtime_order._epoch == 4
    assert len(first_runtime_order._order) == 17
    assert len(second_runtime_order._order) == 17


@pytest.mark.parametrize(
    ("system_config", "initial", "mode", "expected"),
    [
        ({}, 1024, "sequential", 4096),
        ({}, 1024, "react", 16384),
        ({"measured_max_load_concurrency": 2048}, 1024, "sequential", 2048),
        ({"batching": {"max_load_concurrency": 1536}}, 1024, "react", 1536),
        ({"measured_max_load_concurrency": 64}, 1024, "sequential", 1024),
    ],
)
def test_population_hard_cap_is_shared_policy(
    system_config, initial, mode, expected
):
    assert sr._measured_population_hard_cap(
        system_config, initial, mode
    ) == expected


def _s43_system_config(**overrides):
    config = {
        "batch_size_request": 256,
        "batch_size_decode": 256,
        "batching": {
            "request": 256,
            "decode": 256,
            "dynamic_timeout_s": 0.05,
        },
        "vllm": {
            "engines": {
                "generator": {"max_num_seqs": 256},
            }
        },
    }
    config.update(overrides)
    return config


def _generator_runtime(api_server_count, **system_overrides):
    subprocess = SimpleNamespace(
        key=SimpleNamespace(
            max_num_seqs=256,
            api_server_count=api_server_count,
        )
    )
    stage = {
        "stage": "generator",
        "instance": SimpleNamespace(_subprocess=subprocess),
        "node": object(),
        "params": {},
    }
    return sr.MeasuredServingRuntime(
        owner=_Owner(),
        stages=[stage],
        qa_data=pd.DataFrame({"query": ["q"]}),
        node_lines={},
        system_config=_s43_system_config(**system_overrides),
        config={"pipeline_runtime": {"rag_dataflow": "sequential"}},
    )


@pytest.mark.parametrize(
    ("api_server_count", "expected"),
    [(1, 308), (4, 308)],
)
def test_s43_llm_http_admission_is_global_across_api_frontends(
    api_server_count, expected
):
    runtime = _generator_runtime(api_server_count)
    service = runtime._build_services()[0]

    assert runtime.load_concurrency_initial == 1024
    assert runtime.load_concurrency_hard_cap == 4096
    assert isinstance(service, sr.LLMContinuousStageService)
    assert service.max_inflight == expected


def test_llm_http_admission_is_clamped_by_population_hard_cap():
    runtime = _generator_runtime(4, measured_max_load_concurrency=256)

    service = runtime._build_services()[0]

    assert runtime.load_concurrency_hard_cap == 256
    assert service.max_inflight == 256


def _pd_generator_stage():
    subprocess = SimpleNamespace(
        key=SimpleNamespace(
            max_num_seqs=64,
            prefill_max_num_seqs=2,
            decode_max_num_seqs=256,
        ),
        stage_admission_limit=lambda: 259,
    )
    return {
        "stage": "generator",
        "instance": SimpleNamespace(_subprocess=subprocess),
        "node": object(),
        "params": {},
    }


def test_disagg_pd_stage_admission_includes_one_prefill_feeder():
    stage = _pd_generator_stage()

    assert sr._stage_max_inflight(
        stage, _s43_system_config(), "generator"
    ) == 259
    assert sr._stage_max_inflight(
        stage,
        _s43_system_config(),
        "generator",
        population_hard_cap=128,
    ) == 128


def test_disagg_pd_runtime_uses_role_aware_stage_admission():
    stage = _pd_generator_stage()
    runtime = sr.MeasuredServingRuntime(
        owner=_Owner(),
        stages=[stage],
        qa_data=pd.DataFrame({"query": ["q"]}),
        node_lines={},
        system_config=_s43_system_config(),
        config={"pipeline_runtime": {"rag_dataflow": "sequential"}},
    )

    service = runtime._build_services()[0]

    assert isinstance(service, sr.LLMContinuousStageService)
    assert service.max_inflight == 259


def test_pd_role_stats_follow_measured_window_and_reach_summary():
    class FakePdSubprocess:
        base_url = None

        def __init__(self):
            self.started = 0
            self.ended = 0

        def mark_role_admission_window_start(self):
            self.started += 1

        def mark_role_admission_window_end(self):
            self.ended += 1

        def role_admission_stats(self):
            return {
                "prefill": {
                    "engine_max_num_seqs": 2,
                    "admission_limit": 3,
                    "max_inflight_observed": 3,
                    "avg_queue_wait_s": 0.01,
                },
                "decode": {
                    "engine_max_num_seqs": 256,
                    "admission_limit": 256,
                    "max_inflight_observed": 200,
                    "avg_queue_wait_s": 0.02,
                },
            }

    subprocess = FakePdSubprocess()
    stage = {
        "stage": "generator",
        "instance": SimpleNamespace(_subprocess=subprocess),
        "node": object(),
        "params": {},
    }
    service = sr.LLMContinuousStageService(
        _Owner(), stage, max_inflight=259
    )
    service.mark_window_start()
    service.mark_window_end()

    stats = service.stats()

    assert subprocess.started == 1
    assert subprocess.ended == 1
    assert stats["pd_role_admission"]["prefill"]["admission_limit"] == 3
    assert stats["pd_role_admission"]["decode"]["admission_limit"] == 256

    system_config = _s43_system_config()
    system_config["layout"] = {
        "engines": {
            "generator": {
                "pd_serving": "disagg_pd",
                "devices": ["cuda:1", "cuda:2", "cuda:3"],
                "num_chips": 3,
                "tp": 1,
                "pp": 1,
            }
        }
    }
    runtime = sr.MeasuredServingRuntime(
        owner=_Owner(),
        stages=[stage],
        qa_data=pd.DataFrame({"query": ["q"]}),
        node_lines={},
        system_config=system_config,
        config={"pipeline_runtime": {"rag_dataflow": "sequential"}},
    )
    runtime.services = [service]
    summary = {"qps": 1.0}

    runtime._annotate_summary(summary, n_measured=1, total_wall=1.0)

    assert summary["generator_max_inflight"] == 259
    assert summary["generator_pd_prefill_engine_max_num_seqs"] == 2
    assert summary["generator_pd_prefill_admission_limit"] == 3
    assert summary["generator_pd_decode_engine_max_num_seqs"] == 256
    assert summary["generator_pd_decode_admission_limit"] == 256


def test_query_expansion_http_admission_falls_back_to_system_config():
    system_config = _s43_system_config()
    system_config["vllm"]["engines"]["query_expansion"] = {
        "max_num_seqs": 64,
        "api_server_count": 2,
    }
    stage = {
        "stage": "query_expansion",
        "instance": SimpleNamespace(generator=SimpleNamespace()),
        "node": object(),
        "params": {},
    }
    runtime = sr.MeasuredServingRuntime(
        owner=_Owner(),
        stages=[stage],
        qa_data=pd.DataFrame({"query": ["q"]}),
        node_lines={},
        system_config=system_config,
        config={"pipeline_runtime": {"rag_dataflow": "sequential"}},
    )

    service = runtime._build_services()[0]

    assert isinstance(service, sr.QueryExpansionContinuousStageService)
    assert service.max_inflight == 77


def test_client_fill_plateau_does_not_stop_population_growth(monkeypatch):
    """Client occupancy can plateau outside vLLM and is not a stop proof."""
    from rag_stack_evaluator.static_rag_evaluator.measured.vllm_deployment import TrialInvalid

    real_sleep = asyncio.sleep

    async def fast_adapter_sleep(delay):
        # Accelerate only the adapter's one-second samples and driver staggering;
        # leave warmup/measured wall-cap timers sleeping until normal cancellation.
        if delay <= 2.0:
            await real_sleep(0.001)
        else:
            await real_sleep(delay)

    monkeypatch.setattr(sr.asyncio, "sleep", fast_adapter_sleep)
    # This test covers population growth/fail-closed semantics, not the
    # production stage-cycle span (covered below). Keep its synthetic loop tiny.
    monkeypatch.setattr(sr, "_SATURATION_STABILITY_MIN_SPAN", 1)
    monkeypatch.setattr(sr, "_SATURATION_STABILITY_BATCH_CYCLES", 1)
    monkeypatch.setattr(sr, "_SATURATION_STABILITY_MAX_SPAN", 1)
    monkeypatch.setattr(sr, "_WARMUP_WALL_CAP_S", 3.0)
    runtime = sr.MeasuredServingRuntime(
        owner=_Owner(),
        stages=[],
        qa_data=pd.DataFrame({"query": ["q"]}),
        node_lines={},
        system_config={
            "batch_size_request": 1,
            "measured_load_concurrency": 2,
            "measured_warmup_queries": 1,
            "batching": {
                "request": 1,
                "decode": 1,
                "dynamic_timeout_s": 0.001,
            },
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
    service = sr.LLMContinuousStageService(
        _Owner(),
        {"stage": "generator", "instance": object(), "params": {}},
        max_inflight=runtime.load_concurrency_hard_cap,
    )
    # Simulate a frontend/client occupancy plateau with no vLLM waiting samples
    # and no worker backlog. The adapter must continue to its finite cap instead
    # of treating this plateau as saturation or an early-stop signal.
    service.inflight = runtime.load_concurrency_initial
    runtime.services = [service]

    async def fake_request(state):
        await real_sleep(0.0001)

    with pytest.raises(
        TrialInvalid,
        match="measured_(population_not_saturated|warmup_gate_timeout)",
    ):
        asyncio.run(runtime._run_closed_loop_saturated(fake_request))

    assert runtime.load_concurrency_hard_cap == 8
    assert runtime.load_concurrency == 8
    assert runtime._saturation.get("saturated") is not True


def _saturation_runtime(*, warmup_queries, measured_queries, population=4):
    return sr.MeasuredServingRuntime(
        owner=_Owner(),
        stages=[],
        qa_data=pd.DataFrame({"query": ["q"]}),
        node_lines={},
        system_config={
            "batch_size_request": 1,
            "measured_load_concurrency": population,
            "measured_max_load_concurrency": population,
            "measured_warmup_queries": warmup_queries,
            "measured_queries": measured_queries,
            "batching": {
                "request": 1,
                "decode": 1,
                "dynamic_timeout_s": 0.001,
            },
            "vllm": {"engines": {"generator": {"max_num_seqs": 1}}},
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


def _fake_llm_service(runtime, recent_waiting):
    service = sr.LLMContinuousStageService(
        _Owner(),
        {"stage": "generator", "instance": object(), "params": {}},
        max_inflight=runtime.load_concurrency_hard_cap,
    )
    service.inflight = runtime.load_concurrency_initial
    service.recent_engine_waiting = recent_waiting
    # The tests supply deterministic telemetry directly; never start HTTP/NVML
    # probes merely to exercise the saturation state machine.
    service.start_engine_probe = lambda interval_s=1.0: None
    service.stop_engine_probe = lambda: None
    service.mark_probe_window = lambda: None
    return service


def _collocated_outer_stage_service(
    *, engine_cap=1, stage_gate=None, inflight=None, waiting=1, subprocess=None
):
    if subprocess is None:
        subprocess = SimpleNamespace(
            key=SimpleNamespace(max_num_seqs=engine_cap)
        )
    expected_gate = engine_cap + sr._admission_slack(engine_cap)
    service = sr.LLMContinuousStageService(
        _Owner(),
        {
            "stage": "generator",
            "instance": SimpleNamespace(_subprocess=subprocess),
            "params": {},
        },
        max_inflight=expected_gate if stage_gate is None else stage_gate,
    )
    service.inflight = (
        service.max_inflight if inflight is None else inflight
    )
    service.waiting = waiting
    service.recent_engine_waiting = lambda _n: [0.0] * 5
    service.start_engine_probe = lambda interval_s=1.0: None
    service.stop_engine_probe = lambda: None
    service.mark_probe_window = lambda: None
    return service


def test_collocated_outer_backlog_requires_unclamped_deployed_stage_gate():
    service = _collocated_outer_stage_service(engine_cap=8)

    assert sr._collocated_outer_backlog_engine_cap(service) == 8

    clamped = _collocated_outer_stage_service(
        engine_cap=8,
        stage_gate=8,
        inflight=8,
    )
    assert sr._collocated_outer_backlog_engine_cap(clamped) is None


def test_collocated_outer_backlog_requires_explicit_waiter():
    service = _collocated_outer_stage_service(engine_cap=8, waiting=0)

    assert sr._collocated_outer_backlog_engine_cap(service) is None


def test_pd_role_admission_never_duplicates_collocated_outer_backlog():
    class FakePdPair:
        key = SimpleNamespace(max_num_seqs=8)

        @staticmethod
        def stage_admission_limit():
            # Deliberately equal the collocated gate: the role rows, not a
            # coincidental numeric mismatch, must keep this out of the proof.
            return 10

        @staticmethod
        def role_admission_stats():
            return {
                "prefill": {
                    "engine_max_num_seqs": 2,
                    "admission_limit": 3,
                },
                "decode": {
                    "engine_max_num_seqs": 7,
                    "admission_limit": 7,
                },
            }

    service = _collocated_outer_stage_service(
        engine_cap=8,
        subprocess=FakePdPair(),
    )

    assert service.max_inflight == 10
    assert sr._collocated_outer_backlog_engine_cap(service) is None


def test_current_collocated_outer_backlog_opens_window(monkeypatch):
    """Three of five current full-gate waiter samples prove standing demand."""
    real_sleep = asyncio.sleep

    async def accelerated_probe_sleep(delay):
        if delay == 1.0:
            await real_sleep(0.002)
        else:
            await real_sleep(delay)

    monkeypatch.setattr(sr.asyncio, "sleep", accelerated_probe_sleep)
    monkeypatch.setattr(sr, "_WARMUP_WALL_CAP_S", 0.5)
    monkeypatch.setattr(sr, "_MEASURED_WALL_CAP_S", 0.5)
    runtime = _saturation_runtime(
        warmup_queries=1024,
        measured_queries=16,
        population=2,
    )
    service = _collocated_outer_stage_service(engine_cap=1)
    runtime.services = [service]
    admissions = []

    async def fake_request(state):
        admissions.append(state.is_measured)
        await real_sleep(0.0001)

    result, summary = asyncio.run(
        runtime._run_closed_loop_saturated(fake_request)
    )

    assert len(result) == 1
    assert summary["saturation_evidence"] == "llm_stage_backlog:generator"
    assert summary["saturation_candidate_identity"] == (
        "llm_stage_backlog:generator"
    )
    assert summary["population_saturated"] is True
    assert summary["saturation_stability_span_queries"] == 32
    assert summary["saturation_stability_span_source"] == "stage_cycles"
    assert summary["measurement_start_gate"] == (
        "warmup_completion_rate_stable"
    )
    assert admissions and any(admissions)


def test_wall_budget_span_keeps_e48_like_low_qps_finite():
    evidence = sr._wall_budget_stability_span(
        preferred_span=256,
        candidate_completions=30,
        candidate_age_s=120.0,
        measured_wall_cap_s=360.0,
    )

    assert evidence["effective_span_queries"] == 15
    assert evidence["minimum_span_queries"] == 10
    assert evidence["candidate_completions"] == 30
    assert evidence["candidate_age_s"] == pytest.approx(120.0)
    assert evidence["candidate_rate_qps"] == pytest.approx(0.25)
    assert evidence["available_span_queries"] == 15
    assert evidence["measurement_span_queries"] == 22


def test_wall_budget_span_matches_existing_dense_stationarity_proof():
    """The low-QPS fallback must not make its wall proof unreachable.

    This is the saved eval9 completion-count shape: the existing five-bin CV is
    stable, but the old four-boundary budget selected W=68 and guaranteed fewer
    than three spans per bin. The span derived from the same finite wall budget
    must make the existing dense-stream selector applicable without changing its
    threshold.
    """
    evidence = sr._wall_budget_stability_span(
        preferred_span=256,
        candidate_completions=137,
        candidate_age_s=90.11065665999195,
        measured_wall_cap_s=360.0,
    )

    assert evidence["effective_span_queries"] == 32
    assert evidence["measurement_span_queries"] == 32
    wall_counts = [105, 106, 137, 119, 98]
    timestamps = [
        window * 72.0 + (offset + 0.5) * 72.0 / count
        for window, count in enumerate(wall_counts)
        for offset in range(count)
    ]
    stationarity = sr._qps_stationarity_evidence(
        timestamps,
        0.0,
        360.0,
        completion_span_queries=evidence["effective_span_queries"],
    )

    assert stationarity["wall_time_dense_completion_stream"] is True
    assert min(stationarity["wall_subwindow_completion_spans"]) >= 3.0
    assert stationarity["method"] == "wall_time_subwindows"
    assert stationarity["selection_reason"] == "dense_completion_stream"
    assert stationarity["selected_cv"] == pytest.approx(
        stationarity["wall_subwindow_cv"]
    )
    assert stationarity["selected_cv"] <= 0.15


def test_wall_budget_span_fails_closed_below_existing_rate_resolution():
    evidence = sr._wall_budget_stability_span(
        preferred_span=256,
        candidate_completions=18,
        candidate_age_s=120.0,
        measured_wall_cap_s=360.0,
    )

    assert evidence["minimum_span_queries"] == 10
    assert evidence["available_span_queries"] == 9
    assert evidence["effective_span_queries"] is None


def test_minimum_completion_span_accepts_constant_cadence():
    # Two groups of ten completions each contain nine equal intervals. The
    # warmup proof must compare those like-for-like; sharing the boundary
    # completion would compare nine intervals with ten and falsely report
    # 10.526% drift against the existing 10% threshold.
    timestamps = [float(index) for index in range(20)]

    assert sr._completion_rate_stable(timestamps, 10)


def test_completion_span_rejects_real_rate_change():
    timestamps = [float(index) for index in range(10)]
    timestamps.extend(10.0 + 2.0 * index for index in range(1, 11))

    assert not sr._completion_rate_stable(timestamps, 10)


def test_persistent_low_qps_engine_backlog_uses_one_wall_budget_span(
    monkeypatch,
):
    """A finite e48-like run proves backlog and rate without requiring 2*256."""
    real_sleep = asyncio.sleep

    async def accelerated_probe_sleep(delay):
        if delay == 1.0:
            await real_sleep(0.005)
        else:
            await real_sleep(delay)

    monkeypatch.setattr(sr.asyncio, "sleep", accelerated_probe_sleep)
    monkeypatch.setattr(sr, "_WARMUP_WALL_CAP_S", 0.3)
    monkeypatch.setattr(sr, "_MEASURED_WALL_CAP_S", 0.5)
    monkeypatch.delenv("RAG_STACK_WARMUP_WALL_CAP_S", raising=False)
    monkeypatch.delenv("RAG_STACK_MEASURED_WALL_CAP_S", raising=False)
    runtime = _saturation_runtime(
        warmup_queries=256,
        measured_queries=80,
        population=1,
    )
    runtime.system_config["vllm"]["engines"]["generator"][
        "max_num_seqs"
    ] = 64
    runtime.services = [_fake_llm_service(runtime, lambda _n: [1.0] * 5)]

    async def steady_slow_request(_state):
        await real_sleep(0.004)

    _result, summary = asyncio.run(
        runtime._run_closed_loop_saturated(steady_slow_request)
    )

    effective_span = summary["saturation_stability_span_queries"]
    assert summary["saturation_evidence"] == "engine_backlog:generator"
    assert summary["population_saturated"] is True
    assert summary["saturation_candidate_identity"] == (
        "engine_backlog:generator"
    )
    assert summary["saturation_stability_span_source"] == "wall_budget"
    assert summary["saturation_stability_preferred_span_queries"] == 256
    assert 10 <= effective_span < 32
    assert summary["saturation_stability_effective_span_queries"] == (
        effective_span
    )
    assert summary["saturation_candidate_completions_at_span_freeze"] >= (
        2 * effective_span
    )
    support_at = summary["workload_support_complete_warmup_completed"]
    proof_at = summary["warmup_fresh_rate_proof_start_completion"]
    assert support_at is not None
    assert proof_at is not None and proof_at >= support_at
    # Candidate age/completions and the low-QPS W are rebuilt after the support
    # epoch reset; the cold-prefix completion cannot contribute to the freeze.
    assert summary["saturation_candidate_completions_at_span_freeze"] <= (
        summary["warmup_completed"] - support_at
    )
    assert summary["warmup_fresh_rate_observed_completions"] >= (
        summary["warmup_fresh_rate_required_completions"]
    )
    assert summary["warmup_wall_cap_s"] == pytest.approx(0.3)
    assert summary["measured_wall_cap_s"] == pytest.approx(0.5)
    assert summary["saturation_candidate_rate_qps_at_span_freeze"] > 0.0
    assert summary["saturation_wall_budget_minimum_span_queries"] == 10
    assert summary["measurement_rate_stability_waves"] >= 2.0
    assert summary["qps_stationarity_completion_span_queries_requested"] == 256
    assert summary["qps_stationarity_method"] == "wall_time_subwindows"
    assert summary["qps_stationarity_selection_reason"] == (
        "completion_span_unavailable"
    )
    assert summary["qps_subwindow_cv"] <= 0.15
    assert summary["measured_gt_admissibility"]["admissible"] is True


def test_same_llm_candidate_keeps_only_frozen_w_across_late_support(
    monkeypatch,
):
    """A pre-support W is dimensional only; its timestamps cannot open a window."""
    real_sleep = asyncio.sleep

    async def accelerated_probe_sleep(delay):
        if delay == 1.0:
            await real_sleep(0.002)
        else:
            await real_sleep(delay)

    monkeypatch.setattr(sr.asyncio, "sleep", accelerated_probe_sleep)
    monkeypatch.setattr(sr, "_WARMUP_WALL_CAP_S", 0.2)
    monkeypatch.setattr(sr, "_MEASURED_WALL_CAP_S", 0.2)
    monkeypatch.delenv("RAG_STACK_WARMUP_WALL_CAP_S", raising=False)
    monkeypatch.delenv("RAG_STACK_MEASURED_WALL_CAP_S", raising=False)

    span_frozen = False

    def fixed_low_qps_span(
        *, preferred_span, candidate_completions, candidate_age_s,
        measured_wall_cap_s,
    ):
        nonlocal span_frozen
        span_frozen = True
        return {
            "effective_span_queries": 2,
            "minimum_span_queries": 1,
            "candidate_completions": candidate_completions,
            "candidate_age_s": candidate_age_s,
            "candidate_rate_qps": (
                candidate_completions / candidate_age_s
                if candidate_age_s > 0.0 else 0.0
            ),
            "available_span_queries": candidate_completions // 2,
            "measurement_span_queries": 2,
        }

    monkeypatch.setattr(sr, "_wall_budget_stability_span", fixed_low_qps_span)
    # Until W has really frozen, expose only row 0. Row 1 completes immediately
    # afterward, making support strictly later than candidate/W establishment.
    monkeypatch.setattr(
        sr._BalancedDatasetAdmissionOrder,
        "idx_for_seq",
        lambda self, seq: 0 if not span_frozen else seq % 2,
    )

    runtime = _saturation_runtime(
        warmup_queries=256,
        measured_queries=4,
        population=1,
    )
    runtime.qa_data = pd.DataFrame({"query": ["q0", "q1"]})
    runtime.n_rows = 2
    probe_epochs = 0

    def persistent_backlog(_n):
        nonlocal probe_epochs
        probe_epochs += 1
        return [1.0] * 5

    runtime.services = [
        _fake_llm_service(runtime, persistent_backlog)
    ]
    admissions = []
    probe_epochs_at_support = None

    async def steady_request(state):
        nonlocal probe_epochs_at_support
        admissions.append((state.seq, state.idx, state.is_measured))
        await real_sleep(0.001)
        if state.idx == 1 and probe_epochs_at_support is None:
            probe_epochs_at_support = probe_epochs

    _result, summary = asyncio.run(
        runtime._run_closed_loop_saturated(steady_request)
    )

    support_at = summary["workload_support_complete_warmup_completed"]
    proof_at = summary["warmup_fresh_rate_proof_start_completion"]
    assert span_frozen is True
    assert support_at is not None and support_at > 4
    assert probe_epochs_at_support is not None
    # One mixed epoch detects the support transition; only the following full
    # five-sample post-support epoch may re-establish current-candidate evidence.
    assert probe_epochs >= probe_epochs_at_support + 2
    assert proof_at == support_at
    assert summary[
        "warmup_stability_span_dimension_preserved_across_workload_support"
    ] is True
    assert summary["saturation_candidate_identity"] == "engine_backlog:generator"
    assert summary["saturation_stability_effective_span_queries"] == 2
    assert summary["warmup_fresh_rate_required_completions"] == 4
    assert summary["warmup_fresh_rate_observed_completions"] >= 4
    assert summary["warmup_completed"] >= support_at + 4
    assert next(row for row in admissions if row[2])[0] >= support_at + 4


def test_startup_engine_waiting_burst_is_not_permanent_saturation(monkeypatch):
    """A first-stage launch wave must be gone when rate stability is proved."""
    from rag_stack_evaluator.static_rag_evaluator.measured.vllm_deployment import TrialInvalid

    real_sleep = asyncio.sleep

    async def accelerated_probe_sleep(delay):
        if delay == 1.0:
            await real_sleep(0.001)
        else:
            await real_sleep(delay)

    monkeypatch.setattr(sr.asyncio, "sleep", accelerated_probe_sleep)
    monkeypatch.setattr(sr, "_WARMUP_WALL_CAP_S", 0.2)
    runtime = _saturation_runtime(
        warmup_queries=1024,
        measured_queries=4,
        population=2,
    )
    probe_reads = 0

    def recent_waiting(_n):
        nonlocal probe_reads
        probe_reads += 1
        # Persistent for the first five-sample adapter epoch only, then gone.
        return [1.0] * 5 if probe_reads == 1 else [0.0] * 5

    runtime.services = [_fake_llm_service(runtime, recent_waiting)]
    admissions = []

    async def fake_request(state):
        admissions.append(state.is_measured)
        await real_sleep(0.0005)

    with pytest.raises(
        TrialInvalid,
        match="measured_(population_not_saturated|warmup_gate_timeout)",
    ):
        asyncio.run(runtime._run_closed_loop_saturated(fake_request))

    assert probe_reads >= 2
    assert runtime._saturation.get("evidence") != "engine_backlog:generator"
    assert runtime._saturation.get("saturated") is not True
    assert admissions and not any(admissions)


def test_current_worker_backlog_and_bounded_rate_span_open_window(monkeypatch):
    """B=32 worker proof uses W=4B=128, not a 1024-client turnover."""
    real_sleep = asyncio.sleep

    async def accelerated_probe_sleep(delay):
        if delay == 1.0:
            await real_sleep(0.002)
        else:
            await real_sleep(delay)

    monkeypatch.setattr(sr.asyncio, "sleep", accelerated_probe_sleep)
    monkeypatch.setattr(sr, "_WARMUP_WALL_CAP_S", 0.5)
    monkeypatch.setattr(sr, "_MEASURED_WALL_CAP_S", 0.5)
    runtime = _saturation_runtime(
        warmup_queries=1024,
        measured_queries=256,
        population=4,
    )

    async def run_case():
        llm = _fake_llm_service(runtime, lambda _n: [0.0] * 5)
        worker = sr.BatchedPureStageService(
            _Owner(),
            {
                "stage": "passage_compressor",
                "instance": object(),
                "node": object(),
                "params": {},
            },
            batch_size=32,
            timeout_s=0.0,
        )
        worker.backlog = lambda: 33
        runtime.services = [llm, worker]
        admissions = []

        async def fake_request(state):
            admissions.append(state.is_measured)
            await real_sleep(0.0001)

        result, summary = await runtime._run_closed_loop_saturated(fake_request)
        return result, summary, admissions

    result, summary, admissions = asyncio.run(run_case())

    assert len(result) == 1
    assert summary["saturation_evidence"] == "worker_backlog:passage_compressor"
    assert summary["population_saturated"] is True
    assert summary["saturation_stability_span_queries"] == 128
    assert summary["saturation_stability_span_source"] == "stage_cycles"
    assert summary["saturation_stability_preferred_span_queries"] == 128
    assert summary["saturation_stability_effective_span_queries"] == 128
    assert summary["measurement_start_gate"] == "warmup_completion_rate_stable"
    assert summary["warmup_completed"] >= 256
    assert summary["warmup_completed"] < 1024
    assert summary["measurement_rate_stability_waves"] == pytest.approx(2.0)
    first_measured = admissions.index(True)
    assert first_measured >= 256
    assert not any(admissions[:first_measured])


def test_current_pd_outer_backlog_opens_window(monkeypatch):
    """PD outer queue + every role admission slot proves standing demand."""
    real_sleep = asyncio.sleep

    async def accelerated_probe_sleep(delay):
        if delay == 1.0:
            await real_sleep(0.002)
        else:
            await real_sleep(delay)

    monkeypatch.setattr(sr.asyncio, "sleep", accelerated_probe_sleep)
    # Leave enough CPU-test wall time for the required support-transition epoch
    # plus one complete post-support five-sample epoch with 300 live drivers.
    # This does not relax either the production stability threshold or W=256.
    monkeypatch.setattr(sr, "_WARMUP_WALL_CAP_S", 0.8)
    monkeypatch.setattr(sr, "_MEASURED_WALL_CAP_S", 0.5)
    runtime = _saturation_runtime(
        warmup_queries=1024,
        measured_queries=16,
        population=300,
    )
    service = sr.LLMContinuousStageService(
        _Owner(),
        {"stage": "generator", "instance": object(), "params": {}},
        max_inflight=259,
    )
    service.inflight = service.max_inflight
    service.waiting = 1
    service.recent_engine_waiting = lambda _n: [0.0] * 5
    service.start_engine_probe = lambda interval_s=1.0: None
    service.stop_engine_probe = lambda: None
    service.mark_probe_window = lambda: None

    class FakePdPair:
        @staticmethod
        def role_admission_stats():
            return {
                "prefill": {
                    "engine_max_num_seqs": 2,
                    "admission_limit": 3,
                    "current_inflight": 3,
                    "current_waiting": 0,
                },
                "decode": {
                    "engine_max_num_seqs": 256,
                    "admission_limit": 256,
                    "current_inflight": 256,
                    "current_waiting": 0,
                },
            }

    service._subprocess = lambda: FakePdPair()
    runtime.services = [service]
    admissions = []

    async def fake_request(state):
        admissions.append(state.is_measured)
        await real_sleep(0.0001)

    result, summary = asyncio.run(
        runtime._run_closed_loop_saturated(fake_request)
    )

    assert len(result) == 1
    assert summary["saturation_evidence"] == (
        "pd_stage_backlog:generator"
    )
    assert summary["population_saturated"] is True
    assert summary["saturation_stability_span_queries"] == 256
    assert summary["saturation_stability_span_source"] == "stage_cycles"
    assert summary["saturation_stability_preferred_span_queries"] == 256
    assert summary["saturation_stability_effective_span_queries"] == 256
    assert summary["measurement_start_gate"] == "warmup_completion_rate_stable"
    assert admissions and any(admissions)


def test_population_increment_only_reports_cap_at_the_real_hard_cap():
    assert sr._next_population_increment(1125, 4096, 1.0) == 1
    assert sr._next_population_increment(4095, 4096, 1.0) == 1
    assert sr._next_population_increment(4096, 4096, 1.0) == 0


def test_population_adapter_waits_for_initial_and_adaptive_ramps(monkeypatch):
    """Sleeping drivers never participate in fill, candidate, or verdict proof."""
    from rag_stack_evaluator.static_rag_evaluator.measured.vllm_deployment import TrialInvalid

    real_sleep = asyncio.sleep

    async def compressed_driver_sleep(delay):
        task = asyncio.current_task()
        if task is not None and task.get_name().startswith("measured-driver-"):
            await real_sleep(delay * 0.001)
        else:
            await real_sleep(delay)

    monkeypatch.setattr(sr.asyncio, "sleep", compressed_driver_sleep)
    monkeypatch.setattr(sr, "_POPULATION_ADAPTER_SAMPLE_S", 0.001)
    monkeypatch.setattr(sr, "_SATURATION_STABILITY_MIN_SPAN", 1)
    monkeypatch.setattr(sr, "_SATURATION_STABILITY_BATCH_CYCLES", 1)
    monkeypatch.setattr(sr, "_SATURATION_STABILITY_MAX_SPAN", 1)
    monkeypatch.setattr(sr, "_WARMUP_WALL_CAP_S", 0.4)
    runtime = _saturation_runtime(
        warmup_queries=1024,
        measured_queries=8,
        population=64,
    )
    runtime.load_concurrency_hard_cap = 128

    telemetry_ramps = []

    def recent_waiting(_n):
        ramp = runtime._population_ramps[-1]
        telemetry_ramps.append(ramp.kind)
        assert ramp.complete_event.is_set()
        assert ramp.pending_drivers == 0
        return [0.0] * 5

    runtime.services = [_fake_llm_service(runtime, recent_waiting)]
    increment_ramps = []
    real_increment = sr._next_population_increment

    def checked_increment(population, hard_cap, mean_fill):
        increment_ramps.append([
            (ramp.kind, ramp.complete_event.is_set(), ramp.pending_drivers)
            for ramp in runtime._population_ramps
        ])
        return real_increment(population, hard_cap, mean_fill)

    monkeypatch.setattr(sr, "_next_population_increment", checked_increment)

    async def fake_request(_state):
        await real_sleep(0.0002)

    with pytest.raises(
        TrialInvalid,
        match="measured_(population_not_saturated|warmup_gate_timeout)",
    ):
        asyncio.run(runtime._run_closed_loop_saturated(fake_request))

    assert telemetry_ramps
    assert len(runtime._population_ramps) == 2
    initial, adaptive = runtime._population_ramps
    assert initial.kind == "initial_population"
    assert initial.spread_s == pytest.approx(0.1)
    assert adaptive.kind == "adaptive_increment"
    assert adaptive.spread_s == pytest.approx(2.0)
    assert increment_ramps
    assert all(
        complete and pending == 0
        for snapshot in increment_ramps
        for _kind, complete, pending in snapshot
    )


def test_no_growth_window_waits_only_for_existing_initial_ramp(monkeypatch):
    """Ramp synchronization does not introduce a global-population turnover."""
    real_sleep = asyncio.sleep

    async def compressed_driver_sleep(delay):
        task = asyncio.current_task()
        if task is not None and task.get_name().startswith("measured-driver-"):
            await real_sleep(delay * 0.001)
        else:
            await real_sleep(delay)

    monkeypatch.setattr(sr.asyncio, "sleep", compressed_driver_sleep)
    monkeypatch.setattr(sr, "_POPULATION_ADAPTER_SAMPLE_S", 0.001)
    monkeypatch.setattr(sr, "_SATURATION_STABILITY_MIN_SPAN", 1)
    monkeypatch.setattr(sr, "_SATURATION_STABILITY_BATCH_CYCLES", 1)
    monkeypatch.setattr(sr, "_SATURATION_STABILITY_MAX_SPAN", 1)
    monkeypatch.setattr(sr, "_WARMUP_WALL_CAP_S", 0.4)
    monkeypatch.setattr(sr, "_MEASURED_WALL_CAP_S", 0.2)
    runtime = _saturation_runtime(
        warmup_queries=1024,
        measured_queries=320,
        population=64,
    )
    runtime.services = [
        _fake_llm_service(runtime, lambda _n: [1.0] * 5)
    ]

    async def fake_request(_state):
        await real_sleep(0.001)

    _result, summary = asyncio.run(
        runtime._run_closed_loop_saturated(fake_request)
    )

    assert summary["population_ramp_count"] == 1
    assert summary["population_ramp_added_drivers"] == 0
    assert summary["population_ramp_complete_before_measurement"] is True
    assert summary["population_ramp_pending_drivers_at_measurement_start"] == 0
    assert summary["warmup_completed"] < summary["warmup_queries"]


def test_population_ramp_cancellation_resolves_sleeping_drivers(monkeypatch):
    """Abort cleanup resolves, cancels, and consumes ramp sleepers."""
    monkeypatch.setattr(sr, "_WARMUP_WALL_CAP_S", 0.2)
    runtime = _saturation_runtime(
        warmup_queries=1024,
        measured_queries=4,
        population=64,
    )
    runtime.services = [
        _fake_llm_service(runtime, lambda _n: [0.0] * 5)
    ]

    async def fatal_request(_state):
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        asyncio.run(runtime._run_closed_loop_saturated(fatal_request))

    assert len(runtime._population_ramps) == 1
    ramp = runtime._population_ramps[0]
    assert ramp.complete_event.is_set()
    assert ramp.pending_drivers == 0
    assert ramp.cancelled_before_activation > 0
    assert (
        ramp.activated_drivers + ramp.cancelled_before_activation
        == ramp.scheduled_drivers
    )


def test_qps_subwindow_cv_distinguishes_stationary_from_ramp():
    stationary = [
        window + (offset + 0.5) / 20.0
        for window in range(5)
        for offset in range(20)
    ]
    ramp = [
        window + (offset + 0.5) / count
        for window, count in enumerate((10, 15, 20, 25, 30))
        for offset in range(count)
    ]

    assert sr._qps_subwindow_cv(stationary, 0.0, 5.0) == pytest.approx(0.0)
    assert sr._qps_subwindow_cv(ramp, 0.0, 5.0) > 0.15


def test_wall_subwindow_workload_diagnostics_preserve_token_mix():
    diagnostics = sr._wall_subwindow_workload_diagnostics(
        [0.5, 1.5, 2.5, 4.5, 9.9, 10.0],
        [10, 20, 30, 40, 50, 60],
        0.0,
        10.0,
    )

    assert diagnostics["window_subwindow_completions"] == [2, 1, 1, 0, 2]
    assert diagnostics["window_subwindow_span_s"] == pytest.approx(2.0)
    assert diagnostics["window_subwindow_output_tokens"] == [30, 30, 40, 0, 110]
    assert diagnostics["window_subwindow_mean_output_tokens"] == pytest.approx(
        [15.0, 30.0, 40.0, 0.0, 55.0]
    )
    assert diagnostics["window_subwindow_output_tokens_per_s"] == pytest.approx(
        [15.0, 15.0, 20.0, 0.0, 55.0]
    )


def test_wall_subwindow_workload_diagnostics_require_aligned_cohort():
    with pytest.raises(ValueError, match="completion/token alignment"):
        sr._wall_subwindow_workload_diagnostics(
            [1.0, 2.0],
            [10],
            0.0,
            5.0,
        )


def test_qps_stationarity_uses_completion_spans_for_stable_batch_waves():
    # Reproduce the measured eval_0045 shape: one 207-query phase edge, then
    # seven stable 256-query waves every 49.08 s. Five 72 s wall bins alias the
    # wave period into alternating one/two-batch counts and falsely look noisy.
    timestamps = [1.0] * 207
    for cycle in range(7):
        timestamps.extend([50.0 + cycle * 49.08] * 256)

    assert sr._qps_subwindow_cv(timestamps, 0.0, 360.0) == pytest.approx(
        0.29706771478041505
    )
    evidence = sr._qps_stationarity_evidence(
        timestamps,
        0.0,
        360.0,
        completion_span_queries=256,
    )

    assert evidence["method"] == "completion_span_rates"
    assert evidence["selection_reason"] == "sparse_batch_wave"
    assert evidence["wall_time_dense_completion_stream"] is False
    assert evidence["completion_span_queries"] == 256
    assert evidence["selected_cv"] == pytest.approx(0.0, abs=1e-12)
    assert evidence["completion_span_tail_change_cv"] == pytest.approx(
        0.0, abs=1e-12
    )
    assert len(evidence["completion_span_rates_qps"]) == 6


def test_low_qps_sub_batch_span_does_not_define_final_stationarity():
    # Reproduce the failed eval_0037 shape: five equal wall windows have
    # stable completion counts, but each low-QPS W=41 span cuts through a
    # much larger dynamic-batch wave and therefore sees alternating burst/gap
    # durations. The original stage-cycle W=256 cannot form four boundaries
    # in this finite window, so the equal-wall proof is authoritative.
    wall_counts = [120, 133, 121, 133, 123]
    wall_span_s = 270.0
    timestamps = []
    for window, count in enumerate(wall_counts):
        burst_start = window * wall_span_s + 10.0
        timestamps.extend(
            burst_start + offset * 10.0 / count
            for offset in range(count)
        )

    reduced = sr._qps_stationarity_evidence(
        timestamps,
        0.0,
        5 * wall_span_s,
        completion_span_queries=41,
    )
    preferred = sr._qps_stationarity_evidence(
        timestamps,
        0.0,
        5 * wall_span_s,
        completion_span_queries=256,
    )

    assert reduced["selection_reason"] == "sparse_batch_wave"
    assert reduced["selected_cv"] > 0.15
    assert preferred["selection_reason"] == "completion_span_unavailable"
    assert preferred["method"] == "wall_time_subwindows"
    assert preferred["selected_cv"] == pytest.approx(
        preferred["wall_subwindow_cv"]
    )
    assert preferred["selected_cv"] == pytest.approx(0.046004370622823615)
    assert preferred["selected_cv"] <= 0.15


def test_qps_completion_spans_still_reject_rate_drift():
    timestamps = [completed_at for completed_at in (50, 90, 140, 200, 270, 350)
                  for _ in range(256)]

    evidence = sr._qps_stationarity_evidence(
        timestamps,
        0.0,
        360.0,
        completion_span_queries=256,
    )

    assert evidence["method"] == "completion_span_rates"
    assert evidence["selected_cv"] > 0.15


def test_qps_stationarity_uses_wall_bins_for_dense_microburst_stream():
    # Reproduce eval_0043's dense wall-bin counts while giving each bin a fast
    # microburst followed by a slower continuous tail. Fixed-completion
    # durations jitter; with at least three full spans in every bin, wall-time
    # CV is authoritative.
    wall_counts = [19435, 19667, 19938, 19650, 19956]
    wall_span_s = 72.0
    timestamps = []
    for window, count in enumerate(wall_counts):
        fast_count = int(count * 0.8)
        slow_count = count - fast_count
        start = window * wall_span_s
        timestamps.extend(
            start + (offset + 0.5) * (wall_span_s / 2.0) / fast_count
            for offset in range(fast_count)
        )
        timestamps.extend(
            start + wall_span_s / 2.0
            + (offset + 0.5) * (wall_span_s / 2.0) / slow_count
            for offset in range(slow_count)
        )

    evidence = sr._qps_stationarity_evidence(
        timestamps,
        0.0,
        360.0,
        completion_span_queries=256,
    )

    assert evidence["wall_time_dense_completion_stream"] is True
    assert evidence["wall_subwindow_completions"] == wall_counts
    assert min(evidence["wall_subwindow_completion_spans"]) >= 3.0
    assert evidence["completion_span_cv"] > 0.15
    assert evidence["wall_subwindow_cv"] == pytest.approx(
        0.009925478766809303
    )
    assert evidence["method"] == "wall_time_subwindows"
    assert evidence["selection_reason"] == "dense_completion_stream"
    assert evidence["selected_cv"] == evidence["wall_subwindow_cv"]


def test_dense_wall_proof_rejects_tail_change_and_keeps_aligned_audit():
    # A long stable prefix can dilute nine full spans at half throughput below
    # the global-CV threshold. The aligned tail-vs-history score must retain the
    # sustained end-of-window regime change instead of publishing it as steady.
    completed_at = 1.0
    batch_times = [completed_at]
    for _ in range(100):
        completed_at += 1.0
        batch_times.append(completed_at)
    for _ in range(9):
        completed_at += 2.0
        batch_times.append(completed_at)
    timestamps = [timestamp for timestamp in batch_times for _ in range(256)]

    evidence = sr._qps_stationarity_evidence(
        timestamps,
        0.0,
        completed_at,
        completion_span_queries=256,
    )

    assert evidence["method"] == "wall_time_subwindows"
    assert evidence["selection_reason"] == "dense_completion_stream"
    assert evidence["completion_span_cv"] < 0.15
    assert evidence["completion_span_tail_change_cv"] == pytest.approx(1.0 / 3.0)
    assert evidence["completion_span_tail_change_suffix_spans"] == 9
    assert evidence["selected_cv"] > 0.15


def test_qps_completion_spans_fall_back_when_evidence_is_too_short():
    timestamps = [completed_at for completed_at in (50, 100, 150)
                  for _ in range(256)]

    evidence = sr._qps_stationarity_evidence(
        timestamps,
        0.0,
        180.0,
        completion_span_queries=256,
    )

    assert evidence["method"] == "wall_time_subwindows"
    assert evidence["selected_cv"] == evidence["wall_subwindow_cv"]
    assert evidence["completion_span_cv"] is None
    assert evidence["completion_span_tail_change_cv"] is None


def test_qps_completion_spans_do_not_hide_a_terminal_stall():
    timestamps = [completed_at for completed_at in (50, 100, 150, 200)
                  for _ in range(256)]

    evidence = sr._qps_stationarity_evidence(
        timestamps,
        0.0,
        300.0,
        completion_span_queries=256,
    )

    assert evidence["completion_span_tail_s"] == pytest.approx(100.0)
    assert evidence["method"] == "wall_time_subwindows"
    assert evidence["selection_reason"] == "terminal_stall_fallback"
    assert evidence["completion_span_cv"] == pytest.approx(0.0, abs=1e-12)
    assert evidence["completion_span_tail_change_cv"] is None


def test_warmup_wall_cap_fails_closed_before_measurement(monkeypatch):
    from rag_stack_evaluator.static_rag_evaluator.measured.vllm_deployment import TrialInvalid

    monkeypatch.setattr(sr, "_WARMUP_WALL_CAP_S", 0.02)
    runtime = _saturation_runtime(
        warmup_queries=1024,
        measured_queries=4,
        population=2,
    )
    runtime.services = [_fake_llm_service(runtime, lambda _n: [0.0] * 5)]
    admissions = []

    async def slow_request(state):
        admissions.append(state.is_measured)
        await asyncio.sleep(0.01)

    started = time.perf_counter()
    with pytest.raises(TrialInvalid, match="measured_warmup_gate_timeout"):
        asyncio.run(runtime._run_closed_loop_saturated(slow_request))
    elapsed = time.perf_counter() - started

    assert elapsed < 0.2
    assert admissions and not any(admissions)
