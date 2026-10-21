"""Measured phase, warmup and completion-trace contracts with fake requests."""

import asyncio
import time
from types import SimpleNamespace

import pandas as pd
import pytest

from rag_stack_evaluator.static_rag_evaluator import recording


def test_measured_runtime_uses_closed_loop_measured_phase():
    from rag_stack_evaluator.static_rag_evaluator.measured.serving_runtime import (
        MeasuredServingRuntime,
    )

    class Owner:
        def _generator_chip_count(self, system_config):
            return 1

        def _add_deployment_metadata(self, summary, node_lines, system_config):
            return None

    system_config = {
        "batch_size_request": 3,
        "measured_load_concurrency": 5,
        "measured_warmup_queries": 3,
        "measured_queries": 6,
        "batching": {"dynamic_timeout_s": 0.0},
        "retrieval": {"faiss_num_threads": 3, "faiss_ivf_parallel_mode": 0, "num_servers": 1},
        "vllm": {
            "kv_cache_dtype": "auto",
            "engines": {"generator": {"max_num_seqs": 3}},
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
    }
    runtime = MeasuredServingRuntime(
        owner=Owner(),
        stages=[],
        qa_data=pd.DataFrame({"query": ["a", "b"]}),
        node_lines={},
        system_config=system_config,
        config={"pipeline_runtime": {"rag_dataflow": "sequential"}},
    )
    active = 0
    max_active = 0
    starts = []

    async def fake_request(state):
        nonlocal active, max_active
        starts.append((state.seq, state.is_measured, state.record_trace, state.qid))
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.001)
        active -= 1

    result, summary = asyncio.run(runtime._run_closed_loop_saturated(fake_request))

    assert len(result) == 2
    assert max_active == 5
    assert [s[1] for s in starts[:5]] == [False] * 5
    first_measured_admit = next(i for i, s in enumerate(starts) if s[1])
    assert first_measured_admit >= 5
    assert all(not s[1] for s in starts[:first_measured_admit])
    assert all(s[1] for s in starts[first_measured_admit:])
    # Initial duplicates are provisional quality candidates. Performance
    # tracing continues through the completion window; completed non-selected
    # requests are streamed out of the bounded recorder.
    assert all(s[2] for s in starts[:5])
    assert all(s[2] for s in starts[5:])
    assert (
        len({s[3] for s in starts[first_measured_admit:]})
        == len(starts[first_measured_admit:])
    )
    assert summary["scheduling_policy"] == "closed-loop-saturated-stage-services"
    assert summary["batch_size_request"] == 3
    assert summary["stage_batch_cap"] == 3
    assert summary["load_concurrency"] == 5
    assert summary["service_concurrency"] == 5
    assert summary["warmup_queries"] == 3
    assert summary["measured_queries"] == 6
    assert summary["measurement_waves"] == pytest.approx(1.2)
    assert summary["measurement_request_batch_waves"] == 2.0
    # Scope bookkeeping moved out of performance summaries: the canonical quality-
    # trace envelope is self-describing. Capture diagnostics stay.
    assert "trace_scope" not in summary
    assert summary["trace_capture_scope"] == (
        "all_admissions_until_global_qa_coverage"
    )
    assert summary["trace_projection_basis"] == (
        "first_completed_invocation_per_dataset_idx"
    )
    assert summary["quality_trace_warmup_winner_queries"] == 2
    assert summary["measurement_rate_stability_waves"] == 2.0
    assert summary["measured_gt_admissibility"]["checks"][
        "measurement_rate_stability_waves_ge_2"
    ]["passed"] is True
    assert "measurement_waves_ge_2" not in summary[
        "measured_gt_admissibility"
    ]["checks"]
    # The two-row workload support completes at warmup completion 2. Only the
    # following TWO spans of warmup_queries completions may prove stability;
    # the pre-support completions cannot be recycled into that proof.
    assert summary["workload_support_complete_warmup_completed"] == 2
    assert summary["warmup_fresh_rate_proof_start_completion"] == 2
    assert summary["warmup_fresh_rate_required_completions"] == 6
    assert summary["warmup_fresh_rate_observed_completions"] == 6
    assert summary["warmup_completed"] == 8
    assert summary["workload_support_complete_before_measurement"] is True


def test_warmup_rate_proof_starts_fresh_after_workload_support(monkeypatch):
    """A stable cold prefix cannot be recycled into the measured-window gate."""
    from rag_stack_evaluator.static_rag_evaluator.measured import serving_runtime as sr

    monkeypatch.setattr(sr, "_WARMUP_WALL_CAP_S", 0.5)
    monkeypatch.setattr(sr, "_MEASURED_WALL_CAP_S", 0.2)
    monkeypatch.setattr(sr, "_PHASE_CANCEL_GRACE_S", 0.05)

    # Keep row 1 out of the first ten admissions. That prefix has enough
    # completions to satisfy the old W=2 local-rate gate, but it does not cover
    # the two-row workload and therefore must not open the measured window.
    monkeypatch.setattr(
        sr._BalancedDatasetAdmissionOrder,
        "idx_for_seq",
        lambda self, seq: 0 if seq < 10 else seq % 2,
    )
    runtime = _stall_runtime(sr, measured_queries=4, warmup_queries=2)
    admissions = []

    async def fake_request(state):
        admissions.append((state.seq, state.idx, state.is_measured))
        await asyncio.sleep(0.0005)

    _result, summary = asyncio.run(
        runtime._run_closed_loop_saturated(fake_request)
    )

    support_at = summary["workload_support_complete_warmup_completed"]
    proof_at = summary["warmup_fresh_rate_proof_start_completion"]
    assert support_at >= 10
    assert proof_at == support_at
    assert summary["warmup_fresh_rate_required_completions"] == 4
    assert summary["warmup_fresh_rate_observed_completions"] >= 4
    assert summary["warmup_completed"] >= support_at + 4
    first_measured = next(row for row in admissions if row[2])
    assert first_measured[0] >= support_at + 4


def test_warmup_cap_fails_closed_before_workload_support(monkeypatch):
    """The wall budget never turns a one-row prefix into a valid GT window."""
    from rag_stack_evaluator.static_rag_evaluator.measured import serving_runtime as sr
    from rag_stack_evaluator.static_rag_evaluator.measured.vllm_deployment import TrialInvalid

    monkeypatch.setattr(sr, "_WARMUP_WALL_CAP_S", 0.02)
    monkeypatch.setattr(sr, "_PHASE_CANCEL_GRACE_S", 0.01)
    monkeypatch.setattr(
        sr._BalancedDatasetAdmissionOrder,
        "idx_for_seq",
        lambda self, seq: 0,
    )
    runtime = _stall_runtime(sr, measured_queries=2, warmup_queries=2)

    async def fake_request(_state):
        await asyncio.sleep(0.0005)

    with pytest.raises(
        TrialInvalid,
        match=r"measured_warmup_gate_timeout: .*workload_support_complete=False",
    ):
        asyncio.run(runtime._run_closed_loop_saturated(fake_request))


def _stall_runtime(sr, measured_queries=2, warmup_queries=3):
    class Owner:
        def _generator_chip_count(self, system_config):
            return 1

        def _add_deployment_metadata(self, summary, node_lines, system_config):
            return None

    system_config = {
        "batch_size_request": 3,
        "measured_load_concurrency": 3,
        "measured_warmup_queries": warmup_queries,
        "measured_queries": measured_queries,
        "batching": {"dynamic_timeout_s": 0.0},
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
    }
    return sr.MeasuredServingRuntime(
        owner=Owner(),
        stages=[],
        qa_data=pd.DataFrame({"query": ["a", "b"]}),
        node_lines={},
        system_config=system_config,
        config={"pipeline_runtime": {"rag_dataflow": "sequential"}},
    )


def test_measured_runtime_projects_trace_to_dataset_quality_winners(monkeypatch):
    """Trace is one first-completion winner per QA row, in dataset order."""
    from rag_stack_evaluator.static_rag_evaluator.measured import serving_runtime as sr

    monkeypatch.setattr(sr, "_WARMUP_WALL_CAP_S", 0.2)
    monkeypatch.setattr(sr, "_MEASURED_WALL_CAP_S", 0.2)
    monkeypatch.setattr(sr, "_PHASE_CANCEL_GRACE_S", 0.05)
    monkeypatch.setattr(sr, "_PERFORMANCE_TRACE_SAMPLE_CAPACITY", 3)

    recorder = recording.TraceRecorder()
    runtime = _stall_runtime(sr, measured_queries=5, warmup_queries=0)
    runtime.trace_recorder = recorder
    admissions = []
    first_epoch = sr._dataset_admission_epoch_order(2, 0)
    second_epoch = sr._dataset_admission_epoch_order(2, 1)

    async def fake_request(state):
        admissions.append((state.seq, state.idx, state.record_trace))
        if state.record_trace:
            recorder.record(
                state.qid, "semantic_retrieval_encode", input_tokens=1,
            )
        if state.seq < 2:
            # The complete first admission epoch is slow. Both rows are traced
            # provisionally but lose to the same rows in the next deterministic
            # epoch and must not supply the quality row/trace.
            delay = 0.05
        elif state.seq < 4:
            delay = 0.005
        else:
            delay = 0.001
        await asyncio.sleep(delay)
        if state.record_trace:
            recorder.record(
                state.qid, "generator", input_tokens=1, output_tokens=1,
            )
        state.df = state.df.copy()
        state.df["generated_texts"] = [state.qid]

    result, summary = asyncio.run(
        runtime._run_closed_loop_saturated(fake_request)
    )

    winner_qids = [
        f"measured-{2 + second_epoch.index(idx)}"
        for idx in range(2)
    ]
    assert result["generated_texts"].tolist() == winner_qids
    assert recorder.qids == tuple(winner_qids)
    assert summary["quality_trace_exact"] is True
    assert summary["quality_trace_dataset_order"] is True
    assert summary["quality_trace_matches_final_result"] is True
    assert summary["workload_admission_order"] == (
        "balanced_epoch_permutation_v1"
    )
    assert summary["quality_trace_expected_queries"] == 2
    assert summary["quality_trace_recorded_queries"] == 2
    assert summary["quality_trace_provisional_queries"] > 2
    assert summary["quality_trace_discarded_duplicate_queries"] > 0
    assert summary["quality_trace_fill_queries"] == 0
    assert summary["measured_queries"] == 5
    assert summary["quality_trace_recorded_queries"] != summary["measured_queries"]
    # One row is already covered when the second row of epoch 1 is admitted,
    # but global coverage is still pending. It remains a traced batch peer.
    assert (3, second_epoch[1], True) in admissions
    assert sorted(first_epoch) == [0, 1]
    assert sorted(second_epoch) == [0, 1]
    assert all(record_trace for _seq, _idx, record_trace in admissions)
    assert [
        [entry["stage"] for entry in query_trace]
        for query_trace in recorder.to_trace_v1()
    ] == [
        ["semantic_retrieval_encode", "generator"] for _ in recorder.qids
    ]

    performance_trace = summary["__performance_execution_trace__"]
    assert performance_trace["trace_kind"] == (
        "measurement_phase_completion_sample"
    )
    assert performance_trace["sampling"]["population_queries"] == 5
    assert performance_trace["sampling"]["sample_queries"] == 3
    assert len(performance_trace["queries"]) == 3

    # Full phase window is authoritative. Completion delta remains diagnostic.
    assert summary["measurement_rate_mode"] == "phase_window"
    assert summary["qps"] == pytest.approx(
        summary["measured_queries"] / summary["measurement_phase_window_s"]
    )
    assert summary["qps"] == pytest.approx(summary["measurement_phase_window_qps"])
    assert summary["measurement_completion_delta_queries"] == max(
        0, summary["measured_queries"] - 1
    )


def test_measured_winner_trace_ships_canonical_envelope():
    """P1 producer contract (cardinality invariant): qa_rows=N closed-loop
    winners wrap into the canonical quality-trace envelope with exactly N queries in
    dataset order — question_id = the dataset row idx, invocation_id = the
    winning serving qid (a warmup admission may win) — and a coverage
    shortfall or an incomplete row raises instead of shipping."""
    from types import SimpleNamespace

    from rag_stack.rag_ir import validate_quality_trace_envelope
    from rag_stack_evaluator.static_rag_evaluator.measured import serving_runtime as sr
    from rag_stack_evaluator.static_rag_evaluator.measured.evaluator import (
        MeasuredEvaluator,
    )

    recorder = recording.TraceRecorder()
    runtime = _stall_runtime(sr, measured_queries=6, warmup_queries=3)
    runtime.trace_recorder = recorder

    async def fake_request(state):
        if state.record_trace:
            recorder.record(
                state.qid, "semantic_retrieval_encode", input_tokens=5,
            )
            recorder.record(
                state.qid, "generator", input_tokens=42, output_tokens=7,
            )
        await asyncio.sleep(0.001)

    result, summary = asyncio.run(
        runtime._run_closed_loop_saturated(fake_request)
    )
    assert "trace_scope" not in summary
    trace = recorder.to_trace_v1()
    assert len(trace) == 2

    def _measured_out(run):
        base = SimpleNamespace(_run_pipeline=lambda *a, **k: run)
        return MeasuredEvaluator(base).evaluate(
            {}, cache=object(), system_config={},
        )

    run = SimpleNamespace(
        quality={"exact_match": 1.0},
        trace=trace,
        n_evaluated=2,
        previous_result=result,
        sequential_perf=summary,
    )
    envelope = _measured_out(run)["quality"]["__execution_dag__"]
    validate_quality_trace_envelope(envelope)
    assert [q["question_id"] for q in envelope["queries"]] == ["0", "1"]
    # The invocation ids ARE the per-row winners the runtime returned for
    # quality — and with a fast 3-query warmup they are warmup admissions.
    assert [q["invocation_id"] for q in envelope["queries"]] == [
        str(qid) for qid in result["__qid__"].tolist()
    ]
    assert any(
        str(q["invocation_id"]).startswith("warmup-")
        for q in envelope["queries"]
    )

    # Coverage shortfall (fewer trace rows than dataset rows) raises.
    short = SimpleNamespace(
        quality={}, trace=trace[:1], n_evaluated=2,
        previous_result=result, sequential_perf=summary,
    )
    with pytest.raises(RuntimeError, match="partial trace"):
        _measured_out(short)

    # An incomplete winner row (no terminal generator call) is a bug and
    # must surface loudly, not ship.
    incomplete = SimpleNamespace(
        quality={},
        trace=[query_trace[:1] for query_trace in trace],
        n_evaluated=2,
        previous_result=result,
        sequential_perf=summary,
    )
    with pytest.raises(ValueError, match="incomplete"):
        _measured_out(incomplete)


def test_measured_gt_admissibility_contract_is_fail_closed():
    from rag_stack_evaluator.static_rag_evaluator.measured.artifact_contract import (
        validate_measured_gt_admissibility,
    )
    from rag_stack_evaluator.static_rag_evaluator.measured.serving_runtime import (
        build_measured_gt_admissibility,
    )

    summary = {
        "measurement_start_gate": "warmup_completion_rate_stable",
        "measurement_rate_stability_waves": 2.0,
        "population_saturated": True,
        "driver_errors_window": 0,
        "qps_subwindow_cv": 0.15,
    }
    verdict = build_measured_gt_admissibility(summary)

    assert set(verdict) == {"admissible", "reasons", "checks"}
    assert verdict["admissible"] is True
    assert verdict["reasons"] == []
    assert set(verdict["checks"]) == {
        "warmup_rate_stable_not_cap_forced",
        "measurement_rate_stability_waves_ge_2",
        "population_saturated",
        "driver_errors_window_zero",
        "qps_subwindow_cv_le_0_15",
    }
    assert all(
        set(check) == {"observed", "required", "passed"}
        for check in verdict["checks"].values()
    )
    with pytest.raises(ValueError, match="invalid fields"):
        validate_measured_gt_admissibility({
            **verdict,
            "obsolete_header": 2,
        })
    bad = dict(summary)
    bad.update({
        "measurement_start_gate": "warmup_cap_forced",
        "measurement_rate_stability_waves": 1.99,
        "population_saturated": None,
        "driver_errors_window": 1,
        "qps_subwindow_cv": None,
    })
    rejected = build_measured_gt_admissibility(bad)
    assert rejected["admissible"] is False
    assert set(rejected["reasons"]) == {
        "warmup_rate_stability_gate_not_met",
        "measurement_rate_stability_waves_below_minimum",
        "population_not_saturated",
        "driver_errors_in_measurement_window",
        "qps_subwindow_cv_missing_or_above_maximum",
    }

    malformed = build_measured_gt_admissibility({
        "measurement_start_gate": "warmup_completion_rate_stable",
        "measurement_rate_stability_waves": [],
        "population_saturated": True,
        "driver_errors_window": 0.5,
        "qps_subwindow_cv": float("nan"),
    })
    assert malformed["admissible"] is False
    assert malformed["checks"][
        "measurement_rate_stability_waves_ge_2"
    ]["observed"] is None
    assert malformed["checks"]["driver_errors_window_zero"]["observed"] is None
    assert malformed["checks"]["qps_subwindow_cv_le_0_15"]["observed"] is None


def test_automatic_measured_target_ignores_adapter_population_growth():
    from rag_stack_evaluator.static_rag_evaluator.measured import serving_runtime as sr

    class Owner:
        def _generator_chip_count(self, system_config):
            return 1

        def _add_deployment_metadata(self, summary, node_lines, system_config):
            return None

    runtime = sr.MeasuredServingRuntime(
        owner=Owner(),
        stages=[],
        qa_data=pd.DataFrame({"query": ["a", "b"]}),
        node_lines={},
        system_config={
            "batch_size_request": 3,
            "measured_load_concurrency": 3,
            "measured_warmup_queries": 1,
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
    assert runtime.measured_target == 15
    # Simulate the warmup adapter growing the population before the window.
    runtime.load_concurrency = 10

    async def fake_request(state):
        await asyncio.sleep(0.0001)

    _result, summary = asyncio.run(
        runtime._run_closed_loop_saturated(fake_request)
    )

    assert summary["measured_target_explicit"] is False
    # Population growth must not inflate the nominal target. The existing
    # stationarity guard may independently extend an automatic window when its
    # five-subwindow CV still shows a ramp.
    assert summary["measurement_initial_target_queries"] == 15
    assert summary["measured_target_queries"] >= 15
    assert summary["measured_queries"] == summary["measured_target_queries"]
    assert summary["measurement_waves"] == pytest.approx(
        summary["measured_queries"] / 10.0
    )
    assert summary["measurement_rate_stability_waves"] == pytest.approx(
        summary["measured_queries"]
    )


class _VirtualTimeLoop(asyncio.SelectorEventLoop):
    """Advance to the next timer instead of sleeping in fake-request tests."""

    def __init__(self):
        self._clock = 1000.0
        super().__init__()

    def time(self):
        return self._clock

    def _run_once(self):
        if not self._ready and self._scheduled:
            self._clock = max(self._clock, self._scheduled[0]._when)
        super()._run_once()


@pytest.fixture
def run_with_virtual_time(monkeypatch):
    from rag_stack_evaluator.static_rag_evaluator.measured import serving_runtime as sr

    with asyncio.Runner(loop_factory=_VirtualTimeLoop) as runner:
        loop = runner.get_loop()
        # Patch only the evaluator module; asyncio owns the corresponding
        # timer clock. Other modules retain their real perf_counter.
        monkeypatch.setattr(sr, "time", SimpleNamespace(
            perf_counter=loop.time, monotonic=loop.time,
            sleep=time.sleep, strftime=time.strftime,
        ))
        yield runner.run


def test_measured_runtime_keeps_warmup_inflight_across_ramp(monkeypatch, run_with_virtual_time):
    """Cold requests survive until a fresh stable-rate proof opens measurement."""
    from rag_stack_evaluator.static_rag_evaluator.measured import serving_runtime as sr

    # The current contract never opens a window after its warmup budget has
    # expired without proof. Give the 30 ms initial stall enough budget to
    # recover and produce two stable 32-completion spans.
    monkeypatch.setattr(sr, "_WARMUP_WALL_CAP_S", 0.2)
    monkeypatch.setattr(sr, "_MEASURED_WALL_CAP_S", 0.2)
    monkeypatch.setattr(sr, "_PHASE_DRAIN_GRACE_S", 0.05)

    # warmup target 64 caps the stability span at 32, so 64 completions prove
    # two stable spans before the measured window opens.
    runtime = _stall_runtime(sr, warmup_queries=64)
    starts = []
    completed = []
    first_completion_at = sr.time.perf_counter() + 0.03

    async def fake_request(state):
        starts.append((state.seq, state.is_measured, state.record_trace))
        # Distinct virtual completion times model a fixed serving rate and
        # avoid simultaneous completions at the measurement-window boundary.
        done_at = first_completion_at + state.seq * 0.0001
        await asyncio.sleep(done_at - asyncio.get_running_loop().time())
        completed.append(state.seq)

    result, summary = run_with_virtual_time(runtime._run_closed_loop_saturated(fake_request))

    assert {0, 1, 2}.issubset(completed)
    assert len(result) == 2
    assert [s[1] for s in starts[:3]] == [False, False, False]
    assert summary["measured_queries"] == 2
    assert summary["qps"] > 0.0
    assert summary["measurement_rate_mode"] == "phase_window"
    assert summary["qps"] == pytest.approx(summary["measurement_phase_window_qps"])
    assert summary["measurement_completion_delta_qps"] > 0.0
    assert summary["warmup_stall_recovery"] is False
    assert summary["measurement_start_gate"] == "warmup_completion_rate_stable"
    assert "two stable spans of 32 completions" in summary["measurement_start_reason"]
    assert summary["warmup_completed"] >= 64


def test_measured_runtime_stall_cap_fails_closed_without_stability(monkeypatch):
    """A stall that outlives the warmup cap produces no measured window."""
    from rag_stack_evaluator.static_rag_evaluator.measured import serving_runtime as sr
    from rag_stack_evaluator.static_rag_evaluator.measured.vllm_deployment import TrialInvalid

    # cap 5 ms, stall 30 ms: there cannot be completion-rate evidence before
    # the budget expires, so the production fail-closed contract rejects it.
    monkeypatch.setattr(sr, "_WARMUP_WALL_CAP_S", 0.005)
    monkeypatch.setattr(sr, "_MEASURED_WALL_CAP_S", 0.2)
    monkeypatch.setattr(sr, "_PHASE_DRAIN_GRACE_S", 0.05)

    runtime = _stall_runtime(sr)

    async def fake_request(state):
        await asyncio.sleep(0.03 if state.seq < 3 else 0.001)

    with pytest.raises(TrialInvalid, match="measured_warmup_gate_timeout"):
        asyncio.run(runtime._run_closed_loop_saturated(fake_request))


def test_measured_runtime_finishes_all_qa_after_cap(monkeypatch):
    from rag_stack_evaluator.static_rag_evaluator.measured import serving_runtime as sr

    class Owner:
        def _generator_chip_count(self, system_config):
            return 1

        def _add_deployment_metadata(self, summary, node_lines, system_config):
            return None

    monkeypatch.setattr(sr, "_MEASURED_WALL_CAP_S", 0.01)
    monkeypatch.setattr(sr, "_PHASE_CANCEL_GRACE_S", 0.01)

    runtime = sr.MeasuredServingRuntime(
        owner=Owner(),
        stages=[],
        qa_data=pd.DataFrame({"query": ["a", "b"]}),
        node_lines={},
        system_config={
            "batch_size_request": 2,
            "measured_load_concurrency": 4,
            "measured_warmup_queries": 0,
            "measured_queries": 10,
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

    seen = []

    async def slow_request(state):
        seen.append((state.seq, state.idx))
        await asyncio.sleep(0.05 if state.seq < 2 else 1.0)
        state.df = state.df.copy()
        state.df["generated_texts"] = [f"answer-{state.idx}"]

    started = time.perf_counter()
    result, summary = asyncio.run(runtime._run_closed_loop_saturated(slow_request))
    elapsed = time.perf_counter() - started

    assert len(result) == 2
    assert sorted(result["generated_texts"].tolist()) == ["answer-0", "answer-1"]
    assert {idx for _seq, idx in seen} == {0, 1}
    assert elapsed < 0.5
    assert summary["measured_queries"] == 0
    assert summary["quality_rows_completed"] == 2
    assert summary["measurement_stop_reason"] == "measured cap 0s"


def test_measured_runtime_fills_missing_qa_after_drain_without_waiting_for_stragglers(monkeypatch):
    from rag_stack_evaluator.static_rag_evaluator.measured import serving_runtime as sr

    class Owner:
        def _generator_chip_count(self, system_config):
            return 1

        def _add_deployment_metadata(self, summary, node_lines, system_config):
            return None

    monkeypatch.setattr(sr, "_MEASURED_WALL_CAP_S", 0.005)
    monkeypatch.setattr(sr, "_QUALITY_DRAIN_GRACE_S", 0.01)
    monkeypatch.setattr(sr, "_PHASE_CANCEL_GRACE_S", 0.01)

    runtime = sr.MeasuredServingRuntime(
        owner=Owner(),
        stages=[],
        qa_data=pd.DataFrame({"query": ["a", "b"]}),
        node_lines={},
        system_config={
            "batch_size_request": 2,
            "measured_load_concurrency": 4,
            "measured_warmup_queries": 0,
            "measured_queries": 10,
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
    recorder = recording.TraceRecorder()
    runtime.trace_recorder = recorder
    first_live_idx = sr._dataset_admission_epoch_order(2, 0)[0]
    expected_fill_idx = 1 - first_live_idx

    seen = []

    async def slow_live_fast_fill(state):
        seen.append((state.qid, state.seq, state.idx, state.record_trace))
        if state.record_trace:
            recorder.record(
                state.qid, "semantic_retrieval_encode", input_tokens=1,
            )
        if str(state.qid).startswith("final-") or state.seq == 0:
            await asyncio.sleep(0.001)
        else:
            await asyncio.sleep(1.0)
        if state.record_trace:
            recorder.record(
                state.qid, "generator", input_tokens=1, output_tokens=1,
            )
        state.df = state.df.copy()
        state.df["generated_texts"] = [f"answer-{state.idx}"]

    started = time.perf_counter()
    result, summary = asyncio.run(runtime._run_closed_loop_saturated(slow_live_fast_fill))
    elapsed = time.perf_counter() - started

    assert len(result) == 2
    assert sorted(result["generated_texts"].tolist()) == ["answer-0", "answer-1"]
    assert any(
        str(qid).startswith("final-")
        and idx == expected_fill_idx
        and record_trace
        for qid, _seq, idx, record_trace in seen
    )
    assert elapsed < 0.5
    assert summary["quality_rows_completed"] == 2
    assert summary["quality_trace_exact"] is True
    assert summary["quality_trace_fill_queries"] == 1
    assert len(recorder.qids) == 2
    assert str(recorder.qids[expected_fill_idx]).startswith("final-")
