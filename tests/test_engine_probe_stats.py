"""Engine probe stats: windowing, prefix-cache rate from counter deltas."""

from rag_stack.static_rag_evaluator.measured.serving_runtime import (
    LLMContinuousStageService,
)


def _svc(samples, window_idx=0):
    svc = LLMContinuousStageService.__new__(LLMContinuousStageService)
    svc._probe_samples = samples
    svc._probe_window_idx = window_idx
    return svc


def _sample(run, wait, pcq=None, pch=None, pcr=None, clk=None):
    return {"run": run, "wait": wait, "pcq": pcq, "pch": pch,
            "pcr": pcr, "clk": clk}


def test_stats_windowed_and_prefix_rate_from_counters():
    samples = [
        _sample(10, 0, pcq=100, pch=50, clk=1500),   # warmup — excluded
        _sample(250, 5, pcq=1000, pch=600, clk=1400),
        _sample(256, 9, pcq=3000, pch=2200, clk=1380),
    ]
    out = _svc(samples, window_idx=1).engine_probe_stats()
    assert out["engine_probe_samples"] == 2
    assert out["engine_running_min"] == 250
    assert out["engine_running_max"] == 256
    assert out["engine_waiting_mean"] == 7
    # rate over the WINDOW delta: (2200-600)/(3000-1000) = 0.8
    assert abs(out["engine_prefix_cache_hit_rate"] - 0.8) < 1e-9
    assert out["engine_gpu_sm_clock_min_mhz"] == 1380


def test_stats_gauge_fallback_and_empty():
    samples = [_sample(10, 0, pcr=0.6), _sample(12, 0, pcr=0.8)]
    out = _svc(samples).engine_probe_stats()
    assert abs(out["engine_prefix_cache_hit_rate"] - 0.7) < 1e-9
    assert "engine_gpu_sm_clock_mean_mhz" not in out
    assert _svc([]).engine_probe_stats() == {}


def test_recent_engine_waiting():
    samples = [_sample(1, w) for w in (0, 0, 3, 4, 5, 6)]
    assert _svc(samples).recent_engine_waiting(5) == [0, 3, 4, 5, 6]
