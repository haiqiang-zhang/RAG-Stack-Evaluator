"""effective_util weight floor: the global memory scheduler stays authoritative
by never squeezing a vLLM engine below the util its weights need.

Repro: a collocated 14B query-expander shared a card with the generator; the
blanket 24 GiB co-tenant reserve (meant for HF stages) double-counted the
generator and lowered the query-expander to 0.237 util — below its 28 GiB of
weights — so vLLM refused to launch ('not enough memory to serve even a single
token'). The floor pins it back up; working (light) engines are untouched.
"""
from __future__ import annotations

import pytest

import rag_stack_evaluator.static_rag_evaluator.measured.gpu_mem as gm
from rag_stack_evaluator.static_rag_evaluator.measured.gpu_mem import (
    _model_weight_gib,
    _weight_floor_util,
    effective_util,
)

TOTAL = 94.0


def test_weight_gib_parses_param_count():
    assert abs(_model_weight_gib("Qwen/Qwen2.5-14B-Instruct") - 26.1) < 1.0
    assert abs(_model_weight_gib("Qwen2.5-1.5B") - 2.8) < 0.5
    assert _model_weight_gib("some-model-no-count") == 0.0
    assert _model_weight_gib(None) == 0.0


def test_weight_floor_util():
    # 14B: ~26 GiB weights + 4 GiB min KV over 94 GiB ≈ 0.32
    f = _weight_floor_util("Qwen2.5-14B-Instruct", TOTAL)
    assert 0.30 < f < 0.36
    # 1.5B: tiny floor
    assert _weight_floor_util("Qwen2.5-1.5B", TOTAL) < 0.10


@pytest.fixture
def no_settle(monkeypatch):
    monkeypatch.setattr(gm, "_SETTLE_S", 0.0)  # skip the eviction-lag wait loop


def _patch_free(monkeypatch, free, total=TOTAL):
    monkeypatch.setattr(gm, "_free_total_gib", lambda cuda_ids: (free, total))


def test_qe_14b_not_squeezed_below_weights(no_settle, monkeypatch):
    # The repro: 46 GiB free after the generator took its share; the 24 GiB
    # reserve would size QE to (46-24)/94 = 0.237, below its ~0.32 weight floor.
    _patch_free(monkeypatch, free=46.0)
    util = effective_util(["0"], requested=0.45, model="Qwen/Qwen2.5-14B-Instruct")
    floor = _weight_floor_util("Qwen/Qwen2.5-14B-Instruct", TOTAL)
    assert util >= floor - 1e-9          # never below the weight floor
    assert util * TOTAL >= 26.0          # fits the 14B weights
    assert util <= 46.0 / TOTAL + 1e-9   # never demands more than is free


def test_light_engine_working_case_unchanged(no_settle, monkeypatch):
    # A 1.5B engine on a card with 66 GiB free (a co-tenant present): the 24 GiB
    # reserve still sizes it to ~(66-24)/94 = 0.447 — the floor (tiny) does NOT
    # raise it, so light-engine throughput is IDENTICAL to before the floor.
    _patch_free(monkeypatch, free=66.0)
    util_with = effective_util(["0"], requested=0.90, model="Qwen2.5-1.5B")
    util_without = effective_util(["0"], requested=0.90, model=None)
    assert abs(util_with - util_without) < 1e-9   # floor is a no-op here
    assert abs(util_with - round((66.0 - gm._RESERVE_GIB) / TOTAL, 3)) < 1e-9


def test_ample_free_returns_requested(no_settle, monkeypatch):
    _patch_free(monkeypatch, free=90.0)
    # plenty free → keep requested; floor never lowers it
    assert effective_util(["0"], requested=0.45, model="Qwen2.5-14B") == 0.45


def test_no_model_preserves_old_behavior(no_settle, monkeypatch):
    _patch_free(monkeypatch, free=46.0)
    # without a model there is no floor → exact pre-change value
    util = effective_util(["0"], requested=0.45, model=None)
    assert abs(util - round((46.0 - gm._RESERVE_GIB) / TOTAL, 3)) < 1e-9
