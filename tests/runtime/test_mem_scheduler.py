"""Global GPU-memory scheduler: deterministic, layout-driven vLLM util sizing.

Pure (no GPU, no vLLM). Verifies that a vLLM engine sharing a card with HF aux
stages (encoder / reranker / compressor) reserves VRAM for them from the layout
``gpu_occupants`` — the fix for the "1.5B engine OOMs on a 93 GiB H100 because a
co-resident reranker loaded after it sized" bug.
"""
from __future__ import annotations

from rag_stack_evaluator.static_rag_evaluator.measured.vllm_deployment import (
    VllmDeploymentManager,
    _aux_reserve_overrides,
    _COMPRESSOR_RESERVE_GIB,
    _HF_AUX_RESERVE_GIB,
    _RERANKER_RESERVE_GIB,
    _SCHED_SAFETY_GIB,
)

import math

TOTAL = 93.0  # H100


def _floor3(x: float) -> float:
    """The scheduler floors (never rounds up past the budget)."""
    return math.floor(x * 1000) / 1000


def _mgr():
    return VllmDeploymentManager(available_gpus=["cuda:0", "cuda:1", "cuda:2", "cuda:3"])


def _sys(occupants):
    return {"layout": {"gpu_occupants": occupants}}


# ---------------------------------------------------------------------------
# _hf_aux_reserve_gib — deterministic per-card reserve from the layout
# ---------------------------------------------------------------------------

def test_reserve_sums_hf_aux_on_card():
    occ = {"cuda:0": ["query_expansion", "semantic_retrieval",
                      "passage_reranker", "passage_compressor"]}
    r = VllmDeploymentManager._hf_aux_reserve_gib(_sys(occ), ["cuda:0"])
    # QE is a vLLM engine → NOT reserved; the 3 HF aux are.
    assert r == (_HF_AUX_RESERVE_GIB["semantic_retrieval"]
                 + _HF_AUX_RESERVE_GIB["passage_reranker"]
                 + _HF_AUX_RESERVE_GIB["passage_compressor"])


def test_reserve_zero_without_hf_aux():
    occ = {"cuda:1": ["generator"], "cuda:2": ["generator"]}
    assert VllmDeploymentManager._hf_aux_reserve_gib(
        _sys(occ), ["cuda:1", "cuda:2"]) == 0.0


def test_reserve_uses_tightest_card():
    # engine spans cuda:1 (encoder only) and cuda:2 (reranker) → worst card wins
    occ = {"cuda:1": ["generator", "semantic_retrieval"],
           "cuda:2": ["generator", "passage_reranker"]}
    r = VllmDeploymentManager._hf_aux_reserve_gib(_sys(occ), ["cuda:1", "cuda:2"])
    assert r == max(_HF_AUX_RESERVE_GIB["semantic_retrieval"],
                    _HF_AUX_RESERVE_GIB["passage_reranker"])


def test_reserve_missing_layout_is_zero():
    assert VllmDeploymentManager._hf_aux_reserve_gib({}, ["cuda:0"]) == 0.0
    assert VllmDeploymentManager._hf_aux_reserve_gib(None, ["cuda:0"]) == 0.0


# ---------------------------------------------------------------------------
# _coresident_util — the OOM scenario and the working ones
# ---------------------------------------------------------------------------

def test_qe_on_crowded_aux_card_leaves_room():
    """pd_1p2d_left OOM repro: QE-vLLM + encoder+reranker+compressor on 1 card.
    The engine must reserve their footprint so it can't claim ~0.90 and OOM."""
    occ = {"cuda:0": ["query_expansion", "semantic_retrieval",
                      "passage_reranker", "passage_compressor"]}
    util = _mgr()._coresident_util(
        "query_expansion", ["cuda:0"], {"cuda:0": 1},
        system_cfg=_sys(occ), card_total_gib=TOTAL,
    )
    reserve = (_HF_AUX_RESERVE_GIB["semantic_retrieval"]
               + _HF_AUX_RESERVE_GIB["passage_reranker"]
               + _HF_AUX_RESERVE_GIB["passage_compressor"]
               + _SCHED_SAFETY_GIB)
    assert abs(util - _floor3((TOTAL - reserve) / TOTAL)) < 1e-6
    assert util < 0.90                               # DID leave room
    assert util * TOTAL + reserve <= TOTAL + 1e-6    # fits the card


def test_generator_without_aux_keeps_full_util():
    """PD decode cards with no HF aux must NOT be throttled — full 0.90."""
    occ = {"cuda:1": ["generator"], "cuda:2": ["generator"], "cuda:3": ["generator"]}
    util = _mgr()._coresident_util(
        "generator", ["cuda:1", "cuda:2", "cuda:3"], {},
        system_cfg=_sys(occ), card_total_gib=TOTAL,
    )
    assert util == 0.90


def test_ride_generator_reserves_only_actual_aux():
    """tp4_rr: generator spans all 4 cards, one aux rides each. It reserves the
    PRECISE aux footprint (not a blanket 24 GiB), so it keeps most of the card
    for KV — the accuracy win over the old fixed reserve."""
    occ = {f"cuda:{i}": ["generator", "passage_reranker"] for i in range(4)}
    util = _mgr()._coresident_util(
        "generator", [f"cuda:{i}" for i in range(4)], {},
        system_cfg=_sys(occ), card_total_gib=TOTAL,
    )
    reserve = _HF_AUX_RESERVE_GIB["passage_reranker"] + _SCHED_SAFETY_GIB
    assert abs(util - _floor3((TOTAL - reserve) / TOTAL)) < 1e-6
    assert util > 0.85   # far more KV than the old 24 GiB blanket ((93-24)/93=0.74)


def test_two_vllm_tenants_split_after_aux_reserve():
    """A card shared by 2 vLLM engines still halves the post-aux budget."""
    occ = {"cuda:0": ["generator", "query_expansion", "passage_reranker"]}
    util = _mgr()._coresident_util(
        "generator", ["cuda:0"], {"cuda:0": 2},
        system_cfg=_sys(occ), card_total_gib=TOTAL,
    )
    frac = (TOTAL - _HF_AUX_RESERVE_GIB["passage_reranker"]
            - _SCHED_SAFETY_GIB) / TOTAL
    assert abs(util - _floor3(frac / 2)) < 1e-6


# ---------------------------------------------------------------------------
# 07-12 msmarco-rerun regressions: r20 aux-PROCESS footprints (own CUDA
# context + allocator arena grown by saturated serving) dwarf the pre-r20
# in-process estimates. Frozen against the two observed tenancy OOMs.
# ---------------------------------------------------------------------------

def test_encoder_reserve_covers_r20_aux_process_footprint():
    """The MSMARCO batch-256 replay observed 8.66 GiB resident followed by
    a 3.00 GiB forward allocation.  The reserve must cover that real peak."""
    assert _HF_AUX_RESERVE_GIB["semantic_retrieval"] >= 11.66
    occ = {"cuda:0": ["query_expansion_prefill", "query_expansion_decode",
                      "semantic_retrieval_encode"]}
    util = _mgr()._coresident_util(
        "query_expansion", ["cuda:0"], {"cuda:0": 1},
        system_cfg=_sys(occ), card_total_gib=TOTAL,
    )
    reserve = _HF_AUX_RESERVE_GIB["semantic_retrieval"] + _SCHED_SAFETY_GIB
    assert abs(util - _floor3((TOTAL - reserve) / TOTAL)) < 1e-6
    assert util < 0.90
    # the encoder's observed peak demand fits in what the plan holds back
    assert TOTAL - util * TOTAL >= 11.66


def test_colbert_reserve_covers_pair_forward_arena():
    """mixedgp-s43 repro: colbert aux attempted single 3.26 GiB batch allocs;
    the r15 colbert-specific override (2 GiB, weights-sized) starved it. The
    override must now cover weights + arena + batch alloc."""
    assert _RERANKER_RESERVE_GIB["colbert_reranker"] >= 6.0
    nested = {"node_lines": [{"nodes": [{
        "stage": "passage_reranker",
        "modules": [{"component": "colbert_reranker"}],
    }]}]}
    overrides = _aux_reserve_overrides(nested)
    assert overrides.get("passage_reranker") == _RERANKER_RESERVE_GIB["colbert_reranker"]


def test_llmlingua2_reserve_covers_aux_process_peak():
    """Dragonball batch-256 replay observed 6.87 GiB resident followed by a
    0.375 GiB forward allocation. The central reserve must cover that peak."""
    observed_peak_gib = 6.87 + 0.375
    assert _COMPRESSOR_RESERVE_GIB["llmlingua2"] >= observed_peak_gib
    nested = {"node_lines": [{"nodes": [{
        "stage": "passage_compressor",
        "modules": [{"component": "llmlingua2"}],
    }]}]}
    overrides = _aux_reserve_overrides(nested)
    assert overrides["passage_compressor"] == _COMPRESSOR_RESERVE_GIB["llmlingua2"]
    occ = {"cuda:2": [
        "passage_compressor",
        "generator_prefill",
        "generator_decode",
    ]}
    util = _mgr()._coresident_util(
        "generator", ["cuda:2"], {"cuda:2": 1},
        system_cfg=_sys(occ), card_total_gib=TOTAL,
        aux_overrides=overrides,
    )
    reserve = _COMPRESSOR_RESERVE_GIB["llmlingua2"] + _SCHED_SAFETY_GIB
    assert abs(util - _floor3((TOTAL - reserve) / TOTAL)) < 1e-6
    assert util < 0.90
    assert TOTAL - util * TOTAL >= observed_peak_gib


def test_no_layout_falls_back_to_old_behavior():
    """Missing card_total or system_cfg → old vLLM-tenant-only policy (0.90 or
    /tenants), so nothing outside the measured path is disturbed."""
    m = _mgr()
    assert m._coresident_util("generator", ["cuda:0"], {"cuda:0": 1}) == 0.90
    assert m._coresident_util("generator", ["cuda:0"], {"cuda:0": 2}) == 0.45
