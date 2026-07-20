"""Launch-time GPU-memory probing for vLLM subprocess launches.

vLLM's ``--gpu-memory-utilization`` is a fraction of a GPU's TOTAL memory and is
checked at startup as ``free_mem >= util*total`` — it does NOT look at how much
is actually free. Two failure modes follow on a shared / reused card, and BOTH
are "phantom" (the card has room, or will in a moment):

  1. **Eviction lag** — when a trial replaces a prior engine, the old process is
     killed but its VRAM releases a few seconds later. A new engine launched
     immediately sees stale-low free memory → ``free < desired utilization`` →
     refuses to start, even though the memory frees right after.
  2. **Co-resident engine** — a legitimately collocated engine holds part of the
     card. A fixed util fraction may exceed what's left → same refusal, with
     space that simply belongs to the neighbour.

``effective_util`` fixes both: it waits briefly for memory to settle (handles
1), then if the requested fraction still doesn't fit, it lowers util to match
what's actually free (handles 2 — the engine launches with a smaller KV cache
instead of failing). Only when the card is GENUINELY full does the engine fail —
which is the correct, non-phantom outcome.
"""
from __future__ import annotations

import logging
import re
import subprocess
import time
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger("RAG-Stack")

_RESERVE_GIB = 24.0  # extra free memory when vLLM shares a GPU with HF stages
_CLEAN_FREE_SLACK_GIB = 2.0
_SETTLE_S = 25.0     # max wait for an evicted engine's VRAM to release
_FLOOR_UTIL = 0.05
_BYTES_PER_PARAM = 2.0   # bf16/fp16 weights
_MIN_KV_GIB = 4.0        # smallest KV cache that still serves tokens


def _model_weight_gib(model: Optional[str]) -> float:
    """Rough vLLM weight footprint (GiB) from the parameter count in the model
    name, e.g. 'Qwen/Qwen2.5-14B-Instruct' -> 14e9 * 2 bytes ≈ 28 GiB. Returns
    0 when no count is parseable (→ no weight floor)."""
    if not model:
        return 0.0
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", str(model))
    if not m:
        return 0.0
    return float(m.group(1)) * 1e9 * _BYTES_PER_PARAM / (1024 ** 3)


def _weight_floor_util(model: Optional[str], total_gib: float) -> float:
    """Minimum gpu_memory_utilization for ``model`` to load its weights plus a
    token of KV. The global memory scheduler plans utils that respect this; the
    live probe below must never squeeze BELOW it — squeezing a 14B engine under
    its 28 GiB of weights is what made a collocated query-expander fail to
    launch ('not enough memory to serve even a single token')."""
    if total_gib <= 0:
        return 0.0
    w = _model_weight_gib(model)
    if w <= 0:
        return 0.0
    return min(0.95, (w + _MIN_KV_GIB) / total_gib)


def _free_total_gib(cuda_ids: Sequence[str]) -> Tuple[float, float]:
    """(min free across cuda_ids, max total) in GiB, via nvidia-smi. cuda_ids are
    PHYSICAL device indices (what the launcher sets CUDA_VISIBLE_DEVICES to)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        ).stdout
    except Exception:  # noqa: BLE001
        return 0.0, 24.0
    free, total = {}, {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        idx, f, t = parts
        try:
            free[idx] = float(f) / 1024.0
            total[idx] = float(t) / 1024.0
        except ValueError:
            continue
    want = [str(c) for c in cuda_ids]
    fr = min((free.get(c, 0.0) for c in want), default=0.0)
    tot = max((total.get(c, 24.0) for c in want), default=24.0)
    return fr, tot


def effective_util(
    cuda_ids: List[str], requested: float, model: Optional[str] = None
) -> float:
    """The gpu_memory_utilization to actually launch a vLLM engine with on
    ``cuda_ids``, given the live free memory.

    Returns ``requested`` once the GPUs are free enough for it (waiting up to
    ``_SETTLE_S`` for evicted VRAM to release). If a co-resident tenant persists,
    returns a LOWER util sized to the free memory (so the launch succeeds with a
    smaller KV cache rather than hitting ``free < desired``). Bottoms out at
    ``_FLOOR_UTIL`` — below that the card is genuinely full and vLLM will (rightly)
    fail to fit the model.

    ``model`` gives the weight floor: this probe never lowers a vLLM engine
    below the util its weights (+ a token of KV) need. The blanket co-tenant
    reserve is meant to leave room for HF stages, but it also fires when the
    co-tenant is another vLLM engine (already sized by the global scheduler's
    tenant split) — double-counting that once squeezed a 14B query-expander
    below its 28 GiB of weights and failed its launch. The floor keeps the
    global scheduler's plan authoritative: shrink KV for real contention, never
    below what it takes to load the model.
    """
    if not cuda_ids:
        return requested
    deadline = time.monotonic() + _SETTLE_S
    free, total = _free_total_gib(cuda_ids)
    tenant_present = free < total - _CLEAN_FREE_SLACK_GIB
    reserve = _RESERVE_GIB if tenant_present else 0.3
    desired_free = requested * total + reserve
    while free < desired_free and time.monotonic() < deadline:
        time.sleep(2.0)
        free, total = _free_total_gib(cuda_ids)
        tenant_present = free < total - _CLEAN_FREE_SLACK_GIB
        reserve = _RESERVE_GIB if tenant_present else 0.3
        desired_free = requested * total + reserve
    floor = _weight_floor_util(model, total)
    if free >= desired_free:
        return max(requested, floor) if floor <= round(free / total, 3) else requested
    # A tenant persists — size to what's free instead of demanding `requested`,
    # but never below the model's weight floor (capped by what's actually free).
    eff = max(_FLOOR_UTIL, min(requested, round((free - reserve) / total, 3)))
    if floor > 0.0 and eff < floor:
        eff = min(floor, round(free / total, 3))
    logger.info(
        f"gpu_mem: cuda:{cuda_ids} only {free:.1f}/{total:.1f} GiB free; "
        f"lowering gpu_memory_utilization {requested} → {eff} to fit "
        f"(weight floor {floor:.3f})."
    )
    return eff
