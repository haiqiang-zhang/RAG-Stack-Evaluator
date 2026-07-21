"""Per-trial vLLM deployment policy for the measured-performance path.

This is the Layer-3 "policy / translation" layer that sits ABOVE the existing
two layers and orchestrates them (it launches nothing itself):

    VllmDeploymentManager   (this file — per-trial policy: which keys to build)
            │ calls cache.register_main_vllm / register_main_vllm_pd / register_aux_vllm
            ▼
    ModelCache              (cache.py — ownership/reuse: create-or-reuse per key)
            │ holds / relaunches
            ▼
    VllmSubprocess / VllmPdPair   (mechanism: actually run a vLLM, collocated or 1P1D)

Given a per-trial ``system_config`` (sampled placement / TP·PP / serving_mode)
and a RESOLVED nested pipeline config, it decides whether to run the main
generator collocated (single subprocess) or disaggregated (1P1D pair), runs the
GPU-budget checks, registers the main + auxiliary (query-expansion) vLLMs on the
supplied ``ModelCache``, and stamps the resolved device lists back into
``system_config`` (consumed by the measured serving runtime metadata and
placement enforcement).

Imports NO ``rag_stack.optimizer`` code, so it can be reused by the controller's
measured provider AND by an isolated baseline harness without breaking isolation.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

from rag_stack_evaluator.static_rag_evaluator.measured.vllm_subprocess import (
    RetryableVllmStartupError,
    VllmStartupKey,
    resolve_vllm_api_server_count,
)
from rag_stack_evaluator.static_rag_evaluator.measured.vllm_pd_pair import VllmPdPairKey
from rag_stack.system_layout import (
    engine_devices,
    engine_info,
    engine_role_devices,
    engine_role_parallelism,
    vllm_decode_max_num_seqs,
    vllm_kv_cache_dtype,
    vllm_max_num_batched_tokens,
    vllm_max_num_seqs,
    vllm_prefill_max_num_seqs,
)

logger = logging.getLogger("RAG-Stack")

_MEASURED_VLLM_REQUESTED_UTIL = 0.90
_MEASURED_VLLM_MAX_MODEL_LEN = -1

# ---------------------------------------------------------------------------
# Global GPU-memory scheduler (deterministic, layout-driven).
#
# vLLM sizes its own weights+KV within ``gpu_memory_utilization`` but is BLIND
# to co-resident in-process HF stages (query encoder / reranker / compressor).
# The old sizing left room for them only via a live free-memory probe at each
# engine's launch — fragile: when an HF aux on the same card had not finished
# loading at probe time (or loaded lazily on first use during the run), the
# vLLM engine claimed ~0.90 of a card that LOOKED empty, and the aux then OOM'd
# it — even a 1.5B engine, on a 93 GiB H100.
#
# Because the resolved layout already declares EXACTLY which stages sit on each
# card (``gpu_occupants``), we can plan every engine's util up front instead of
# probing: reserve a conservative per-stage VRAM budget (model weights + peak
# large-batch forward activation + headroom) for the HF aux sharing an engine's
# tightest card, and hand vLLM the rest. The live ``effective_util`` probe stays
# as a backstop that only LOWERS util further, never raises it.
# Sizing note (07-12, msmarco rerun campaign evidence): since r20 the encoder
# and rerankers run in SEPARATE aux processes — each carries its own CUDA
# context (~0.5-0.8 GiB) and a caching-allocator arena grown by the saturated
# serving protocol's large batches, on top of weights. The mpnet-class encoder
# was observed at 6.25 GiB resident + 1.31 GiB single batch alloc (eval_0016
# tenancy OOM, 100% warmup failure); the old 2 GiB reserve was pre-r20 truth.
_HF_AUX_RESERVE_GIB = {
    "semantic_retrieval": 12.0,  # MPNet aux process at request batch 256:
                                 # 8.66 GiB resident + a 3.00 GiB forward alloc
                                 # observed in the MSMARCO replay (07-14)
    "passage_reranker": 6.0,     # unknown reranker fallback
    "passage_compressor": 5.0,   # unknown compressor fallback
}
# Component-aware reranker footprints (r15): the flat 6 GiB was sized for
# colbert-class models; a 3B-parameter f32 cross-encoder (tart / monot5)
# needs ~11.4 GiB of weights alone plus multi-GiB activations for the
# 256-pair forwards — under the flat reserve the vLLM engine claims the
# card first and the aux load deterministically OOMs (s45_eval_0028
# tenancies, twice). Weights = params x 4 bytes (records confirm f32),
# activations sized for 256x512 pair forwards.
_RERANKER_RESERVE_GIB = {
    "tart": 16.0,                       # flan-t5-xl ~2.85B f32 + forwards
    "monot5": 16.0,                     # monot5-3b ~2.85B f32 + forwards
    "flag_embedding_reranker": 4.0,     # bge-reranker-large 560M f32
    "colbert_reranker": 6.0,            # BERT-base 110M weights are small but
                                        # the aux-process pair-forward arena is
                                        # not: 3.26 GiB single allocs observed
                                        # under saturated serving (07-12)
    "sentence_transformer_reranker": 1.0,  # MiniLM-L2 ~22M
}
# Encoder models whose f32 weights + long-context activations exceed the
# 2 GiB mpnet-class default.
_ENCODER_RESERVE_GIB = {
    "huggingface_bge_m3": 10.0,         # 570M f32 (+1.9 GiB over mpnet) and
                                        # 8k-context activations on top of the
                                        # 8 GiB aux-process default
}
_COMPRESSOR_RESERVE_GIB = {
    "llmlingua2": 8.0,                  # 6.87 GiB resident + a 0.375 GiB
                                        # batch-256 forward allocation observed
                                        # in the Dragonball replay (07-18)
}
_HF_AUX_ENGINES = frozenset(_HF_AUX_RESERVE_GIB)


def _aux_reserve_overrides(nested_cfg: Dict[str, Any]) -> Dict[str, float]:
    """Per-user-stage reserve overrides derived from the trial's ACTUAL
    components (nested_cfg node_lines) — the honest footprints above."""
    overrides: Dict[str, float] = {}
    for nl in (nested_cfg or {}).get("node_lines", []) or []:
        for node in nl.get("nodes", []) or []:
            stage = node.get("stage")
            mod = _resolved_module(node) or {}
            if stage == "passage_reranker":
                comp = str(mod.get("component") or "")
                if comp in _RERANKER_RESERVE_GIB:
                    overrides[stage] = _RERANKER_RESERVE_GIB[comp]
            if stage == "passage_compressor":
                comp = str(mod.get("component") or "")
                if comp in _COMPRESSOR_RESERVE_GIB:
                    overrides[stage] = _COMPRESSOR_RESERVE_GIB[comp]
    for vdb in (nested_cfg or {}).get("vectordb", []) or []:
        emb = str((vdb or {}).get("embedding_model") or "")
        if emb in _ENCODER_RESERVE_GIB:
            overrides["semantic_retrieval"] = max(
                overrides.get("semantic_retrieval", 0.0),
                _ENCODER_RESERVE_GIB[emb],
            )
    return overrides
_SCHED_SAFETY_GIB = 4.0          # slack for activation spikes / late allocs
_SCHED_FLOOR_UTIL = 0.05


class TrialInvalid(Exception):
    """A sampled deployment can't be realized on the GPU budget (TP·PP exceeds
    available GPUs, or a vLLM launch failed). The caller marks the trial
    invalid/penalized rather than aborting the whole sweep."""


def _check_nixl_hetero_tp_supported(
    model: Any,
    *,
    prefill_tp: int,
    decode_tp: int,
    prefill_pp: int = 1,
    decode_pp: int = 1,
) -> None:
    """Fail fast on a model/PD shape GenZ says NIXL cannot realize.

    GenZ owns the model-structure facts (attention heads, KV heads, layers,
    Mamba state) and the connector capability.  Measured mode deliberately
    consumes the same verdict as CM candidate generation; keeping a second
    model-name/KV-head table here previously let the two paths disagree.
    """
    from GenZ.parallelism import (
        NIXL_KV_TRANSFER_CAPABILITIES,
        check_disaggregated_parallelism,
    )

    verdict = check_disaggregated_parallelism(
        model,
        prefill_tp=int(prefill_tp),
        prefill_pp=int(prefill_pp),
        decode_tp=int(decode_tp),
        decode_pp=int(decode_pp),
        capabilities=NIXL_KV_TRANSFER_CAPABILITIES,
    )
    if not verdict.feasible:
        raise TrialInvalid(
            f"GenZ rejected NIXL P/D parallelism for model={model!r}: "
            f"{verdict.code}: {verdict.reason}"
        )


def _check_unified_parallelism_supported(
    model: Any, *, tensor_parallel: int, pipeline_parallel: int
) -> None:
    """Apply the same GenZ model-structure gate to a unified vLLM engine."""
    from GenZ.parallelism import check_model_parallelism

    verdict = check_model_parallelism(
        model,
        tensor_parallel=int(tensor_parallel),
        pipeline_parallel=int(pipeline_parallel),
    )
    if not verdict.feasible:
        raise TrialInvalid(
            f"GenZ rejected unified parallelism for model={model!r}: "
            f"{verdict.code}: {verdict.reason}"
        )


def _resolved_module(node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return a node's single resolved module from the current schema."""
    mods = node.get("modules")
    if isinstance(mods, list) and mods:
        return mods[0]
    raise KeyError(f"node {node.get('stage')!r} requires modules: [...]")


def find_generator_module(nested_cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The resolved vLLM generator module (carries the SAMPLED model)."""
    for nl in nested_cfg.get("node_lines", []) or []:
        for node in nl.get("nodes", []) or []:
            if node.get("stage") != "generator":
                continue
            mod = _resolved_module(node)
            if mod and mod.get("component") == "vllm":
                return mod
    return None


def find_query_expansion_module(nested_cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The resolved query-expansion module IFF enabled + vLLM backend (its
    model is an independent search dim → needs its own aux vLLM)."""
    for nl in nested_cfg.get("node_lines", []) or []:
        for node in nl.get("nodes", []) or []:
            if node.get("stage") != "query_expansion":
                continue
            mod = _resolved_module(node)
            if mod and mod.get("generator_backend") == "vllm":
                return mod
    return None


class VllmDeploymentManager:
    """Translates a per-trial ``system_config`` + resolved config into the right
    ``ModelCache`` registrations. Construct once per run; call
    :meth:`prepare_trial` before each evaluator call."""

    def __init__(self, available_gpus: List[str]):
        if not available_gpus:
            raise ValueError("available_gpus must be a non-empty list")
        self.available_gpus = [str(g) for g in available_gpus]

    def _vllm_knobs(
        self,
        system_cfg: Dict[str, Any],
        *,
        engine: str = "generator",
    ) -> Dict[str, Any]:
        return dict(
            max_num_seqs=vllm_max_num_seqs(system_cfg, engine),
            # None ⇒ don't pass --max-num-batched-tokens; let vLLM choose its own
            # (much larger) default. Only pin it if the config explicitly sets it.
            max_num_batched_tokens=vllm_max_num_batched_tokens(system_cfg),
            # Measured runtime policy, not a system-design knob: let vLLM
            # auto-fit context length instead of inheriting a model default
            # such as 32k that may not fit the trial placement.
            max_model_len=_MEASURED_VLLM_MAX_MODEL_LEN,
            kv_cache_dtype=vllm_kv_cache_dtype(system_cfg),
        )

    def _pd_seq_caps(
        self,
        system_cfg: Dict[str, Any],
        *,
        engine: str = "generator",
    ) -> tuple[int, int]:
        """Role-local admission caps for a 1P1D engine."""
        return (
            vllm_prefill_max_num_seqs(system_cfg, engine),
            vllm_decode_max_num_seqs(system_cfg, engine),
        )

    @staticmethod
    def _requested_gpu_memory_utilization(engine: str) -> float:
        """Measured-mode launch policy, not a system-design knob.

        The subprocess launcher will call ``effective_util`` against live
        ``nvidia-smi`` state and lower this request if a co-tenant already
        occupies VRAM. Keeping this outside ``system_config`` prevents cached
        benchmark cases from pinning stale co-residency fractions.
        """
        return _MEASURED_VLLM_REQUESTED_UTIL

    @staticmethod
    def _device_tenant_counts(device_groups: List[List[str]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for group in device_groups:
            for dev in {str(d) for d in group if d}:
                counts[dev] = counts.get(dev, 0) + 1
        return counts

    @staticmethod
    def _hf_aux_reserve_gib(
        system_cfg: Dict[str, Any], devices: List[str],
        aux_overrides: Optional[Dict[str, float]] = None,
    ) -> float:
        """Deterministic VRAM to hold back for HF aux stages co-resident with a
        vLLM engine on ``devices``, read from the resolved layout's
        ``gpu_occupants`` (the global plan, NOT a live probe). Sizes on the
        TIGHTEST card the engine spans, since a single gpu_memory_utilization
        applies to every card the engine uses.

        ``gpu_occupants`` speaks PerformanceStage names in the resolved
        system_config (e.g. ``semantic_retrieval_encode``,
        ``generator_prefill``) but user-stage names off ``derive_layout``. Map
        each occupant to its owning USER stage via the taxonomy and reserve per
        HF aux user stage — so ``semantic_retrieval_encode`` correctly counts as
        the ``semantic_retrieval`` encoder (the exact miss that would have made
        this a silent no-op at runtime)."""
        from rag_stack.search_space.placement import stage_engine

        layout = (system_cfg or {}).get("layout") or {}
        occ = layout.get("gpu_occupants") or {}
        table = dict(_HF_AUX_RESERVE_GIB)
        if aux_overrides:
            # r15: component-aware footprints from the trial's nested config
            # (a 3B f32 cross-encoder is not a colbert) — see
            # _RERANKER_RESERVE_GIB.
            table.update(aux_overrides)
        worst = 0.0
        for d in devices:
            reserve = 0.0
            for occupant in occ.get(str(d), []) or []:
                if occupant in table:                    # user-stage name
                    reserve += table[occupant]
                    continue
                try:                                      # PerformanceStage name
                    user_stage = stage_engine(occupant)
                except Exception:  # noqa: BLE001 — unknown name → no reserve
                    continue
                reserve += table.get(user_stage, 0.0)
            worst = max(worst, reserve)
        return worst

    def _coresident_util(
        self,
        engine: str,
        devices: List[str],
        tenant_counts: Dict[str, int],
        system_cfg: Optional[Dict[str, Any]] = None,
        card_total_gib: float = 0.0,
        aux_overrides: Optional[Dict[str, float]] = None,
    ) -> float:
        """Global memory scheduler: the gpu_memory_utilization to request for
        ``engine`` on ``devices``, planned from the full layout up front.

        Two deterministic reservations come off the requested 0.90:
          * **HF aux** co-resident on the card (encoder / reranker / compressor)
            — read from ``gpu_occupants`` so it holds regardless of load order,
            unlike the old launch-time free-memory probe that missed
            not-yet-loaded aux and OOM'd the engine.
          * **Later vLLM tenants** sharing the card — divide by the max vLLM
            tenant count so the first engine leaves room for the others.

        ``effective_util`` later still probes live free memory and may lower
        this further; it never raises it. Measured-runtime policy, not a knob.
        """
        requested = self._requested_gpu_memory_utilization(engine)
        max_tenants = max((tenant_counts.get(str(d), 1) for d in devices), default=1)
        hf_reserve = (
            self._hf_aux_reserve_gib(system_cfg, devices, aux_overrides)
            if system_cfg else 0.0
        )
        budget_frac = requested
        if hf_reserve > 0.0 and card_total_gib > 0.0:
            budget_frac = min(
                requested,
                max(
                    _SCHED_FLOOR_UTIL,
                    (card_total_gib - hf_reserve - _SCHED_SAFETY_GIB)
                    / card_total_gib,
                ),
            )
        # floor, not round: a planned claim must never round UP past the
        # budget (round-half-up overshot the card by ~46 MiB on 93 GiB).
        util = max(_SCHED_FLOOR_UTIL, math.floor(budget_frac / max_tenants * 1000) / 1000)
        if util < requested:
            logger.info(
                "mem-sched: %s on %s — reserve %.1f GiB HF-aux + %.1f safety "
                "of %.0f GiB, ÷%d vLLM tenant(s) -> gpu_memory_utilization "
                "%.3f (from %.2f)",
                engine, devices, hf_reserve, _SCHED_SAFETY_GIB,
                card_total_gib, max_tenants, util, requested,
            )
        return util

    @staticmethod
    def _unified_parallelism(
        system_cfg: Dict[str, Any],
        devices: List[str],
        *,
        engine: str,
    ) -> tuple[int, int]:
        """Resolve vLLM TP/PP for a unified engine.

        The resolved layout is the only source of truth. If the layout carries
        an invalid TP/PP pair for the concrete device list, normalize it in the
        nested engine record.
        """
        info = engine_info(system_cfg, engine)
        n = max(1, len(devices))
        pp = int(info.get("pp", 1) or 1)
        if pp < 1 or n % pp != 0 or pp > n:
            logger.warning(
                "%s pipeline parallelism PP=%s is incompatible with %s devices; "
                "falling back to PP=1.",
                engine,
                pp,
                n,
            )
            pp = 1
        tp = int(info.get("tp", n // pp) or (n // pp))
        if tp * pp != n:
            logger.warning(
                "%s TP*PP=%s*%s does not match %s devices; using TP=%s, PP=%s.",
                engine,
                tp,
                pp,
                n,
                n // pp,
                pp,
            )
            tp = n // pp
        info["tp"] = tp
        info["pp"] = pp
        return tp, pp

    @staticmethod
    def _role_parallelism(
        system_cfg: Dict[str, Any],
        devices: List[str],
        *,
        engine: str,
        role: str,
    ) -> tuple[int, int]:
        if not devices:
            raise TrialInvalid(f"{engine}.{role} has no resolved device list")
        tp, pp = engine_role_parallelism(system_cfg, engine, role)
        n = len(devices)
        if tp * pp != n:
            raise TrialInvalid(
                f"{engine}.{role} TP*PP={tp}*{pp} does not match {n} device(s)"
            )
        return tp, pp

    def prepare_trial(
        self,
        *,
        cache: Any,
        nested_cfg: Dict[str, Any],
        system_cfg: Dict[str, Any],
        force_disagg: bool = False,
    ) -> Dict[str, Any]:
        """Register the derived deployment on ``cache`` for this trial.

        The layout was already derived (CPU-only) by
        :meth:`PerformanceContext.resolve_system_config` — per-engine device lists,
        unified-vs-1P1D serving, and ``total_slots_needed``. This launches the
        matching vLLM(s): the generator (single subprocess if unified, a
        ``VllmPdPair`` if 1P1D) and, when present, the query-expansion engine
        (own aux subprocess / aux PD pair) — each with TP = len(its devices).

        Raises :class:`TrialInvalid` if the layout falls outside the configured
        GPU-slot bounds, overflows the GPU budget, or a launch fails. Mutates
        and returns ``system_cfg`` only inside its nested layout records."""
        gpus = self.available_gpus
        gen_knobs = self._vllm_knobs(system_cfg, engine="generator")

        # Budget: the derived layout's total GPU footprint must fit.
        layout = system_cfg.get("layout") or {}
        total_slots = int(layout.get("total_gpu_slots", 0) or 0)
        min_slots = int(layout.get("min_total_gpu_slots", 1) or 1)
        max_slots = int(layout.get("max_total_gpu_slots", len(gpus)) or len(gpus))
        if total_slots < min_slots:
            raise TrialInvalid(
                f"derived layout needs {total_slots} GPU slots < "
                f"configured min_num_gpus={min_slots}"
            )
        if total_slots > max_slots:
            raise TrialInvalid(
                f"derived layout needs {total_slots} GPU slots > "
                f"configured max_num_gpus={max_slots}"
            )
        if total_slots > len(gpus):
            raise TrialInvalid(
                f"derived layout needs {total_slots} GPU slots > "
                f"available_gpus={len(gpus)} ({gpus})"
            )

        # ---- main generator ----
        gen = find_generator_module(nested_cfg) or {}
        gen_model = gen.get("model", "Qwen/Qwen2.5-3B-Instruct")
        gen_dtype = gen.get("dtype", "bfloat16")
        gen_layout = engine_info(system_cfg, "generator")
        gen_devices = engine_devices(system_cfg, "generator") or [gpus[0]]
        gen_serving = str(gen_layout.get("pd_serving", "collocated_pd"))
        gen_disagg = bool(force_disagg) or gen_serving == "disagg_pd"
        gen_prefill_devices = (
            engine_role_devices(system_cfg, "generator", "prefill")
            if gen_disagg else None
        )
        gen_decode_devices = (
            engine_role_devices(system_cfg, "generator", "decode")
            if gen_disagg else None
        )
        gen_prefill_parallelism = gen_decode_parallelism = None
        if gen_disagg:
            gen_prefill_parallelism = self._role_parallelism(
                system_cfg, list(gen_prefill_devices or []),
                engine="generator", role="prefill",
            )
            gen_decode_parallelism = self._role_parallelism(
                system_cfg, list(gen_decode_devices or []),
                engine="generator", role="decode",
            )
            _check_nixl_hetero_tp_supported(
                gen_model,
                prefill_tp=gen_prefill_parallelism[0],
                prefill_pp=gen_prefill_parallelism[1],
                decode_tp=gen_decode_parallelism[0],
                decode_pp=gen_decode_parallelism[1],
            )
            gen_tp, gen_pp = gen_prefill_parallelism
        else:
            gen_tp, gen_pp = self._unified_parallelism(
                system_cfg, gen_devices, engine="generator"
            )
            _check_unified_parallelism_supported(
                gen_model,
                tensor_parallel=gen_tp,
                pipeline_parallel=gen_pp,
            )

        # ---- aux query-expansion metadata (if present) ----
        # Resolve this before launching the main generator so memory util can
        # account for all colocated vLLM tenants on each slot.
        qe = find_query_expansion_module(nested_cfg)
        qe_model = qe_dtype = qe_devices = qe_tp = qe_pp = qe_serving = qe_disagg = qe_knobs = None
        qe_prefill_parallelism = qe_decode_parallelism = None
        qe_prefill_devices = qe_decode_devices = None
        if qe is not None:
            qe_layout = engine_info(system_cfg, "query_expansion")
            qe_model = qe.get("model", gen_model)
            qe_dtype = qe.get("dtype", gen_dtype)
            qe_devices = engine_devices(system_cfg, "query_expansion") or [gen_devices[0]]
            qe_serving = str(qe_layout.get("pd_serving", "collocated_pd"))
            qe_disagg = qe_serving == "disagg_pd"
            if qe_disagg:
                qe_prefill_devices = engine_role_devices(
                    system_cfg, "query_expansion", "prefill"
                )
                qe_decode_devices = engine_role_devices(
                    system_cfg, "query_expansion", "decode"
                )
                qe_prefill_parallelism = self._role_parallelism(
                    system_cfg, qe_prefill_devices,
                    engine="query_expansion", role="prefill",
                )
                qe_decode_parallelism = self._role_parallelism(
                    system_cfg, qe_decode_devices,
                    engine="query_expansion", role="decode",
                )
                _check_nixl_hetero_tp_supported(
                    qe_model,
                    prefill_tp=qe_prefill_parallelism[0],
                    prefill_pp=qe_prefill_parallelism[1],
                    decode_tp=qe_decode_parallelism[0],
                    decode_pp=qe_decode_parallelism[1],
                )
            else:
                qe_tp, qe_pp = self._unified_parallelism(
                    system_cfg, qe_devices, engine="query_expansion"
                )
                _check_unified_parallelism_supported(
                    qe_model,
                    tensor_parallel=qe_tp,
                    pipeline_parallel=qe_pp,
                )
            qe_knobs = self._vllm_knobs(system_cfg, engine="query_expansion")

        vllm_tenant_groups: List[List[str]] = []
        if gen_disagg:
            vllm_tenant_groups.append(list(gen_prefill_devices or gen_devices))
            vllm_tenant_groups.append(list(gen_decode_devices or gen_devices))
        else:
            vllm_tenant_groups.append(list(gen_devices))
        if qe is not None:
            if qe_disagg:
                vllm_tenant_groups.append(list(qe_prefill_devices or qe_devices))
                vllm_tenant_groups.append(list(qe_decode_devices or qe_devices))
            else:
                vllm_tenant_groups.append(list(qe_devices))
        tenant_counts = self._device_tenant_counts(vllm_tenant_groups)
        # Card capacity for the global memory scheduler (homogeneous box —
        # probe once). Physical indices are the ":N" suffix of the layout's
        # cuda ids; total is invariant across cards.
        from rag_stack_evaluator.static_rag_evaluator.measured.gpu_mem import _free_total_gib
        _, card_total_gib = _free_total_gib(
            [str(g).split(":")[-1] for g in self.available_gpus]
        )
        aux_overrides = _aux_reserve_overrides(nested_cfg)
        gen_util = self._coresident_util(
            "generator",
            list(gen_prefill_devices or gen_devices)
            + (list(gen_decode_devices or []) if gen_disagg else []),
            tenant_counts,
            system_cfg=system_cfg,
            card_total_gib=card_total_gib,
            aux_overrides=aux_overrides,
        )

        self._register_engine(
            cache=cache, role="main", model=gen_model, dtype=gen_dtype,
            util=gen_util, devices=gen_devices, disagg=gen_disagg,
            prefill_devices=gen_prefill_devices,
            decode_devices=gen_decode_devices,
            knobs=gen_knobs, tensor_parallel_size=gen_tp,
            pipeline_parallel_size=gen_pp,
            prefill_parallelism=gen_prefill_parallelism,
            decode_parallelism=gen_decode_parallelism,
            pd_seq_caps=(
                self._pd_seq_caps(system_cfg, engine="generator")
                if gen_disagg else None
            ),
        )

        if qe is not None:
            qe_util = self._coresident_util(
                "query_expansion",
                list(qe_prefill_devices or qe_devices)
                + (list(qe_decode_devices or []) if qe_disagg else []),
                tenant_counts,
                system_cfg=system_cfg,
                card_total_gib=card_total_gib,
                aux_overrides=aux_overrides,
            )
            self._register_engine(
                cache=cache, role="aux", model=qe_model, dtype=qe_dtype,
                util=qe_util, devices=qe_devices, disagg=qe_disagg,
                prefill_devices=qe_prefill_devices,
                decode_devices=qe_decode_devices,
                knobs=qe_knobs, tensor_parallel_size=qe_tp,
                pipeline_parallel_size=qe_pp,
                prefill_parallelism=qe_prefill_parallelism,
                decode_parallelism=qe_decode_parallelism,
                pd_seq_caps=(
                    self._pd_seq_caps(system_cfg, engine="query_expansion")
                    if qe_disagg else None
                ),
            )
        else:
            # No query expansion this trial — tear down any aux engine left over
            # from a prior trial (eviction-on-register only fires when aux IS
            # used), so it can't leak GPU memory into this/later trials.
            cache.evict_aux()

        return system_cfg

    def _register_engine(
        self,
        *,
        cache: Any,
        role: str,                       # "main" (generator) | "aux" (query_expansion)
        model: str,
        dtype: str,
        util: float,
        devices: List[str],
        disagg: bool,
        prefill_devices: Optional[List[str]],
        decode_devices: Optional[List[str]],
        knobs: Dict[str, Any],
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        prefill_parallelism: Optional[tuple[int, int]] = None,
        decode_parallelism: Optional[tuple[int, int]] = None,
        pd_seq_caps: Optional[tuple[int, int]] = None,
    ) -> None:
        """Launch one LLM engine (unified vLLM or 1P1D PD pair) on ``cache``.

        TP = len(devices) (unified) or len(prefill_devices)/len(decode_devices)
        (1P1D). ``role`` picks the ModelCache slot (main generator vs aux QE).
        Raises :class:`TrialInvalid` on launch failure."""
        if disagg:
            pf = list(prefill_devices or [])
            dc = list(decode_devices or [])
            if not pf or not dc:
                raise TrialInvalid("disaggregated vLLM launch requires P/D devices")
            if prefill_parallelism is None or decode_parallelism is None:
                raise TrialInvalid("disaggregated vLLM launch requires role-local TP/PP")
            prefill_tp, prefill_pp = prefill_parallelism
            decode_tp, decode_pp = decode_parallelism
            prefill_max_num_seqs, decode_max_num_seqs = (
                pd_seq_caps if pd_seq_caps is not None
                else (int(knobs["max_num_seqs"]), int(knobs["max_num_seqs"]))
            )
            prefill_str, decode_str = ",".join(pf), ",".join(dc)
            # Do not evict HF models here: the measured provider has already
            # cleared stale models and pre-warmed the current trial's colocated
            # in-process stages so vLLM sizes KV cache around them.
            pd_key = VllmPdPairKey(
                model=model,
                prefill_device=prefill_str,
                decode_device=decode_str,
                gpu_memory_utilization=util,
                prefill_max_num_seqs=int(prefill_max_num_seqs),
                decode_max_num_seqs=int(decode_max_num_seqs),
                prefill_tensor_parallel_size=int(prefill_tp),
                prefill_pipeline_parallel_size=int(prefill_pp),
                decode_tensor_parallel_size=int(decode_tp),
                decode_pipeline_parallel_size=int(decode_pp),
                dtype=dtype,
                **knobs,
            )
            register = (cache.register_main_vllm_pd if role == "main"
                        else cache.register_aux_vllm_pd)
            try:
                register(pd_key)
            except RetryableVllmStartupError:
                # A host-port race is infrastructure state, not a property of
                # this sampled deployment. Preserve the typed exception so the
                # controller retries after the provider's teardown funnel.
                raise
            except Exception as exc:  # noqa: BLE001
                raise TrialInvalid(
                    f"{role} PD pair launch failed (key={pd_key}): {exc}"
                ) from exc
        else:
            device_str = ",".join(devices)
            # Do not evict HF models here; see the disaggregated branch above.
            key = VllmStartupKey(
                model=model,
                device=device_str,
                gpu_memory_utilization=util,
                tensor_parallel_size=int(tensor_parallel_size),
                pipeline_parallel_size=int(pipeline_parallel_size),
                dtype=dtype,
                api_server_count=resolve_vllm_api_server_count(),
                **knobs,
            )
            register = (cache.register_main_vllm if role == "main"
                        else cache.register_aux_vllm)
            try:
                register(key)
            except RetryableVllmStartupError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise TrialInvalid(
                    f"{role} vLLM launch failed (key={key}): {exc}"
                ) from exc
