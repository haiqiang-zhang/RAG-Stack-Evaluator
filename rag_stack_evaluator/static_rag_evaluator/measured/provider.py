"""Measured-performance provider for the controller's ``performance_source:
measured`` mode.

Encapsulates the "run the real pipeline on real GPUs and measure performance"
path so the controller stays a thin branch. One real run yields BOTH quality and
performance (via :meth:`MeasuredEvaluator.evaluate`, which composes the
pure-quality evaluator's shared pipeline core), so this provider returns both;
the controller hands the optimizer whichever objectives the current iteration
requested.

Responsibilities:
  * own the run-spanning :class:`ModelCache` (context manager). FAISS/BM25 stay
    reusable; in-process GPU models are evicted at each trial start and then
    pre-warmed for the current resolved placement before vLLM launch, so vLLM
    memory sizing sees the real colocated HF footprint. vLLM engines are torn
    down at the END of every trial (process-group kill, see ``evaluate``'s
    ``finally``) so a crashed run can't orphan their GPU memory into the
    next/resumed run — the reload cost is the price of zero cross-trial leakage;
  * drive :class:`VllmDeploymentManager` to launch the right vLLM(s) per trial
    (collocated / 1P1D disagg / aux query-expansion) from the sampled
    ``system_config``;
  * call the controller's (already corpus-swapped) evaluator and map the perf
    dict to a single scalar matching the cost model's selection policy
    (``max_throughput`` → qps, ``min_latency`` → -median latency), so the
    measured ``performance`` objective is directly comparable to the cost-model
    one (same direction, ref_point = 0);
  * memoize per resolved (config, system_config) so a decoupled optimizer's
    cheap-then-full requests for the same config cost ONE real run.

It owns no optimizer code, so the cost-model path is entirely unaffected.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from rag_stack_evaluator.static_rag_evaluator.measured.cache import ModelCache
from rag_stack_evaluator.static_rag_evaluator.measured.artifact_contract import (
    MeasuredGTInadmissibleError,
    require_measured_gt_admissible,
)
from rag_stack_evaluator.static_rag_evaluator.measured.evaluator import (
    MeasuredEvaluator,
    apply_measured_generation_defaults,
)
from rag_stack_evaluator.static_rag_evaluator.measured.vllm_deployment import (
    VllmDeploymentManager,
    TrialInvalid,
)
from rag_stack_evaluator.static_rag_evaluator.measured.vllm_subprocess import reclaim_orphaned_vllm
from rag_stack.security import safe_json_dump

logger = logging.getLogger("RAG-Stack")


@dataclass
class ProviderResult:
    """One measured evaluation: performance scalar + the quality metrics +
    the raw perf dict (persisted to ``performance.json`` for resume) + the
    independent completion-window workload trace."""
    performance_score: float
    performance_execution_trace: Dict[str, Any]
    quality: Dict[str, float] = field(default_factory=dict)
    raw_performance: Dict[str, Any] = field(default_factory=dict)


def _perf_scalar(perf: Dict[str, Any], selection: str) -> float:
    """Map the measured perf dict to a higher-is-better scalar that matches the
    cost-model selection policy, so cost_model and measured runs are on the same
    axis (and bounded below by 0 for throughput → matches ref_point=0)."""
    sel = (selection or "max_throughput").lower()
    latency = perf.get("latency_s")
    median_latency = (
        latency.get("median") if isinstance(latency, dict) else latency
    )
    if "latency" in sel or "min_latency" in sel:
        # min_latency policy → maximize negative latency.
        if median_latency is not None:
            return -float(median_latency)
        qps = perf.get("qps")
        return float(qps) if qps is not None else 0.0
    # default / max_throughput → maximize qps; fall back to -latency.
    qps = perf.get("qps")
    if qps is not None:
        return float(qps)
    if median_latency is not None:
        return -float(median_latency)
    return 0.0


def _config_hash(pipeline_config: Dict[str, Any], system_config: Dict[str, Any]) -> str:
    blob = json.dumps([pipeline_config, system_config], sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _persist_inadmissible_diagnostic(
    run_dir: Optional[str],
    result: ProviderResult,
) -> Optional[str]:
    """Persist a rejected measured window without publishing it as GT.

    Retries intentionally do not memoize these results. Keeping the raw
    stationarity payload and completion-window trace makes repeated rejection
    diagnosable instead of reducing it to one reason string.
    """

    if not run_dir:
        return None
    os.makedirs(run_dir, exist_ok=True)
    attempt = 1
    while True:
        path = os.path.join(
            run_dir, f"inadmissible_attempt_{attempt:03d}.json"
        )
        if not os.path.exists(path):
            break
        attempt += 1
    with open(path, "w") as stream:
        safe_json_dump({
            "status": "inadmissible",
            "performance_score": result.performance_score,
            "performance": result.raw_performance,
            "performance_execution_trace": result.performance_execution_trace,
        }, stream, indent=2, default=str)
    return path


_PROCESS_DEFAULTS_APPLIED = False


def ensure_measured_process_defaults() -> None:
    """Normalize every process-level knob that changes measured physics.

    THE single measured entry is MeasuredProvider — optimizer trials,
    accuracy-case runs and replay scripts all funnel through it — so this
    is the ONE place such knobs may be set. Launcher scripts and shells
    must not: two instruments once ran faiss with 32 vs 1 OMP threads for
    the SAME config (replay-script setdefault vs module default), which
    changes the retrieval truth outright at msmarco scale.

    - OMP thread default: 1 — the CM calibration anchor (t=1). Forced,
      not setdefault: the launching shell must not matter. Threaded
      search stays available ONLY via the priced per-query knob
      system.retrieval.faiss_num_threads. Child processes (aux workers,
      vLLM servers) inherit the env.
    - faiss process default re-pinned directly (its import-time read may
      predate this call).
    - HF tokenizers fork-parallelism off.

    Idempotent."""
    global _PROCESS_DEFAULTS_APPLIED
    if _PROCESS_DEFAULTS_APPLIED:
        return
    _PROCESS_DEFAULTS_APPLIED = True
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        import faiss

        faiss.omp_set_num_threads(1)
    except Exception:  # noqa: BLE001 — faiss may be absent in this process
        pass


class MeasuredProvider:
    """Run-spanning measured-performance evaluator. Use as a context manager so
    the ModelCache (and any vLLM subprocesses it owns) tears down on exit/crash:

        with MeasuredProvider(evaluator, gpus, n_queries, selection) as mp:
            for trial...: mp.evaluate(pipeline_config, system_config, run_dir)

    THE single measured entry (r23): optimizer trials pass resolved system
    values, replay/benchmark scripts pass a system_config dict loaded from
    JSON — same schema, same protocol. Process env and the client-population
    protocol are owned HERE (ensure_measured_process_defaults + the serving
    runtime's saturation adapter); callers must not pin either."""

    provides_quality = True

    def __init__(
        self,
        evaluator: Any,
        available_gpus: list,
        *,
        n_queries: Optional[int] = None,
        selection: str = "max_throughput",
    ):
        # Wrap the controller's (corpus-pre-swapped) pure-quality evaluator with the
        # measured orchestration layer; one real run yields quality + performance.
        self._measured = MeasuredEvaluator(evaluator)
        self._deploy = VllmDeploymentManager(available_gpus)
        self._n_queries = n_queries
        self._selection = selection
        self._cache: Optional[ModelCache] = None
        self._memo: Dict[str, ProviderResult] = {}
        # Set when an eval fails; the NEXT evaluate() re-runs the full HF-model
        # eviction first. The in-except eviction can't fully release VRAM — the
        # in-flight exception's traceback frames still reference the models —
        # but by the next call those frames are dead and gc+empty_cache works.
        self._dirty = False

    def __enter__(self) -> "MeasuredProvider":
        ensure_measured_process_defaults()
        # Clear vLLM servers orphaned by a previous run that died via
        # SIGKILL/OOM-kill (its teardown never ran) BEFORE we launch anything —
        # those re-parented EngineCore processes pin GPU memory and would
        # otherwise starve this run's (or a resume's) launches. Only our own
        # orphans (env-marked, re-parented to init) are touched.
        reclaim_orphaned_vllm(logger)
        self._cache = ModelCache().__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._cache is not None:
            self._cache.__exit__(exc_type, exc, tb)
            self._cache = None

    def evaluate(
        self,
        pipeline_config: Dict[str, Any],
        system_config: Dict[str, Any],
        *,
        run_dir: Optional[str] = None,
        force_disagg: bool = False,
        require_admissible: bool = True,
    ) -> ProviderResult:
        """Real measured eval for one resolved config. Raises
        :class:`TrialInvalid` when the deployment can't fit the GPU budget /
        fails to launch (the controller penalizes the trial). Memoized by
        (config, system_config)."""
        if self._cache is None:
            raise RuntimeError("MeasuredProvider.evaluate called outside its context manager")

        pipeline_config = apply_measured_generation_defaults(pipeline_config)
        key = _config_hash(pipeline_config, system_config)
        cached = self._memo.get(key)
        if cached is not None:
            logger.info("MeasuredProvider: memo hit — returning cached objectives")
            return cached

        if self._dirty:
            # Previous eval failed: its exception traceback (alive at the time
            # of the in-except eviction) pinned the model refs. Those frames
            # are dead now — re-evict so gc + empty_cache actually return the
            # VRAM before this trial deploys.
            self._cache.evict_all_hf_models()
            self._cache.evict_vllm()
            self._dirty = False
        else:
            # Clear stale in-process GPU models from prior trials. The current
            # trial's HF stages will be pre-warmed inside MeasuredEvaluator just
            # before vLLM launch; that is the footprint vLLM must size around.
            self._cache.evict_all_hf_models()

        try:
            def _launch_vllm(prepared_system_config: Dict[str, Any]) -> None:
                # Launch the right vLLM(s) for this trial after current-trial
                # HF stages have been warmed. This keeps gpu_memory_utilization
                # a measured-runtime decision based on live colocated memory.
                self._deploy.prepare_trial(
                    cache=self._cache,
                    nested_cfg=pipeline_config,
                    system_cfg=prepared_system_config,
                    force_disagg=force_disagg,
                )

            if run_dir:
                os.makedirs(run_dir, exist_ok=True)
            result = self._measured.evaluate(
                config=pipeline_config,
                cache=self._cache,
                system_config=system_config,
                n_queries=self._n_queries,
                run_dir=run_dir,
                launch_vllm=_launch_vllm,
                generation_defaults_applied=True,
            )
            quality = dict(result.get("quality", {}) or {})
            perf = dict(result.get("performance", {}) or {})
            performance_execution_trace = result.get(
                "performance_execution_trace"
            )
            if not isinstance(performance_execution_trace, dict):
                raise RuntimeError(
                    "MeasuredEvaluator returned no performance_execution_trace"
                )
            res = ProviderResult(
                performance_score=_perf_scalar(perf, self._selection),
                quality=quality,
                raw_performance=perf,
                performance_execution_trace=performance_execution_trace,
            )
            # An inadmissible window is useful diagnostic output (eval-one
            # persists it), but it is not an optimizer observation and must
            # never poison the run-spanning memo. The controller applies the
            # same publication gate and retries; leaving this key absent makes
            # that retry a real measurement instead of a memo hit.
            try:
                require_measured_gt_admissible(perf)
            except MeasuredGTInadmissibleError as exc:
                logger.warning(
                    "MeasuredProvider: result is diagnostic-only and was not "
                    "memoized: %s", exc,
                )
                if require_admissible:
                    try:
                        diagnostic_path = _persist_inadmissible_diagnostic(
                            run_dir, res,
                        )
                        if diagnostic_path is not None:
                            logger.warning(
                                "MeasuredProvider: inadmissible diagnostic saved: %s",
                                diagnostic_path,
                            )
                    except Exception as diagnostic_exc:  # noqa: BLE001
                        logger.warning(
                            "MeasuredProvider: could not persist inadmissible "
                            "diagnostic: %s", diagnostic_exc,
                        )
                    raise
            else:
                self._memo[key] = res
            return res
        except Exception as exc:
            # Failed eval (OOM mid-pipeline, launch failure, …): drop every
            # cached HF model so partially-loaded residue can't poison the
            # controller's retries or later trials (on 24 GiB cards a few GiB
            # of residue flips genuinely-feasible arms into launch failures).
            # NOTE: this in-except eviction is only PARTIAL — the in-flight
            # exception's traceback frames still reference the models, so gc
            # can't free them yet. The _dirty flag finishes the job at the
            # start of the next evaluate() call.
            try:
                self._cache.evict_all_hf_models()
            except Exception:  # noqa: BLE001 — never mask the real error
                pass
            self._dirty = True
            # CUDA OOM while loading/running an aux stage next to a planned
            # vLLM tenant is DETERMINISTIC for this deployment (the layout's
            # aux set exceeds the HF reserve — e.g. a slot-projected layout
            # packing every aux onto the engine cards, s44_eval_0035 r12).
            # Retrying reproduces it forever; classify as the same
            # weight-floor INVALID the docs promise so campaigns prune the
            # arm instead of error-looping.
            if "CUDA out of memory" in str(exc):
                raise TrialInvalid(
                    f"aux-stage CUDA OOM under the planned vLLM tenancy "
                    f"(deployment infeasible on this hardware): {exc}"
                ) from exc
            raise
        finally:
            # SINGLE per-trial teardown funnel. EVERY measured eval — success,
            # TrialInvalid, or any other exception — ends HERE, killing all vLLM
            # processes this trial spawned (main + aux, unified subprocess or PD
            # pair, each reaped as a whole process group incl. EngineCore
            # workers). We deliberately do NOT keep engines warm across trials:
            # an abrupt run death (SIGKILL/OOM-kill) skips teardown and orphans
            # 17 GiB+ servers into the next/resumed run (the failure this fixes).
            # A guaranteed clean slate per trial trades reload latency for zero
            # cross-trial GPU leakage.
            try:
                self._cache.evict_vllm()
            except Exception:  # noqa: BLE001 — teardown is best-effort
                pass
