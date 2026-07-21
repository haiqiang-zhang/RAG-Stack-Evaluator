"""Measured-performance orchestration, layered on the pure-quality evaluator.

The measured half of the GT/measured split: StaticRAGEvaluatorQualityOnly owns the
mode-agnostic pipeline core (corpus -> vectordb -> node_lines -> quality) and returns
ONLY quality; MeasuredEvaluator composes it and adds the measured-only concerns —
injecting the run-spanning ModelCache + derived device placement into the node
modules BEFORE the pipeline runs (via the ``before_run`` hook), and deriving real
per-query latency / TTFT / TPOT / QPS from the run artifacts AFTER. The cost-model
(pure-quality) path never imports or touches this module.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from rag_stack.rag_ir import make_quality_trace_envelope
from rag_stack_evaluator.static_rag_evaluator.measured.performance_trace import (
	validate_performance_trace_envelope,
)
from rag_stack_evaluator.static_rag_evaluator.measured.serving_runtime import (
	MeasuredServingRuntime,
)
logger = logging.getLogger("RAG-Stack")

# GPU stages that consume a multi-device list in module_param (HF models with
# no vLLM subprocess): retrieval encoder/reranker = data-parallel replicas,
# compressor = component-specific DP/sharding.
_DEVICE_LIST_NODE_TYPES = frozenset({
	"semantic_retrieval", "passage_reranker", "passage_compressor",
})

# Node types whose models live IN-PROCESS (not in a vLLM subprocess) and are
# therefore (re)loaded lazily on first use INSIDE the pipeline run: the
# retrieval embedding (released every trial — see _release_gpu_memory) and the
# cache-managed HF reranker / compressor (loaded on first trial that uses
# them). These are pre-warmed via the before_timing hook so model-load time
# never pollutes the measured window. generator / query_expansion are NOT
# warmed: their models run in vLLM subprocesses launched before timing, and
# constructing their modules in-process loads nothing.
_WARM_NODE_TYPES = frozenset(
	{"semantic_retrieval", "passage_reranker", "passage_compressor"}
)

MEASURED_GENERATION_MAX_TOKENS = 512
MEASURED_SOURCE_MAX_TOKENS_KEY = "_measured_source_max_tokens"


def _measured_reranker_forward_batch(
	system_config: Dict[str, Any],
	node: Any,
	request_batch_size: int,
) -> int:
	"""Forward microbatch for GPU rerankers in measured serving.

	``batch_size_request`` is the service/request batch. MonoT5's ``batch`` is a
	passage-pair forward microbatch; tying it to the request batch can OOM when
	MonoT5 is colocated with vLLM. The policy itself lives in
	``rag_stack.cost_model.reranker_policy`` — the CM owns the policy and
	measured consumes the SAME table so both sides tile the same number of
	pairs per forward (single source, r26).
	"""
	from rag_stack.cost_model.reranker_policy import (
		reranker_forward_batch_from_system,
	)

	component = str(getattr(getattr(node, "module", None), "component", ""))
	return reranker_forward_batch_from_system(
		component,
		int(request_batch_size),
		system_config,
	)


def apply_measured_generation_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
	"""Return a measured-run config with a uniform generation output cap.

	This is measured runtime policy, not a CM/workload token statistic. Existing
	benchmark cases may carry react ``max_tokens=256``; measured runs should use
	the same hard cap as sequential runs so trace token counts, not config-family
	defaults, explain output length differences.
	"""
	cfg = deepcopy(config)
	for node_line in cfg.get("node_lines", []) or []:
		for node in node_line.get("nodes", []) or []:
			stage = node.get("stage")
			modules: List[Dict[str, Any]] = []
			for mod in node.get("modules", []) or []:
				if isinstance(mod, dict):
					modules.append(mod)
			for mod in modules:
				source_max_tokens = mod.pop(MEASURED_SOURCE_MAX_TOKENS_KEY, None)
				generation_max_tokens = (
					int(source_max_tokens)
					if source_max_tokens is not None
					else MEASURED_GENERATION_MAX_TOKENS
				)
				if (
					stage == "generator"
					and mod.get("component") == "vllm"
				):
					mod["max_tokens"] = generation_max_tokens
				elif (
					stage == "query_expansion"
					and mod.get("generator_backend") == "vllm"
				):
					mod["max_tokens"] = generation_max_tokens
	return cfg


class MeasuredEvaluator:
	"""Measured-performance wrapper around a pure-quality StaticRAGEvaluatorQualityOnly.

	Owns NO pipeline logic: it delegates the run to ``base._run_pipeline`` with a
	pre-run hook, then derives the cost-model-aligned performance dict. One real
	run yields BOTH quality and performance.
	"""

	def __init__(self, base: Any):
		# base: a StaticRAGEvaluatorQualityOnly (corpus pre-swapped per eval by the controller).
		self._base = base

	def evaluate(
		self,
		config: dict,
		cache: Any,
		system_config: Optional[Dict[str, Any]] = None,
		n_queries: Optional[int] = None,
		run_dir: str | None = None,
		launch_vllm: Optional[Callable[[Dict[str, Any]], None]] = None,
		generation_defaults_applied: bool = False,
	) -> dict:
		"""Run the pipeline on real hardware; return
		``{quality, performance, config_resolved}``.

		:param cache: REQUIRED run-spanning ModelCache owning the cross-call
		    resources (vLLM subprocess, embedding, reranker, FAISS, BM25).
		:param system_config: deployment knobs (placement_<component>,
		    tensor_parallel_vllm, batch_size_*, kv_cache_dtype, ...).
		:param n_queries: If set, evaluate on the first ``n_queries`` rows.
		"""
		if cache is None:
			raise ValueError(
				"MeasuredEvaluator.evaluate requires a non-None ModelCache. "
				"Use StaticRAGEvaluatorQualityOnly.evaluate() for the quality-only path."
			)
		# MeasuredProvider prepares the config before hashing and deployment, so
		# applying the policy again here would consume replay-only source markers
		# twice (notably changing a vllm_api QE cap from 4096 to 512). Direct
		# MeasuredEvaluator callers still get the policy by default.
		config = (
			deepcopy(config)
			if generation_defaults_applied
			else apply_measured_generation_defaults(config)
		)
		system_config = dict(system_config or {})

		def _before_run(node_lines):
			self._inject_cache_and_devices(node_lines, cache, system_config)

		def _before_timing(node_lines):
			# Load current-trial in-process GPU stages first so measured vLLM
			# launch sizes KV cache against the REAL colocated footprint.
			self._warm_inprocess_gpu_models(node_lines, strict=True)
			if launch_vllm is not None:
				launch_vllm(system_config)

		run = self._base._run_pipeline(
			config, run_dir=run_dir, n_queries=n_queries,
			before_run=_before_run,
			before_timing=_before_timing,
			sequential_runner=lambda **kwargs: self._run_sequential_service(
				**kwargs, system_config=system_config
			),
		)
		if getattr(run, "sequential_perf", None) is not None:
			# Measured path: sequential and ReAct both run through the same
			# saturated service runtime. No stage-summary recomposition, no
			# per-query module/index reload, and no separate ReAct timing branch.
			performance = dict(run.sequential_perf)
		else:
			raise RuntimeError(
				"measured service runner returned no performance payload"
			)
		# Extract the independent completion-window workload trace before the
		# performance summary leaves this layer. It must never ride as a metric or
		# silently fall back to the quality winners.
		performance_execution_trace = performance.pop(
			"__performance_execution_trace__", None
		)
		if performance_execution_trace is None:
			raise RuntimeError(
				"measured service runner returned no performance_execution_trace"
			)
		validate_performance_trace_envelope(performance_execution_trace)

		# Surface one trace per dataset QA row so DynamicAssembly prices the
		# quality workload. This is intentionally not the saturated performance
		# completion cohort, whose cardinality is measured_queries.
		quality_out = dict(run.quality)
		trace = getattr(run, "trace", None)
		if not isinstance(trace, list) or len(trace) != run.n_evaluated:
			raise RuntimeError(
				"measured service runner returned a partial trace_v1 payload: "
				f"expected {run.n_evaluated} dataset rows, got "
				f"{len(trace) if isinstance(trace, list) else 0}"
			)
		if any(
			not isinstance(query_trace, list) or not query_trace
			for query_trace in trace
		):
			raise RuntimeError(
				"measured service runner returned an empty per-query trace_v1 entry"
			)
		# Ship THE frozen canonical quality-trace envelope: question_id = the dataset
		# row's stable id (row idx — the runtime projected the recorder to one
		# first-completion winner per row, in dataset order), invocation_id =
		# that winner's serving qid (may be a warmup-* admission), read from
		# the first-completion dataframe the runtime returned (its per-row
		# ``__qid__`` IS the winner identity). The envelope factory re-checks
		# cardinality and per-row completeness (terminal generator call).
		quality_out["__execution_dag__"] = make_quality_trace_envelope(
			trace,
			question_ids=[str(idx) for idx in range(run.n_evaluated)],
			invocation_ids=[
				str(qid) for qid in run.previous_result["__qid__"].tolist()
			],
		)
		return {
			"quality": quality_out,
			"performance": performance,
			"performance_execution_trace": performance_execution_trace,
			"config_resolved": {
				"system_config": system_config,
				"n_queries": run.n_evaluated,
			},
		}

	def _run_sequential_service(
		self,
		*,
		config: dict,
		node_lines: Dict[str, List[Any]],
		run_dir: str,
		qa_data: pd.DataFrame,
		system_config: Dict[str, Any],
		trace_recorder: Optional[Any] = None,
	) -> tuple[pd.DataFrame, Dict[str, Any]]:
		"""Run measured RAG as saturated stage services.

		The runtime handles both sequential and ReAct flows. Modules are
		constructed once, requests cycle through the QA set in a closed-loop
		saturated driver, generator/QR submit to vLLM continuous batching, and
		non-vLLM stages use explicit count/timeout batching.
		"""
		del run_dir  # the shared core already resolved corpus/index/run dirs.

		def _run_sync() -> tuple[pd.DataFrame, Dict[str, Any]]:
			# ``_run_sync`` may execute in the defensive ThreadPoolExecutor
			# branch below. Contextvars do not cross that boundary, so bind the
			# explicitly passed recorder for node wrappers in this execution
			# context and restore whatever the caller had afterwards.
			from rag_stack_evaluator.static_rag_evaluator.recording import (
				get_current_recorder,
				set_current_recorder,
			)
			previous_recorder = get_current_recorder()
			if trace_recorder is not None:
				set_current_recorder(trace_recorder)
			stages: List[Dict[str, Any]] = []
			try:
				stages = self._build_service_stages(node_lines)
				runtime = MeasuredServingRuntime(
					owner=self,
					stages=stages,
					qa_data=qa_data,
					node_lines=node_lines,
					system_config=system_config,
					config=config,
					trace_recorder=trace_recorder,
				)
				return asyncio.run(
					runtime.run()
				)
			finally:
				for stage in stages:
					try:
						inst = stage.get("instance")
						close = getattr(inst, "close", None)
						if close is not None and type(inst).__name__ == "AuxProcessStage":
							close()  # kill the aux child process + its VRAM
						del stage["instance"]
					except Exception:  # noqa: BLE001
						pass
				set_current_recorder(previous_recorder)

		try:
			asyncio.get_running_loop()
		except RuntimeError:
			return _run_sync()
		# Defensive path for callers already inside an event loop.
		with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
			return pool.submit(_run_sync).result()

	def _build_service_stages(
		self,
		node_lines: Dict[str, List[Any]],
	) -> List[Dict[str, Any]]:
		"""Instantiate each node module exactly once for the service lifetime."""
		stages: List[Dict[str, Any]] = []
		for node_line_name, nodes in node_lines.items():
			for node in nodes:
				callables, params = node.get_param_combinations()
				if len(callables) != 1:
					raise ValueError(
						f"{node.stage} expects exactly one module after sampling, "
						f"got {len(callables)}"
					)
				cls = callables[0]
				kwargs = dict(params[0])
				# Host-heavy stages run in a dedicated child process (A2, r12;
				# r20 adds retrieval): in-process they share the GIL + CPU
				# cores with the event loop / vLLM client / faiss and their
				# host-bound forwards inflate 2.8-90x over their isolated
				# cost. See aux_process.py.
				from rag_stack_evaluator.static_rag_evaluator.measured.aux_process import (
					AuxProcessStage,
					process_isolated_stage,
				)
				if process_isolated_stage(str(node.stage)):
					instance: Any = AuxProcessStage(
						cls, self._base.project_dir, kwargs,
						stage=str(node.stage),
					)
				else:
					instance = cls(self._base.project_dir, **kwargs)
				stages.append(
					{
						"node_line_name": node_line_name,
						"node": node,
						"stage": node.stage,
						"params": kwargs,
						"instance": instance,
					}
				)
		return stages

	@staticmethod
	def _sampling_params(params: Dict[str, Any]) -> Dict[str, Any]:
		from rag_stack_evaluator.static_rag_evaluator.measured.vllm_subprocess import (
			MEASURED_REQUEST_FORMAT_KEY,
			_request_format,
		)

		out = {
			"temperature": params.get("temperature", 1.0),
			"max_tokens": params.get("max_tokens", 512),
		}
		out[MEASURED_REQUEST_FORMAT_KEY] = _request_format(params)
		if params.get("stop"):
			out["stop"] = params["stop"]
		if params.get("top_p") is not None:
			out["top_p"] = params["top_p"]
		return out

	@staticmethod
	def _merge_service_node_result(
		node: Any,
		previous_result: pd.DataFrame,
		result: pd.DataFrame,
	) -> pd.DataFrame:
		"""Mirror node run.py column semantics without file/summary writes."""
		from rag_stack_evaluator.static_rag_evaluator.evaluation.retrieval import (
			RETRIEVAL_METRIC_FUNC_DICT,
		)
		from rag_stack_evaluator.static_rag_evaluator.utils.cast import drop_retrieval_columns

		stage = node.stage
		prev = previous_result.copy().reset_index(drop=True)
		res = result.copy().reset_index(drop=True)
		metrics = node.strategy.get("metrics") or []
		metric_names = [
			(m.get("metric_name") or m.get("metric")) if isinstance(m, dict) else m
			for m in metrics
		]
		metric_names = [m for m in metric_names if isinstance(m, str)]

		if stage == "semantic_retrieval":
			prev = prev.drop(
				columns=list(RETRIEVAL_METRIC_FUNC_DICT.keys()), errors="ignore"
			)
			res = res.rename(
				columns={
					"retrieved_contents": "retrieved_contents_semantic",
					"retrieved_ids": "retrieved_ids_semantic",
					"retrieve_scores": "retrieve_scores_semantic",
				}
			)
			return pd.concat([prev, res], axis=1)
		if stage == "lexical_retrieval":
			prev = prev.drop(
				columns=list(RETRIEVAL_METRIC_FUNC_DICT.keys()), errors="ignore"
			)
			res = res.rename(
				columns={
					"retrieved_contents": "retrieved_contents_lexical",
					"retrieved_ids": "retrieved_ids_lexical",
					"retrieve_scores": "retrieve_scores_lexical",
				}
			)
			return pd.concat([prev, res], axis=1)
		if stage == "hybrid_retrieval":
			prev = prev.drop(
				columns=list(RETRIEVAL_METRIC_FUNC_DICT.keys()), errors="ignore"
			)
			return pd.concat([prev, res], axis=1)
		if stage in {"passage_reranker", "passage_filter", "passage_augmenter"}:
			res = res.rename(
				columns={m: f"{stage}_{m}" for m in metric_names if m in res.columns}
			)
			return pd.concat([drop_retrieval_columns(prev), res], axis=1)
		if stage == "passage_compressor":
			if "retrieved_contents" in res.columns:
				prev["retrieved_contents"] = res["retrieved_contents"].values
				res = res.drop(columns=["retrieved_contents"])
			combined = pd.concat([prev, res], axis=1)
			return combined.rename(
				columns={m: f"passage_compressor_{m}" for m in metric_names}
			)
		if stage == "query_expansion":
			overlap_cols = res.columns.intersection(prev.columns)
			if len(overlap_cols) > 0:
				res = res.drop(columns=overlap_cols)
			return pd.concat([prev, res], axis=1)
		return pd.concat([prev, res], axis=1)

	@staticmethod
	def _generator_chip_count(system_config: Dict[str, Any]) -> int:
		from rag_stack.system_layout import generator_chip_count

		return generator_chip_count(system_config)

	@staticmethod
	def _add_deployment_metadata(
		summary: Dict[str, Any],
		node_lines: Dict[str, List[Any]],
		system_config: Dict[str, Any],
	) -> None:
		from rag_stack.search_space.placement import GPU_STAGES
		from rag_stack.system_layout import (
			engine_info,
			engine_parallelism,
			engine_role_devices,
			engine_role_parallelism,
			generator_chip_count,
			stage_devices,
		)

		node_devices: Dict[str, List[str]] = {}
		for nodes in node_lines.values():
			for node in nodes:
				nt = node.stage
				if nt not in GPU_STAGES:
					continue
				devs = stage_devices(system_config, nt)
				if devs:
					node_devices[nt] = devs
		gpu_occupants: Dict[str, List[str]] = {}
		for nt, devs in node_devices.items():
			for dev in devs:
				gpu_occupants.setdefault(dev, []).append(nt)
		strategy_groups = [
			sorted(set(gpu_occupants[g])) for g in sorted(gpu_occupants.keys())
		]
		summary["collocation_strategy"] = " | ".join(
			"+".join(g) for g in strategy_groups
		)
		node_to_devs = {nt: set(devs) for nt, devs in node_devices.items()}
		all_used_devices = set(gpu_occupants.keys()) or {"cuda:0"}
		any_overlap = any(
			node_to_devs[a] & node_to_devs[b]
			for a in node_to_devs
			for b in node_to_devs
			if a < b
		)
		if len(all_used_devices) == 1:
			summary["placement_policy"] = "collocated"
		elif not any_overlap:
			summary["placement_policy"] = "disaggregated"
		else:
			summary["placement_policy"] = "mixed"
		summary["placement_per_node"] = {
			nt: ",".join(devs) for nt, devs in node_devices.items()
		}
		gen = engine_info(system_config, "generator")
		summary["serving_mode"] = str(gen.get("pd_serving", "collocated_pd"))
		if gen.get("pd_serving") == "disagg_pd":
			# The NIXL P/D launcher deliberately remains single-frontend until its
			# role-local API scaling is validated separately.
			summary["generator_api_server_count"] = 1
		else:
			from rag_stack_evaluator.static_rag_evaluator.measured.vllm_subprocess import (
				resolve_vllm_api_server_count,
			)

			summary["generator_api_server_count"] = (
				resolve_vllm_api_server_count()
			)
		summary["num_chips_total_vllm"] = generator_chip_count(system_config)
		if gen.get("pd_serving") == "disagg_pd":
			pf_tp, pf_pp = engine_role_parallelism(system_config, "generator", "prefill")
			dec_tp, dec_pp = engine_role_parallelism(system_config, "generator", "decode")
			summary["tensor_parallel_prefill_vllm"] = pf_tp
			summary["pipeline_parallel_prefill_vllm"] = pf_pp
			summary["tensor_parallel_decode_vllm"] = dec_tp
			summary["pipeline_parallel_decode_vllm"] = dec_pp
			summary["num_chips_prefill_vllm"] = len(
				engine_role_devices(system_config, "generator", "prefill")
			)
			summary["num_chips_decode_vllm"] = len(
				engine_role_devices(system_config, "generator", "decode")
			)
		else:
			tp, pp = engine_parallelism(system_config, "generator")
			summary["tensor_parallel_vllm"] = tp
			summary["pipeline_parallel_vllm"] = pp
			summary["num_chips_vllm"] = tp * pp

	def _warm_inprocess_gpu_models(
		self,
		node_lines: Dict[str, List[Node]],
		*,
		strict: bool = False,
	) -> None:
		"""Pre-warm in-process GPU models so the timed window measures ONLY
		online inference, never model (re)loads.

		Runs as the pipeline core's ``before_timing`` hook — after
		``_release_gpu_memory`` (which clears the per-trial embedding instance)
		and immediately before the timed node-line loop. For each
		``_WARM_NODE_TYPES`` node we construct its module once exactly the way
		the timed run will (same class, same merged params — mirrors
		``BaseModule.run_evaluator``) and discard the instance: construction is
		what triggers the lazy loads, and the loaded weights survive the
		instance (embedding → registry ``LazyInit._instance``; reranker /
		compressor / FAISS → the run-spanning ``ModelCache``), so the timed
		run's own construction becomes a cheap cache hit. In measured deployment
		this is strict: a warm-up failure means the current trial's resolved
		placement cannot realize its in-process GPU stages, so continuing would
		let vLLM claim memory that belongs to those stages.
		"""
		import time as _time
		from rag_stack_evaluator.static_rag_evaluator.measured.aux_process import (
			process_isolated_stage,
		)
		for nodes in node_lines.values():
			for node in nodes:
				if node.stage not in _WARM_NODE_TYPES:
					continue
				if process_isolated_stage(str(node.stage)):
					# The stage's child process loads its own copy at stage
					# build (blocking, outside the timed window) and cannot
					# see parent-side caches. A parent-side warm load would
					# only pin a never-used twin of the model in parent VRAM
					# (embedding registry / ModelCache keep it alive all
					# trial) plus an extra CUDA context time-slicing a
					# co-resident GPU (r20: 016 retrieval-in-child measured
					# 3x its in-process service with the twin present).
					logger.info(
						f"measured warm-up: {node.stage} skipped — "
						f"loads inside its own worker process"
					)
					continue
				try:
					callables, params = node.get_param_combinations()
					cls, kwargs = callables[0], dict(params[0])
					t0 = _time.perf_counter()
					instance = cls(self._base.project_dir, **kwargs)
					del instance
					logger.info(
						f"measured warm-up: {node.stage} ({cls.__name__}) "
						f"ready in {_time.perf_counter() - t0:.1f}s "
						f"(model load excluded from timing)"
					)
				except Exception as e:  # noqa: BLE001 — see strict handling below
					logger.warning(
						f"measured warm-up failed for {node.stage}: {e} — "
						f"its first-load cost will be counted in that stage's time"
					)
					if strict:
						raise

	@staticmethod
	def _inject_cache_and_devices(
		node_lines: Dict[str, List[Node]],
		cache: Any,
		system_config: Dict[str, Any],
	) -> None:
		"""Set the module-level current cache and inject per-component devices
		into each node's module_param. The cache is intentionally NOT put into
		module_param (would break the downstream `str(dict) → eval()`
		round-trip that some serializers perform). Modules pull the cache from
		`rag_stack_evaluator.static_rag_evaluator.measured.cache.get_current()`.

		Placement is read directly from
		``system_config["placement_<component>"]`` (independently sampled
		per component by the upstream optimizer — cost-model-aligned design;
		deployment metadata is attached by the measured serving runtime).
		"""
		from rag_stack_evaluator.static_rag_evaluator.measured.cache import set_current as _set_current_cache
		from rag_stack.search_space.placement import GPU_STAGES
		from rag_stack.system_layout import (
			stage_device,
			stage_devices,
			retrieval_config,
			request_batch,
		)
		from rag_stack.runtime_parallelism import (
			validate_measured_parallelism_width,
		)

		# A historical ``system_config_resolved`` can enter this evaluator
		# directly, bypassing both normal search-space projection and
		# PerformanceContext's CM-replay overlay gate.  Validate the actual
		# selected node component against its resolved stage devices before
		# setting global cache state or mutating any module parameters.
		reranker_nodes = [
			node
			for nodes in node_lines.values()
			for node in nodes
			if str(node.stage) == "passage_reranker"
		]
		if reranker_nodes:
			reranker_devices = stage_devices(
				system_config, "passage_reranker",
			)
			if len(reranker_devices) > 1 and len(reranker_nodes) != 1:
				components = [
					str(getattr(getattr(node, "module", None), "component", ""))
					for node in reranker_nodes
				]
				raise NotImplementedError(
					"measured evaluator resolved system_config: multi-GPU "
					"passage_reranker requires exactly one active component; "
					f"found {components!r}"
				)
		for nodes in node_lines.values():
			for node in nodes:
				stage = str(node.stage)
				if stage not in {"semantic_retrieval", "passage_reranker"}:
					continue
				devices = stage_devices(system_config, stage)
				component = getattr(
					getattr(node, "module", None), "component", None,
				)
				validate_measured_parallelism_width(
					stage=stage,
					component=(str(component) if component else None),
					width=len(devices),
					source="measured evaluator resolved system_config",
				)
		_set_current_cache(cache)
		# Per-node GPU placement is DERIVED from the collocation partition (see
		# PerformanceContext.resolve_system_config → rag_stack/placement.py): each
		# GPU stage gets its first device via `placement_<stage>` and, for
		# data-parallel / sharded engines, the full device list via
		# `placement_<stage>_devices`. These are injected (overriding any YAML
		# device hint) so the executed deployment matches the derived layout.
		default_device = "cuda:0"
		retrieval = retrieval_config(system_config)
		batch_size_request = request_batch(system_config)
		# ONE global service-batch cap → the real per-node batch params. The
		# measured serving runtime derives a separate closed-loop load population
		# to saturate the deployment. GPU HF modules may also expose a forward
		# microbatch used only for executable tiling.
		# stage → list of (module_param, system_config_key) injections. ALL
		# sources are SYSTEM-space knobs (system.system_design_space), never
		# pipeline/algo-space module params — so they don't enter the quality
		# config hash (they affect performance only, not quality).
		batch_param_for = {
			"semantic_retrieval": [("embedding_batch", "batch_size_request")],
			"passage_compressor": [
				# QUERIES-per-batch: the global request batch (no-microbatch),
				# pooled per fused forward.
				("query_batch_size", "batch_size_request"),
				# CHUNKS-per-BERT-forward: the compressor's real GPU batch, a
				# SEPARATE system dim (`compressor_max_batch_size`) because the
				# feasible forward size is memory/collocation-bound, not a query
				# count. This is a hardware tile, NOT query microbatching.
				("max_batch_size", "compressor_max_batch_size"),
			],
		}
		for nodes in node_lines.values():
			for node in nodes:
				nt = node.stage
				if nt in GPU_STAGES:
					# Derived placement wins over any YAML `device` hint.
					node.module.module_param["device"] = stage_device(
						system_config, nt, default_device,
					)
					# Per-engine device list — only for the HF engines that
					# consume it (retrieval/reranker DP replicas, compressor). The
					# vLLM engines (generator/query_expansion) get their devices
					# via the deployment manager's vLLM key, not module_param.
					if nt in _DEVICE_LIST_NODE_TYPES:
						devs = stage_devices(system_config, nt)
						if devs:
							node.module.module_param["devices"] = list(devs)
				for param_name, sys_key in batch_param_for.get(nt, []):
					if sys_key == "batch_size_request":
						node.module.module_param[param_name] = int(batch_size_request)
					elif system_config.get(sys_key) is not None:
						node.module.module_param[param_name] = int(system_config[sys_key])
				if nt == "passage_reranker":
					node.module.module_param["batch"] = _measured_reranker_forward_batch(
						system_config,
						node,
						batch_size_request,
					)
				# System-level FAISS runtime knobs (system.retrieval) → the
				# retrieval module, which passes them to the store per query
				# (search-time only; never part of the index-build signature).
				if nt == "semantic_retrieval":
					for param_name, sys_key in (
						("num_threads", "faiss_num_threads"),
						("parallel_mode", "faiss_ivf_parallel_mode"),
					):
						if retrieval.get(sys_key) is not None:
							node.module.module_param[param_name] = int(retrieval[sys_key])

		# Pin the retrieval embedding to its derived GPU. The FAISS/vectordb is
		# CPU, so the encode stage owns the embedding device. Without this the
		# llama_index HuggingFace embedding defaults to cuda:0 and
		# collides with whatever vLLM engine sits there (a real OOM). Stamps the
		# registry so BOTH the ingest-time and query-time embedding builds land on
		# the right card.
		from rag_stack_evaluator.static_rag_evaluator.embedding.base import set_embedding_device
		set_embedding_device(stage_device(system_config, "semantic_retrieval", default_device))
