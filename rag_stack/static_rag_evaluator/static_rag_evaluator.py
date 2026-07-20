# Portions derived from AutoRAG (https://github.com/Marker-Inc-Korea/AutoRAG), Apache-2.0.
# Modified by the RAG-Stack authors for namespace and runtime integration; see LICENSE.autorag and NOTICE.

import logging
import json
import os
import shutil
import time
from copy import deepcopy
from dataclasses import dataclass
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Sequence

import pandas as pd
import yaml

from rag_stack.base_evaluator import BaseEvaluator
from rag_stack.static_rag_evaluator.cache_paths import faiss_index_root
from rag_stack.static_rag_evaluator.evaluation.generation import (
	GENERATION_METRIC_FUNC_DICT,
)
from rag_stack.static_rag_evaluator.evaluation.retrieval import (
	RETRIEVAL_METRIC_FUNC_DICT,
)
from rag_stack.static_rag_evaluator.evaluation.metric import (
	retrieval_token_recall,
	retrieval_token_precision,
	retrieval_token_f1,
)
from rag_stack.static_rag_evaluator.evaluation.util import cast_metrics
from rag_stack.static_rag_evaluator.node_line import run_node_line
from rag_stack.static_rag_evaluator.trace_builder import (
	build_static_execution_dag,
	build_static_fanout_dag,
	rag_ir_dynamic_enabled,
)
from rag_stack.static_rag_evaluator.nodes.lexicalretrieval.bm25 import bm25_ingest
from rag_stack.static_rag_evaluator.nodes.retrieval.base import get_bm25_pkl_name
from rag_stack.static_rag_evaluator.nodes.semanticretrieval.vectordb import (
	vectordb_ingest_api,
	filter_exist_ids,
	vectordb_ingest_huggingface,
)
from rag_stack.static_rag_evaluator.schema import Node
from rag_stack.static_rag_evaluator.schema.node import (
	module_type_exists,
	extract_values_from_nodes,
	extract_values_from_nodes_strategy,
)
from rag_stack.static_rag_evaluator.schema.metricinput import MetricInput
from rag_stack.static_rag_evaluator.utils.cast import cast_retrieved_contents
from rag_stack.static_rag_evaluator.utils.util import (
	convert_env_in_dict,
	get_event_loop,
	to_list,
)
from rag_stack.static_rag_evaluator.vectordb import load_vectordb

logger = logging.getLogger("RAG-Stack")

@dataclass
class PipelineRun:
	"""Output of the mode-agnostic pipeline core (:meth:`StaticRAGEvaluatorQualityOnly._run_pipeline`).

	Carries everything either mode needs after the shared run: the quality dict,
	plus the raw materials the CM path turns into an execution DAG
	(``previous_result`` / ``node_lines`` / ``token_stats``) and the measured path
	turns into a performance dict (``run_dir`` / ``node_lines`` / ``pipeline_wall_s``
	/ ``n_evaluated``). One struct = one shared pipeline run for both modes.
	"""
	quality: Dict[str, Any]
	previous_result: pd.DataFrame
	node_lines: Dict[str, List[Node]]
	run_dir: str
	pipeline_wall_s: float
	n_evaluated: int
	token_stats: Any = None
	# UNIFIED trace recorded at execution time by the orthogonal recording layer
	# (recording.TraceRecorder) — one ordered call list per query, mode-agnostic
	# (measured | quality-only) and shape-agnostic (sequential | agentic).
	# Quality paths carry THE canonical quality-trace envelope (question_id = dataset
	# row position; already fired through on_trace_ready). The measured path
	# carries the raw trace_v1 list — MeasuredEvaluator wraps it with
	# question_id = dataset idx + invocation_id = winner qid. None when
	# recording produced no calls.
	trace: Any = None
	# Per-query MEASURED timing from the service runner. Populated only by
	# MeasuredEvaluator; quality-only keeps using run_node_line / run_react.
	sequential_perf: Any = None


def _last_node_type_for(node_lines: Dict[str, List[Node]]) -> Optional[str]:
	"""Last node's type in insertion order — picks the generation-vs-retrieval
	metric dict for the sequential pipeline."""
	last_type: Optional[str] = None
	for nodes in node_lines.values():
		if nodes:
			last_type = nodes[-1].stage
	return last_type


def _publish_quality_trace(
	trace: Optional[List[List[dict]]],
	trace_qids: Sequence[Any],
	qa_rows: int,
	on_trace_ready: Optional[Callable[[dict], None]],
) -> Optional[dict]:
	"""Wrap a quality-run trace into THE frozen canonical envelope and fire the hook.

	The quality-only paths (sequential / react) record under the permanent
	``__qid__`` = dataset row position, so the recorder's emission qids ARE the
	stable question identities. Cardinality is the producer's responsibility:
	a recorded trace that does not cover every QA row is a harness bug and
	raises here instead of shipping a silently partial workload to the CM.

	``on_trace_ready`` (judge/CM overlap) receives the SAME envelope object
	this function returns — the controller identity-matches (``is``) the
	overlap result against ``quality["__execution_dag__"]``, so building a
	second envelope downstream would silently disable the overlap. The hook
	stays advisory: its errors never break the eval.

	Returns ``None`` when recording produced no calls (the caller falls back
	to the legacy post-hoc CM payloads).
	"""
	if not trace or not any(trace):
		return None
	if len(trace) != qa_rows:
		raise RuntimeError(
			"quality trace does not cover every dataset QA row: expected "
			f"{qa_rows} queries, recorded {len(trace)}"
		)
	from rag_stack.rag_ir import make_quality_trace_envelope

	envelope = make_quality_trace_envelope(
		trace,
		question_ids=[str(qid) for qid in trace_qids],
	)
	if on_trace_ready is not None:
		try:
			on_trace_ready(envelope)
		except Exception as hook_exc:  # noqa: BLE001 — advisory hook
			logger.warning(f"on_trace_ready hook failed: {hook_exc}")
	return envelope


class StaticRAGEvaluatorQualityOnly(BaseEvaluator):
	"""
	Evaluator that runs a single RAG pipeline configuration and evaluates the final output.

	Unlike the original Evaluator which optimizes across multiple module combinations,
	StaticRAGEvaluatorQualityOnly expects exactly one module per node with fixed parameters.
	It runs the pipeline end-to-end and evaluates only the final node's output
	using the top-level evaluation config.

	Scope & philosophy
	------------------
	This class produces **quality metrics only** (retrieval / generation scores).
	It is NOT a production runtime — its ``BaseModule.run_evaluator`` helper
	recreates a fresh module instance for every ``pure()`` call, which is
	convenient for isolated offline scoring but does not reflect how RAG is
	actually deployed.

	The cost model and its calibration framework
	(``rag_stack/cost_model/llm_sim/calibration/``) target the **real
	deployment scenario**: models are loaded once and amortized across many
	per-query inference calls. Calibration therefore times only ``pure()``
	(model load is a one-shot startup cost, excluded from the
	``(ceff, meff)`` fit). Do not conflate the two code paths — quality
	evaluator = quality, cost model = performance.
	"""

	def __init__(
		self,
		dataset: "Optional[GeneratedDataset]" = None,
		project_dir: Optional[str] = None,
		*,
		dataset_manager: "Optional[Any]" = None,
	):
		"""
		Initialize a StaticRAGEvaluatorQualityOnly.

		:param dataset: A GeneratedDataset containing QA and Corpus data. Used
		    to build a standalone :class:`DatasetManager` when none is supplied.
		:param project_dir: Path to the project directory. Default is the current directory.
		:param dataset_manager: optional shared :class:`DatasetManager` (owner
		    paths — Controller / QualityStore — pass theirs so the corpus and the
		    chunk cache are the SAME instance the cost-model path queries). When
		    omitted, one is built from ``dataset`` for standalone / test use.
		"""
		from rag_stack.static_rag_evaluator.dataset import DatasetManager

		if dataset_manager is not None:
			self._dataset = dataset_manager
		elif dataset is not None:
			pdir = project_dir if project_dir is not None else os.getcwd()
			self._dataset = DatasetManager.from_dataset(dataset, pdir)
		else:
			raise ValueError(
				"StaticRAGEvaluatorQualityOnly requires either a dataset or a dataset_manager."
			)
		# The manager owns the canonical qa/corpus + their data/ materialization.
		self.qa_data = self._dataset.qa_data
		self.corpus_data = self._dataset.corpus_data
		self.project_dir = self._dataset.project_dir
		if not os.path.exists(self.project_dir):
			os.makedirs(self.project_dir)


	def evaluate(
		self,
		config: dict,
		run_dir: str | None = None,
		metrics_override: Optional[List[str]] = None,
		on_trace_ready=None,
	) -> dict:
		"""Run a RAG pipeline configuration and return QUALITY metrics only.

		The **rag_stack canonical entry point**: GT produces quality, the cost
		model produces performance, and never the twain shall meet. Returns a flat
		dict of quality metric values (plus a CM-only ``__execution_dag__`` payload
		the controller pops before objectives).

		For real on-hardware performance measurement use
		:class:`rag_stack.static_rag_evaluator.measured.evaluator.MeasuredEvaluator`,
		which composes this evaluator's shared core via :meth:`_run_pipeline`.

		:param config: Parsed config dict (vectordb / node_lines / gt_evaluation).
		:param run_dir: Optional artifact dir. If None, uses ``project_dir/_static_run``
		    (cleared each call); when provided it is used as-is and NOT cleared.
		:param metrics_override: when given, compute ONLY these metrics instead of the
		    config's ``eval_backend_setting.metrics`` — lets the controller skip the expensive
		    LLM-judge sub-metrics on evals that don't need them (e.g. ax-gen steps, which only
		    need the objective metric). The combined-quality objective stays computable as long
		    as its metric(s) are included. Applied as a shallow config copy so the pipeline
		    still runs identically — only the scoring metric set changes.
		"""
		if metrics_override is not None:
			config = dict(config)
			ebs = dict(config.get("eval_backend_setting") or {})
			ebs["metrics"] = list(metrics_override)
			config["eval_backend_setting"] = ebs
		run = self._run_pipeline(
			config, run_dir=run_dir, on_trace_ready=on_trace_ready,
		)
		quality = run.quality
		# DEFAULT CM input: the UNIFIED recorded trace (recording.py) — call-level, mode-
		# agnostic, byte+token — already wrapped by ``_run_pipeline`` into THE canonical
		# quality-trace envelope (the same object the on_trace_ready hook saw, so
		# the controller's judge/CM overlap identity-match holds). Falls through
		# to the legacy paths below only when recording produced nothing.
		if run.trace is not None:
			quality["__execution_dag__"] = run.trace
			return quality
		# CM feedback payload (cost-model path ONLY): a trace_v1 execution DAG when
		# rag_ir_mode=dynamic, else the legacy aggregate fanout (static engine).
		# The measured path never builds this.
		execution_dag = None
		if rag_ir_dynamic_enabled(config):
			try:
				payload = build_static_execution_dag(
					run.previous_result,
					run.node_lines,
					config=config,
					token_stats=run.token_stats,
				)
				# rag_ir: hand the reconstructed trace to the controller as a canonical
				# envelope (question_id = dataset row position, one trace per
				# ``previous_result`` row) so the sequential path is priced by
				# the SAME trace-driven cost model (compute_trace_performance)
				# as the agentic (react) path — one comparable performance scale
				# across modes. Fall back to the aggregate dict (→
				# StaticAssembly) only when the trace is empty.
				traces = payload.get("trace_v1") if isinstance(payload, dict) else None
				if traces and any(traces):
					from rag_stack.rag_ir import make_quality_trace_envelope
					execution_dag = make_quality_trace_envelope(
						traces,
						question_ids=[str(idx) for idx in range(len(traces))],
					)
				else:
					execution_dag = payload
			except Exception as e:  # noqa: BLE001
				logger.warning(
					f"static rag_ir CM data build failed: {e}; "
					f"falling back to static CM data."
				)
		if execution_dag is None:
			execution_dag = build_static_fanout_dag(run.previous_result)
		if execution_dag is not None:
			quality["__execution_dag__"] = execution_dag
		return quality

	@staticmethod
	def _sweep_stale_gpu_engines() -> None:
		"""Deterministic per-eval GPU sweep (quality path only).

		The previous eval's in-process vLLM engines are torn down by
		``Vllm.__del__`` — which only runs when the GC finalizes the node
		graph (reference cycles → arbitrary timing), and vllm's ``del`` does
		not always stop the EngineCore subprocess. When finalization lags
		into the NEXT eval's engine load, engines co-reside and the trial
		dies with a spurious CUDA OOM recorded as INVALID — silently biasing
		the search. Force the boundary: collect now, reap ghost EngineCore
		children, and return freed memory to the driver.
		"""
		import gc
		gc.collect()
		try:
			from rag_stack.static_rag_evaluator.engine_slot import quality_engine_slot
			from rag_stack.static_rag_evaluator.nodes.generator.vllm import (
				_force_kill_engine_core_orphans,
			)
			# Spare the quality slot's live engine (adjacent-eval reuse);
			# everything else EngineCore-shaped is an orphan and dies.
			_force_kill_engine_core_orphans(
				keep_pids=quality_engine_slot().keep_pids()
			)
		except Exception:  # noqa: BLE001
			pass
		try:
			import torch
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
				torch.cuda.synchronize()
		except Exception:  # noqa: BLE001
			pass

	def _run_pipeline(
		self,
		config: dict,
		run_dir: str | None = None,
		*,
		n_queries: Optional[int] = None,
		before_run: Optional[Callable[[Dict[str, List[Node]]], None]] = None,
		before_timing: Optional[Callable[[Dict[str, List[Node]]], None]] = None,
		sequential_runner: Optional[Callable[..., tuple[pd.DataFrame, Any]]] = None,
		on_trace_ready: Optional[Callable[[dict], None]] = None,
	) -> PipelineRun:
		"""Mode-agnostic pipeline core shared by the quality and measured paths.

		Resolves the per-eval corpus + vectordb, parses node lines, ingests
		BM25/vectordb, runs the sequential pipeline, and scores the final result —
		returning everything either mode needs in a :class:`PipelineRun`. Carries NO
		mode flags: the only extension points are ``before_run`` / ``before_timing``.

		:param on_trace_ready: Optional judge/CM overlap hook (quality paths only).
		    Called with the canonical quality-trace ENVELOPE — the SAME object
		    :meth:`evaluate` later returns under ``quality["__execution_dag__"]``
		    (the controller matches the overlap result by identity). Advisory:
		    hook errors never break the eval.

		:param n_queries: If set, evaluate only the first ``n_queries`` QA rows
		    (restored afterwards). ``None`` evaluates all.
		:param before_run: Optional hook called with the parsed ``node_lines`` after
		    parsing and BEFORE ingest/run — the measured path uses it to inject the
		    ModelCache + derived device map into each node's module_param.
		:param before_timing: Optional hook called with ``node_lines`` AFTER
		    ``_release_gpu_memory`` and immediately BEFORE the timed pipeline loop —
		    the measured path uses it to pre-warm in-process GPU models (embedding /
		    reranker / compressor) so model-load time never lands inside the
		    measured window. Must run after the release (which clears the embedding
		    instance) or the warmed embedding would be cleared again.
		:param sequential_runner: Optional measured-only service runner for the
		    sequential mode. When omitted, the legacy quality node-line runner is
		    used unchanged.
		"""
		config = deepcopy(config)

		# Optional qa_data subsample for SMOKE / measured per-trial runs.
		_orig_qa_data = None
		_clear_trace_context = None
		if n_queries is not None and n_queries < len(self.qa_data):
			_orig_qa_data = self.qa_data
			self.qa_data = self.qa_data.head(n_queries).copy().reset_index(drop=True)
		try:
			# Quality mode builds fresh engines per eval — sweep the previous
			# eval's engines FIRST so they can never co-reside with this
			# eval's loads. (Measured mode owns engine lifetimes via the
			# ModelCache and always passes the hooks, so it never sweeps.)
			if sequential_runner is None and before_run is None:
				self._sweep_stale_gpu_engines()
			os.makedirs(os.path.join(self.project_dir, "resources"), exist_ok=True)
			os.environ["PROJECT_DIR"] = self.project_dir

			# Expand env vars AFTER PROJECT_DIR is set
			convert_env_in_dict(config)

			# Per-eval corpus: resolve THIS config's chunker and make it the active
			# corpus (chunk cache is hash-keyed → cheap hit when an owner already built
			# it; deterministic chunk_hash → FAISS path matches the cost-model path).
			chunker_params = (config.get("corpus_runtime") or {}).get("chunker") or {}
			logger.info("[pipeline] resolving corpus/chunk cache")
			corpus_view = self._dataset.resolve_corpus(chunker_params)
			self._dataset.activate(corpus_view)
			self.corpus_data = self._dataset.corpus_data
			chunk_hash = corpus_view.chunk_hash
			logger.info(
				f"[pipeline] corpus ready: chunk_hash={chunk_hash} "
				f"qa_rows={len(self.qa_data)} corpus_rows={len(self.corpus_data)}"
			)

			# Resolve vectordb paths (per-param-combo subdirs); inject chunk_hash so
			# different chunkings get separate FAISS index dirs.
			vectordb_configs = config.get("vectordb", [])
			# Resolve a relative IVF-PQ nlist_factor → concrete nlist now that
			# this eval's chunk count is known, BEFORE path resolution so the
			# cache subdir keys on the real nlist (mirrors the cost-model path's
			# DatasetManager.apply_corpus_view). No-op for absolute-nlist configs.
			self._dataset.resolve_nlist_factor(vectordb_configs, corpus_view.n_vectors)
			vectordb_configs = self._resolve_vectordb_paths(vectordb_configs, chunk_hash=chunk_hash)
			# Propagate the build-time OMP knob (system.retrieval.
			# faiss_indexing_thread) onto each faiss store so add_embedding
			# threads it through faiss_build_threads. Absent → None → cpu-2.
			_idx_threads = (
				(config.get("system") or {}).get("retrieval") or {}
			).get("faiss_indexing_thread")
			if _idx_threads is not None:
				for _vdb in vectordb_configs:
					_vdb.setdefault("faiss_indexing_thread", _idx_threads)
			vectordb_config_path = os.path.join(
				self.project_dir, "resources", "vectordb.yaml"
			)
			with open(vectordb_config_path, "w") as f:
				yaml.safe_dump({"vectordb": deepcopy(vectordb_configs)}, f)

			evaluation_config = config.get("eval_backend_setting", {})
			default_metrics = evaluation_config.get("metrics", [])

			# Parse node lines, adding default strategy for nodes without one
			node_lines = self._parse_node_lines(config, default_metrics)
			n_nodes = sum(len(nodes) for nodes in node_lines.values())
			logger.info(
				f"[pipeline] parsed {len(node_lines)} node line(s), "
				f"{n_nodes} node(s), run_dir={run_dir or '<auto>'}"
			)

			# Mode-specific pre-run hook (measured: inject ModelCache + device map).
			if before_run is not None:
				before_run(node_lines)

			# Ingest data into BM25 and VectorDB
			ingest_t0 = time.perf_counter()
			logger.info("[pipeline] ingest start")
			self._ingest_bm25(node_lines)
			self._ingest_vectordb(vectordb_configs, node_lines)
			logger.info(
				f"[pipeline] ingest complete in {time.perf_counter() - ingest_t0:.1f}s"
			)

			# Release embedding GPU memory before running the pipeline. Always released
			# (even in measured mode): the llama_index embedding's async wrapper binds an
			# active_span_id token to the loop it was created in, so reusing it across
			# trials crashes on a cross-Context token reset. A fresh embedding per trial
			# (~5-10s) is the accepted cost; vLLM/reranker/FAISS caches remain.
			self._release_gpu_memory()

			# Ensure qa/corpus parquet exist in project data dir (retrieval nodes read them)
			data_dir = os.path.join(self.project_dir, "data")
			os.makedirs(data_dir, exist_ok=True)
			qa_path = os.path.join(data_dir, "qa.parquet")
			corpus_path = os.path.join(data_dir, "corpus.parquet")
			if not os.path.exists(qa_path):
				self.qa_data.to_parquet(qa_path, index=False)
			# DatasetManager.activate() owns per-eval corpus materialization. Writing
			# the same 8.8M-row frame again here duplicated gigabytes of NFS I/O and
			# changed only parquet encoding/mtime, not benchmark data.
			if not os.path.isfile(corpus_path):
				raise RuntimeError(
					"DatasetManager.activate() did not materialize data/corpus.parquet"
				)

			# Create a run directory for execution artifacts
			if run_dir is None:
				run_dir = os.path.join(self.project_dir, "_static_run")
				if os.path.exists(run_dir):
					shutil.rmtree(run_dir)
			os.makedirs(run_dir, exist_ok=True)

			# Mode-specific pre-timing hook (measured: pre-warm in-process GPU
			# models so their load time stays OUT of the timed window below).
			if before_timing is not None:
				before_timing(node_lines)

			# Mode: sequential (default) or an agentic method. `rag_dataflow` is
			# the collaborative co-mode knob (one shared node_lines, one knob to
			# pick sequential vs react per trial); it takes precedence over the
			# block `mode`. For static_gt the only agentic method is `react`.
			_rt = config.get("pipeline_runtime") or {}
			mode = str(_rt.get("rag_dataflow") or _rt.get("mode", "sequential")).lower()
			# Orthogonal recording layer (mode-agnostic): bind a run-scoped recorder
			# and stamp a PERMANENT per-query id (row position) onto qa_data so each
			# node's model calls attribute their trace entry to the right query.
			# Cleared after the run. See static_rag_evaluator/recording.py.
			from rag_stack.static_rag_evaluator.recording import (
				TraceRecorder, set_current_recorder, clear_current_recorder)
			_clear_trace_context = clear_current_recorder
			_recorder = TraceRecorder()
			set_current_recorder(_recorder)
			if "__qid__" not in self.qa_data.columns:
				self.qa_data["__qid__"] = range(len(self.qa_data))
			_pipeline_start = time.perf_counter()
			sequential_perf = None
			if sequential_runner is not None:
				logger.info(
					f"[pipeline] mode={mode} — running measured service loop"
				)
				previous_result, sequential_perf = sequential_runner(
					config=config,
					node_lines=node_lines,
					run_dir=run_dir,
					qa_data=self.qa_data.copy(),
					# Pass the object explicitly: the measured runner has a
					# defensive ThreadPoolExecutor path, and contextvars do not
					# implicitly cross that thread boundary.
					trace_recorder=_recorder,
				)
			elif mode == "react":
				logger.info("[pipeline] mode=react (agentic) — running ReAct loop")
				# The quality-only ReAct loop records its trace into the SAME
				# run-scoped recorder as the sequential nodes (record_io).
				previous_result = self._run_react(config, node_lines, vectordb_configs)
			else:
				# Sequential pipeline: run each node line once, top-to-bottom.
				previous_result = self.qa_data.copy()
				for node_line_name, nodes in node_lines.items():
					node_line_dir = os.path.join(run_dir, node_line_name)
					os.makedirs(node_line_dir, exist_ok=True)
					line_t0 = time.perf_counter()
					node_names = " -> ".join(node.stage for node in nodes)
					logger.info(
						f"[pipeline] node_line={node_line_name} start: {node_names}"
					)
					previous_result = run_node_line(nodes, node_line_dir, previous_result)
					logger.info(
						f"[pipeline] node_line={node_line_name} complete in "
						f"{time.perf_counter() - line_t0:.1f}s rows={len(previous_result)}"
					)
			_pipeline_wall_s = time.perf_counter() - _pipeline_start

			# Collect the recorded trace (one ordered call list per query) and unbind.
			_trace_v1 = _recorder.to_trace_v1() or None
			_trace_qids = _recorder.qids
			clear_current_recorder()

			if sequential_runner is not None:
				# Measured path: raw trace_v1 rides on PipelineRun;
				# MeasuredEvaluator wraps it into the canonical envelope with
				# question_id = dataset idx and invocation_id = the winner
				# qid (the serving runtime already projected the recorder to
				# one first-completion winner per dataset row).
				_trace = _trace_v1
			else:
				# Quality paths ship THE frozen canonical envelope NOW — the trace
				# is FINAL here, so the quality-only judge/CM overlap hook may fire
				# with the SAME envelope object later returned via
				# ``quality["__execution_dag__"]`` (controller identity-match;
				# retries re-fire with a fresh envelope). Measured serving takes the
				# branch above and always finishes its timed runtime before scoring.
				_trace = _publish_quality_trace(
					_trace_v1, _trace_qids, len(self.qa_data), on_trace_ready,
				)

			last_node_type = _last_node_type_for(node_lines)

			# Evaluate final output with the top-level evaluation config
			metric_names = [
				m.get("metric_name") or m.get("metric")
				for m in default_metrics
				if isinstance(m, dict)
			]
			if evaluation_config.get("performance_only"):
				logger.info(
					"[pipeline] final scoring SKIPPED (performance_only)"
				)
				quality = {}
			else:
				logger.info(
					f"[pipeline] final scoring start: last_node={last_node_type} "
					f"metrics={metric_names}"
				)
				# Fingerprint dedup (M3): a HIT marker means the generator inherited
				# a donor's answers for byte-identical inputs — every quality metric
				# is a pure function of those inputs+answers, so inherit the donor's
				# scores wholesale and skip judging (the trace above is real: the
				# node re-recorded the generator calls at donor token counts).
				from rag_stack.static_rag_evaluator import gen_fingerprint as _gfp
				_hit = _gfp.read_marker(run_dir, "fp_hit.json") if run_dir else None
				_donor_quality = None
				if _hit:
					rec = _gfp.load_complete_record(
						os.environ.get("PROJECT_DIR", self.project_dir), _hit["fp"])
					if rec is not None:
						_donor_quality = dict(rec["quality"])
				if _donor_quality is not None:
					quality = _donor_quality
					quality["__fp_hit__"] = str(_hit.get("donor", "?"))
					logger.info(
						f"[pipeline] final scoring SKIPPED (fingerprint hit, "
						f"donor={_hit.get('donor', '?')}); quality inherited: "
						f"{ {k: round(v, 4) for k, v in _donor_quality.items() if isinstance(v, (int, float))} }"
					)
				else:
					quality = self._evaluate_final_result(previous_result, evaluation_config, last_node_type)
					_pend = _gfp.read_marker(run_dir, "fp_pending.json") if run_dir else None
					if _pend:
						_gfp.attach_quality(
							os.environ.get("PROJECT_DIR", self.project_dir),
							_pend["fp"], quality,
						)
			logger.info(
				f"[pipeline] final scoring complete: metrics={list(quality.keys())} "
				f"pipeline_wall={_pipeline_wall_s:.1f}s"
			)

			n_evaluated = len(self.qa_data)
			return PipelineRun(
				quality=quality,
				previous_result=previous_result,
				node_lines=node_lines,
				run_dir=run_dir,
				pipeline_wall_s=_pipeline_wall_s,
				n_evaluated=n_evaluated,
				token_stats=corpus_view.token_stats,
				trace=_trace,
				sequential_perf=sequential_perf,
			)
		finally:
			# Runtime/projection failures can occur before the normal trace
			# readout. Never leak that attempt's recorder into a retry.
			if _clear_trace_context is not None:
				_clear_trace_context()
			# Restore qa_data if we subsampled
			if _orig_qa_data is not None:
				self.qa_data = _orig_qa_data

	def _run_react(self, config, node_lines, vectordb_configs):
		"""Run the quality-only ReAct loop using AutoRAG's retriever + generator.

		Extracts the retriever (semantic_retrieval node) + generator (generator
		node) configs from the resolved node_lines, instantiates them directly
		(the vectordb is already ingested by the shared core), and drives the
		Thought/Action/Observation loop. Measured ReAct is served by
		``MeasuredServingRuntime`` via the ``sequential_runner`` hook.
		"""
		from rag_stack.static_rag_evaluator.agentic_react import run_react
		from rag_stack.static_rag_evaluator.nodes.semanticretrieval.vectordb import VectorDB

		all_nodes = [n for nodes in node_lines.values() for n in nodes]
		retr_node = next(
			(n for n in all_nodes if n.stage in ("semantic_retrieval", "hybrid_retrieval")),
			None,
		)
		gen_node = next((n for n in all_nodes if n.stage == "generator"), None)
		rerank_node = next((n for n in all_nodes if n.stage == "passage_reranker"), None)
		if retr_node is None or gen_node is None:
			raise ValueError(
				"react mode requires a semantic_retrieval node and a generator node."
			)

		# Retrieval knobs (top_k is node-level; vectordb/nprobe/ef_search on module).
		top_k = int(retr_node.node_params.get("top_k", 4))
		rmod = retr_node.module.module_param
		vdb_name = rmod.get("vectordb")
		nprobe = rmod.get("nprobe")
		ef_search = rmod.get("ef_search")
		emb_id = next(
			(vc.get("embedding_model") for vc in (vectordb_configs or [])
			 if vc.get("name") == vdb_name),
			None,
		)

		# Generator: sampling params (temperature/max_tokens) go to _pure; the rest
		# are engine kwargs (Vllm.__init__ separates them internally).
		gmod = dict(gen_node.module.module_param)
		model = gmod.get("model")
		gen_params = {
			k: gmod.get(k) for k in ("temperature", "max_tokens", "top_p")
			if gmod.get(k) is not None
		}
		engine_kwargs = {k: v for k, v in gmod.items() if k != "model"}
		_runtime = config.get("pipeline_runtime") or {}
		_mi = _runtime.get("max_iter")
		if _mi is None:
			raise ValueError(
				"react requires pipeline.max_iter (max Thought/Action rounds; "
				"react-only sub-config of the rag_dataflow knob) — "
				"max_agent_steps_safety was removed."
			)
		max_iter = int(_mi)

		reranker = None
		reranker_params = None
		reranker_model_id = None
		if rerank_node is not None:
			reranker_params = {
				**dict(rerank_node.node_params),
				**dict(rerank_node.module.module_param),
			}
			reranker_model_id = str(
				reranker_params.get("model")
				or reranker_params.get("model_name")
				or rerank_node.module.component
			)
			reranker = rerank_node.module.module(
				self.project_dir,
				**dict(rerank_node.module.module_param),
			)

		retriever = VectorDB(self.project_dir, vectordb=vdb_name)
		# Instantiate the generator component selected by the resolved node. ReAct
		# historically hard-coded the in-process ``Vllm`` implementation here,
		# which both bypassed ``vllm_api`` and leaked its server-only ``uri`` into
		# vLLM's EngineArgs. The ordinary sequential path already dispatches via
		# ``module.module``; keep the agentic path consistent with it.
		generator = gen_node.module.module(
			self.project_dir,
			model=model,
			**engine_kwargs,
		)
		try:
			return run_react(
				qa_data=self.qa_data,
				retriever=retriever,
				generator=generator,
				reranker=reranker,
				generator_model=str(model),
				embedding_model_id=str(emb_id or "embedding"),
				top_k=top_k,
				gen_params=gen_params,
				reranker_params=reranker_params,
				reranker_model_id=reranker_model_id,
				nprobe=nprobe,
				ef_search=ef_search,
				max_iter=max_iter,
			)
		finally:
			for obj in (generator, retriever, reranker):
				try:
					del obj
				except Exception:  # noqa: BLE001
					pass

	@staticmethod
	def _release_gpu_memory():
		"""Release GPU memory held by embedding models after vectordb ingestion."""
		import gc
		from rag_stack.static_rag_evaluator.embedding.base import embedding_models
		# Clear any cached embedding model instances
		for key, lazy in embedding_models.items():
			if hasattr(lazy, '_instance') and lazy._instance is not None:
				del lazy._instance
				lazy._instance = None
		gc.collect()
		try:
			import torch
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
		except ImportError:
			pass

	@staticmethod
	def _parse_node_lines(
		config: dict, default_metrics: List[str]
	) -> Dict[str, List[Node]]:
		"""
		Parse node lines from config dict into Node objects.

		If a node has no explicit ``strategy`` in the YAML, it is given an
		empty strategy so that per-node evaluation is skipped.  Only the
		final pipeline output is evaluated via ``gt_evaluation``.
		"""
		node_lines_config = config["node_lines"]
		node_line_dict = {}
		for node_line in node_lines_config:
			nodes = []
			for node_config in node_line["nodes"]:
				node_dict = deepcopy(node_config)
				if "strategy" not in node_dict:
					node_dict["strategy"] = {
						"metrics": [],
						"strategy": "mean",
					}
				nodes.append(Node.from_dict(node_dict))
			node_line_dict[node_line["node_line_name"]] = nodes
		return node_line_dict

	def _ingest_bm25(self, node_lines: Dict[str, List[Node]]):
		"""Ingest BM25 corpus if any node uses BM25 retrieval."""
		if not any(
			module_type_exists(nodes, "bm25") for nodes in node_lines.values()
		):
			return

		logger.info("Embedding BM25 corpus...")
		bm25_tokenizer_list = list(
			chain.from_iterable(
				self._find_bm25_tokenizer(nodes) for nodes in node_lines.values()
			)
		)
		if len(bm25_tokenizer_list) == 0:
			bm25_tokenizer_list = ["porter_stemmer"]

		for bm25_tokenizer in bm25_tokenizer_list:
			bm25_dir = os.path.join(
				self.project_dir, "resources", get_bm25_pkl_name(bm25_tokenizer)
			)
			os.makedirs(os.path.dirname(bm25_dir), exist_ok=True)
			bm25_ingest(bm25_dir, self.corpus_data, bm25_tokenizer=bm25_tokenizer)
		logger.info("BM25 corpus embedding complete.")

	# Keys in vectordb config that are metadata (not index structure params).
	# Note: ``embedding_model`` is intentionally NOT here — different models
	# produce different vectors, so they MUST live in different index dirs.
	# It is sanitized for path use in `_resolve_vectordb_paths`.
	_VECTORDB_META_KEYS = frozenset({
		"name", "db_type", "collection_name", "path",
		"N", "faiss_indexing_thread", "embedding_dtype_bytes",
	})

	def _ingest_vectordb(
		self, vectordb_configs: List[dict], node_lines: Dict[str, List[Node]]
	):
		"""Ingest VectorDB corpus if any node uses VectorDB retrieval.

		Each unique combination of index-structure parameters gets its own
		subdirectory under the base ``path``, so changing e.g. nlist/M triggers
		a rebuild while identical params reuse the cached index.
		"""
		if not any(
			module_type_exists(nodes, "vectordb") for nodes in node_lines.values()
		):
			return

		active_vectordb_configs = self._active_vectordb_configs(
			vectordb_configs, node_lines
		)
		if not active_vectordb_configs:
			return

		if self._cached_vectordb_ingest_ready(active_vectordb_configs, node_lines):
			logger.info(
				"[pipeline] vectordb ingest fast-skip: cached local FAISS index "
				"covers the active corpus"
			)
			return

		vectordb_list = self._load_all_vectordb(active_vectordb_configs)
		# _load_all_vectordb preserves order → zip to recover each store's embedding
		# model id (for the global embedding cache key, alongside dataset_name).
		for vectordb, vdb_cfg in zip(vectordb_list, active_vectordb_configs):
			loop = get_event_loop()
			target_corpus = loop.run_until_complete(
				filter_exist_ids(vectordb, self.corpus_data)
			)
			if vectordb.embedding.__class__.class_name() == "HuggingFaceEmbedding":
				vectordb_ingest_huggingface(
					vectordb, target_corpus,
					dataset_name=self._dataset.dataset_name,
					embedding_id=str(vdb_cfg.get("embedding_model") or "embedding"),
				)
			else:
				loop = get_event_loop()
				loop.run_until_complete(
					vectordb_ingest_api(vectordb, target_corpus)
				)

		# Piggyback: cache each IVF store's cell-imbalance for the cost model
		# (the index is already built here, so this is near-free). Once per
		# (corpus, embedding, nlist); fully guarded — never affects evaluation.
		self._save_imbalance_profiles(vectordb_list, active_vectordb_configs)

	def _cached_vectordb_ingest_ready(
		self, vectordb_configs: List[dict], node_lines: Dict[str, List[Node]]
	) -> bool:
		"""True when active local FAISS stores are already complete on disk.

		The old cached path still loaded each FAISS index and called
		``filter_exist_ids`` just to discover that every corpus id was present.
		For large corpora this dominates benchmark setup. Here we only inspect
		the active store's metadata; unknown/non-local stores fall back to the
		original ingest path.
		"""
		active = self._active_vectordb_configs(vectordb_configs, node_lines)
		if not active:
			return False
		expected_rows = len(self.corpus_data)
		return all(
			self._cached_faiss_vectordb_ready(vdb_cfg, expected_rows)
			for vdb_cfg in active
		)

	@staticmethod
	def _active_vectordb_configs(
		vectordb_configs: List[dict], node_lines: Dict[str, List[Node]]
	) -> List[dict]:
		if not vectordb_configs:
			return []
		names = set()
		for nodes in node_lines.values():
			for node in nodes:
				if node.module.component.lower() != "vectordb":
					continue
				vdb_name = node.module.module_param.get("vectordb")
				if isinstance(vdb_name, str):
					names.add(vdb_name)
				elif isinstance(vdb_name, list):
					names.update(str(v) for v in vdb_name if v)
		if not names:
			return [vectordb_configs[0]]
		matched = [cfg for cfg in vectordb_configs if cfg.get("name") in names]
		return matched

	@staticmethod
	def _cached_faiss_paths(vdb_cfg: dict) -> Optional[tuple[str, str, str]]:
		db_type = str(vdb_cfg.get("db_type") or "").lower()
		path = vdb_cfg.get("path")
		if not path:
			return None
		collection = str(vdb_cfg.get("collection_name") or "default")
		if db_type in {"faiss_ivf", "faissivf"}:
			suffix = "ivf"
		elif db_type in {"faiss_hnsw", "faisshnsw"}:
			suffix = "hnsw"
		else:
			return None
		return (
			os.path.join(path, f"{collection}.{suffix}.faiss"),
			os.path.join(path, f"{collection}.{suffix}.meta.json"),
			suffix,
		)

	@classmethod
	def _cached_faiss_vectordb_ready(
		cls, vdb_cfg: dict, expected_rows: int
	) -> bool:
		from rag_stack.static_rag_evaluator.vectordb._faiss_cache import (
			faiss_cache_build_lock,
			faiss_cache_metadata_if_ready,
		)

		paths = cls._cached_faiss_paths(vdb_cfg)
		if paths is None:
			return False
		index_path, meta_path, suffix = paths
		# The completion manifest is published last by the writer. Holding the
		# same per-key lock closes the check-vs-publish race without loading the
		# multi-gigabyte FAISS binary merely to skip an already-complete ingest.
		with faiss_cache_build_lock(str(vdb_cfg.get("path") or "")):
			meta = faiss_cache_metadata_if_ready(
				index_path,
				meta_path,
				expected_rows=int(expected_rows),
				# The subsequent retrieval node opens the same pair read-only.
				# Retain the parsed/normalized id map so it is not deserialized
				# a second time in this eval (or in later evals).
				process_cache=True,
			)
			if meta is None:
				return False
			if suffix == "ivf":
				if str(meta.get("index_type")) != str(
					vdb_cfg.get("index_type", "pq")
				):
					return False
				try:
					if int(meta.get("nlist")) != int(vdb_cfg.get("nlist")):
						return False
				except (TypeError, ValueError):
					return False
			elif suffix == "hnsw":
				for key in ("M", "ef_construction"):
					if key not in vdb_cfg:
						continue
					try:
						if int(meta.get(key)) != int(vdb_cfg.get(key)):
							return False
					except (TypeError, ValueError):
						return False
			return True

	def _save_imbalance_profiles(self, vectordb_list, vectordb_configs) -> None:
		"""Compute + cache the IVF cell-imbalance from each just-built index.

		Lets the cost model use the real data-aware scan count on CM-init /
		system-migration (it just LOADS this profile, no index build). Keyed by
		the shared ``imbalance_key`` (corpus × embedding, chunk-/nlist-agnostic)
		so the controller's load side finds it. Done once per (corpus, embedding,
		nlist) — the first config at a given nlist populates it, the rest reuse
		it across the chunk sweep. Best-effort: any failure is swallowed.
		"""
		try:
			import numpy as np
			import faiss
			from rag_stack.cost_model.faiss_ivf_sim.imbalance import (
				imbalance_key, load_imbalance_profile, save_imbalance_profile,
				compute_cell_imbalance, imbalance_anchor_nlist,
			)
		except Exception:  # noqa: BLE001
			return
		try:
			# Explicit, mandatory dataset name (config dataset.dataset_name), shared
			# via the DatasetManager — the SAME id the controller's _cm_corpus_id()
			# uses, so the load side finds this profile. Never path-derived.
			dataset_name = getattr(self._dataset, "dataset_name", "") or ""
			if not dataset_name:
				return
			q_texts = self.qa_data["query"].astype(str).tolist()[:512]
		except Exception:  # noqa: BLE001
			return
		if not q_texts:
			return
		NPROBES = [1, 4, 16, 64, 128, 256, 512, 1024]
		for vectordb, vdb_cfg in zip(vectordb_list, vectordb_configs):
			try:
				idx = getattr(vectordb, "index", None)
				nlist = int(getattr(vectordb, "nlist", 0) or 0)
				if idx is None or nlist <= 0:
					continue
				try:
					faiss.extract_index_ivf(idx)  # IVF indices only
				except Exception:  # noqa: BLE001
					continue
				key = imbalance_key(dataset_name, str(vdb_cfg.get("embedding_model") or ""))
				ivf = faiss.extract_index_ivf(idx)
				if nlist != imbalance_anchor_nlist(int(ivf.ntotal)):
					continue
				prof = load_imbalance_profile(key) or {}
				if prof.get("nlist"):
					continue  # single-curve policy: THE corpus×embedding curve exists
				loop = get_event_loop()
				q_vecs = np.asarray(
					loop.run_until_complete(
						vectordb.embedding.aget_text_embedding_batch(q_texts)),
					dtype="float32")
				f = compute_cell_imbalance(idx, q_vecs, [p for p in NPROBES if p <= nlist])
				prof.setdefault("d", int(ivf.d))
				prof.setdefault("N", int(ivf.ntotal))
				prof.setdefault("nlist", {})[str(nlist)] = {str(p): v for p, v in f.items()}
				save_imbalance_profile(key, prof)
				logger.info("[cm] cached cell-imbalance profile %s (nlist=%d)", key, nlist)
			except Exception as exc:  # noqa: BLE001
				logger.debug("[cm] imbalance profile skipped: %s", exc)

	@classmethod
	def _resolve_vectordb_paths(
		cls,
		vectordb_configs: List[dict],
		chunk_hash: str = "none",
	) -> List[dict]:
		"""Resolve each vectordb's on-disk location + index-param subdir.

		Index-structure params (everything except metadata keys, INCLUDING
		``embedding_model``) are sanitized + sorted + joined into a
		deterministic ``<param-sig>`` subdir under a leading ``<chunk_hash>``
		segment, so different chunkings/params never collide::

		    <root>/<chunk_hash>/M32_embedding_modelmpnet_nbits8_nlist1024

		The ROOT is content-addressed for faiss — ``faiss_index_root()``, the
		shared global cache, resolved WITHOUT reference to any per-project
		``path`` — and the declared project-local ``path`` for non-faiss
		stores (Chroma, …). A non-faiss store that declares no ``path`` is
		left unresolved. ``chunk_hash="none"`` is the no-chunking-sweep
		fallback.
		"""
		import re
		def _sanitize(v):
			# Model names contain '/' (e.g. BAAI/bge-large) — replace with '-'
			# so the path stays a single segment.
			return re.sub(r"[^A-Za-z0-9._-]+", "-", str(v))
		resolved = []
		for vdb in vectordb_configs:
			vdb = deepcopy(vdb)
			db_type = str(vdb.get("db_type") or "").lower().replace("_", "")
			# faiss indexes are content-addressed (chunk_hash + index params)
			# and byte-identical across runs/seeds, so they live under ONE
			# SHARED global root — resolved independently of any per-project
			# ``path`` (was project-local: ~38% cross-run duplication + a
			# rebuild per project). Non-faiss stores (Chroma, …) key off the
			# project-local ``path`` they declare, if any.
			root = (faiss_index_root()
			        if db_type in {"faissivf", "faisshnsw"}
			        else vdb.get("path"))
			if root is not None:
				param_parts = sorted(
					f"{k}{_sanitize(v)}" for k, v in vdb.items()
					if k not in cls._VECTORDB_META_KEYS
					and isinstance(v, (int, float, str, bool))
				)
				# chunk_hash segment always present so chunk-keyed indexes stay
				# separate; the param-signature subdir keys distinct builds.
				base = os.path.join(root, chunk_hash)
				vdb["path"] = os.path.join(base, "_".join(param_parts)) if param_parts else base
			resolved.append(vdb)
		return resolved

	@staticmethod
	def _load_all_vectordb(vectordb_configs: List[dict]):
		"""Load all vectordb instances from config dicts."""
		if not vectordb_configs:
			return []
		result = []
		for vdb_config in deepcopy(vectordb_configs):
			vdb_config.pop("name")
			db_type = vdb_config.pop("db_type")
			result.append(load_vectordb(db_type, **vdb_config))
		return result

	GENERATION_NODE_TYPES = {"generator"}
	RETRIEVAL_NODE_TYPES = {
		"semantic_retrieval", "dense_retrieval",
		"lexical_retrieval", "sparse_retrieval",
		"hybrid_retrieval",
		"passage_reranker", "passage_filter", "passage_augmenter",
	}

	# Unified metric dict — all gt_evaluation.metrics are dispatched against the
	# final pipeline DataFrame (which carries both retrieval and generation
	# columns), regardless of which stage the metric "logically" measures.
	# Each metric func introspects its required MetricInput fields; rows that
	# don't have them (e.g. retrieval_gt_contents missing from QA data) yield
	# None, which the aggregation step filters out.
	# Content-based retrieval metrics (compare text-token overlap, not UUIDs).
	# Survive chunker sweeps because they use retrieval_gt_contents (precomputed
	# from the original corpus) instead of retrieval_gt UUIDs.
	_RETRIEVAL_TOKEN_METRIC_FUNC_DICT: dict = {
		"retrieval_token_recall": retrieval_token_recall,
		"retrieval_token_precision": retrieval_token_precision,
		"retrieval_token_f1": retrieval_token_f1,
	}
	_ALL_METRIC_FUNC_DICT: dict = {
		**RETRIEVAL_METRIC_FUNC_DICT,
		**_RETRIEVAL_TOKEN_METRIC_FUNC_DICT,
		**GENERATION_METRIC_FUNC_DICT,
	}

	def _evaluate_final_result(
		self, final_result: pd.DataFrame, evaluation_config: dict, last_node_type: Optional[str] = None
	) -> dict:
		"""
		Evaluate the final pipeline output using the top-level evaluation section.
		All metrics in `gt_evaluation.metrics` are dispatched against the final
		DataFrame. A retrieval-side metric (e.g. ``retrieval_recall``) finds the
		retrieved-id columns the pipeline left behind; a generation-side metric
		(``bleu``, ``deepeval_*``) finds ``generated_texts``. Metrics whose
		required MetricInput fields are absent simply produce ``None`` and are
		filtered out at aggregation time.

		:param final_result: The DataFrame output from the last node in the pipeline.
		:param evaluation_config: Dict with 'metrics' (list) and 'strategy' (str).
		:param last_node_type: kept for backward-compat with callers; no longer
		    used to select metric dict (the dict is unified).
		:return: Dict of {metric_name: aggregated_score}.
		"""
		# performance_only: first-class perf-only switch for measured replays /
		# benchmarks — the serving protocol (closed loop, 100-QA gate, caps) is
		# unaffected; ONLY scoring is skipped. Optimize runs must not set it
		# (quality objective would go empty — validator rejects).
		if evaluation_config.get("performance_only"):
			return {}
		metrics = evaluation_config.get("metrics", [])
		strategy = evaluation_config.get("strategy", "mean")

		if not metrics:
			return {}

		metric_func_dict = self._ALL_METRIC_FUNC_DICT
		metric_inputs = self._create_metric_inputs(final_result, evaluation_config)
		metric_names, metric_params = cast_metrics(metrics)

		# DeepEval metrics already manage per-test-case async execution via
		# DeepEval's AsyncConfig. Run those metric batches sequentially at the
		# rag-stack level so multiple LLM-judge metrics do not multiply provider
		# concurrency. Cheap local metrics can still run in parallel.
		from concurrent.futures import ThreadPoolExecutor

		local_jobs = []
		llm_judge_jobs = []
		for metric_name, metric_param in zip(metric_names, metric_params):
			if metric_name not in metric_func_dict:
				logger.warning(
					f"Metric '{metric_name}' is not registered. "
					f"Available: {sorted(metric_func_dict.keys())}. Skipping."
				)
				continue
			job = (metric_name, metric_param)
			if metric_name.startswith("deepeval_"):
				llm_judge_jobs.append(job)
			else:
				local_jobs.append(job)

		# Sub-metric query subset: the EXPENSIVE deepeval SUB-metrics (everything except the
		# combined-quality objective) are diagnostic context the agent reads as directional
		# hints, so a query subset gives a fine estimate at a fraction of the judge cost. The
		# objective metric + the free local metrics always run on ALL queries.
		obj_metric_names = set((evaluation_config.get("combined_quality") or {}).get("metrics") or [])
		subset_n = int(evaluation_config.get("submetric_query_subset") or 0)

		def _run(job):
			name, param = job
			mi = metric_inputs
			if (subset_n > 0 and name.startswith("deepeval_")
					and name not in obj_metric_names and len(metric_inputs) > subset_n):
				mi = metric_inputs[:subset_n]
			return name, metric_func_dict[name](metric_inputs=mi, **param)

		def _record(name, scores):
			valid_scores = [
				s for s in scores
				if s is not None and not (isinstance(s, float) and s != s)
			]
			all_scores[name] = (
				sum(valid_scores) / len(valid_scores)
				if (strategy == "mean" and valid_scores) else None
			)

		all_scores = {}
		if local_jobs:
			with ThreadPoolExecutor(max_workers=len(local_jobs)) as ex:
				for name, scores in ex.map(_run, local_jobs):
					_record(name, scores)
		if llm_judge_jobs:
			logger.info(
				f"Evaluating {len(llm_judge_jobs)} DeepEval metric(s) "
				"sequentially; DeepEval owns per-case concurrency."
			)
			for job in llm_judge_jobs:
				name, scores = _run(job)
				_record(name, scores)

		return all_scores

	def _create_metric_inputs(self, final_result: pd.DataFrame, evaluation_config: dict) -> List[MetricInput]:
		"""
		Create MetricInput objects by combining the QA ground truth data
		with the final pipeline output.

		Retrieval ground truth is TEXT, carried in ``retrieval_gt_contents``
		(resolved by the dataset loader to ``references`` when present, else
		``generation_gt`` — the answer). Token-overlap metrics
		(``retrieval_token_*``) score it against the retrieved chunk text, so it
		is chunker-invariant — no chunk-UUID ``retrieval_gt`` and no per-eval
		ref→UUID remapping (both removed when chunk_size became a search-space
		dimension). ``references`` is also kept on MetricInput for metrics that
		want the raw evidence text directly.
		"""
		gt_cols = [
			c for c in [
				"query", "generation_gt",
				"retrieval_gt_contents", "references", "keypoints",
			] if c in self.qa_data.columns
		]
		merged = self.qa_data[gt_cols].copy()
		# Ensure list-type columns are plain lists (not numpy arrays)
		for col in ["generation_gt", "retrieval_gt_contents", "references", "keypoints"]:
			if col in merged.columns:
				merged[col] = merged[col].apply(
					lambda x: x.tolist() if hasattr(x, "tolist") else x
				)

		if "generated_texts" in final_result.columns:
			merged["generated_texts"] = final_result["generated_texts"].values
		# ID source for retrieval_recall etc.: prefer the pre-rerank semantic
		# IDs so the metric reports pure retriever quality (pre-rerank stages
		# only). Falls back to post-rerank retrieved_ids when semantic stage
		# didn't run (e.g. pure-lexical pipelines).
		if "retrieved_ids_semantic" in final_result.columns:
			merged["retrieved_ids"] = final_result["retrieved_ids_semantic"].values
		elif "retrieved_ids" in final_result.columns:
			merged["retrieved_ids"] = final_result["retrieved_ids"].values

		# Auto-collect retrieved_contents using the same cast logic as all other nodes
		try:
			merged["retrieved_contents"] = cast_retrieved_contents(final_result)
		except ValueError:
			pass  # no retrieval columns — metrics requiring context will get None

		# Preserve the pre-reranker / pre-compressor semantic-retrieval output so
		# retrieval_token_* (and pre-rerank retrieval_recall) can isolate raw
		# retriever quality. The rag stack assumes EVERY pipeline runs semantic
		# retrieval, so this column is an INVARIANT here: semantic_retrieval emits
		# `retrieved_contents_semantic` and every downstream node preserves it
		# (concat-based) — the only node that ever dropped it was a retrieval-side
		# node's drop_retrieval_columns, now fixed to drop only the unsuffixed
		# names. Assert rather than soft-skip: a missing column means a node
		# regressed that contract, and silently skipping nulls retrieval_token_*
		# which then aggregates to a misleading 0.0 in all_evaluations.csv.
		assert "retrieved_contents_semantic" in final_result.columns, (
			"retrieved_contents_semantic missing from the final pipeline result. "
			"Every pipeline is assumed to run semantic retrieval, and that column "
			"must survive to evaluation for retrieval_token_*/pre-rerank "
			"retrieval_recall. A retrieval-side node likely dropped it — check "
			"drop_retrieval_columns (utils/cast.py) and any node that rebuilds the "
			f"result frame. Present columns: {sorted(final_result.columns)}"
		)
		merged["retrieved_contents_semantic"] = final_result["retrieved_contents_semantic"].values

		return MetricInput.from_dataframe(merged)

	@staticmethod
	def _find_bm25_tokenizer(nodes: List[Node]):
		bm25_tokenizer_list = extract_values_from_nodes(nodes, "bm25_tokenizer")
		strategy_tokenizer_list = list(
			chain.from_iterable(
				extract_values_from_nodes_strategy(nodes, "bm25_tokenizer")
			)
		)
		return list(set(bm25_tokenizer_list + strategy_tokenizer_list))
