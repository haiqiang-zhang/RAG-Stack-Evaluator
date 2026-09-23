"""FlashRAG-backed GT quality evaluator — parallel to StaticRAGEvaluatorQualityOnly.

Implements the same :class:`~rag_stack_evaluator.base.BaseEvaluator` contract
so the Controller's quality-evaluator factory can swap between backends
based on YAML config. The class is responsible for:

  1. **Data preparation**: convert rag-stack parquet QA + corpus to FlashRAG
     JSONL format (cached by content hash via :mod:`dataset_adapter`).
  2. **Index orchestration**: ensure a FAISS index for the requested
     ``(retrieval_method, faiss_type)`` exists on disk before FlashRAG
     tries to load it (FlashRAG itself never auto-builds — see plan).
  3. **Config translation**: rag-stack pipeline config dict → FlashRAG
     ``Config`` dict (via :mod:`config_translator`).
  4. **Monitor wiring**: when ``flashrag.monitor`` is enabled, install a
     :class:`~rag_stack_evaluator.flashrag_quality_evaluator.monitor.Monitor` into
     FlashRAG's hook so call-site patches record per-query DAGs, normalized
     into the frozen canonical quality-trace envelope (see
     :mod:`rag_stack.rag_ir.trace`) and emitted under
     ``__execution_dag__`` in the result dict for the trace-driven cost model.
  5. **Evaluation**: run the FlashRAG pipeline, return a flat
     ``{metric_name: float}`` dict matching what the rag-stack optimizer
     consumes.

Side-effects are scoped to ``<project_dir>/_flashrag/`` so a static-backend
run never touches FlashRAG state.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional, Sequence

from rag_stack_evaluator.base import BaseEvaluator
from rag_stack_evaluator.flashrag_quality_evaluator import (
    config_translator,
    dataset_adapter,
    pipeline_factory,
)
from rag_stack_evaluator.flashrag_quality_evaluator.monitor import Monitor

logger = logging.getLogger("RAG-Stack")


def _method_kwargs_for(pipeline_cls: type, pipeline_config: dict) -> Dict[str, Any]:
    """Constructor kwargs for a pipeline class from ``pipeline_runtime``.

    The search space sweeps per-method scalar knobs (ircot ``max_iter``;
    flare ``threshold`` / ``look_ahead_steps`` / ``max_iter_num``; reasoning
    pipelines' ``max_retrieval_num``; …) into
    ``config["pipeline_runtime"]`` — but FlashRAG pipelines accept these
    ONLY as ``__init__`` kwargs (they do not read them from ``Config``).
    Match the runtime knobs against the constructor signature by name; a
    knob that matches nothing is logged loudly, because a silently dropped
    knob turns its whole search dimension into a no-op.
    """
    import inspect

    runtime = dict(pipeline_config.get("pipeline_runtime") or {})
    runtime.pop("mode", None)
    if not runtime:
        return {}
    params = inspect.signature(pipeline_cls.__init__).parameters
    kwargs: Dict[str, Any] = {}
    leftover = []
    for key, value in runtime.items():
        if key in params:
            # numpy scalars from the decoded config → plain python.
            kwargs[key] = value.item() if hasattr(value, "item") else value
        else:
            leftover.append(key)
    if leftover:
        logger.warning(
            f"pipeline_runtime knobs {sorted(leftover)} do not match any "
            f"{pipeline_cls.__name__}.__init__ parameter — they will have NO "
            f"effect. Accepted parameters: {sorted(params)[1:]}."
        )
    if kwargs:
        logger.info(
            f"Forwarding method knobs to {pipeline_cls.__name__}: {kwargs}"
        )
    return kwargs


class FlashRAGQualityEvaluator(BaseEvaluator):
    """Drop-in replacement for :class:`StaticRAGEvaluatorQualityOnly` that runs
    queries through FlashRAG's static / iterative / agentic pipelines.

    Construction mirrors the static evaluator so the Controller can swap
    instances transparently. Both evaluators own a shared
    :class:`~rag_stack_evaluator.static_rag_evaluator.dataset.DatasetEvalManager`, so the
    per-eval chunked corpus is resolved inside :meth:`evaluate` (not mutated
    in from outside).

    The Controller can pass ``flashrag_options`` (the ``eval_backend_setting.flashrag``
    YAML sub-block) to customize: pipeline_type, framework, generator_model,
    nprobe/ef_search, model2path, monitor mode, etc.
    """

    def __init__(
        self,
        dataset: Any = None,
        project_dir: Optional[str] = None,
        flashrag_options: Optional[Dict[str, Any]] = None,
        *,
        dataset_manager: Optional[Any] = None,
    ):
        if dataset_manager is not None:
            self._dataset = dataset_manager
        elif dataset is not None:
            from rag_stack_evaluator.static_rag_evaluator.dataset import (
                DatasetEvalManager,
            )

            pdir = project_dir if project_dir is not None else os.getcwd()
            self._dataset = DatasetEvalManager.from_dataset(dataset, pdir)
        else:
            raise ValueError(
                "FlashRAGQualityEvaluator requires either a dataset or a dataset_manager."
            )
        self.qa_data = self._dataset.qa_data
        self.corpus_data = self._dataset.corpus_data
        self.project_dir = self._dataset.project_dir
        os.makedirs(self.project_dir, exist_ok=True)

        self.flashrag_options: Dict[str, Any] = dict(flashrag_options or {})

        # Fail fast if flashrag isn't importable.
        try:
            import flashrag  # noqa: F401
        except ImportError as exc:
            from pathlib import Path

            dependency_path = Path(__file__).resolve().parents[2] / "FlashRAG"
            raise RuntimeError(
                "FlashRAG backend selected but `flashrag` is not importable. "
                "Initialize the evaluator's nested submodules and run "
                f"`python -m pip install -e '{dependency_path}[core]'` "
                "in the evaluator environment."
            ) from exc

    # ------------------------------------------------------------------
    # BaseEvaluator API.
    # ------------------------------------------------------------------

    def evaluate(
        self,
        config: dict,
        run_dir: Optional[str] = None,
        metrics_override: Optional[Sequence[str]] = None,
        on_trace_ready: Optional[Callable[[dict], None]] = None,
    ) -> dict:
        """Run one config through FlashRAG and return its quality metrics.

        Steps:
          1. Materialize JSONL (cached by corpus/QA content hash).
          2. Ensure FAISS index exists for ``(retrieval_method, faiss_type)``.
          3. Translate the rag-stack pipeline config to a FlashRAG
             ``Config`` dict.
          4. Install the Monitor (if ``flashrag.monitor != "off"``).
          5. Build + run the pipeline.
          6. Return ``{metric_name: float, "__execution_dag__": <envelope>}``
             where ``<envelope>`` is the canonical quality-trace envelope (one
             component-call trace per query, ``question_id`` = row position).

        The returned dict's metric keys are FlashRAG's native vocabulary
        (``em``, ``f1``, ``rouge-l``, etc.); the Controller's evaluator-
        agnostic code consumes them as-is.

        ``metrics_override`` replaces the configured scoring metrics without
        mutating the caller's configuration. ``on_trace_ready`` receives the
        same envelope object returned under ``__execution_dag__``. FlashRAG
        performs scoring inside ``pipeline.run``, so this callback runs after
        scoring. Callback failures are advisory, matching the static backend.
        """
        if metrics_override is not None:
            config = dict(config)
            eval_settings = dict(config.get("eval_backend_setting") or {})
            eval_settings["metrics"] = list(metrics_override)
            config["eval_backend_setting"] = eval_settings

        # Per-eval corpus: resolve THIS config's chunker and refresh our corpus
        # so the materialized JSONL + corpus_hash reflect the active chunking
        # (mirrors the static evaluator; chunk cache is hash-keyed → a cheap
        # cache hit when an owner already built it).
        chunker_params = (config.get("corpus_runtime") or {}).get("chunker") or {}
        corpus_view = self._dataset.resolve_corpus(chunker_params)
        self._dataset.activate(corpus_view)
        self.corpus_data = self._dataset.corpus_data

        # ------------------------------------------------------------------
        # 1. Materialize JSONL artifacts (one-time per content hash).
        # ------------------------------------------------------------------
        corpus_hash = dataset_adapter.corpus_hash(self.corpus_data)
        qa_hash = dataset_adapter.qa_hash(self.qa_data)
        dataset_name = self.flashrag_options.get("dataset_name") or f"qa_{qa_hash}"

        flashrag_root = os.path.join(self.project_dir, "_flashrag")
        data_dir = os.path.join(flashrag_root, "datasets")
        qa_dir = os.path.join(data_dir, dataset_name)
        qa_jsonl = os.path.join(qa_dir, "test.jsonl")
        corpus_dir = os.path.join(flashrag_root, "corpus", corpus_hash)
        corpus_jsonl = os.path.join(corpus_dir, "corpus.jsonl")

        dataset_adapter.materialize_qa_jsonl(self.qa_data, qa_jsonl)
        dataset_adapter.materialize_corpus_jsonl(self.corpus_data, corpus_jsonl)

        # ------------------------------------------------------------------
        # 2. Ensure FAISS index (pre-build if missing — FlashRAG cannot).
        # ------------------------------------------------------------------
        index_path = self._ensure_index_for_config(
            pipeline_config=config,
            corpus_jsonl=corpus_jsonl,
            corpus_hash=corpus_hash,
            indexes_root=os.path.join(flashrag_root, "indexes", corpus_hash),
        )

        # ------------------------------------------------------------------
        # 3. Translate rag-stack config → FlashRAG Config dict.
        # ------------------------------------------------------------------
        gt_eval = config.get("eval_backend_setting", {}) or {}
        # YAML-level flashrag opts can override per-instance defaults.
        merged_opts = {**self.flashrag_options, **(gt_eval.get("flashrag") or {})}

        # A-RAG's Keyword tool needs a BM25 index over the same corpus; build it
        # (cached) only for the a_rag pipeline so other methods pay nothing.
        bm25_index_path = None
        if config_translator._resolve_pipeline_type(config, merged_opts) == "a_rag":
            bm25_index_path = dataset_adapter.ensure_bm25_index(
                corpus_jsonl_path=corpus_jsonl,
                out_dir=os.path.join(flashrag_root, "indexes", corpus_hash),
            )

        if run_dir is None:
            run_dir = os.path.join(self.project_dir, "_flashrag_run")
        os.makedirs(run_dir, exist_ok=True)

        flashrag_dict = config_translator.translate(
            pipeline_config=config,
            qa_jsonl_path=qa_jsonl,
            corpus_jsonl_path=corpus_jsonl,
            index_path=index_path,
            dataset_name=dataset_name,
            data_dir=data_dir,
            run_dir=run_dir,
            gt_evaluation=gt_eval,
            flashrag_opts=merged_opts,
            model2path=merged_opts.get("model2path"),
            bm25_index_path=bm25_index_path,
        )

        # ------------------------------------------------------------------
        # 4. Install monitor (default "full"; "off" skips entirely).
        # ------------------------------------------------------------------
        monitor_mode = str(merged_opts.get("monitor", "full")).lower()
        monitor: Optional[Monitor] = Monitor() if monitor_mode != "off" else None
        if monitor is not None:
            try:
                from flashrag.monitor_hook import set_monitor
                set_monitor(monitor)
            except ImportError:
                logger.warning(
                    "flashrag.monitor_hook not found — falling back to no monitor. "
                    "The FlashRAG fork patches must be applied for DAG capture."
                )
                monitor = None

        # ------------------------------------------------------------------
        # 5. Build + run the pipeline.
        # ------------------------------------------------------------------
        try:
            from flashrag.config import Config
            from flashrag.utils import get_dataset

            cfg_obj = Config(config_dict=flashrag_dict)
            ds = get_dataset(cfg_obj)["test"]
            pipeline_type = config_translator._resolve_pipeline_type(config, merged_opts)
            pipeline_cls = pipeline_factory.build_pipeline_class(pipeline_type)
            pipeline = pipeline_cls(cfg_obj, **_method_kwargs_for(pipeline_cls, config))
            out_ds = pipeline.run(ds, do_eval=True)
        finally:
            # Always clear the monitor so subsequent standalone FlashRAG
            # callers (or our own next evaluate) start clean.
            if monitor is not None:
                try:
                    from flashrag.monitor_hook import set_monitor as _set
                    _set(None)
                except ImportError:
                    pass

        # ------------------------------------------------------------------
        # 6. Extract flat metrics + DAG.
        # ------------------------------------------------------------------
        result = _extract_flat_metrics(out_ds, requested=flashrag_dict.get("metrics", []))
        if monitor is not None:
            # Normalize the per-query execution DAGs into taxonomy-stage calls
            # and ship THE frozen canonical quality-trace envelope (one complete
            # component-call trace per query, question_id = row position).
            # This is the cost model's sole input contract — it replays the
            # traces and never inspects pipeline.mode.
            result["__execution_dag__"] = _quality_trace_envelope(
                [_dag_to_dict(d) for d in monitor.build_dags()]
            )
            if on_trace_ready is not None:
                try:
                    on_trace_ready(result["__execution_dag__"])
                except Exception as hook_exc:  # noqa: BLE001
                    logger.warning("on_trace_ready hook failed: %s", hook_exc)
        return result

    # ------------------------------------------------------------------
    # Internals.
    # ------------------------------------------------------------------

    def _ensure_index_for_config(
        self, *, pipeline_config: dict, corpus_jsonl: str, corpus_hash: str,
        indexes_root: str,
    ) -> str:
        """Pre-build the FAISS index for this config if it doesn't exist yet.

        Reads ``retrieval_method`` / ``faiss_type`` / ``embedding_model``
        from the same places :func:`config_translator._build_retriever_block`
        reads them so the cache key matches what FlashRAG will actually load.
        """
        gt_eval = pipeline_config.get("eval_backend_setting", {}) or {}
        opts = {**self.flashrag_options, **(gt_eval.get("flashrag") or {})}

        vdbs = pipeline_config.get("vectordb") or []
        if not vdbs:
            raise ValueError(
                "FlashRAG backend requires at least one `vectordb` block "
                "in the resolved pipeline config."
            )
        vdb = vdbs[0]

        retrieval_method = opts.get("retrieval_method") or vdb.get("embedding_model")
        if not retrieval_method:
            raise ValueError(
                "Could not infer retrieval_method for FlashRAG index build."
            )
        faiss_type = opts.get("faiss_type") or vdb.get("faiss_type", "Flat")

        model_path = None
        if opts.get("model2path") and retrieval_method in opts["model2path"]:
            model_path = opts["model2path"][retrieval_method]
        if not model_path:
            # Pass alias through; FlashRAG's index_builder will resolve via
            # its own model2path defaults or treat it as a direct HF path.
            model_path = str(retrieval_method)

        return dataset_adapter.ensure_index(
            corpus_jsonl_path=corpus_jsonl,
            retrieval_method=str(retrieval_method),
            faiss_type=str(faiss_type),
            model_path=str(model_path),
            out_dir=indexes_root,
            batch_size=int(opts.get("index_build_batch_size", 512)),
            use_fp16=bool(opts.get("index_build_fp16", False)),
            pooling_method=opts.get("retrieval_pooling_method"),
        )


# ---------------------------------------------------------------------------
# Helpers for converting FlashRAG output dataset → rag-stack-friendly dict.
# ---------------------------------------------------------------------------


def _safe_attr(obj: Any, name: str, default: Any = None) -> Any:
    """``getattr`` that also tolerates ``KeyError``.

    FlashRAG's ``Dataset``/``Item.__getattr__`` delegate unknown names to the
    item's ``data`` dict, which raises ``KeyError`` for a missing key —
    ``getattr``'s default only swallows ``AttributeError``.
    """
    try:
        return getattr(obj, name, default)
    except KeyError:
        return default


def _extract_flat_metrics(out_ds: Any, requested: list) -> Dict[str, float]:
    """Pull the dataset-level metric scores out of a FlashRAG ``Dataset``.

    FlashRAG's :class:`flashrag.evaluator.evaluator.Evaluator` writes a
    per-item ``output["metric_score"]`` dict (``update_evaluation_score``);
    the aggregated dict from ``BasicPipeline.evaluate`` is only printed, not
    attached to the dataset. So averaging the per-item dicts is the normal
    path; the dataset-level ``eval_result`` lookup is kept for forks that do
    attach it.
    """
    eval_result = _safe_attr(out_ds, "eval_result")
    if isinstance(eval_result, dict) and eval_result:
        return {
            k: float(v) for k, v in eval_result.items()
            if (not requested) or k in requested
        }

    # Normal path: average per-item metric_score dicts.
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for item in out_ds:
        ms = _safe_attr(item, "metric_score")
        if not isinstance(ms, dict):
            ms = (_safe_attr(item, "output", {}) or {}).get("metric_score") or {}
        for k, v in ms.items():
            if v is None:
                continue
            sums[k] = sums.get(k, 0.0) + float(v)
            counts[k] = counts.get(k, 0) + 1
    return {
        k: sums[k] / counts[k] for k in sums
        if (not requested) or k in requested
    }


def _dag_to_dict(dag: Any) -> Dict[str, Any]:
    """JSON-friendly serialization of an :class:`ExecutionDAG`."""
    return {
        "query_id": dag.query_id,
        "total_input_tokens": dag.total_input_tokens,
        "total_output_tokens": dag.total_output_tokens,
        "nodes": [
            {
                "node_type": n.node_type,
                "step_idx": n.step_idx,
                "model_id": n.model_id,
                "input_tokens": n.input_tokens,
                "output_tokens": n.output_tokens,
                "latency_ms": n.latency_ms,
                "extras": n.extras,
            }
            for n in dag.nodes
        ],
    }


def _quality_trace_envelope(monitor_dags: list) -> dict:
    """Monitor DAG dicts → THE frozen canonical quality-trace envelope.

    ``normalize_traces`` maps monitor event verbs to taxonomy stages (dense
    ``retrieve`` splits into encode + vectorsearch); ``question_id`` is the
    dataset row position. FlashRAG monitor events carry NO byte counts —
    bytes stay absent (the envelope keeps them optional; zero-filling would
    silently zero the CM's communication volume). The factory validates
    completeness (terminal generator call per query) before shipping.
    """
    from rag_stack.rag_ir import make_quality_trace_envelope
    from rag_stack_evaluator.flashrag_quality_evaluator.trace_adapter import normalize_traces

    traces = normalize_traces(monitor_dags)
    return make_quality_trace_envelope(
        traces,
        question_ids=[str(idx) for idx in range(len(traces))],
    )
