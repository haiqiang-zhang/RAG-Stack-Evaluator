"""rag-stack YAML dict → FlashRAG ``Config`` dict.

The translator is a **pure function**: given a rag-stack pipeline config
(already resolved by the search-space adapter — no sweep specs left) plus
the prepared JSONL / index paths, produce a dict that FlashRAG's
:class:`flashrag.config.Config` can consume.

Design notes:

  * Metrics are passed through unchanged — configs with ``backend: flashrag``
    must use FlashRAG's native metric vocabulary (``em``, ``f1``, ``rouge-l``,
    ``llm_judge``, etc.). The translator validates against FlashRAG's
    supported set and raises on unknowns.
  * Pipeline type comes from ``pipeline_runtime.mode`` (set by the search-
    space resolver) or falls back to ``eval_backend_setting.flashrag.pipeline_type``,
    finally to ``sequential``.
  * vectordb search-time knobs (``nprobe`` / ``ef_search``) require the
    FlashRAG fork patch in :meth:`DenseRetriever.load_index`; they're
    forwarded verbatim into the ``Config`` dict and applied at retriever
    load time.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Sequence

logger = logging.getLogger("RAG-Stack")

# FlashRAG's supported metric names (see flashrag/evaluator/metrics.py).
_FLASHRAG_METRICS: frozenset = frozenset({
    "em", "sub_em", "f1", "acc", "recall", "precision",
    "bleu", "rouge_score", "rouge-1", "rouge-2", "rouge-l",
    "llm_judge", "input_tokens", "retrieval_recall", "retrieval_precision",
})
_METRIC_ALIASES = {"rouge_l": "rouge-l"}


def supported_metrics() -> tuple[str, ...]:
    """Return accepted metric names and aliases without importing FlashRAG."""
    return tuple(sorted(_FLASHRAG_METRICS | _METRIC_ALIASES.keys()))


def _normalize_metric_name(name: str) -> str:
    """Allow ``rouge_l`` as a typo-friendly alias for ``rouge-l``.

    FlashRAG itself accepts ``rouge-l`` only; we forgive the underscore form
    that's natural in YAML and rewrite it to the canonical hyphenated form.
    """
    return _METRIC_ALIASES.get(name, name)


def _resolve_metrics(gt_evaluation: dict) -> list:
    """Pull the metrics list out of ``gt_evaluation`` and validate it."""
    raw = gt_evaluation.get("metrics") or []
    names: list = []
    for m in raw:
        if isinstance(m, dict):
            n = m.get("metric_name")
        else:
            n = m
        if not n:
            continue
        n = _normalize_metric_name(str(n))
        if n not in _FLASHRAG_METRICS:
            raise ValueError(
                f"Metric '{n}' is not supported by the FlashRAG backend. "
                f"Supported metrics: {sorted(_FLASHRAG_METRICS)}. "
                f"Configs with `global.eval_backend: flashrag` must use "
                f"FlashRAG's native metric vocabulary."
            )
        names.append(n)
    return names


def _resolve_pipeline_type(
    pipeline_config: dict, flashrag_opts: dict,
) -> str:
    """Decide which FlashRAG pipeline class to build for this eval.

    Order of precedence:
      1. ``pipeline_runtime.mode`` (set by the search-space resolver — this
         is what optimizer-suggested configs carry).
      2. ``eval_backend_setting.flashrag.pipeline_type`` (explicit YAML default).
      3. ``"sequential"``.
    """
    runtime_mode = (pipeline_config.get("pipeline_runtime") or {}).get("mode")
    if runtime_mode:
        return str(runtime_mode)
    explicit = flashrag_opts.get("pipeline_type")
    if explicit:
        return str(explicit)
    return "sequential"


def _build_retriever_block(
    pipeline_config: dict,
    index_path: str,
    corpus_jsonl_path: str,
    flashrag_opts: dict,
) -> Dict[str, Any]:
    """Map rag-stack vectordb + semantic_retrieval to FlashRAG retriever keys.

    rag-stack pipelines that use ``backend: flashrag`` must keep their
    retriever config minimal — one ``vectordb`` block + one
    ``semantic_retrieval`` node. The translator picks values from both and
    folds them into FlashRAG's flat retrieval-keys vocabulary.
    """
    # Search the (already-resolved) vectordb block for retrieval params.
    vdbs = pipeline_config.get("vectordb") or []
    if not vdbs:
        raise ValueError(
            "FlashRAG backend requires at least one `vectordb` block "
            "to derive embedding_model + index parameters."
        )
    vdb = vdbs[0]

    retriever: Dict[str, Any] = {
        "index_path": index_path,
        "corpus_path": corpus_jsonl_path,
    }

    # retrieval_method: prefer explicit yaml override, else use the vectordb's
    # embedding_model alias (which gets resolved to a HF path via model2path).
    retrieval_method = flashrag_opts.get("retrieval_method") or vdb.get(
        "embedding_model"
    )
    if not retrieval_method:
        raise ValueError(
            "Could not infer FlashRAG `retrieval_method`. Set "
            "`eval_backend_setting.flashrag.retrieval_method` or provide "
            "`vectordb[0].embedding_model`."
        )
    retriever["retrieval_method"] = str(retrieval_method)

    # retrieval_topk from semantic_retrieval node.
    for node_line in pipeline_config.get("node_lines") or []:
        for node in node_line.get("nodes") or []:
            if node.get("stage") == "semantic_retrieval":
                if "top_k" in node:
                    retriever["retrieval_topk"] = int(node["top_k"])
                break

    # Search-time knobs (require the fork patch in DenseRetriever.load_index).
    for k in ("nprobe", "ef_search"):
        if vdb.get(k) is not None:
            retriever[k] = vdb[k]
        if flashrag_opts.get(k) is not None:
            retriever[k] = flashrag_opts[k]

    # faiss_type is a build-time param (not consumed at load time by FlashRAG)
    # but we forward it anyway in case downstream callers inspect Config.
    if vdb.get("faiss_type") is not None:
        retriever["faiss_type"] = vdb["faiss_type"]
    if flashrag_opts.get("faiss_type") is not None:
        retriever["faiss_type"] = flashrag_opts["faiss_type"]

    return retriever


def _build_generator_block(
    pipeline_config: dict, flashrag_opts: dict,
) -> Dict[str, Any]:
    """Map rag-stack generator node to FlashRAG generator keys.

    The semantics we want from FlashRAG:
      * Pick a generator model (HuggingFace path or alias).
      * Pick a backend framework (``vllm`` / ``openai`` / ``hf`` / ``fschat``).
      * Pass generation params (``temperature``, ``max_new_tokens``, ...).

    Defaults: when ``framework`` is omitted under an agentic pipeline type,
    we suggest ``vllm`` to the user (vLLM is dramatically faster for the
    step-batched agentic loop). Otherwise FlashRAG's own default applies.
    """
    gen: Dict[str, Any] = {}

    # Walk node_lines for the generator module.
    for node_line in pipeline_config.get("node_lines") or []:
        for node in node_line.get("nodes") or []:
            if node.get("stage") != "generator":
                continue
            modules = node.get("modules") or []
            if not modules:
                continue
            mod = modules[0]
            component = str(mod.get("component", ""))
            # Module type → framework alias.
            framework_map = {
                "vllm": "vllm",
                "openai_llm": "openai",
                "openai": "openai",
                "huggingface": "hf",
                "hf": "hf",
                "fschat": "fschat",
            }
            if component in framework_map:
                gen["framework"] = framework_map[component]
            if "model" in mod:
                gen["generator_model"] = str(mod["model"])
            # API generator endpoint/credentials. FlashRAG's OpenaiGenerator
            # builds its client from ``config["openai_setting"]`` — without
            # base_url it talks to api.openai.com, which 400s on any
            # non-OpenAI model id (e.g. an OpenRouter model).
            if framework_map.get(component) == "openai":
                setting: Dict[str, Any] = {}
                if mod.get("api_key") is not None:
                    setting["api_key"] = str(mod["api_key"])
                if mod.get("base_url") is not None:
                    setting["base_url"] = str(mod["base_url"])
                if setting:
                    gen["openai_setting"] = setting
            # Generation params get collected under generation_params.
            gp: Dict[str, Any] = {}
            for k in ("temperature", "top_p", "top_k", "max_tokens", "max_new_tokens"):
                if mod.get(k) is not None:
                    gp[k] = mod[k]
            if gp:
                gen["generation_params"] = gp

    # Yaml-level overrides win.
    for k in (
        "framework", "generator_model", "generator_batch_size",
        "gpu_memory_utilization", "max_retrieval_num", "openai_setting",
        "generator_max_input_len",
    ):
        if flashrag_opts.get(k) is not None:
            gen[k] = flashrag_opts[k]

    return gen


def translate(
    *,
    pipeline_config: dict,
    qa_jsonl_path: str,
    corpus_jsonl_path: str,
    index_path: str,
    dataset_name: str,
    data_dir: str,
    run_dir: str,
    gt_evaluation: dict,
    flashrag_opts: dict,
    model2path: Optional[Dict[str, str]] = None,
    bm25_index_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a FlashRAG ``Config`` dict ready for :class:`flashrag.config.Config`.

    Args:
        pipeline_config: rag-stack resolved pipeline config (post-search-space).
        qa_jsonl_path: ``<data_dir>/<dataset_name>/test.jsonl``.
        corpus_jsonl_path: corpus JSONL written by :func:`dataset_adapter.materialize_corpus_jsonl`.
        index_path: pre-built FAISS index file (from :func:`dataset_adapter.ensure_index`).
        dataset_name: matches the directory name under ``data_dir``.
        data_dir: parent dir of the dataset directory.
        run_dir: where FlashRAG should write its run artifacts.
        gt_evaluation: ``config["eval_backend_setting"]`` block (metrics + flashrag opts).
        flashrag_opts: ``eval_backend_setting.flashrag`` sub-block.
        model2path: optional alias → HF path overrides forwarded to FlashRAG.

    Returns:
        A flat dict consumable by ``Config(config_dict=...)``.
    """
    retriever = _build_retriever_block(
        pipeline_config, index_path, corpus_jsonl_path, flashrag_opts,
    )
    generator = _build_generator_block(pipeline_config, flashrag_opts)
    metrics = _resolve_metrics(gt_evaluation)

    cfg: Dict[str, Any] = {
        "data_dir": data_dir,
        "dataset_name": dataset_name,
        "split": ["test"],
        "test_sample_num": None,   # use the full materialized JSONL
        "disable_save": True,
        "save_dir": run_dir,
        "metrics": metrics,
        **retriever,
        **generator,
    }

    # A-RAG's Keyword tool: the BM25 index dir, read by ARAGPipeline to build a
    # BM25Retriever alongside the dense one. Absent → Keyword tool disabled.
    if bm25_index_path:
        cfg["bm25_index_path"] = str(bm25_index_path)

    # adaptive_rag needs a judger: the Adaptive-RAG query-complexity
    # classifier (a T5 Seq2SeqLM emitting A/B/C). FlashRAG's AdaptiveJudger
    # reads cfg["judger_name"] + cfg["judger_config"]["model_path"] — the
    # checkpoint is an external artifact, so the path MUST come from YAML.
    if _resolve_pipeline_type(pipeline_config, flashrag_opts) == "adaptive_rag":
        judger_path = flashrag_opts.get("adaptive_judger_model_path")
        if not judger_path:
            raise ValueError(
                "pipeline mode 'adaptive_rag' requires "
                "eval_backend_setting.flashrag.adaptive_judger_model_path — the "
                "Adaptive-RAG query-complexity classifier checkpoint (T5 "
                "Seq2SeqLM scoring options A/B/C; see the Adaptive-RAG "
                "paper's released classifier or train one per the repo)."
            )
        cfg["judger_name"] = "adaptive"
        cfg["judger_config"] = {
            "model_path": str(judger_path),
            "batch_size": int(flashrag_opts.get("adaptive_judger_batch_size", 16)),
        }

    if model2path:
        cfg["model2path"] = dict(model2path)
    _ensure_local_generator_path(cfg)

    return cfg


def _ensure_local_generator_path(cfg: Dict[str, Any]) -> None:
    """Map an HF-id generator onto its local snapshot via ``model2path``.

    FlashRAG's ``get_generator`` opens ``<generator_model_path>/config.json``
    directly, so every non-openai framework needs ``generator_model`` to
    resolve to a real local directory (FlashRAG resolves it through
    ``model2path``). The validator's model-cache check snapshot-downloads
    configured models up front, so ``local_files_only`` resolution succeeds
    here without touching the network.
    """
    framework = cfg.get("framework")
    model = cfg.get("generator_model")
    if framework == "openai" or not model or os.path.isdir(model):
        return
    m2p = cfg.setdefault("model2path", {})
    if model in m2p:
        return
    try:
        from huggingface_hub import snapshot_download
        m2p[model] = snapshot_download(model, local_files_only=True)
        logger.info(f"Resolved generator {model!r} -> {m2p[model]}")
    except Exception as e:  # noqa: BLE001
        logger.warning(
            f"generator model {model!r} is neither a local directory nor a "
            f"cached HF snapshot ({e}); FlashRAG will fail to load it under "
            f"framework={framework!r}. Pre-download it or add a model2path "
            f"entry pointing at a local copy."
        )
