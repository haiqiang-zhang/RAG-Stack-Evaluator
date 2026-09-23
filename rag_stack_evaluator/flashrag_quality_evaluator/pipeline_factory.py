"""Map a ``pipeline_type`` string to a FlashRAG pipeline class.

Dead-simple dispatcher. Adding a new pipeline mode is one line in
:data:`PIPELINE_MAP`. The factory does NOT instantiate the pipeline — it
just returns the class so the caller can construct it with whatever
generator/retriever/prompt_template overrides it has in scope (including
the FlashRAG fork's no-op monitor hooks).
"""

from __future__ import annotations

import importlib
from typing import Any, Type


def _lazy(module: str, name: str):
    """Return a thunk that imports ``module.name`` on first call.

    FlashRAG's pipeline modules are heavy (vLLM / transformers / faiss).
    Importing them at module load time would slow down rag-stack startup
    for the static-backend case. We import only when the user actually
    asks for a specific pipeline type.
    """
    def loader() -> Type[Any]:
        mod = importlib.import_module(module)
        return getattr(mod, name)
    return loader


# Currently-supported pipeline modes. Each entry must satisfy BOTH:
#   - the class is dispatchable (the loader resolves), AND
#   - the class's run() has rag-stack ``query_context`` patches so DAG events
#     are attributed back to per-query IDs.
#
# To extend: patch the method's run() with ``with query_context([item.id...], step_idx=...):``
# wrappers around every generator.generate / retriever.batch_search call site,
# then add one line below.
PIPELINE_MAP: dict = {
    # Static — one retrieve, one generate.
    "sequential": _lazy("flashrag.pipeline", "SequentialPipeline"),

    # Conditional — judger routes each query to a sub-pipeline.
    # adaptive_rag = Adaptive-RAG (NAACL'24): T5 query-complexity classifier
    # routes norag / single-hop / multi-hop(IRCoT). Requires
    # eval_backend_setting.flashrag.adaptive_judger_model_path (see config_translator).
    "adaptive_rag": _lazy("flashrag.pipeline.pipeline", "AdaptivePipeline"),

    # Loop — multi-turn retrieval/generation.
    "iter_retgen": _lazy("flashrag.pipeline.active_pipeline", "IterativePipeline"),
    "flare": _lazy("flashrag.pipeline.active_pipeline", "FLAREPipeline"),
    "ircot": _lazy("flashrag.pipeline.active_pipeline", "IRCOTPipeline"),

    # Decompose loop — Self-Ask (arXiv:2210.03350): the model asks itself
    # follow-up sub-questions and retrieves per sub-question until "So the final
    # answer is:". Per-item loop (like FLARE); run_item is query_context-patched
    # in the vendored active_pipeline.py. Knobs: max_iter, single_hop.
    "self_ask": _lazy("flashrag.pipeline.active_pipeline", "SelfAskPipeline"),

    # Agentic — true reasoning loops, the FlashRAG value-add over the static evaluator.
    "searchr1": _lazy("flashrag.pipeline.reasoning_pipeline", "SearchR1Pipeline"),
    "corag":    _lazy("flashrag.pipeline.reasoning_pipeline", "CoRAGPipeline"),
    "search_o1": _lazy(
        "rag_stack_evaluator.flashrag_quality_evaluator.custom_pipelines", "SearchO1Pipeline"
    ),
    "react": _lazy(
        "rag_stack_evaluator.flashrag_quality_evaluator.custom_pipelines", "ReActPipeline"
    ),

    # Agentic, multi-tool — A-RAG (arXiv:2602.03442): the LLM picks among
    # hierarchical retrieval tools (Search=semantic, Read=chunk-read; BM25
    # keyword is the Phase-2 follow-up) in a ReAct-style loop. Implemented IN
    # the FlashRAG backend (flashrag.pipeline.arag_pipeline), per the
    # "all RAG evaluation lives in the backend" philosophy. Knob: max_iter.
    "a_rag": _lazy("flashrag.pipeline.arag_pipeline", "ARAGPipeline"),
}


def supported_pipeline_modes() -> tuple[str, ...]:
    """Return supported mode names without loading optional pipeline code."""
    return tuple(sorted(PIPELINE_MAP))


def build_pipeline_class(pipeline_type: str) -> Type[Any]:
    """Return the FlashRAG pipeline class for ``pipeline_type``.

    Raises ``ValueError`` for unknown types with the supported set in the
    error message — same shape as :func:`config_translator._resolve_metrics`
    for consistency.
    """
    if pipeline_type not in PIPELINE_MAP:
        raise ValueError(
            f"Unknown FlashRAG pipeline_type: {pipeline_type!r}. "
            f"Supported types: {sorted(PIPELINE_MAP)}."
        )
    return PIPELINE_MAP[pipeline_type]()
