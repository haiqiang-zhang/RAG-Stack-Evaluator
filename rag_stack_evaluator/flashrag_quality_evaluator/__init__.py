"""Optional FlashRAG quality backend owned by RAG-Stack-Evaluator.

Package and metadata imports do not load FlashRAG or model libraries. The
implementation is loaded when ``FlashRAGQualityEvaluator`` is requested, and
FlashRAG itself is required only when constructing or running that evaluator.
"""

__all__ = ["FlashRAGQualityEvaluator"]


def __getattr__(name: str):
    if name == "FlashRAGQualityEvaluator":
        from .flashrag_quality_evaluator import FlashRAGQualityEvaluator

        return FlashRAGQualityEvaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
