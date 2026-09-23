"""Lazy construction and public metadata for supported quality backends.

Importing this module does not load model libraries or optional FlashRAG
dependencies. The selected backend owns its implementation and metadata.
"""

from __future__ import annotations

from typing import Any

from rag_stack_evaluator.base import BaseEvaluator


def _normalize_backend(backend: str) -> str:
    name = str(backend).lower()
    if name == "static":
        raise ValueError(
            "global.eval_backend 'static' was renamed to 'static_gt'. "
            "Please update your YAML."
        )
    if name not in {"static_gt", "flashrag"}:
        raise ValueError(
            f"Unknown global.eval_backend: {name!r}. "
            "Supported: 'static_gt' (default) | 'flashrag'."
        )
    return name


def create_quality_evaluator(
    dataset: Any = None,
    project_dir: str | None = None,
    *,
    dataset_manager: Any = None,
    backend: str = "static_gt",
    flashrag_options: dict[str, Any] | None = None,
) -> BaseEvaluator:
    """Construct only the requested backend, using a shared dataset owner."""
    name = _normalize_backend(backend)
    if name == "flashrag":
        from rag_stack_evaluator.flashrag_quality_evaluator import (
            FlashRAGQualityEvaluator,
        )

        return FlashRAGQualityEvaluator(
            dataset,
            project_dir,
            dataset_manager=dataset_manager,
            flashrag_options=flashrag_options,
        )

    from rag_stack_evaluator.static_rag_evaluator import (
        StaticRAGEvaluatorQualityOnly,
    )

    return StaticRAGEvaluatorQualityOnly(
        dataset, project_dir, dataset_manager=dataset_manager,
    )


def supported_pipeline_modes(backend: str = "static_gt") -> tuple[str, ...]:
    """Return supported modes without importing pipeline implementations."""
    if _normalize_backend(backend) == "flashrag":
        from rag_stack_evaluator.flashrag_quality_evaluator.pipeline_factory import (
            supported_pipeline_modes as flashrag_modes,
        )

        return flashrag_modes()
    return ("react", "sequential")


def supported_metrics(backend: str = "static_gt") -> tuple[str, ...]:
    """Return the selected backend's metric vocabulary.

    Static metric discovery imports the static evaluator's metric registry;
    FlashRAG discovery reads its lightweight metadata without loading FlashRAG.
    """
    if _normalize_backend(backend) == "flashrag":
        from rag_stack_evaluator.flashrag_quality_evaluator.config_translator import (
            supported_metrics as flashrag_metrics,
        )

        return flashrag_metrics()

    metrics = {
        "retrieval_token_recall", "retrieval_token_precision", "retrieval_token_f1",
    }
    try:
        from rag_stack_evaluator.static_rag_evaluator.evaluation.generation import (
            GENERATION_METRIC_FUNC_DICT,
        )

        metrics.update(GENERATION_METRIC_FUNC_DICT)
    except ImportError:
        pass
    try:
        from rag_stack_evaluator.static_rag_evaluator.evaluation.retrieval import (
            RETRIEVAL_METRIC_FUNC_DICT,
        )

        metrics.update(RETRIEVAL_METRIC_FUNC_DICT)
    except ImportError:
        pass
    return tuple(sorted(metrics))
