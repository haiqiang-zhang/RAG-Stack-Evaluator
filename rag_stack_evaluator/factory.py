"""Compatibility imports for the RAG-Stack-owned backend registry.

New integrations use ``rag_stack.evaluation.registry``. Backend construction,
capability resolution, and entry-point discovery have a single host owner.
"""

from rag_stack.evaluation.registry import (  # noqa: F401
    BackendSpec, EvaluatorContext, ENTRY_POINT_GROUP, available_backends,
    register_backend, get_backend_spec, supported_metrics, supported_pipeline_modes,
)
from rag_stack.evaluation.registry import create_quality_evaluator as _create


def create_quality_evaluator(
    dataset=None, project_dir=None, *, dataset_manager=None, backend="static_gt",
    flashrag_options=None, backend_options=None,
):
    """Forward to the host factory, translating the retired options keyword."""
    if flashrag_options is not None and backend_options is not None:
        raise ValueError("Pass backend_options or flashrag_options, not both")
    if flashrag_options is not None:
        name = get_backend_spec(backend).name
        if name not in {"static_gt", "flashrag"}:
            raise ValueError("flashrag_options can only configure the FlashRAG backend")
        backend_options = flashrag_options if name == "flashrag" else {}
    return _create(
        dataset, project_dir, dataset_manager=dataset_manager, backend=backend,
        backend_options=backend_options,
    )


def _normalize_backend(backend):
    return get_backend_spec(backend).name


__all__ = [
    "BackendSpec", "EvaluatorContext", "ENTRY_POINT_GROUP", "available_backends",
    "register_backend", "get_backend_spec", "create_quality_evaluator",
    "supported_metrics", "supported_pipeline_modes",
]
