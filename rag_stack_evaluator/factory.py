"""Lazy evaluator construction and capability registration.

External distributions declare a ``rag_stack.evaluators`` entry point whose
name is the configured backend and whose value resolves to a ``BackendSpec``.
Listing backends reads package metadata only; selecting one loads its descriptor.
Model libraries should be imported inside the descriptor's factory callable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import metadata
import re
from typing import Any, Callable

from rag_stack_evaluator.api import Evaluator

ENTRY_POINT_GROUP = "rag_stack.evaluators"
_MetadataValues = tuple[str, ...] | Callable[[], tuple[str, ...]]


@dataclass(frozen=True)
class EvaluatorContext:
    """Construction inputs supplied by the host, independent of a backend API."""

    dataset_manager: Any = None
    project_dir: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    dataset: Any = None


@dataclass(frozen=True)
class BackendSpec:
    """A backend factory and the capabilities used by configuration validation.

    ``metrics`` and ``pipeline_modes`` may be tuples or lazy zero-argument
    providers. ``node_lines`` uses the host's component search-space projection;
    ``methods`` uses method-based pipeline selection. Registering a backend does
    not add new cost-model component types or a measured-performance runner.
    """

    name: str
    factory: Callable[[EvaluatorContext], Evaluator]
    metrics: _MetadataValues
    pipeline_modes: _MetadataValues = ("sequential",)
    pipeline_style: str = "node_lines"
    supports_measured: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", self.name):
            raise ValueError("Backend names must match [a-z][a-z0-9_]*")
        if self.name == "static":
            raise ValueError("Backend name 'static' is reserved; use 'static_gt'")
        if not callable(self.factory):
            raise TypeError("BackendSpec.factory must be callable")
        if self.pipeline_style not in {"node_lines", "methods"}:
            raise ValueError("BackendSpec.pipeline_style must be 'node_lines' or 'methods'")
        if not isinstance(self.supports_measured, bool):
            raise TypeError("BackendSpec.supports_measured must be a boolean")
        for name in ("metrics", "pipeline_modes"):
            values = getattr(self, name)
            if not callable(values):
                _metadata_values(values, label=f"{self.name}.{name}")


def _metadata_values(values: _MetadataValues, *, label: str) -> tuple[str, ...]:
    resolved = values() if callable(values) else values
    if not isinstance(resolved, tuple) or any(
        not isinstance(item, str) or not item.strip() for item in resolved
    ):
        raise TypeError(f"{label} must be a tuple of non-empty names or a callable returning one")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{label} contains duplicate names")
    return resolved


def _create_static(context: EvaluatorContext) -> Evaluator:
    from rag_stack_evaluator.static_rag_evaluator import StaticRAGEvaluatorQualityOnly

    return StaticRAGEvaluatorQualityOnly(
        context.dataset, context.project_dir, dataset_manager=context.dataset_manager,
    )


def _create_flashrag(context: EvaluatorContext) -> Evaluator:
    from rag_stack_evaluator.flashrag_quality_evaluator import FlashRAGQualityEvaluator

    return FlashRAGQualityEvaluator(
        context.dataset, context.project_dir, dataset_manager=context.dataset_manager,
        flashrag_options=context.options,
    )


def _static_metrics() -> tuple[str, ...]:
    # Preserve the built-in registry's lazy discovery and optional dependencies.
    metrics = {"retrieval_token_recall", "retrieval_token_precision", "retrieval_token_f1"}
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


def _flashrag_metrics() -> tuple[str, ...]:
    from rag_stack_evaluator.flashrag_quality_evaluator.config_translator import supported_metrics

    return supported_metrics()


def _flashrag_modes() -> tuple[str, ...]:
    from rag_stack_evaluator.flashrag_quality_evaluator.pipeline_factory import supported_pipeline_modes

    return supported_pipeline_modes()


_BACKENDS: dict[str, BackendSpec] = {
    "static_gt": BackendSpec(
        name="static_gt", factory=_create_static, metrics=_static_metrics,
        pipeline_modes=("react", "sequential"), supports_measured=True,
    ),
    "flashrag": BackendSpec(
        name="flashrag", factory=_create_flashrag, metrics=_flashrag_metrics,
        pipeline_modes=_flashrag_modes, pipeline_style="methods",
    ),
}


def _entry_points() -> tuple[Any, ...]:
    return tuple(metadata.entry_points(group=ENTRY_POINT_GROUP))


def available_backends() -> tuple[str, ...]:
    """List built-in, explicitly registered, and installed names without loading plugins."""
    return tuple(sorted(set(_BACKENDS) | {entry.name for entry in _entry_points()}))


def register_backend(spec: BackendSpec) -> None:
    """Register a descriptor for this process; duplicate names are errors.

    Use entry points for CLI/subprocess use so every worker discovers the same
    backend. Explicit registration is useful for embedded Python applications.
    """
    if not isinstance(spec, BackendSpec):
        raise TypeError("register_backend expects a BackendSpec")
    if spec.name in _BACKENDS or any(entry.name == spec.name for entry in _entry_points()):
        raise ValueError(f"Evaluator backend {spec.name!r} is already registered or installed")
    _BACKENDS[spec.name] = spec


def get_backend_spec(backend: str = "static_gt") -> BackendSpec:
    """Resolve the selected descriptor, loading only its entry point when needed."""
    name = str(backend).strip().lower()
    if name == "static":
        raise ValueError(
            "global.eval_backend 'static' was renamed to 'static_gt'. Please update your YAML."
        )
    if name in _BACKENDS:
        return _BACKENDS[name]
    entries = [entry for entry in _entry_points() if entry.name == name]
    if len(entries) > 1:
        raise ValueError(f"Multiple installed evaluator entry points use the name {name!r}")
    if not entries:
        raise ValueError(
            f"Unknown global.eval_backend: {name!r}. Supported: {', '.join(available_backends())}."
        )
    spec = entries[0].load()
    if not isinstance(spec, BackendSpec):
        raise TypeError(f"Evaluator entry point {name!r} must resolve to a BackendSpec")
    if spec.name != name:
        raise ValueError(f"Evaluator entry point {name!r} declares backend name {spec.name!r}")
    _BACKENDS[name] = spec
    return spec


def _normalize_backend(backend: str) -> str:
    """Compatibility helper for callers of the previous private dispatcher."""
    return get_backend_spec(backend).name


def create_quality_evaluator(
    dataset: Any = None,
    project_dir: str | None = None,
    *,
    dataset_manager: Any = None,
    backend: str = "static_gt",
    flashrag_options: dict[str, Any] | None = None,
    backend_options: dict[str, Any] | None = None,
) -> Evaluator:
    """Construct only the selected backend, using a shared dataset owner.

    ``backend_options`` is the selected backend's options dictionary. The old
    ``flashrag_options`` keyword remains supported; ambiguous inputs are rejected.
    """
    spec = get_backend_spec(backend)
    if backend_options is not None and flashrag_options is not None:
        raise ValueError("Pass backend_options or flashrag_options, not both")
    if flashrag_options is not None and spec.name != "flashrag":
        # Historically callers passed the FlashRAG config unconditionally for
        # built-in static evaluation. Preserve that path, but never feed those
        # options into an unrelated external backend.
        options = {} if spec.name == "static_gt" else None
        if options is None:
            raise ValueError("flashrag_options can only configure the FlashRAG backend")
    else:
        options = backend_options if backend_options is not None else flashrag_options
    if options is not None and not isinstance(options, dict):
        raise TypeError("backend_options must be a dictionary")
    evaluator = spec.factory(EvaluatorContext(
        dataset_manager=dataset_manager, project_dir=project_dir,
        options=dict(options or {}), dataset=dataset,
    ))
    if not callable(getattr(evaluator, "evaluate", None)):
        raise TypeError(f"Factory for {spec.name!r} must return an object with an evaluate method")
    return evaluator


def supported_pipeline_modes(backend: str = "static_gt") -> tuple[str, ...]:
    """Return selected backend modes without constructing its evaluator."""
    spec = get_backend_spec(backend)
    return _metadata_values(spec.pipeline_modes, label=f"{spec.name}.pipeline_modes")


def supported_metrics(backend: str = "static_gt") -> tuple[str, ...]:
    """Return the selected backend's quality metric vocabulary."""
    spec = get_backend_spec(backend)
    return _metadata_values(spec.metrics, label=f"{spec.name}.metrics")


__all__ = [
    "BackendSpec", "EvaluatorContext", "ENTRY_POINT_GROUP", "available_backends",
    "register_backend", "get_backend_spec", "create_quality_evaluator",
    "supported_metrics", "supported_pipeline_modes",
]
