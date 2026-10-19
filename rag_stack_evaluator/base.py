"""Common interface for quality evaluators."""

from abc import ABC, abstractmethod
from typing import Callable, Sequence

from rag_stack_evaluator.api import EvaluationResult


class BaseEvaluator(ABC):
    """Evaluate one resolved pipeline configuration."""

    @abstractmethod
    def evaluate(
        self,
        config: dict,
        run_dir: str | None = None,
        metrics_override: Sequence[str | dict] | None = None,
        on_trace_ready: Callable[[dict], None] | None = None,
    ) -> EvaluationResult | dict:
        """Return quality metrics and optional canonical execution metadata.

        Implementations may return the typed ``EvaluationResult`` or the
        existing flat metric dictionary. Inheritance is optional: external
        backends can implement the structural ``Evaluator`` protocol instead.
        Metric overrides apply only to this call. When a trace is produced,
        the optional callback receives the same envelope returned with the
        metrics; callback failures do not invalidate the quality result.
        """
        raise NotImplementedError
