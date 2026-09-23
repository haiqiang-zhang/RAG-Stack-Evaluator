"""Common interface for quality evaluators."""

from abc import ABC, abstractmethod
from typing import Callable, Sequence


class BaseEvaluator(ABC):
    """Evaluate one resolved pipeline configuration."""

    @abstractmethod
    def evaluate(
        self,
        config: dict,
        run_dir: str | None = None,
        metrics_override: Sequence[str | dict] | None = None,
        on_trace_ready: Callable[[dict], None] | None = None,
    ) -> dict:
        """Return quality metrics and optional canonical execution metadata.

        Metric overrides apply only to this call. When a trace is produced,
        the optional callback receives the same envelope returned with the
        metrics; callback failures do not invalidate the quality result.
        """
        raise NotImplementedError
