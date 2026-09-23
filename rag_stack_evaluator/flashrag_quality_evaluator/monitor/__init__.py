"""Backend-agnostic execution-DAG monitor.

The :class:`Monitor` collects per-call events emitted from FlashRAG (via
the inline ``monitor.record_*`` patches in ``FlashRAG/flashrag/monitor_hook.py``
+ pipeline call sites). It assembles them into :class:`ExecutionDAG`
records — one per query — that downstream cost-model code reads to drive
its per-call cost computation.

The Monitor knows nothing FlashRAG-specific; any backend that emits the
same record_* events feeds into the same data model.
"""

from rag_stack_evaluator.flashrag_quality_evaluator.monitor.monitor import (
    ExecutionDAG,
    ExecutionNode,
    Monitor,
)

__all__ = ["ExecutionDAG", "ExecutionNode", "Monitor"]
