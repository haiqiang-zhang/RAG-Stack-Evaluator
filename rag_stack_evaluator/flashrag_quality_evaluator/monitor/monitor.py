"""Monitor — collects per-call execution events and assembles per-query DAGs.

The :class:`Monitor` is the rag-stack-side receiver for events emitted by
the inline ``monitor.record_*`` calls patched into FlashRAG. It is
backend-agnostic by design: any backend (FlashRAG, future direct
implementations, etc.) that calls the same record_* API can feed the same
Monitor.

Event flow:

  1. FlashRAG call site (e.g. :meth:`ReasoningPipeline.run` at the
     ``self.generator.generate(prompts, ...)`` line) records a batch of
     calls via :meth:`Monitor.record_generate_batch` — passing the
     ordered ``query_ids`` along with the prompts / outputs / token counts
     / latency.
  2. Monitor expands the batch into per-query :class:`ExecutionNode`
     records.
  3. At the end of ``pipeline.run(...)``, the evaluator calls
     :meth:`Monitor.build_dags` which groups events by ``query_id`` and
     returns one :class:`ExecutionDAG` per query, sorted by ``step_idx``.

Token counts are MANDATORY on every ``generate`` / ``retrieve`` / ``rerank``
node — the cost model reads them as its primary cost-driver signal.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Union


@dataclass
class ExecutionNode:
    """One operation in a per-query execution DAG.

    Token counts are mandatory for ``generate`` / ``retrieve`` / ``rerank``
    (the cost-model-relevant node types). ``vectordb`` records the raw
    nearest-neighbor lookup latency without LLM tokens. ``terminate`` is
    a sentinel marking the query's last step (no token info).
    """

    node_type: Literal["generate", "retrieve", "rerank", "vectordb", "terminate"]
    step_idx: int
    query_id: str
    model_id: Optional[str] = None
    input_text: str = ""
    input_tokens: int = 0
    output_text: Union[str, List[Dict[str, Any]]] = ""
    output_tokens: int = 0
    latency_ms: float = 0.0
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionDAG:
    """All :class:`ExecutionNode` events for one query, ordered by step.

    ``total_input_tokens`` / ``total_output_tokens`` are running sums over
    the nodes; downstream cost-model code can read either the per-node
    breakdown or the totals.
    """

    query_id: str
    nodes: List[ExecutionNode] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0


class Monitor:
    """Backend-agnostic event collector → list[:class:`ExecutionDAG`].

    Usage from inside FlashRAG (via the inline patches):

    >>> from flashrag.monitor_hook import get_monitor
    >>> if (mon := get_monitor()) is not None:
    ...     mon.record_generate_batch(
    ...         query_ids=[item.id for item in active_items],
    ...         step_idx=current_step_idx,
    ...         model_id=self.generator.model_name,
    ...         prompts=exist_prompts,
    ...         outputs=step_outputs,
    ...         input_token_counts=[...],
    ...         output_token_counts=[...],
    ...         latency_ms=elapsed_ms,
    ...     )

    Usage from rag-stack:

    >>> monitor = Monitor()
    >>> set_monitor(monitor)
    >>> try:
    ...     out = pipeline.run(dataset)
    ... finally:
    ...     set_monitor(None)
    >>> dags = monitor.build_dags()   # list[ExecutionDAG], one per query
    """

    def __init__(self) -> None:
        self._events: List[ExecutionNode] = []

    # ------------------------------------------------------------------
    # FlashRAG-side recorders (called from inline patches).
    # ------------------------------------------------------------------

    def record_generate_batch(
        self,
        *,
        query_ids: Sequence[str],
        step_idx: int,
        model_id: Optional[str],
        prompts: Sequence[str],
        outputs: Sequence[str],
        input_token_counts: Sequence[int],
        output_token_counts: Sequence[int],
        latency_ms: float,
        extras: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record one LLM batch call. Expands into one event per query.

        ``latency_ms`` is the wall-clock for the whole batch; we attach it
        to every per-query event so downstream consumers can either treat
        it as shared overhead or amortize it themselves.
        """
        for qid, prompt, out, in_t, out_t in zip(
            query_ids, prompts, outputs, input_token_counts, output_token_counts,
        ):
            self._events.append(
                ExecutionNode(
                    node_type="generate",
                    step_idx=step_idx,
                    query_id=str(qid),
                    model_id=model_id,
                    input_text=prompt,
                    input_tokens=int(in_t),
                    output_text=out,
                    output_tokens=int(out_t),
                    latency_ms=float(latency_ms),
                    extras=dict(extras or {}),
                )
            )

    def record_retrieve_batch(
        self,
        *,
        query_ids: Sequence[str],
        step_idx: int,
        model_id: Optional[str],
        queries: Sequence[str],
        doc_lists: Sequence[List[Dict[str, Any]]],
        input_token_counts: Sequence[int],
        output_token_counts: Sequence[int],
        latency_ms: float,
        extras: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record one retriever batch_search call (one event per query)."""
        for qid, q, docs, in_t, out_t in zip(
            query_ids, queries, doc_lists, input_token_counts, output_token_counts,
        ):
            self._events.append(
                ExecutionNode(
                    node_type="retrieve",
                    step_idx=step_idx,
                    query_id=str(qid),
                    model_id=model_id,
                    input_text=q,
                    input_tokens=int(in_t),
                    output_text=docs,
                    output_tokens=int(out_t),
                    latency_ms=float(latency_ms),
                    extras=dict(extras or {}),
                )
            )

    def record_rerank_call(
        self,
        *,
        query_id: str,
        step_idx: int,
        model_id: Optional[str],
        query: str,
        docs_in: List[Dict[str, Any]],
        docs_out: List[Dict[str, Any]],
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        extras: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record one reranker call (per-query, single event)."""
        self._events.append(
            ExecutionNode(
                node_type="rerank",
                step_idx=step_idx,
                query_id=str(query_id),
                model_id=model_id,
                input_text=query,
                input_tokens=int(input_tokens),
                output_text=docs_out,
                output_tokens=int(output_tokens),
                latency_ms=float(latency_ms),
                extras={"docs_in_count": len(docs_in), **(extras or {})},
            )
        )

    def record_vectordb_call(
        self,
        *,
        query_ids: Sequence[str],
        step_idx: int,
        db_type: str,
        queries: Sequence[str],
        doc_ids_per_query: Sequence[List[str]],
        latency_ms: float,
        extras: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record the raw vectordb (FAISS / BM25 / etc.) search call.

        Separate from :meth:`record_retrieve_batch` so callers that want
        retriever-vs-vectordb granularity can have both. Token counts are
        not applicable to vectordb directly (the retriever wrapper above
        handles that).
        """
        for qid, q, doc_ids in zip(query_ids, queries, doc_ids_per_query):
            self._events.append(
                ExecutionNode(
                    node_type="vectordb",
                    step_idx=step_idx,
                    query_id=str(qid),
                    model_id=db_type,
                    input_text=q,
                    input_tokens=0,
                    output_text=[{"id": d} for d in doc_ids],
                    output_tokens=0,
                    latency_ms=float(latency_ms),
                    extras=dict(extras or {}),
                )
            )

    def record_terminate(
        self,
        *,
        query_id: str,
        step_idx: int,
        reason: str,
    ) -> None:
        """Mark a query as terminating at this step (no token info)."""
        self._events.append(
            ExecutionNode(
                node_type="terminate",
                step_idx=step_idx,
                query_id=str(query_id),
                extras={"reason": reason},
            )
        )

    # ------------------------------------------------------------------
    # Aggregation.
    # ------------------------------------------------------------------

    def build_dags(self) -> List[ExecutionDAG]:
        """Group events by ``query_id`` and emit one DAG per query.

        Nodes within a DAG are sorted by ``step_idx``; ties keep insertion
        order, which matches the order calls were made within a step.
        """
        groups: Dict[str, List[ExecutionNode]] = defaultdict(list)
        for ev in self._events:
            groups[ev.query_id].append(ev)
        dags: List[ExecutionDAG] = []
        for qid, nodes in groups.items():
            nodes_sorted = sorted(
                enumerate(nodes), key=lambda kv: (kv[1].step_idx, kv[0]),
            )
            ordered = [n for _, n in nodes_sorted]
            dag = ExecutionDAG(
                query_id=qid,
                nodes=ordered,
                total_input_tokens=sum(n.input_tokens for n in ordered),
                total_output_tokens=sum(n.output_tokens for n in ordered),
            )
            dags.append(dag)
        return dags

    def clear(self) -> None:
        """Drop all accumulated events. Useful between successive runs
        sharing the same Monitor instance."""
        self._events.clear()
