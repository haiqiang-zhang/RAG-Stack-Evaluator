"""Translate FlashRAG monitor events into the shared component-trace protocol.

This adapter has no runtime dependency on FlashRAG or the optimizer. Envelope
validation remains part of the shared RAG-Stack IR contract.
"""

from __future__ import annotations


# FlashRAG monitor event verb → taxonomy stage(s). This is the external-library
# adapter table (the monitor's vocabulary is NOT ours and is never modified):
#   * ``retrieve`` expands to the encode + vector-search call PAIR (token
#     fields shared — the event doesn't time them separately); a bm25 retrieve
#     (``model_id == "bm25"``) is a single ``lexical_retrieval`` call instead.
#   * ``encode`` / ``compress`` are mapped for future monitor hooks.
#   * ``vectordb`` is the inner FAISS sub-event of a ``retrieve`` event; it is
#     dropped so the search stage is not double-counted. ``terminate`` is a
#     no-cost sentinel — also dropped.
#   * any OTHER event is a hard error: an uninstrumented vocabulary drifting in
#     must fail loudly, not be silently skipped.
_MONITOR_EVENT_TO_STAGE = {
    "generate": ("generator",),
    "retrieve": ("semantic_retrieval_encode", "semantic_retrieval_vectorsearch"),
    "rerank": ("passage_reranker",),
    "encode": ("semantic_retrieval_encode",),
    "compress": ("passage_compressor",),
}
_MONITOR_DROPPED_EVENTS = frozenset({"vectordb", "terminate"})


def normalize_traces(dags: list[dict]) -> list[list[dict]]:
    """Convert FlashRAG monitor per-query DAG dicts to a trace payload.

    Each returned trace is an ordered (by the monitor's ``step_idx``) list of
    :class:`TraceCall` records in taxonomy-stage vocabulary; text fields are
    dropped, dense ``retrieve`` events split into their encode + vector-search
    pair, and unknown event verbs raise. The result is the sole input contract
    consumed by :class:`rag_stack.cost_model.assembly.RAGCMAssembly`.
    """
    traces: list[list[dict]] = []
    for dag in dags or []:
        nodes = dag.get("nodes", []) or []
        calls: list[dict] = []
        for n in sorted(nodes, key=lambda x: x.get("step_idx", 0)):
            event = n.get("node_type")
            if event in _MONITOR_DROPPED_EVENTS:
                continue
            if event not in _MONITOR_EVENT_TO_STAGE:
                raise KeyError(
                    f"unknown FlashRAG monitor event {event!r} — extend the "
                    f"normalize adapter (_MONITOR_EVENT_TO_STAGE) deliberately"
                )
            stages = _MONITOR_EVENT_TO_STAGE[event]
            if event == "retrieve" and n.get("model_id") == "bm25":
                stages = ("lexical_retrieval",)
            for stage in stages:
                calls.append(dict(
                    stage=stage,
                    input_tokens=int(n.get("input_tokens", 0) or 0),
                    output_tokens=int(n.get("output_tokens", 0) or 0),
                    step_idx=int(n.get("step_idx", 0) or 0),
                    model_id=n.get("model_id"),
                ))
        traces.append(calls)
    return traces

