"""Static evaluator CM data-source helpers.

The static backend does not have FlashRAG's runtime monitor, but after a run
its final DataFrame still contains the concrete texts/tokens that drove the
main cost terms: expanded queries, retrieved passages, prompts, and generated
token ids. This module converts those artifacts into a compact aggregate
payload that the cost model can consume without changing the static
evaluator's public metric contract.
"""

from __future__ import annotations

import logging
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from rag_stack.performance_context import (
    RAG_IR_MODE_KEY,
    RAG_IR_MODE_DYNAMIC,
    RAG_IR_MODE_STATIC,
)

logger = logging.getLogger("RAG-Stack")


def rag_ir_dynamic_enabled(config: dict) -> bool:
    """True when the controller stamped ``rag_ir_mode: dynamic`` — the trial
    is priced by the trace-replay dynamic engine, so the quality run must
    hand back trace + workflow-schema CM feedback."""
    runtime = config.get("_runtime") or {}
    return str(runtime.get(RAG_IR_MODE_KEY, "")).lower() == RAG_IR_MODE_DYNAMIC


def build_static_execution_dag(
    result_df: pd.DataFrame,
    node_lines: Dict[str, List[Any]],
    *,
    config: dict,
    token_stats: Optional[Any] = None,
) -> dict:
    """Build the static-backend aggregate execution payload.

    Shape is deliberately a dict, not a ``trace_v1`` top-level list: the static
    sequential/RAGO path has richer stage semantics than the trace replay path
    currently does (query expansion fanout, compressors, filters). The payload
    still carries a best-effort ``trace_v1`` field for inspection and future
    replay support, while StaticAssembly consumes the aggregate token stats.
    """
    tokenizer = _resolve_tokenizer(config, token_stats)
    stages = [node.stage for nodes in node_lines.values() for node in nodes]
    traces: List[List[dict]] = []

    generator_input_tokens: List[int] = []
    generator_output_tokens: List[int] = []
    query_tokens: List[int] = []
    retrieved_context_tokens: List[int] = []
    reranker_input_tokens: List[int] = []
    qe_input_tokens: List[int] = []
    qe_output_tokens: List[int] = []
    fanouts: List[int] = []

    for row_idx, row in result_df.reset_index(drop=True).iterrows():
        trace: List[dict] = []
        step_idx = 0

        query = _row_text(row, "query")
        q_tokens = _count_text_tokens(query, tokenizer)
        query_tokens.append(q_tokens)

        expanded_queries = _expanded_queries(row, query)
        fanouts.append(max(1, len(expanded_queries)))

        if "query_expansion" in stages:
            qe_in = q_tokens
            qe_out = max(
                1,
                _count_text_tokens(expanded_queries, tokenizer)
                - _count_text_tokens(query, tokenizer),
            )
            qe_input_tokens.append(qe_in)
            qe_output_tokens.append(qe_out)
            trace.append({
                "stage": "query_expansion",
                "input_tokens": int(qe_in),
                "output_tokens": int(qe_out),
                "step_idx": step_idx,
                "model_id": _model_id_for_node_type(node_lines, "query_expansion"),
            })
            step_idx += 1

        retrieved_contents = _retrieved_contents(row)
        retrieved_tokens = _count_text_tokens(retrieved_contents, tokenizer)
        retrieved_context_tokens.append(retrieved_tokens)

        if any(nt in stages for nt in ("semantic_retrieval", "lexical_retrieval", "hybrid_retrieval")):
            # Multiple expanded queries are one logical user-request fanout.
            # RAGO consumes the runtime fanout from the aggregate field; the
            # trace entries are diagnostic and future-compatible. Retrieval is
            # ALWAYS two calls per (sub-)query — the embedding forward and the
            # vector search — matching the measured recorder's granularity.
            _retrieval_model = _model_id_for_node_type(node_lines, "semantic_retrieval")
            for subq in expanded_queries:
                subq_tokens = int(_count_text_tokens(subq, tokenizer))
                trace.append({
                    "stage": "semantic_retrieval_encode",
                    "input_tokens": subq_tokens,
                    "output_tokens": 0,
                    "step_idx": step_idx,
                    "model_id": _retrieval_model,
                })
                trace.append({
                    "stage": "semantic_retrieval_vectorsearch",
                    "input_tokens": subq_tokens,
                    "output_tokens": int(retrieved_tokens),
                    "step_idx": step_idx + 1,
                    "model_id": _retrieval_model,
                })
            step_idx += 2

        if "passage_reranker" in stages:
            rr_tokens = _reranker_tokens(query, retrieved_contents, tokenizer)
            reranker_input_tokens.append(rr_tokens)
            trace.append({
                "stage": "passage_reranker",
                "input_tokens": int(rr_tokens),
                "output_tokens": 0,
                "step_idx": step_idx,
                "model_id": _model_id_for_node_type(node_lines, "passage_reranker"),
            })
            step_idx += 1

        if "passage_compressor" in stages:
            # The compressor (e.g. llmlingua2) runs a per-chunk encoder forward over
            # the retrieved context. Emit it so the trace-driven latency walk prices
            # its compute (DynamicAssembly.compress_s); its DOWNSTREAM effect (fewer
            # generator input tokens) is already captured in gen_in below. The cost
            # model reads n_chunks/chunk_tokens from the config, so input_tokens here
            # is diagnostic (the pre-compression context size).
            trace.append({
                "stage": "passage_compressor",
                "input_tokens": int(retrieved_tokens),
                "output_tokens": 0,
                "step_idx": step_idx,
                "model_id": _model_id_for_node_type(node_lines, "passage_compressor"),
            })
            step_idx += 1

        if "generator" in stages:
            gen_in = _generator_input_tokens(row, tokenizer)
            gen_out = _generator_output_tokens(row, tokenizer)
            generator_input_tokens.append(gen_in)
            generator_output_tokens.append(gen_out)
            trace.append({
                "stage": "generator",
                "input_tokens": int(gen_in),
                "output_tokens": int(gen_out),
                "step_idx": step_idx,
                "model_id": _model_id_for_node_type(node_lines, "generator"),
            })

        traces.append(trace)
        if row_idx == 0 and not trace:
            logger.debug("static execution DAG builder saw no cost-relevant nodes")

    payload: dict = {
        "shape": "aggregate",
        "source": "static_gt",
        "rag_ir_mode": RAG_IR_MODE_DYNAMIC,
        "n_user_queries": int(len(result_df)),
        "token_stats": {
            "avg_query_tokens": _avg_int(query_tokens),
            "avg_retrieved_context_tokens": _avg_int(retrieved_context_tokens),
            "avg_generator_input_tokens": _avg_int(generator_input_tokens),
            "avg_generator_output_tokens": _avg_int(generator_output_tokens),
        },
        "trace_v1": traces,
    }

    if fanouts:
        payload["query_expansion_fanout"] = float(mean(fanouts))
    if qe_input_tokens:
        payload["token_stats"]["avg_query_expansion_input_tokens"] = _avg_int(qe_input_tokens)
        payload["token_stats"]["avg_query_expansion_output_tokens"] = _avg_int(qe_output_tokens)
    if reranker_input_tokens:
        payload["token_stats"]["avg_passage_reranker_input_tokens"] = _avg_int(reranker_input_tokens)

    return payload


def build_static_fanout_dag(result_df: pd.DataFrame) -> Optional[dict]:
    """Return the legacy static aggregate DAG for query-expansion fanout."""
    if "queries" not in result_df.columns or len(result_df) == 0:
        return None
    counts = result_df["queries"].apply(lambda v: len(v) if hasattr(v, "__len__") else 1)
    return {
        "shape": "aggregate",
        "source": "static_gt",
        "rag_ir_mode": RAG_IR_MODE_STATIC,
        "query_expansion_fanout": float(counts.mean()),
        "n_user_queries": int(len(counts)),
    }


def _resolve_tokenizer(config: dict, token_stats: Optional[Any]):
    tokenizer = getattr(token_stats, "_tokenizer", None) if token_stats is not None else None
    if tokenizer is not None:
        return tokenizer
    try:
        from rag_stack.cost_model.token_stats import TokenStats
        return TokenStats._load_tokenizer(config)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"static rag_ir CM data could not load tokenizer ({exc}); "
            f"falling back to whitespace token counts."
        )
        return None


def _avg_int(values: Iterable[int]) -> int:
    vals = [int(v) for v in values if v is not None]
    if not vals:
        return 0
    return max(1, int(round(mean(vals))))


def _count_text_tokens(value: Any, tokenizer: Any) -> int:
    texts = list(_flatten_texts(value))
    if not texts:
        return 0
    if tokenizer is None:
        return sum(len(t.split()) for t in texts)
    total = 0
    for text in texts:
        try:
            total += len(tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            total += len(tokenizer.encode(text))
        except Exception:  # noqa: BLE001
            total += len(text.split())
    return int(total)


def _flatten_texts(value: Any) -> Iterable[str]:
    if value is None:
        return
    if isinstance(value, float) and pd.isna(value):
        return
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, dict):
        for key in ("text", "contents", "content", "query", "prompt"):
            if key in value:
                yield from _flatten_texts(value[key])
                return
        for v in value.values():
            yield from _flatten_texts(v)
        return
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        # A list of ints is token ids, not text.
        if all(isinstance(x, int) for x in value):
            return
        for item in value:
            yield from _flatten_texts(item)
        return
    yield str(value)


def _row_text(row: pd.Series, column: str) -> str:
    if column not in row.index:
        return ""
    value = row.get(column)
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _expanded_queries(row: pd.Series, query: str) -> List[str]:
    if "queries" not in row.index:
        return [query]
    value = row.get("queries")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        out = [str(v) for v in value if v is not None]
        return out or [query]
    return [query]


def _retrieved_contents(row: pd.Series) -> Any:
    for column in ("retrieved_contents", "retrieved_contents_semantic", "retrieved_contents_lexical"):
        if column in row.index:
            return row.get(column)
    return []


def _generator_input_tokens(row: pd.Series, tokenizer: Any) -> int:
    if "prompts" in row.index:
        return max(1, _count_text_tokens(row.get("prompts"), tokenizer))
    # Iterative adapters may only surface the final answer. Fall back to the
    # concrete row context rather than offline corpus averages.
    return max(
        1,
        _count_text_tokens(_row_text(row, "query"), tokenizer)
        + _count_text_tokens(_retrieved_contents(row), tokenizer),
    )


def _generator_output_tokens(row: pd.Series, tokenizer: Any) -> int:
    if "generated_tokens" in row.index:
        toks = row.get("generated_tokens")
        if hasattr(toks, "tolist"):
            toks = toks.tolist()
        if isinstance(toks, (list, tuple)):
            return max(1, len(toks))
    if "generated_texts" in row.index:
        return max(1, _count_text_tokens(row.get("generated_texts"), tokenizer))
    return 1


def _reranker_tokens(query: str, retrieved_contents: Any, tokenizer: Any) -> int:
    contents = list(_flatten_texts(retrieved_contents))
    if not contents:
        return max(1, _count_text_tokens(query, tokenizer))
    q = _count_text_tokens(query, tokenizer)
    return max(1, sum(q + _count_text_tokens(c, tokenizer) for c in contents))


def _model_id_for_node_type(node_lines: Dict[str, List[Any]], stage: str) -> Optional[str]:
    for nodes in node_lines.values():
        for node in nodes:
            if node.stage != stage:
                continue
            params = node.module.module_param
            for key in ("model", "model_name", "embedding_model", "vectordb"):
                value = params.get(key)
                if value:
                    if isinstance(value, list):
                        return str(value[0]) if value else None
                    return str(value)
    return None
