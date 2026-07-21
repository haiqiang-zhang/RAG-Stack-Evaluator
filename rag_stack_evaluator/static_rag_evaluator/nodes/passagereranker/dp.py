"""Data-parallel execution for measured-mode reranker replicas.

A passage reranker is a small cross-encoder with no Hugging-Face tensor-parallel
path, so its measured multi-GPU mode is **data parallelism**: N independent model
copies (one per GPU), the work split contiguously across them and run
concurrently. A HF forward pass releases the GIL, so a ``ThreadPoolExecutor``
gives real wall-clock speedup — and because the reranker's ``_pure`` is wrapped
by ``measure_speed``, measured throughput can rise with N replicas. This is a
runtime capability, not a cost-model composition rule: the current whole-stage
P1 observation does not separate global wrapper work from replica-local model
work, so RAG-CM must not infer qps×N from this dispatcher.

Design contract: with a single replica (the common case) ``run_data_parallel``
short-circuits to one in-line call, so the single-GPU code path is **byte-for-byte
identical** to the previous ``flatten_apply(run_model, ..., model=self.model)``.
Multi-GPU is purely additive.

The generic primitives (``run_data_parallel``, ``_contiguous_chunks``) now live
in :mod:`rag_stack_evaluator.static_rag_evaluator.utils.data_parallel` so other GPU nodes
(e.g. the LLMLingua-2 compressor) can reuse them without importing this
reranker package (which eagerly loads every reranker). This module keeps the
reranker-specific ``rerank_flatten_apply_dp`` wrapper.
"""
from __future__ import annotations

from typing import Any, Callable, List, Sequence

from rag_stack_evaluator.static_rag_evaluator.utils.data_parallel import (  # noqa: F401
    _contiguous_chunks,
    run_data_parallel,
)


def rerank_flatten_apply_dp(
    run_model_fn: Callable,
    nested_list: List[List[Any]],
    replicas: Sequence[Any],
    batch_size: int,
) -> List[List[Any]]:
    """Data-parallel drop-in for ``flatten_apply(run_model_fn, nested_list,
    model=<model>, batch_size=batch_size)`` used by the cross-encoder rerankers.

    Flattens the nested (query, content) pairs, scores the flat list across the
    model ``replicas`` (contiguous split, concurrent), then reconstructs the
    nested shape. With one replica this is exactly the original ``flatten_apply``.
    """
    import pandas as pd

    df = pd.DataFrame({"col1": nested_list})
    df = df.explode("col1")
    flat = df["col1"].tolist()
    scores = run_data_parallel(
        replicas,
        flat,
        lambda model, sub: run_model_fn(sub, model=model, batch_size=batch_size),
    )
    df["result"] = scores
    return df.groupby(level=0, sort=False)["result"].apply(list).tolist()
