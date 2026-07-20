"""Generic data-parallel execution over model replicas (measured mode).

Shared by the GPU-resident HF nodes whose measured multi-GPU mode is *data
parallelism* — N independent model copies (one per GPU), the work split
contiguously across them and run concurrently. A HF forward pass releases the
GIL, so a ``ThreadPoolExecutor`` gives real wall-clock speedup; because the
node's ``_pure`` is wrapped by ``measure_speed``, measured throughput can rise
with N replicas.  This is a measured-runtime capability only.  It is not by
itself evidence that a whole-stage P1 cost-model observation can be decomposed
or extrapolated as qps×N.

Used by the passage reranker (cross-encoder replicas) and the LLMLingua-2
passage compressor (BERT-classifier replicas). Stdlib-only so any node can
import it without pulling in heavy model dependencies.

Design contract: with a single replica (the common case) ``run_data_parallel``
short-circuits to one in-line call, so the single-GPU code path is identical to
a plain ``fn(replica, items)``. Multi-GPU is purely additive.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, List, Sequence


def _contiguous_chunks(items: Sequence[Any], n: int) -> List[List[Any]]:
    """Split ``items`` into ``n`` contiguous near-equal chunks (order preserved,
    concatenation == ``items``). When ``len(items) < n`` the trailing chunks are
    empty."""
    if n <= 1:
        return [list(items)]
    total = len(items)
    base, extra = divmod(total, n)
    chunks: List[List[Any]] = []
    start = 0
    for i in range(n):
        size = base + (1 if i < extra else 0)
        chunks.append(list(items[start:start + size]))
        start += size
    return chunks


def run_data_parallel(
    replicas: Sequence[Any],
    items: Sequence[Any],
    fn: Callable[[Any, List[Any]], Sequence[Any]],
) -> List[Any]:
    """Apply ``fn(replica, sub_items) -> list`` across ``replicas`` and
    concatenate the results in original order.

    ``items`` is split into ``len(replicas)`` contiguous chunks, each processed
    on its own replica in a separate thread (HF forward releases the GIL →
    genuine parallelism across devices). ``len(replicas) <= 1`` (or
    ``len(items) <= 1``) short-circuits to a single in-line call — NO thread
    pool, identical to the non-DP path. A chunk that raises propagates (the
    trial is penalized upstream).
    """
    n = len(replicas)
    if n <= 1 or len(items) <= 1:
        return list(fn(replicas[0], list(items)))
    chunks = _contiguous_chunks(items, n)
    results: List[Any] = [[] for _ in range(n)]
    with ThreadPoolExecutor(max_workers=n) as ex:
        futs = {
            ex.submit(fn, replicas[i], chunk): i
            for i, chunk in enumerate(chunks) if chunk
        }
        for fut, i in futs.items():
            results[i] = list(fut.result())
    out: List[Any] = []
    for r in results:
        out.extend(r)
    return out
