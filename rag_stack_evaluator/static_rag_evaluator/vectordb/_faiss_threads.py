"""OpenMP thread policy for faiss INDEX BUILD (train + add).

Build (IVF kmeans train, PQ encoding, HNSW graph insertion) is CPU-bound and
is NOT part of the cost model's search-latency measurement — so parallelising
it is a free wall-clock win with zero calibration impact. This module is the
single place that sets build threads.

The invariant that matters: build threads must be RESTORED to 1 afterwards.
Search latency/qps is calibrated with faiss anchored at a single thread
(``active_threads=1``); the per-query thread count is set explicitly from
``system.retrieval.num_threads`` inside ``query()``. If a build left OMP at
cpu-2, an unspecified query would silently run multi-threaded and break that
anchor. The context manager guarantees the restore even on early return /
exception.
"""
import contextlib
import logging
import os

import faiss

logger = logging.getLogger("RAG-Stack")


def build_num_threads(explicit=None) -> int:
    """Index-build thread count. An explicit value (YAML
    ``system.retrieval.faiss_indexing_thread``) wins; otherwise the default is
    this machine's CPU count minus 2, floored at 1 (leaves headroom for the
    driver / co-resident work). A non-positive / unparseable explicit value
    falls back to the default."""
    if explicit is not None:
        try:
            n = int(explicit)
            if n >= 1:
                return n
        except (TypeError, ValueError):
            pass
    return max(1, (os.cpu_count() or 3) - 2)


@contextlib.contextmanager
def faiss_build_threads(explicit=None):
    """Run the enclosed faiss build on ``build_num_threads(explicit)`` OMP
    threads, then restore to 1 (search-calibration anchor). Reentrant-safe for
    our use (build is never nested) and exception-safe via ``finally``."""
    n = build_num_threads(explicit)
    faiss.omp_set_num_threads(n)
    logger.debug(
        "faiss index build: OMP threads = %d (%s)",
        n, "explicit" if explicit is not None else "cpu_count-2",
    )
    try:
        yield n
    finally:
        faiss.omp_set_num_threads(1)
