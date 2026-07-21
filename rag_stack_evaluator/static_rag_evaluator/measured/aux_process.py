"""Dedicated worker processes for host-heavy stages (A2, r12; r20 retrieval).

Why: the measured harness used to run every component in ONE process. A
light stage's forward is mostly host work (tokenize, pandas, framework
glue) — for ms_marco_minilm_l2 the isolated forward is 12.8 ms but the
SAME forward inside the serving process takes 35.4 ms (2.8x), because it
shares the GIL and the CPU cores with the event loop, the vLLM HTTP
client and faiss's OMP pool. Moving the stage into its own process gives
it its own GIL and scheduler share; the GPU model lives in the child
(same VRAM, one copy, plus a small CUDA-context overhead).

Scope: the batched host-bound GPU aux stages (passage_reranker,
passage_compressor) and — since r20 — the retrieval stages. Retrieval
used to stay in-process "because faiss_num_threads is a CM-priced knob
whose environment must not change"; the 07-10 overhead audit showed the
opposite: in-process the retrieval node pays a 7-90x GIL/co-residency
stretch over its isolated cost (0.54 ms/query solo vs 30-51 ms in-situ),
so in-process is exactly the environment its calibration never sees. The
child applies the same faiss_num_threads / parallel_mode knobs per query
(they travel in the stage params) and inherits OMP_NUM_THREADS, so the
knob semantics are unchanged — only the GIL contention is gone. LLM
engines are separate vLLM server processes already.

Protocol: parent sends (df, params) pickled over a Pipe, child runs
``instance.pure(df, **params)`` and sends the result back. The parent
times the WHOLE round trip in the same place it used to time the
in-process call, so the exported per-batch service keeps its meaning.
Two payload cuts (r20) keep the IPC term small even at batch 256:

- column pruning: only the columns the stage's ``pure`` actually reads
  are shipped (the full frame carries every accumulated text column;
  at reranker time that is ~19 KB/query of which the reply needs none);
- id-keyed reply for rerankers: rerankers permute/cut (contents, ids,
  scores) together and never rewrite a passage, so the child replies
  with ids+scores only and the parent rebuilds ``retrieved_contents``
  from its own copy of the input. The child verifies id coverage and
  spot-checks row-0 content identity first; any mismatch falls back to
  shipping the full frame (correct-by-construction).

Lifecycle: child is spawned (CUDA-safe) per trial at stage build, killed
at stage teardown; ``daemon=True`` guarantees it dies with the parent
even on abrupt exits. Disable with RAG_STACK_AUX_PROCESS=0.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger("RAG-Stack")

# Batched, host-bound stages that own a (small) GPU model.
AUX_PROCESS_STAGES = frozenset({"passage_reranker", "passage_compressor"})

# Retrieval family (r20): encode is GPU, faiss/bm25/fusion are host-heavy.
RETRIEVAL_PROCESS_STAGES = frozenset(
    {"semantic_retrieval", "lexical_retrieval", "hybrid_retrieval"}
)

# Columns each stage's ``pure`` reads (validate_qa_dataset needs
# qid/query/generation_gt; retrieval casts also touch ``queries``). A stage
# missing from this table ships the full frame — pruning is opt-in per stage.
_QA_COLUMNS = ("qid", "query", "queries", "generation_gt", "__qid__")
_RETRIEVE_COLUMNS = (
    "retrieved_contents", "retrieved_ids", "retrieve_scores",
    "retrieved_contents_semantic", "retrieved_ids_semantic",
    "retrieve_scores_semantic",
    "retrieved_contents_lexical", "retrieved_ids_lexical",
    "retrieve_scores_lexical",
)
_SHIP_COLUMNS: Dict[str, tuple] = {
    "passage_reranker": _QA_COLUMNS + _RETRIEVE_COLUMNS,
    "passage_compressor": _QA_COLUMNS + _RETRIEVE_COLUMNS,
    "semantic_retrieval": _QA_COLUMNS,
    "lexical_retrieval": _QA_COLUMNS,
    "hybrid_retrieval": _QA_COLUMNS + _RETRIEVE_COLUMNS,
}

_PACK_RESULT_COLUMNS = ("retrieved_contents", "retrieved_ids", "retrieve_scores")


def aux_process_enabled() -> bool:
    return os.environ.get(
        "RAG_STACK_AUX_PROCESS", "1"
    ).strip().lower() not in {"0", "false", "no", "off"}


def process_isolated_stage(stage: str) -> bool:
    """Single decision point for 'does this stage run in a child process'."""
    if not aux_process_enabled():
        return False
    return stage in AUX_PROCESS_STAGES or stage in RETRIEVAL_PROCESS_STAGES


def _pack_reranker_result(df_in, result) -> Optional[Any]:
    """Return ``result`` without its ``retrieved_contents`` column when the
    parent can provably rebuild it from the input frame, else None.

    Safe iff the module permutes/cuts (contents, ids, scores) together —
    true for every reranker (they select from the input, never rewrite).
    Guarded twice: every output id must exist in its row's input ids, and
    row 0's contents must equal the id-mapped input contents (a module
    that rewrites text fails the spot check on every batch)."""
    import pandas as pd

    if not isinstance(result, pd.DataFrame):
        return None
    if any(c not in result.columns for c in _PACK_RESULT_COLUMNS):
        return None
    try:
        from rag_stack_evaluator.static_rag_evaluator.utils.cast import (
            cast_retrieved_contents,
            cast_retrieved_ids,
        )

        in_ids = cast_retrieved_ids(df_in)
        in_contents = cast_retrieved_contents(df_in)
    except Exception:  # noqa: BLE001 — unexpected input shape: ship full
        return None
    out_ids_col = result["retrieved_ids"].tolist()
    if len(out_ids_col) != len(in_ids):
        return None
    for out_ids, ids_row in zip(out_ids_col, in_ids):
        known = set(ids_row)
        if any(i not in known for i in out_ids):
            return None
    row0_map = dict(zip(in_ids[0], in_contents[0]))
    rebuilt0 = [row0_map[i] for i in out_ids_col[0]]
    if list(result["retrieved_contents"].iloc[0]) != rebuilt0:
        return None
    return result.drop(columns=["retrieved_contents"])


def _unpack_reranker_result(df_sent, slim):
    """Parent-side inverse of :func:`_pack_reranker_result`."""
    from rag_stack_evaluator.static_rag_evaluator.utils.cast import (
        cast_retrieved_contents,
        cast_retrieved_ids,
    )

    in_ids = cast_retrieved_ids(df_sent)
    in_contents = cast_retrieved_contents(df_sent)
    contents = []
    for out_ids, ids_row, cont_row in zip(
        slim["retrieved_ids"].tolist(), in_ids, in_contents
    ):
        row_map = dict(zip(ids_row, cont_row))
        contents.append([row_map[i] for i in out_ids])
    out = slim.copy()
    # result_to_dataframe emits (contents, ids, scores) — restore that order.
    out.insert(0, "retrieved_contents", contents)
    return out


def _worker_main(conn, cls_module: str, cls_name: str,
                 project_dir: str, kwargs: Dict[str, Any],
                 stage: Optional[str] = None) -> None:
    """Child entry: build the stage instance and serve pure() calls."""
    # The child owns a fresh interpreter: keep its CPU pools modest so it
    # does not oversubscribe the box the way the old shared process did.
    # Thread pools must never busy-wait: on the 32-core box an idle-spinning
    # OMP pool in the encoder child burned ~11 cores and slowed the
    # CO-RESIDENT vLLM host side ~18% at b256 react (0046-A replay) — set
    # passive waiting BEFORE any OMP runtime initializes.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    os.environ.setdefault("KMP_BLOCKTIME", "0")
    try:
        import torch
        if stage in RETRIEVAL_PROCESS_STAGES:
            # Encode is GPU work and the faiss knob drives its own OMP pool
            # per query (independent of torch's) — torch CPU threads here
            # only tokenize a handful of queries per batch.
            torch.set_num_threads(1)
        else:
            torch.set_num_threads(max(4, (os.cpu_count() or 32) // 8))
    except Exception:  # noqa: BLE001 — thread cap is best-effort
        pass
    try:
        if stage in RETRIEVAL_PROCESS_STAGES:
            # The parent pins the query-embedding device via a process-global
            # registry (set_embedding_device) — re-stamp it here or the child
            # builds the encoder on llama_index's default cuda:0 and collides
            # with whatever vLLM engine sits there.
            device = kwargs.get("device")
            if device:
                from rag_stack_evaluator.static_rag_evaluator.embedding.base import (
                    set_embedding_device,
                )
                set_embedding_device(str(device))
        import importlib
        cls = getattr(importlib.import_module(cls_module), cls_name)
        instance = cls(project_dir, **kwargs)
        conn.send(("ready", None))
    except BaseException as exc:  # noqa: BLE001 — report, then die
        try:
            conn.send(("init_error", f"{type(exc).__name__}: {exc}"))
        finally:
            return
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break
        if msg is None:
            break
        df, params = msg
        try:
            result = instance.pure(df, **params)
            report_reader = getattr(
                instance, "pop_last_forward_execution_report", None
            )
            forward_execution_report = (
                report_reader() if callable(report_reader) else None
            )
            slim = (
                _pack_reranker_result(df, result)
                if stage == "passage_reranker" else None
            )
            if slim is not None:
                if forward_execution_report is None:
                    conn.send(("ok_ids", slim))
                else:
                    conn.send((
                        "ok_ids_with_forward_execution",
                        (slim, forward_execution_report),
                    ))
            else:
                if forward_execution_report is None:
                    conn.send(("ok", result))
                else:
                    conn.send((
                        "ok_with_forward_execution",
                        (result, forward_execution_report),
                    ))
        except BaseException as exc:  # noqa: BLE001 — mirror to parent
            conn.send(("error", f"{type(exc).__name__}: {exc}"))
    conn.close()


class AuxProcessStage:
    """Drop-in stand-in for a stage instance whose ``pure`` runs in a
    dedicated child process. Exposes the same ``pure(df, **params)``
    surface the stage services call (from a worker thread — the call
    blocks on the pipe, never the event loop)."""

    def __init__(self, cls: type, project_dir: str, kwargs: Dict[str, Any],
                 stage: Optional[str] = None):
        self._label = f"{cls.__module__}.{cls.__name__}"
        self._stage = str(stage) if stage is not None else None
        self.last_forward_execution_report = None
        ctx = mp.get_context("spawn")  # CUDA-safe
        self._conn, child_conn = ctx.Pipe()
        self._lock = threading.Lock()
        self._proc = ctx.Process(
            target=_worker_main,
            args=(child_conn, cls.__module__, cls.__name__,
                  project_dir, kwargs, self._stage),
            daemon=True,
            name=f"aux:{cls.__name__}",
        )
        self._proc.start()
        child_conn.close()
        status, payload = self._conn.recv()  # blocks until model loaded
        if status != "ready":
            self.close()
            raise RuntimeError(
                f"aux worker {self._label} failed to initialize: {payload}"
            )
        logger.info(
            f"aux process up: {self._label} (pid {self._proc.pid})"
        )

    def pure(self, df, **params):
        ship = df
        keep = _SHIP_COLUMNS.get(self._stage or "")
        if keep:
            ship = df[[c for c in df.columns if c in keep]]
        with self._lock:  # one in-flight call per worker (serialized stage)
            self.last_forward_execution_report = None
            self._conn.send((ship, params))
            status, payload = self._conn.recv()
        if status == "ok":
            return payload
        if status == "ok_ids":
            return _unpack_reranker_result(ship, payload)
        if status == "ok_with_forward_execution":
            result, report = payload
            self.last_forward_execution_report = dict(report)
            return result
        if status == "ok_ids_with_forward_execution":
            slim, report = payload
            self.last_forward_execution_report = dict(report)
            return _unpack_reranker_result(ship, slim)
        raise RuntimeError(
            f"aux worker {self._label} failed: {payload}"
        )

    def close(self) -> None:
        try:
            if self._proc.is_alive():
                try:
                    self._conn.send(None)
                except Exception:  # noqa: BLE001
                    pass
                self._proc.join(timeout=10)
                if self._proc.is_alive():
                    self._proc.kill()
        except Exception:  # noqa: BLE001 — teardown is best-effort
            pass
        finally:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass

    def __del__(self):  # last-resort cleanup
        self.close()
