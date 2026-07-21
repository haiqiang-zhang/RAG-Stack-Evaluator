"""Call-level trace recording — ORTHOGONAL to measured / quality-only mode.

A run-scoped :class:`TraceRecorder` plus thin ``Recording*`` model wrappers turn
``trace_v1`` into a *natural byproduct of pipeline execution*: every real model
call (generate / encode / retrieve / rerank / compress) records itself, keyed by
the query's PERMANENT ``qid``. This replaces both the post-hoc
``build_static_execution_dag`` reconstruction (sequential) and the hand-rolled
``react_trace`` (agentic) with one mechanism.

Two orthogonal axes:

* **mode** (measured vs quality-only) decides *which* model instances run and
  whether we time them.
* **recording** (this module) decides whether calls emit a trace.

They compose freely. The recorder is bound once at the SHARED ``_run_pipeline``
(``set_current_recorder``) and consulted by the wrappers via a contextvar, so the
trace comes out identically in either mode. The wrappers are attached wherever a
mode obtains its models (measured: the ModelCache; quality-only: per-query node
construction) but share this recorder + the ``rag_ir`` ``TraceCall`` schema.

Per-query attribution uses the permanent ``qid`` (NOT a batch position): whoever
builds a batch — the sequential node runner over the dataframe, or the agentic
loop over its frontier — declares the batch's ``qids`` (in submission order) via
:func:`set_current_qids`. Because the id is permanent, multi-round (agentic) and
fan-out (query-expansion → N sub-queries) calls accumulate under the right query.
"""
from __future__ import annotations

import contextvars
import copy
import functools
import threading
from typing import Any, List, Optional, Sequence


# FORMAL tokenization: every token count comes from a real tokenizer — the call's own
# model when resolvable, else this fixed byte-level-BPE default. NEVER whitespace (that is
# only a last-ditch guard if transformers itself is unavailable, and it warns).
_DEFAULT_TOKENIZER_MODEL = "gpt2"
_warned_whitespace = [False]

# Deferred measured traces can contain hundreds of long, multi-round ReAct
# invocations.  Finalising every retained call as one tokenizer batch creates
# a transient allocation proportional to the whole performance reservoir.  A
# bounded batch keeps finalisation memory proportional to one small slice while
# preserving the exact per-text tokenizer/UTF-8 results and trace order.
_TRACE_FINALIZE_BATCH_SIZE = 128


@functools.lru_cache(maxsize=32)
def _load_local_tokenizer(name: str):
    """Load one tokenizer strictly from the local Hugging Face cache.

    Trace materialisation runs after the authoritative measured window.  It
    must therefore be deterministic and must never turn an otherwise complete
    measurement into a sequence of Hugging Face network retries.  Caching by
    the resolved repository name also lets different unknown/component ids
    share the same already-loaded fallback tokenizer.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name, local_files_only=True)


@functools.lru_cache(maxsize=16)
def _tokenizer_for(model_id: Optional[str]):
    """Cached local HF tokenizer for the model, else the fixed local default.

    ``model_id`` is trace provenance and can legitimately be a non-HF
    component identity (for example ``faiss_ivf_index``).  Such identities
    fail locally and immediately before falling back; they are never probed on
    the network.
    """
    try:
        from rag_stack.model_map import resolve_tokenizer_name
        names = []
        if model_id:
            names.append(resolve_tokenizer_name(model_id))
        names.append(_DEFAULT_TOKENIZER_MODEL)
        for name in dict.fromkeys(names):
            try:
                return _load_local_tokenizer(name)
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001
        pass
    return None


def count_tokens_batch(texts: Sequence[Any], model_id: Optional[str] = None) -> List[int]:
    """FORMAL token count per text (the model's tokenizer, else the default real tokenizer).
    Used for INPUT tokens; OUTPUT tokens of an LLM should instead use the exact
    ``len(token_ids)`` the model returned. Whitespace is only a logged last resort."""
    tok = _tokenizer_for(model_id)
    if tok is None and not _warned_whitespace[0]:
        _warned_whitespace[0] = True
        import logging
        logging.getLogger("RAG-Stack").warning(
            "recording: no tokenizer available (transformers load failed); token counts "
            "fall back to whitespace — INSTALL/FIX transformers for formal counts.")
    out: List[int] = []
    # Batch fast path: HF fast tokenizers batch-encode in Rust — identical
    # per-text counts, far cheaper than per-text .encode() in a loop.
    if tok is not None and len(texts) > 1:
        try:
            enc = tok([str(t) for t in texts], add_special_tokens=True)
            return [max(1, len(ids)) for ids in enc["input_ids"]]
        except Exception:  # noqa: BLE001
            pass
    for t in texts:
        s = str(t)
        try:
            out.append(max(1, len(tok.encode(s)) if tok is not None else len(s.split())))
        except Exception:  # noqa: BLE001
            out.append(max(1, len(s.split())))
    return out


def count_bytes_batch(texts: Sequence[Any]) -> List[int]:
    """Exact UTF-8 byte count per text — tokenizer-independent ground truth."""
    return [len(str(t).encode("utf-8")) for t in texts]

# ---------------------------------------------------------------------------
# Run-scoped context: the active recorder + the current batch's permanent qids.
# Contextvars (not globals) so concurrent evals / async never cross-talk, mirror-
# ing measured/cache.py's set_current/get_current pattern.
# ---------------------------------------------------------------------------
_current_recorder: contextvars.ContextVar["TraceRecorder | None"] = (
    contextvars.ContextVar("trace_recorder", default=None)
)
_current_qids: contextvars.ContextVar["list | None"] = (
    contextvars.ContextVar("trace_qids", default=None)
)


def set_current_recorder(recorder: "TraceRecorder | None") -> None:
    _current_recorder.set(recorder)


def get_current_recorder() -> "TraceRecorder | None":
    return _current_recorder.get()


def clear_current_recorder() -> None:
    _current_recorder.set(None)
    _current_qids.set(None)


def set_current_qids(qids: Optional[Sequence[Any]]) -> None:
    """Declare the permanent qids of the batch about to be submitted, in the SAME
    order the model will receive its inputs. Called by the batch builder (the
    sequential node runner once per run = the dataframe's qid order; the agentic
    loop once per round = its frontier). The wrappers zip results → these qids."""
    _current_qids.set(list(qids) if qids is not None else None)


def get_current_qids() -> Optional[list]:
    return _current_qids.get()


def recording_active() -> bool:
    return _current_recorder.get() is not None


def _join_text(item) -> str:
    if isinstance(item, (list, tuple)):
        return " ".join(str(x) for x in item if x is not None)
    return str(item) if item is not None else ""


def _freeze_pending_text(item: Any) -> Any:
    """Take a cheap, stable snapshot without joining or encoding text.

    Serving-stage payloads are strings or one-level lists/tuples of strings.
    Keeping strings by reference and copying only the outer sequence avoids the
    expensive passage join while preventing a caller from later changing list
    membership.  Unusual mutable/non-string values are stringified now so the
    deferred readout is identical to the historical eager ``_join_text`` call.
    """

    if item is None:
        return ""
    if type(item) is str:
        return item
    if isinstance(item, (list, tuple)):
        return tuple(
            value
            if value is None or type(value) is str
            else str(value)
            for value in item
        )
    return str(item)


def record_io(
    stage,
    qids,
    in_texts,
    *,
    out_texts=None,
    out_token_ids=None,
    model_id=None,
):
    """The single node-facing API: record one ``stage`` call per query with FORMAL token
    counts AND exact UTF-8 bytes, for input (and output when it has text). ``stage`` is a
    taxonomy name — a user stage (``generator``, ``query_expansion``, …) or a split
    retrieval stage (``semantic_retrieval_encode`` / ``semantic_retrieval_vectorsearch``).
    Each ``in_texts`` / ``out_texts`` item may be a str or a list-of-str (passages) —
    joined before counting. ``out_token_ids`` (per query) supplies the EXACT output token
    count for LLM calls, overriding tokenization of ``out_texts``. No-op when no
    recorder / no qids bound."""
    rec = get_current_recorder()
    if rec is None or not qids:
        return
    # The measurement path performs NO joining, UTF-8 encoding, or
    # tokenization here. record_batch takes a shallow immutable snapshot of
    # each raw item; only the final retained cohorts are materialised after the
    # measured services close. Exact output-token ids are reduced to their
    # lengths now because the model-owned lists need not remain alive.
    n = len(qids)
    if out_token_ids is not None:
        out_tok = [len(t) if hasattr(t, "__len__") else 1 for t in out_token_ids]
    elif out_texts is not None:
        out_tok = None
    else:
        out_tok = [0] * n
    rec.record_batch(
        stage,
        qids=qids,
        input_tokens=[0] * n,
        output_tokens=out_tok,
        model_id=model_id,
        pending_in_texts=in_texts,
        pending_out_texts=out_texts,
        pending_out_needs_tokens=out_token_ids is None,
    )


class TraceRecorder:
    """Accumulates one ordered call list per permanent ``qid``.

    A run-scoped contextvar prevents cross-run mixing. An internal re-entrant
    lock serializes worker-thread recording with final quality-cohort projection
    and readout; projection freezes the recorder so late cancelled work cannot
    repopulate the trace.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # Python dict insertion order is the canonical first-seen order. A
        # deletion is O(1) and preserves survivor order, so no parallel order
        # list needs to be rebuilt for every measured completion.
        self._by_qid: dict[Any, List[dict]] = {}
        self._ordinal_by_qid: dict[Any, int] = {}
        self._next_ordinal = 0
        self._frozen = False
        self._selected_order: Optional[List[Any]] = None

    @property
    def qids(self) -> tuple[Any, ...]:
        """The exact qid order that :meth:`to_trace_v1` will emit.

        Returning an immutable snapshot keeps callers from mutating the
        recorder's insertion-ordered backing dict. Quality-only evaluation
        does not use the filtering APIs below, so its historical first-seen
        order remains the default.
        """
        with self._lock:
            order = self._selected_order
            return tuple(order if order is not None else self._by_qid)

    @staticmethod
    def _validated_qids(qids: Sequence[Any]) -> List[Any]:
        """Materialise and validate a qid projection before mutating state.

        Duplicate identities are always an error: accepting them would make
        the positional ``trace_v1`` payload contain the same request twice and
        silently break its one-query/one-trace invariant.
        """
        requested = list(qids)
        seen: set[Any] = set()
        duplicates: List[Any] = []
        try:
            for qid in requested:
                if qid in seen:
                    duplicates.append(qid)
                else:
                    seen.add(qid)
        except TypeError as exc:
            raise TypeError("trace qids must be hashable permanent identities") from exc
        if duplicates:
            raise ValueError(f"duplicate trace qids are not allowed: {duplicates!r}")
        return requested

    def select_qids(
        self,
        qids: Sequence[Any],
        *,
        require_all: bool = True,
    ) -> tuple[Any, ...]:
        """Atomically retain only ``qids``, in the caller's exact order.

        Measured runtime uses this to project repeated closed-loop invocations
        down to one quality invocation per dataset row, in dataset order.
        Unknown identities fail closed by default. With ``require_all=False``
        they are omitted so the caller can publish an explicit integrity
        verdict instead of retaining unrelated duplicates.
        """
        with self._lock:
            # Freeze BEFORE validation/finalization. Cancellation-resistant
            # worker threads may still unwind after the async cancel grace;
            # every selected quality invocation is already done, so any new
            # write is necessarily outside the selected dataset cohort. A
            # failed projection stays frozen (fail closed).
            self._frozen = True
            requested = self._validated_qids(qids)
            missing = [qid for qid in requested if qid not in self._by_qid]
            if missing and require_all:
                raise KeyError(f"trace qids were never recorded: {missing!r}")
            selected = [qid for qid in requested if qid in self._by_qid]
            # Defer the actual pruning to to_trace_v1(). It runs only after
            # the measured runtime has closed its services, so the expensive
            # deferred tokenization (winners only — unselected closed-loop
            # peers are dropped before it) never keeps GPU services alive.
            self._selected_order = selected
            return tuple(selected)

    def discard_qids(
        self,
        qids: Sequence[Any],
        *,
        require_all: bool = True,
    ) -> tuple[Any, ...]:
        """Atomically discard identities while preserving survivor order.

        The return value is the discarded identities in their previous trace
        order.  As with :meth:`select_qids`, duplicate input is rejected and
        unknown qids fail closed unless explicitly allowed.
        """
        with self._lock:
            if self._selected_order is not None:
                raise RuntimeError("cannot discard qids after trace projection")
            requested = self._validated_qids(qids)
            missing = [qid for qid in requested if qid not in self._by_qid]
            if missing and require_all:
                raise KeyError(f"trace qids were never recorded: {missing!r}")
            # Work is proportional only to the requested identities, never to
            # the retained population. Most measured calls discard one qid, so
            # this is O(1); sorting a multi-qid request preserves the documented
            # previous trace order without scanning/rebuilding the whole dict.
            present = [qid for qid in requested if qid in self._by_qid]
            removed = tuple(sorted(
                present, key=self._ordinal_by_qid.__getitem__,
            ))
            for qid in present:
                self._by_qid.pop(qid)
                self._ordinal_by_qid.pop(qid)
            return removed

    def trace_for_qids(
        self,
        qids: Sequence[Any],
        *,
        require_all: bool = True,
    ) -> List[List[dict]]:
        """Read a non-destructive trace projection in caller order.

        Measured serving freezes/selects the quality winners before services
        close, but must also emit an independent performance reservoir.  The
        underlying records are intentionally not pruned until
        :meth:`to_trace_v1`; this method reads that second cohort after service
        teardown while leaving the pending quality projection byte-identical.
        """

        with self._lock:
            requested = self._validated_qids(qids)
            missing = [qid for qid in requested if qid not in self._by_qid]
            if missing and require_all:
                raise KeyError(f"trace qids were never recorded: {missing!r}")
            selected = [qid for qid in requested if qid in self._by_qid]
            self._finalize_pending_unlocked(selected)
            return copy.deepcopy([self._by_qid[qid] for qid in selected])

    def record(
        self,
        qid: Any,
        stage: str,
        *,
        input_tokens: int,
        output_tokens: int = 0,
        input_bytes: int = 0,
        output_bytes: int = 0,
        model_id: Optional[str] = None,
        pending_in_text: Any = None,
        pending_out_text: Any = None,
        pending_out_needs_tokens: bool = True,
    ) -> None:
        frozen_in = (
            _freeze_pending_text(pending_in_text)
            if pending_in_text is not None
            else None
        )
        frozen_out = (
            _freeze_pending_text(pending_out_text)
            if pending_out_text is not None
            else None
        )
        with self._lock:
            if self._frozen:
                return
            self._record_unlocked(
                qid,
                stage,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                input_bytes=input_bytes,
                output_bytes=output_bytes,
                model_id=model_id,
                frozen_in=frozen_in,
                frozen_out=frozen_out,
                pending_out_needs_tokens=pending_out_needs_tokens,
            )

    def _record_unlocked(
        self,
        qid: Any,
        stage: str,
        *,
        input_tokens: int,
        output_tokens: int,
        input_bytes: int,
        output_bytes: int,
        model_id: Optional[str],
        frozen_in: Any,
        frozen_out: Any,
        pending_out_needs_tokens: bool,
    ) -> None:
        lst = self._by_qid.get(qid)
        if lst is None:
            lst = self._by_qid[qid] = []
            self._ordinal_by_qid[qid] = self._next_ordinal
            self._next_ordinal += 1
        entry = {
            "stage": str(stage),
            "input_tokens": int(max(0, input_tokens)),
            "output_tokens": int(max(0, output_tokens)),
            "input_bytes": int(max(0, input_bytes)),
            "output_bytes": int(max(0, output_bytes)),
            "step_idx": len(lst),
            "model_id": model_id,
        }
        # Raw immutable payloads stay private until final cohort readout. These
        # keys are removed before trace_v1/performance trace publication.
        if frozen_in is not None:
            entry["_pend_in"] = frozen_in
        if frozen_out is not None:
            entry["_pend_out"] = frozen_out
            if pending_out_needs_tokens:
                entry["_pend_out_tokens"] = True
        lst.append(entry)

    def record_batch(
        self,
        stage: str,
        *,
        input_tokens: Sequence[int],
        output_tokens: Optional[Sequence[int]] = None,
        input_bytes: Optional[Sequence[int]] = None,
        output_bytes: Optional[Sequence[int]] = None,
        model_id: Optional[str] = None,
        qids: Optional[Sequence[Any]] = None,
        pending_in_texts: Optional[Sequence[Any]] = None,
        pending_out_texts: Optional[Sequence[Any]] = None,
        pending_out_needs_tokens: bool = True,
    ) -> None:
        """Record one entry per batch element under its permanent qid. ``qids``
        defaults to :func:`get_current_qids`; the i-th input/output maps to the
        i-th qid (submission order). Length mismatch falls back to recording the
        min overlap (defensive — a node that drops rows shouldn't crash the run)."""
        if qids is None:
            qids = get_current_qids()
        if not qids:
            return  # no attribution available → skip (don't guess positions)
        z = [0] * len(input_tokens)
        outs = output_tokens if output_tokens is not None else z
        ib = input_bytes if input_bytes is not None else z
        ob = output_bytes if output_bytes is not None else z
        n = min(len(qids), len(input_tokens), len(outs))
        # Materialise only the outer batch container. This preserves the old
        # iterable semantics (e.g. a Series with non-positional labels) while
        # leaving all text materialisation deferred.
        pending_in_values = (
            list(pending_in_texts) if pending_in_texts is not None else None
        )
        pending_out_values = (
            list(pending_out_texts) if pending_out_texts is not None else None
        )
        frozen_ins = [
            _freeze_pending_text(pending_in_values[i])
            if pending_in_values is not None and i < len(pending_in_values)
            else None
            for i in range(n)
        ]
        frozen_outs = [
            _freeze_pending_text(pending_out_values[i])
            if pending_out_values is not None and i < len(pending_out_values)
            else None
            for i in range(n)
        ]
        with self._lock:
            if self._frozen:
                return
            for i in range(n):
                self._record_unlocked(
                    qids[i], stage,
                    input_tokens=input_tokens[i], output_tokens=outs[i],
                    input_bytes=ib[i] if i < len(ib) else 0,
                    output_bytes=ob[i] if i < len(ob) else 0,
                    model_id=model_id,
                    frozen_in=frozen_ins[i],
                    frozen_out=frozen_outs[i],
                    pending_out_needs_tokens=pending_out_needs_tokens,
                )

    def _finalize_pending(self) -> None:
        with self._lock:
            self._finalize_pending_unlocked(self._by_qid)

    def _finalize_pending_unlocked(self, qids: Sequence[Any]) -> None:
        """Materialise only ``qids`` after measurement, exactly once.

        Joining, UTF-8 byte counting, and tokenizer work are all deferred to
        this final cohort boundary. Tokenization stays grouped by model and
        uses the same helpers as the historical eager path, so published
        bytes/tokens are bit-identical.
        """

        token_groups: dict[Optional[str], list] = {}
        byte_only_items: list[tuple[dict, str, Any]] = []
        finalized_entries: list[dict] = []
        for qid in qids:
            for entry in self._by_qid[qid]:
                has_pending = False
                if "_pend_in" in entry:
                    token_groups.setdefault(entry.get("model_id"), []).append(
                        (
                            entry,
                            "input_bytes",
                            "input_tokens",
                            entry["_pend_in"],
                        )
                    )
                    has_pending = True
                if "_pend_out" in entry:
                    if entry.get("_pend_out_tokens"):
                        token_groups.setdefault(entry.get("model_id"), []).append(
                            (
                                entry,
                                "output_bytes",
                                "output_tokens",
                                entry["_pend_out"],
                            )
                        )
                    else:
                        byte_only_items.append(
                            (entry, "output_bytes", entry["_pend_out"])
                        )
                    has_pending = True
                if has_pending:
                    finalized_entries.append(entry)

        for model_id, items in token_groups.items():
            for start in range(0, len(items), _TRACE_FINALIZE_BATCH_SIZE):
                chunk = items[start:start + _TRACE_FINALIZE_BATCH_SIZE]
                texts = [_join_text(item[3]) for item in chunk]
                byte_counts = count_bytes_batch(texts)
                token_counts = count_tokens_batch(texts, model_id)
                for item, byte_count, token_count in zip(
                    chunk, byte_counts, token_counts
                ):
                    entry, byte_field, token_field, _pending = item
                    entry[byte_field] = int(max(0, byte_count))
                    entry[token_field] = int(max(0, token_count))
        for start in range(0, len(byte_only_items), _TRACE_FINALIZE_BATCH_SIZE):
            chunk = byte_only_items[start:start + _TRACE_FINALIZE_BATCH_SIZE]
            texts = [_join_text(item[2]) for item in chunk]
            byte_counts = count_bytes_batch(texts)
            for item, byte_count in zip(chunk, byte_counts):
                entry, byte_field, _pending = item
                entry[byte_field] = int(max(0, byte_count))
        for entry in finalized_entries:
            entry.pop("_pend_in", None)
            entry.pop("_pend_out", None)
            entry.pop("_pend_out_tokens", None)

    def to_trace_v1(self) -> List[List[dict]]:
        """The ``trace_v1`` payload: one ordered call list per query, in
        first-seen qid order (= dataframe order for the sequential path)."""
        with self._lock:
            # Apply a pending select_qids() projection FIRST, so the deferred
            # tokenization below pays only for the emitted winners — the
            # unselected closed-loop peers are dropped untokenized.
            if self._selected_order is not None:
                order = self._selected_order
                self._by_qid = {qid: self._by_qid[qid] for qid in order}
                self._ordinal_by_qid = {
                    qid: index for index, qid in enumerate(order)
                }
                self._selected_order = None
            order = tuple(self._by_qid)
            self._finalize_pending_unlocked(order)
            return [self._by_qid[qid] for qid in order]

    def __len__(self) -> int:
        with self._lock:
            order = self._selected_order
            return len(order if order is not None else self._by_qid)
