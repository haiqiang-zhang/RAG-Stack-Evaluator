"""Frozen trace contract for measured performance-workload sampling.

This is deliberately separate from the quality-trace contract.  A quality
trace contains the first completed invocation for every dataset row; this
payload contains a bounded, deterministic sample of *all* requests that
completed inside the authoritative measurement window.  It carries call shape
only -- never QPS, latency, timestamps, queueing, or service measurements.
"""

from __future__ import annotations

from typing import Any, Sequence

from rag_stack.rag_ir import (
    make_quality_trace_envelope,
    validate_quality_trace_envelope,
)


PERFORMANCE_TRACE_KIND = "measurement_phase_completion_sample"
PERFORMANCE_TRACE_SAMPLING_ALGORITHM = "sha256_bottom_k"
PERFORMANCE_TRACE_PRIORITY_SEED = "rag-stack-performance-trace-v1"
PERFORMANCE_TRACE_POPULATION_SCOPE = "measurement_phase_completion_window"

_ENVELOPE_FIELDS = frozenset({"trace_kind", "sampling", "queries"})
_SAMPLING_FIELDS = frozenset({
    "algorithm",
    "priority_seed",
    "capacity",
    "population_queries",
    "sample_queries",
    "population_scope",
})
_QUERY_FIELDS = frozenset({"invocation_id", "calls"})


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"performance trace {field} must be a positive integer")
    return value


def _validate_calls_strict(
    traces: Sequence[list[dict]],
    invocation_ids: Sequence[Any],
) -> None:
    """Validate the original call objects before any producer projection.

    The quality factory deliberately projects recorder-private fields. That is
    useful for quality production, but a persisted performance trace must fail
    closed if a latency/QPS/timestamp field appears on any call. Build the
    shared quality envelope by hand so its strict validator sees every key.
    """

    if len(traces) != len(invocation_ids):
        raise ValueError(
            "performance trace invocation_ids/traces cardinality mismatch"
        )
    validate_quality_trace_envelope({
        "queries": [
            {
                "question_id": f"performance-sample-{index}",
                "invocation_id": invocation_id,
                "calls": calls,
            }
            for index, (invocation_id, calls) in enumerate(
                zip(invocation_ids, traces)
            )
        ],
    })


def make_performance_trace_envelope(
    traces: Sequence[list[dict]],
    *,
    invocation_ids: Sequence[Any],
    capacity: int,
    population_queries: int,
) -> dict[str, Any]:
    """Build the independent measured-performance trace envelope.

    ``traces`` must already be the deterministic reservoir sample.  Reusing
    the frozen quality-call projector/validator here keeps the per-call schema
    identical without giving the two envelopes the same workload semantics.
    """

    trace_list = list(traces)
    ids = list(invocation_ids)
    if len(trace_list) != len(ids):
        raise ValueError(
            "performance trace invocation_ids/traces cardinality mismatch "
            f"({len(ids)} != {len(trace_list)})"
        )
    capacity = _positive_int(capacity, field="sampling.capacity")
    population_queries = _positive_int(
        population_queries, field="sampling.population_queries"
    )
    if not trace_list:
        raise ValueError("performance trace sample is empty")
    expected_sample = min(capacity, population_queries)
    if len(trace_list) != expected_sample:
        raise ValueError(
            "performance trace sample must contain the complete bottom-k cohort "
            f"({len(trace_list)} != min({capacity}, {population_queries})="
            f"{expected_sample})"
        )

    # Fail on unknown call fields BEFORE the quality factory projects/copies
    # the validated records into a detached envelope.
    _validate_calls_strict(trace_list, ids)
    projected = make_quality_trace_envelope(
        trace_list,
        question_ids=[f"performance-sample-{index}" for index in range(len(ids))],
        invocation_ids=ids,
    )
    envelope = {
        "trace_kind": PERFORMANCE_TRACE_KIND,
        "sampling": {
            "algorithm": PERFORMANCE_TRACE_SAMPLING_ALGORITHM,
            "priority_seed": PERFORMANCE_TRACE_PRIORITY_SEED,
            "capacity": capacity,
            "population_queries": population_queries,
            "sample_queries": len(trace_list),
            "population_scope": PERFORMANCE_TRACE_POPULATION_SCOPE,
        },
        "queries": [
            {
                "invocation_id": query["invocation_id"],
                "calls": query["calls"],
            }
            for query in projected["queries"]
        ],
    }
    validate_performance_trace_envelope(envelope)
    return envelope


def validate_performance_trace_envelope(envelope: Any) -> None:
    """Fail closed unless ``envelope`` has the exact performance-trace shape."""

    if not isinstance(envelope, dict):
        raise TypeError("performance trace envelope must be a dict")
    unknown = set(envelope) - _ENVELOPE_FIELDS
    if unknown:
        raise ValueError(
            f"performance trace envelope has unsupported fields: {sorted(unknown)}"
        )
    if envelope.get("trace_kind") != PERFORMANCE_TRACE_KIND:
        raise ValueError(
            f"performance trace envelope requires trace_kind={PERFORMANCE_TRACE_KIND!r}"
        )

    sampling = envelope.get("sampling")
    if not isinstance(sampling, dict):
        raise TypeError("performance trace 'sampling' must be an object")
    unknown_sampling = set(sampling) - _SAMPLING_FIELDS
    if unknown_sampling:
        raise ValueError(
            "performance trace sampling has unsupported fields: "
            f"{sorted(unknown_sampling)}"
        )
    missing_sampling = _SAMPLING_FIELDS - set(sampling)
    if missing_sampling:
        raise ValueError(
            "performance trace sampling missing fields: "
            f"{sorted(missing_sampling)}"
        )
    if sampling["algorithm"] != PERFORMANCE_TRACE_SAMPLING_ALGORITHM:
        raise ValueError("performance trace has unsupported sampling algorithm")
    if sampling["priority_seed"] != PERFORMANCE_TRACE_PRIORITY_SEED:
        raise ValueError("performance trace has unsupported priority seed")
    if sampling["population_scope"] != PERFORMANCE_TRACE_POPULATION_SCOPE:
        raise ValueError("performance trace has unsupported population scope")
    capacity = _positive_int(sampling["capacity"], field="sampling.capacity")
    population = _positive_int(
        sampling["population_queries"], field="sampling.population_queries"
    )
    sample_count = _positive_int(
        sampling["sample_queries"], field="sampling.sample_queries"
    )

    queries = envelope.get("queries")
    if not isinstance(queries, list) or not queries:
        raise ValueError("performance trace 'queries' must be a non-empty list")
    if len(queries) != sample_count:
        raise ValueError("performance trace sample_queries does not match queries")
    expected_sample = min(capacity, population)
    if sample_count != expected_sample:
        raise ValueError(
            "performance trace sample_queries must equal the complete bottom-k "
            f"cohort ({sample_count} != min({capacity}, {population})="
            f"{expected_sample})"
        )

    invocation_ids: list[Any] = []
    traces: list[list[dict]] = []
    seen: set[Any] = set()
    for position, query in enumerate(queries):
        if not isinstance(query, dict):
            raise TypeError(f"performance trace query {position} must be an object")
        unknown_query = set(query) - _QUERY_FIELDS
        if unknown_query:
            raise ValueError(
                f"performance trace query {position} has unsupported fields: "
                f"{sorted(unknown_query)}"
            )
        invocation_id = query.get("invocation_id")
        if invocation_id is None or (
            isinstance(invocation_id, str) and not invocation_id.strip()
        ):
            raise ValueError(
                f"performance trace query {position} requires invocation_id"
            )
        try:
            duplicate = invocation_id in seen
        except TypeError as exc:
            raise TypeError("performance trace invocation_id must be hashable") from exc
        if duplicate:
            raise ValueError(
                f"performance trace has duplicate invocation_id={invocation_id!r}"
            )
        seen.add(invocation_id)
        calls = query.get("calls")
        if not isinstance(calls, list) or not calls:
            raise ValueError(
                f"performance trace query {position} requires non-empty calls"
            )
        invocation_ids.append(invocation_id)
        traces.append(calls)

    # Defence in depth for the EXACT original call objects: call whitelists,
    # stage taxonomy, terminal completeness, and step order all fail closed.
    _validate_calls_strict(traces, invocation_ids)


def performance_trace_calls(
    envelope: Any,
) -> tuple[list[list[dict]], list[Any]]:
    """Return validated calls/identities for an explicit consumer adapter."""

    validate_performance_trace_envelope(envelope)
    queries = envelope["queries"]
    return (
        [query["calls"] for query in queries],
        [query["invocation_id"] for query in queries],
    )
