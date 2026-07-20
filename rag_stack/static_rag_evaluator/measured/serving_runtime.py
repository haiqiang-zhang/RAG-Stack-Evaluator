"""Measured serving runtime for saturated RAG services.

This module owns the measured-mode scheduling semantics. It deliberately models
closed-loop saturated serving rather than the quality runner's node-line
execution:

* generator and vLLM-backed query expansion submit independent async requests to
  vLLM; server-side continuous batching is left to vLLM.
* retrieval, reranker, compressor, and prompt maker use request-batch capped
  dynamic batching.
* sequential and ReAct share the same services; only the flow policy differs.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import faulthandler
import hashlib
import logging
import math
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from rag_stack.static_rag_evaluator.measured import performance as perf_mod
from rag_stack.static_rag_evaluator.measured.artifact_contract import (
	validate_measured_gt_admissibility,
)
from rag_stack.static_rag_evaluator.measured.performance_trace import (
	PERFORMANCE_TRACE_PRIORITY_SEED,
	make_performance_trace_envelope,
)
from rag_stack.system_layout import (
	decode_batch,
	dynamic_batch_timeout_s,
	engine_info,
	request_batch,
	stage_devices,
	vllm_max_num_seqs,
)

logger = logging.getLogger("RAG-Stack")

# Wall-clock caps for the closed-loop measurement. A genuinely slow config (a
# heavy reranker/compressor that is the GPU bottleneck) would otherwise take
# ~1.5h to run the full load_concurrency×5 target at ~1 qps.  The warmup cap is
# a FAIL-CLOSED budget: it must never turn an unsaturated or still-settling
# warmup into a measured window.  Once the saturation + rate-stability gate has
# passed, the client population stays continuous across warmup→measured and
# authoritative qps is completions divided by the exact measured phase window.
# A completion-delta rate is retained as a diagnostic, but no longer shortens
# the denominator by dropping the first completion.
# Overridable per-config via
# system.warmup_wall_cap_s / system.measured_wall_cap_s.
_WARMUP_WALL_CAP_S = 240.0      # 4 min — enough to fill the pipeline to steady state
# qps is a STEADY-STATE rate: once the window sits in steady state (warmup
# gate), its length only trades variance, not mean — 6 min at ~1 qps still
# yields hundreds of completions. User decision 07-06: measure the rate, not
# a completion count.
_MEASURED_WALL_CAP_S = 360.0    # 6 min steady-state window
_PHASE_DRAIN_GRACE_S = 120.0    # hard guard before the first startup completion
_PHASE_CANCEL_GRACE_S = 10.0    # after all QA rows are real, cancel duplicate stragglers
_QUALITY_DRAIN_GRACE_S = 30.0   # after perf cap, briefly accept natural QA completions
_NOFILE_SOFT_TARGET = 65535

_GT_MIN_RATE_STABILITY_WAVES = 2.0
_GT_MAX_QPS_SUBWINDOW_CV = 0.15
_WARMUP_RATE_STABILITY_RELATIVE_TOLERANCE = 0.10
_SATURATION_STABILITY_MIN_SPAN = 32
_SATURATION_STABILITY_BATCH_CYCLES = 4
_SATURATION_STABILITY_MAX_SPAN = 256
_PERFORMANCE_TRACE_SAMPLE_CAPACITY = 512
_GT_RATE_STABLE_GATES = frozenset({
	"warmup_completion_rate_stable",
	"post_stall_completion_rate_stable",
})

# Existing stationarity-proof dimensions, named once so the low-QPS fallback
# and the final proof cannot derive mutually incompatible completion spans.
_GT_QPS_WALL_SUBWINDOWS = 5
_GT_QPS_MIN_COMPLETION_SPANS_PER_WALL_SUBWINDOW = 3
_POPULATION_ADAPTER_SAMPLE_S = 1.0

# A closed-loop benchmark repeatedly reuses the finite QA dataset. Replaying
# the file order verbatim (``seq % n_rows``) phase-locks heterogeneous rows to
# fixed P/D and dynamic-batch boundaries. Use one exact, deterministic
# permutation per dataset-sized admission epoch instead. The fixed seed is
# scheduling policy, not a user/calibration knob.
_DATASET_ADMISSION_ORDER_SEED = b"rag-stack-measured-admission-v1"


def _dataset_admission_epoch_order(
	n_rows: int,
	epoch: int,
) -> tuple[int, ...]:
	"""Return one stable, exact permutation for an admission epoch."""
	if isinstance(n_rows, bool) or not isinstance(n_rows, int) or n_rows <= 0:
		raise ValueError("dataset admission order requires positive n_rows")
	if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
		raise ValueError("dataset admission epoch must be a non-negative integer")
	prefix = b":".join((
		_DATASET_ADMISSION_ORDER_SEED,
		str(n_rows).encode("ascii"),
		str(epoch).encode("ascii"),
	)) + b":"
	return tuple(sorted(
		range(n_rows),
		key=lambda idx: (
			hashlib.sha256(prefix + str(idx).encode("ascii")).digest(),
			idx,
		),
	))


class _BalancedDatasetAdmissionOrder:
	"""Map monotonic sequence IDs to balanced, epoch-varying row order.

	Only the current epoch is retained. Runtime sequence IDs are monotonic under
	the closed-loop state lock, so this bounds memory at one ``n_rows`` tuple and
	amortizes permutation construction over a full dataset cycle.
	"""

	def __init__(self, n_rows: int) -> None:
		if isinstance(n_rows, bool) or not isinstance(n_rows, int) or n_rows <= 0:
			raise ValueError("dataset admission order requires positive n_rows")
		self.n_rows = n_rows
		self._epoch = -1
		self._order: tuple[int, ...] = ()

	def idx_for_seq(self, seq: int) -> int:
		if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
			raise ValueError("dataset admission sequence must be non-negative")
		epoch, offset = divmod(seq, self.n_rows)
		if epoch != self._epoch:
			self._order = _dataset_admission_epoch_order(self.n_rows, epoch)
			self._epoch = epoch
		return self._order[offset]


@dataclass
class _PopulationRamp:
	"""One bounded driver-start transition used by the saturation controller.

	Scheduled driver tasks are not active closed-loop population until their
	start delay has elapsed.  Keeping that distinction explicit prevents the
	population adapter from reading a half-created population and treating its
	low fill as evidence that another population increment is needed.
	"""

	kind: str
	scheduled_drivers: int
	population_before: int
	population_after: int
	spread_s: float
	started_ts: float
	complete_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
	activated_drivers: int = 0
	cancelled_before_activation: int = 0
	completed_ts: Optional[float] = None

	@property
	def pending_drivers(self) -> int:
		return max(
			0,
			self.scheduled_drivers
			- self.activated_drivers
			- self.cancelled_before_activation,
		)

	def _resolve_one(self, *, activated: bool, now: float) -> None:
		if self.pending_drivers <= 0:
			raise RuntimeError("population ramp resolved more drivers than scheduled")
		if activated:
			self.activated_drivers += 1
		else:
			self.cancelled_before_activation += 1
		if self.pending_drivers == 0:
			self.completed_ts = float(now)
			self.complete_event.set()

	def mark_activated(self, now: float) -> None:
		self._resolve_one(activated=True, now=now)

	def mark_cancelled_before_activation(self, now: float) -> None:
		self._resolve_one(activated=False, now=now)

	def diagnostic(self, warmup_start: float) -> Dict[str, Any]:
		return {
			"kind": self.kind,
			"scheduled_drivers": self.scheduled_drivers,
			"population_before": self.population_before,
			"population_after": self.population_after,
			"spread_s": self.spread_s,
			"started_offset_s": max(0.0, self.started_ts - warmup_start),
			"completed_offset_s": (
				max(0.0, self.completed_ts - warmup_start)
				if self.completed_ts is not None else None
			),
			"activated_drivers": self.activated_drivers,
			"cancelled_before_activation": self.cancelled_before_activation,
			"pending_drivers": self.pending_drivers,
			"complete": self.complete_event.is_set(),
		}


def _request_failure_is_fatal(exc: BaseException) -> bool:
	"""Whether recycling a failed closed-loop slot can never recover.

	The provider classifies a propagated CUDA OOM as an infeasible deployment.
	Keeping it inside a driver instead retries the same aux-worker OOM until the
	warmup wall cap and obscures the actual deployment failure.
	"""
	return "CUDA out of memory" in str(exc)


def _completion_rate_stable(
	done_timestamps: List[float],
	span_queries: int,
	*,
	start_index: int = 0,
) -> bool:
	"""Compare two equally sized, non-overlapping completion spans.

	Each group of ``span_queries`` completions contains
	``span_queries - 1`` inter-completion intervals. Sharing the first
	completion of the recent group with the previous duration would compare
	``w - 1`` intervals with ``w`` intervals. At the minimum wall-budget span
	``w=10``, that creates a false 10.526% drift for a constant stream and
	fails the existing 10% stability gate.
	"""
	w = max(1, int(span_queries))
	start = max(0, int(start_index))
	if len(done_timestamps) - start < 2 * w:
		return False
	d_recent = done_timestamps[-1] - done_timestamps[-w]
	d_prev = done_timestamps[-w - 1] - done_timestamps[-2 * w]
	if d_recent < 0.5 and d_prev < 0.5:
		return True
	if d_recent <= 0.0 or d_prev <= 0.0:
		return False
	return (
		abs(d_recent - d_prev) / ((d_recent + d_prev) / 2.0)
		<= _WARMUP_RATE_STABILITY_RELATIVE_TOLERANCE
	)


def _wall_budget_stability_span(
	*,
	preferred_span: int,
	candidate_completions: int,
	candidate_age_s: float,
	measured_wall_cap_s: float,
) -> Dict[str, Any]:
	"""Derive one finite low-QPS evidence span from observations already made.

	The normal stage-cycle span remains preferred.  This fallback is only
	eligible after its caller has spent half of the finite overall warmup wall
	budget.  Candidate-local age and completions size W; two spans must already
	fit in that candidate's completed suffix.  The measured budget first tries to
	make the existing five-bin proof applicable (three full completion spans per
	bin, sized within the existing warmup-rate tolerance). If the finite budget
	is too small at the existing minimum rate resolution, it retains the
	sparse-wave proof's four-boundary minimum. These are validation dimensions,
	not calibration inputs.
	"""
	preferred = max(1, int(preferred_span))
	completed = max(0, int(candidate_completions))
	age_s = max(0.0, float(candidate_age_s))
	measured_cap_s = max(0.0, float(measured_wall_cap_s))
	minimum_span = int(math.ceil(
		1.0 / _WARMUP_RATE_STABILITY_RELATIVE_TOLERANCE
	))
	rate_qps = float(completed) / age_s if age_s > 0.0 else 0.0
	available_span = completed // 2
	expected_measurement_completions = int(math.floor(rate_qps * measured_cap_s))
	# Four completion boundaries yield three independent full-span rates in
	# _qps_stationarity_evidence. This is the fail-closed affordability bound.
	completion_proof_span = expected_measurement_completions // 4
	# The wall-time proof is applicable only when every one of its five bins has
	# at least three full spans. Account for the rate variation already allowed
	# by the warmup proof; otherwise a candidate can pass warmup at the low edge
	# of that tolerance but be assigned a span that makes the final wall proof
	# unreachable by construction.
	dense_proof_denominator = int(math.ceil(
		(
			_GT_QPS_WALL_SUBWINDOWS
			* _GT_QPS_MIN_COMPLETION_SPANS_PER_WALL_SUBWINDOW
		)
		/ (1.0 - _WARMUP_RATE_STABILITY_RELATIVE_TOLERANCE)
	))
	dense_proof_span = expected_measurement_completions // dense_proof_denominator
	measurement_span = (
		min(completion_proof_span, dense_proof_span)
		if dense_proof_span >= minimum_span
		else completion_proof_span
	)
	budget_span = min(available_span, measurement_span)
	effective_span = (
		min(preferred, budget_span)
		if budget_span >= minimum_span
		else None
	)
	return {
		"effective_span_queries": effective_span,
		"minimum_span_queries": minimum_span,
		"candidate_completions": completed,
		"candidate_age_s": age_s,
		"candidate_rate_qps": rate_qps,
		"available_span_queries": available_span,
		"measurement_span_queries": measurement_span,
	}


class _DeterministicTraceReservoir:
	"""Bounded bottom-k sample over successful completion-window qids.

	SHA-256 priorities make the result independent of completion order and
	therefore avoid the old first-completion selection bias.  The fixed seed
	makes the same completion cohort byte-for-byte reproducible.
	"""

	def __init__(self, capacity: int) -> None:
		if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
			raise ValueError("performance trace reservoir capacity must be positive")
		self.capacity = capacity
		self._priorities: Dict[Any, tuple[bytes, str]] = {}

	@staticmethod
	def _priority(qid: Any) -> tuple[bytes, str]:
		identity = str(qid)
		payload = f"{PERFORMANCE_TRACE_PRIORITY_SEED}\0{identity}".encode("utf-8")
		return hashlib.sha256(payload).digest(), identity

	def offer(self, qid: Any) -> Optional[Any]:
		"""Offer one completed invocation; return an evicted qid, if any."""
		if qid in self._priorities:
			raise ValueError(f"duplicate performance completion qid: {qid!r}")
		priority = self._priority(qid)
		if len(self._priorities) < self.capacity:
			self._priorities[qid] = priority
			return None
		worst_qid, worst_priority = max(
			self._priorities.items(), key=lambda item: item[1]
		)
		if priority >= worst_priority:
			return qid
		del self._priorities[worst_qid]
		self._priorities[qid] = priority
		return worst_qid

	def __contains__(self, qid: Any) -> bool:
		return qid in self._priorities

	@property
	def qids(self) -> tuple[Any, ...]:
		return tuple(
			qid for qid, _priority in sorted(
				self._priorities.items(), key=lambda item: item[1]
			)
		)


def _qps_subwindow_cv(
	done_timestamps: List[float],
	window_start: float,
	window_end: float,
	*,
	n_subwindows: int = 5,
) -> Optional[float]:
	"""Completion-count CV over equal wall-time subwindows."""
	if not done_timestamps or window_end <= window_start or n_subwindows <= 0:
		return None
	span = (window_end - window_start) / n_subwindows
	if span <= 0.0:
		return None
	counts = [0] * n_subwindows
	for completed_at in done_timestamps:
		index = int((completed_at - window_start) / span)
		if 0 <= index < n_subwindows:
			counts[index] += 1
		elif index >= n_subwindows:
			counts[-1] += 1
	mean = sum(counts) / n_subwindows
	if mean <= 0.0:
		return None
	return (
		sum((count - mean) ** 2 for count in counts) / n_subwindows
	) ** 0.5 / mean


def _wall_subwindow_workload_diagnostics(
	done_timestamps: List[float],
	output_tokens: List[int],
	window_start: float,
	window_end: float,
	*,
	n_subwindows: int = _GT_QPS_WALL_SUBWINDOWS,
) -> Dict[str, Any]:
	"""Summarize completion and token mix in equal wall-time subwindows.

	The caller supplies completion timestamps and output-token counts from the
	same already aligned performance cohort. These fields are diagnostics only;
	they do not participate in stationarity selection or GT admissibility.
	"""
	if len(done_timestamps) != len(output_tokens):
		raise ValueError(
			"wall-subwindow diagnostics lost completion/token alignment"
		)
	if window_end <= window_start or n_subwindows <= 0:
		return {}
	span = (window_end - window_start) / n_subwindows
	if span <= 0.0:
		return {}
	counts = [0] * n_subwindows
	token_totals = [0] * n_subwindows
	for completed_at, tokens in zip(done_timestamps, output_tokens):
		index = int((completed_at - window_start) / span)
		if index >= n_subwindows:
			index = n_subwindows - 1
		if index < 0:
			continue
		counts[index] += 1
		token_totals[index] += int(tokens)
	token_means = [
		total / count if count > 0 else 0.0
		for total, count in zip(token_totals, counts)
	]
	return {
		"window_subwindow_completions": counts,
		"window_subwindow_span_s": span,
		"window_subwindow_output_tokens": token_totals,
		"window_subwindow_mean_output_tokens": token_means,
		"window_subwindow_output_tokens_per_s": [
			total / span for total in token_totals
		],
	}


def _qps_stationarity_evidence(
	done_timestamps: List[float],
	window_start: float,
	window_end: float,
	*,
	completion_span_queries: Optional[int],
) -> Dict[str, Any]:
	"""Choose a stationarity proof without aliasing dynamic-batch waves.

	Equal wall-time bins are appropriate for continuous completion streams, but
	a stable large-batch worker can put one batch in one bin and two in the next.
	The saturation gate already derives a bounded completion span from the
	bottleneck's batch cap. Sparse batch waves use completion-aligned rates, while
	a stream with at least three complete spans in every wall bin keeps the wall
	proof: individual completion-span durations are too jitter-sensitive there.
	The first phase-partial span is excluded. A tail longer than any observed full
	span disables the completion-aligned proof so a real terminal stall still fails
	closed through the wall-bin fallback.
	"""
	wall_cv = _qps_subwindow_cv(
		done_timestamps,
		window_start,
		window_end,
	)
	evidence: Dict[str, Any] = {
		"selected_cv": wall_cv,
		"method": "wall_time_subwindows",
		"selection_reason": "completion_span_unavailable",
		"wall_subwindow_cv": wall_cv,
		"wall_subwindow_completions": [],
		"wall_subwindow_completion_spans": [],
		"wall_time_dense_completion_stream": False,
		"completion_span_cv": None,
		"completion_span_tail_change_cv": None,
		"completion_span_tail_change_suffix_spans": None,
		"completion_span_queries": None,
		"completion_span_rates_qps": [],
		"completion_span_durations_s": [],
		"completion_span_tail_s": None,
	}
	if (
		isinstance(completion_span_queries, bool)
		or not isinstance(completion_span_queries, int)
		or completion_span_queries <= 0
	):
		return evidence

	ordered = sorted(done_timestamps)
	span_queries = int(completion_span_queries)
	wall_counts = [0] * _GT_QPS_WALL_SUBWINDOWS
	wall_span = (window_end - window_start) / len(wall_counts)
	if wall_span > 0.0:
		for completed_at in ordered:
			index = int((completed_at - window_start) / wall_span)
			if 0 <= index < len(wall_counts):
				wall_counts[index] += 1
			elif index >= len(wall_counts):
				wall_counts[-1] += 1
	wall_completion_spans = [count / span_queries for count in wall_counts]
	# Reuse the existing minimum of three full completion-span observations.
	# This is an applicability boundary between two proofs, not a tunable knob.
	dense_completion_stream = all(
		spans >= _GT_QPS_MIN_COMPLETION_SPANS_PER_WALL_SUBWINDOW
		for spans in wall_completion_spans
	)
	evidence.update({
		"wall_subwindow_completions": wall_counts,
		"wall_subwindow_completion_spans": wall_completion_spans,
		"wall_time_dense_completion_stream": dense_completion_stream,
	})
	boundaries = [
		ordered[index - 1]
		for index in range(span_queries, len(ordered) + 1, span_queries)
	]
	# Four boundaries yield three independent full-span rate observations.
	if len(boundaries) < 4:
		return evidence
	durations = [
		current - previous
		for previous, current in zip(boundaries, boundaries[1:])
	]
	if any(duration <= 0.0 for duration in durations):
		return evidence
	tail_s = max(0.0, window_end - boundaries[-1])
	evidence["completion_span_tail_s"] = tail_s
	terminal_stall = tail_s > max(durations)

	rates = [span_queries / duration for duration in durations]
	mean = sum(rates) / len(rates)
	if mean <= 0.0:
		return evidence
	completion_cv = (
		sum((rate - mean) ** 2 for rate in rates) / len(rates)
	) ** 0.5 / mean
	# A global CV can dilute a sustained regime change near the end of a long
	# window. Compare every bounded tail of at least three full spans with its
	# preceding history, while keeping at least three spans on both sides. The
	# two-point CV is dimensionless and uses the same admissibility threshold as
	# the global completion-span CV; it is evidence, not a calibration knob.
	tail_change_cv: Optional[float] = None
	tail_change_suffix_spans: Optional[int] = None
	total_rate = sum(rates)
	suffix_rate = sum(rates[-2:])
	for suffix_spans in range(3, len(rates) // 2 + 1):
		suffix_rate += rates[-suffix_spans]
		prefix_spans = len(rates) - suffix_spans
		prefix_mean = (total_rate - suffix_rate) / prefix_spans
		suffix_mean = suffix_rate / suffix_spans
		combined_mean = (prefix_mean + suffix_mean) / 2.0
		if combined_mean <= 0.0:
			continue
		change_cv = abs(prefix_mean - suffix_mean) / (2.0 * combined_mean)
		if tail_change_cv is None or change_cv > tail_change_cv:
			tail_change_cv = change_cv
			tail_change_suffix_spans = suffix_spans
	completion_selected_cv = max(completion_cv, tail_change_cv or 0.0)
	evidence.update({
		"completion_span_cv": completion_cv,
		"completion_span_tail_change_cv": tail_change_cv,
		"completion_span_tail_change_suffix_spans": tail_change_suffix_spans,
		"completion_span_queries": span_queries,
		"completion_span_rates_qps": rates,
		"completion_span_durations_s": durations,
	})
	if terminal_stall:
		evidence["selection_reason"] = "terminal_stall_fallback"
	elif dense_completion_stream:
		evidence["selection_reason"] = "dense_completion_stream"
	else:
		evidence.update({
			"selected_cv": completion_selected_cv,
			"method": "completion_span_rates",
			"selection_reason": "sparse_batch_wave",
		})
	return evidence


def build_measured_gt_admissibility(summary: Dict[str, Any]) -> Dict[str, Any]:
	"""Build the structurally validated, fail-closed GT-admissibility verdict.

	Legacy artifacts used ``steady_state_ok`` as a CV-only signal. Future
	summaries mirror this stricter verdict into that compatibility field: CM
	ground truth additionally needs a rate-stable warmup, at least two bounded
	stage-rate spans in the measured window, demonstrated saturation, and an
	intact closed population.  The adaptive client population is deliberately
	not a workload shape: a large population may sit behind one slow, fully
	backlogged stage, so requiring global population turnovers rejects valid
	steady-state windows. Quality-workload trace integrity is deliberately
	orthogonal and is reported under ``quality_trace_*`` fields.
	"""
	gate = summary.get("measurement_start_gate")
	waves_raw = summary.get("measurement_rate_stability_waves")
	waves = (
		float(waves_raw)
		if isinstance(waves_raw, (int, float))
		and not isinstance(waves_raw, bool)
		else None
	)
	if waves is not None and not math.isfinite(waves):
		waves = None
	saturated = summary.get("population_saturated")
	driver_errors_raw = summary.get("driver_errors_window")
	driver_errors = (
		driver_errors_raw
		if isinstance(driver_errors_raw, int)
		and not isinstance(driver_errors_raw, bool)
		else None
	)
	cv_raw = summary.get("qps_subwindow_cv")
	cv = (
		float(cv_raw)
		if isinstance(cv_raw, (int, float))
		and not isinstance(cv_raw, bool)
		else None
	)
	if cv is not None and not math.isfinite(cv):
		cv = None

	checks: Dict[str, Dict[str, Any]] = {
		"warmup_rate_stable_not_cap_forced": {
			"passed": gate in _GT_RATE_STABLE_GATES,
			"observed": gate,
			"required": sorted(_GT_RATE_STABLE_GATES),
		},
		"measurement_rate_stability_waves_ge_2": {
			"passed": waves is not None and waves >= _GT_MIN_RATE_STABILITY_WAVES,
			"observed": waves,
			"required": {
				"operator": ">=", "value": _GT_MIN_RATE_STABILITY_WAVES,
			},
		},
		"population_saturated": {
			"passed": saturated is True,
			"observed": saturated,
			"required": True,
		},
		"driver_errors_window_zero": {
			"passed": driver_errors == 0,
			"observed": driver_errors,
			"required": 0,
		},
		"qps_subwindow_cv_le_0_15": {
			"passed": (
				cv is not None and 0.0 <= cv <= _GT_MAX_QPS_SUBWINDOW_CV
			),
			"observed": cv,
			"required": {"operator": "<=", "value": _GT_MAX_QPS_SUBWINDOW_CV},
		},
	}
	reason_for_check = {
		"warmup_rate_stable_not_cap_forced": "warmup_rate_stability_gate_not_met",
		"measurement_rate_stability_waves_ge_2": (
			"measurement_rate_stability_waves_below_minimum"
		),
		"population_saturated": "population_not_saturated",
		"driver_errors_window_zero": "driver_errors_in_measurement_window",
		"qps_subwindow_cv_le_0_15": "qps_subwindow_cv_missing_or_above_maximum",
	}
	reasons = [
		reason_for_check[name]
		for name, check in checks.items()
		if not check["passed"]
	]
	verdict = {
		"admissible": not reasons,
		"reasons": reasons,
		"checks": checks,
	}
	validate_measured_gt_admissibility(verdict)
	return verdict


def _phase_wall_cap_s(system_config: Dict[str, Any], phase: str) -> float:
	import os
	key = "warmup_wall_cap_s" if phase == "warmup" else "measured_wall_cap_s"
	default = _WARMUP_WALL_CAP_S if phase == "warmup" else _MEASURED_WALL_CAP_S
	env = os.environ.get("RAG_STACK_" + key.upper())
	if env is not None:
		try:
			return max(1.0, float(env))
		except ValueError:
			pass
	cfg = system_config or {}
	val = cfg.get(key)
	if val is None and isinstance(cfg.get("batching"), dict):
		val = cfg["batching"].get(key)
	try:
		return max(1.0, float(val)) if val is not None else default
	except (TypeError, ValueError):
		return default


def _raise_nofile_limit() -> None:
	try:
		import os
		import resource
	except Exception:
		return
	raw_target = os.environ.get("RAG_STACK_NOFILE_LIMIT")
	try:
		target = max(1024, int(raw_target or _NOFILE_SOFT_TARGET))
	except (TypeError, ValueError):
		target = _NOFILE_SOFT_TARGET
	try:
		soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
	except (OSError, ValueError):
		return
	if soft >= target:
		return
	hard_cap = target if hard == resource.RLIM_INFINITY else int(hard)
	new_soft = min(target, hard_cap)
	if new_soft <= soft:
		logger.warning(
			"[measured service] nofile soft limit is low "
			f"({soft}) and hard limit is {hard}; high-concurrency measured runs "
			"may hit 'too many open files'"
		)
		return
	try:
		resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
		logger.info(
			"[measured service] raised nofile soft limit "
			f"from {soft} to {new_soft}"
		)
	except (OSError, ValueError) as exc:
		logger.warning(f"[measured service] failed to raise nofile limit: {exc}")


@dataclass
class StageResult:
	df: pd.DataFrame
	gen_perf: Optional[Dict[str, Any]]
	elapsed_s: float
	queue_wait_s: float
	service_s: float
	batch_size: int


@dataclass
class RequestState:
	seq: int
	idx: int
	qid: Any
	df: pd.DataFrame
	is_measured: bool
	record_trace: bool = False
	admit_ts: float = 0.0
	done_ts: float = 0.0
	first_token_ts: Optional[float] = None
	n_output_tokens: int = 0
	agent_generate_calls: int = 0
	agent_retrieval_calls: int = 0
	agent_truncated: bool = False
	stage_times: Dict[str, float] = field(default_factory=dict)
	stage_queue_wait_s: Dict[str, float] = field(default_factory=dict)
	stage_batch_sizes: Dict[str, List[int]] = field(default_factory=dict)

	def add_stage(self, stage: str, result: StageResult) -> None:
		self.stage_times[stage] = self.stage_times.get(stage, 0.0) + result.elapsed_s
		self.stage_queue_wait_s[stage] = (
			self.stage_queue_wait_s.get(stage, 0.0) + result.queue_wait_s
		)
		self.stage_batch_sizes.setdefault(stage, []).append(int(result.batch_size))
		if result.gen_perf is None:
			return
		send = result.gen_perf.get("request_send_ts")
		first = result.gen_perf.get("first_token_ts")
		if send is not None and first is not None:
			first_abs = float(first)
			if self.first_token_ts is None:
				self.first_token_ts = first_abs
		self.n_output_tokens += max(
			int(result.gen_perf.get("n_output_tokens", 1) or 1),
			1,
		)

	def to_perf(self) -> perf_mod.QueryPerf:
		return perf_mod.QueryPerf(
			request_send_ts=self.admit_ts,
			first_token_ts=self.first_token_ts,
			last_token_ts=self.done_ts,
			n_output_tokens=max(int(self.n_output_tokens), 1),
			per_stage_times=dict(self.stage_times),
			agent_generate_calls=int(self.agent_generate_calls),
			agent_retrieval_calls=int(self.agent_retrieval_calls),
			agent_truncated=bool(self.agent_truncated),
		)


class BaseStageService:
	def __init__(self, owner: Any, stage: Dict[str, Any], name: Optional[str] = None):
		self.owner = owner
		self.stage = stage
		self.name = name or str(stage["stage"])
		# Measurement-window counter snapshots. Service counters accumulate
		# from process start; the exported stats must cover the MEASURED
		# window only (same semantics as qps, which is a completion delta
		# over the window) — otherwise warmup's cold small batches and the
		# drain tail pollute avg_service_s, the CM calibration target.
		self._win_start_counters: Optional[Dict[str, float]] = None
		self._win_end_counters: Optional[Dict[str, float]] = None

	def _counter_snapshot(self) -> Dict[str, float]:
		return {}

	def mark_window_start(self) -> None:
		self._win_start_counters = self._counter_snapshot()
		self._win_end_counters = None

	def mark_window_end(self) -> None:
		if self._win_start_counters is not None and self._win_end_counters is None:
			self._win_end_counters = self._counter_snapshot()

	def _windowed_counters(self) -> Dict[str, float]:
		"""Counter deltas over the measured window; lifetime totals when the
		window was never marked (back-compat for direct service use)."""
		if self._win_start_counters is None:
			return self._counter_snapshot()
		end = self._win_end_counters or self._counter_snapshot()
		return {k: end[k] - self._win_start_counters.get(k, 0.0) for k in end}

	async def run(self, state: RequestState, previous: pd.DataFrame) -> StageResult:
		raise NotImplementedError

	async def close(self) -> None:
		return None

	def stats(self) -> Dict[str, Any]:
		return {}


class DirectPureStageService(BaseStageService):
	"""Direct async wrapper for lightweight CPU stages."""

	async def run(self, state: RequestState, previous: pd.DataFrame) -> StageResult:
		node = self.stage["node"]
		params = dict(self.stage["params"])
		instance = self.stage["instance"]

		def _work():
			# Deep copy, merge and trace byte-work all run off the event loop.
			# Keep executor wait separate from the full stage-service interval;
			# trace capture is instrumentation and therefore stays outside service.
			work_start = time.perf_counter()
			result = instance.pure(previous.copy(deep=True), **params)
			merged = self.owner._merge_service_node_result(node, previous, result)
			service_done = time.perf_counter()
			if state.record_trace:
				_record_pure_stage(node, params, merged)
			return work_start, service_done, merged

		submit = time.perf_counter()
		work_start, service_done, merged = await asyncio.to_thread(_work)
		done = time.perf_counter()
		return StageResult(
			df=merged,
			gen_perf=None,
			elapsed_s=done - submit,
			queue_wait_s=work_start - submit,
			service_s=service_done - work_start,
			batch_size=1,
		)


class BatchedPureStageService(BaseStageService):
	"""Count/timeout batch service for retrieval and dynamic-batch HF stages."""

	def __init__(
		self,
		owner: Any,
		stage: Dict[str, Any],
		*,
		batch_size: int,
		timeout_s: float,
		name: Optional[str] = None,
	):
		super().__init__(owner, stage, name=name)
		self.batch_size = max(1, int(batch_size))
		self.timeout_s = max(0.0, float(timeout_s))
		self._condition = asyncio.Condition()
		self._queue: List[Dict[str, Any]] = []
		self._closed = False
		# A serialized stage worker must not queue behind hundreds of one-row
		# tasks in asyncio's process-wide default executor.  One private worker
		# preserves the existing single-batch/AuxProcess Pipe contract while
		# removing cross-stage FIFO convoying.
		self._executor = concurrent.futures.ThreadPoolExecutor(
			max_workers=1,
			thread_name_prefix=f"measured-{self.name}",
		)
		self._executor_shutdown = False
		self._task = asyncio.create_task(self._worker())
		self.batches = 0
		self.rows = 0
		self.partial_batches = 0
		self.max_batch_observed = 0
		self.win_max_batch_observed = 0
		self.total_queue_wait_s = 0.0
		self.total_service_s = 0.0
		self.total_scheduler_wait_s = 0.0
		self.total_executor_wait_s = 0.0
		self.total_prepare_s = 0.0
		self.total_pure_service_s = 0.0
		self.total_postprocess_s = 0.0
		self.total_trace_s = 0.0
		self.total_event_loop_resume_s = 0.0

	def _counter_snapshot(self) -> Dict[str, float]:
		return {
			"batches": float(self.batches),
			"rows": float(self.rows),
			"partial_batches": float(self.partial_batches),
			"total_queue_wait_s": float(self.total_queue_wait_s),
			"total_service_s": float(self.total_service_s),
			"total_scheduler_wait_s": float(self.total_scheduler_wait_s),
			"total_executor_wait_s": float(self.total_executor_wait_s),
			"total_prepare_s": float(self.total_prepare_s),
			"total_pure_service_s": float(self.total_pure_service_s),
			"total_postprocess_s": float(self.total_postprocess_s),
			"total_trace_s": float(self.total_trace_s),
			"total_event_loop_resume_s": float(self.total_event_loop_resume_s),
		}

	def mark_window_start(self) -> None:
		super().mark_window_start()
		self.win_max_batch_observed = 0

	def backlog(self) -> int:
		"""Instantaneous submitted-not-yet-batched queue depth. A backlog
		persistently deeper than one full batch means this worker cannot
		keep up — the saturation evidence for worker-bound deployments
		(the LLM dashboards are mute when e.g. the reranker is the
		bottleneck)."""
		return len(self._queue)

	async def run(self, state: RequestState, previous: pd.DataFrame) -> StageResult:
		loop = asyncio.get_running_loop()
		fut = loop.create_future()
		# Contract: the submitted df must not be mutated by the caller until
		# the future resolves. Under that contract no defensive copy is
		# needed here — the worker thread deep-copies before instance.pure
		# and pd.concat(ignore_index=True) builds fresh buffers. The old
		# per-submission copy(deep=True) ran ON THE EVENT LOOP for every
		# call and was a measurable source of loop stall at high dispatch
		# rates (r12 driver-feed investigation).
		item = {
			"state": state,
			"previous": previous,
			"n_rows": len(previous),
			"future": fut,
			"enqueue_s": time.perf_counter(),
			"enqueue_loop_s": loop.time(),
		}
		async with self._condition:
			if self._closed:
				raise RuntimeError(f"stage service {self.name!r} is closed")
			self._queue.append(item)
			self._condition.notify()
		try:
			return await fut
		except asyncio.CancelledError:
			fut.cancel()
			async with self._condition:
				removed = False
				for idx, queued in enumerate(self._queue):
					if queued is item:
						del self._queue[idx]
						removed = True
						break
				if removed:
					# The oldest enqueue (and therefore the batch deadline) may
					# have changed. Wake the worker so it recomputes from the new
					# queue head instead of timing out an empty/stale batch.
					self._condition.notify_all()
			raise

	async def close(self) -> None:
		if self._executor_shutdown:
			return
		async with self._condition:
			self._closed = True
			pending = list(self._queue)
			self._queue.clear()
			for item in pending:
				item["future"].cancel()
			self._condition.notify_all()
		# A running executor future cannot be cancelled safely: its AuxProcess
		# request may still own the stage Pipe.  Finish the worker even if the
		# caller is cancelled, shut the executor down, then restore cancellation.
		cancelled_during_close = False
		try:
			while not self._task.done():
				try:
					await asyncio.shield(self._task)
				except asyncio.CancelledError:
					if not self._task.cancelled():
						cancelled_during_close = True
					if self._task.done():
						break
		finally:
			self._shutdown_executor()

		# Retrieve and propagate a worker failure after its executor is closed.
		# Calling result() also avoids an un-retrieved-task warning on teardown.
		self._task.result()
		if cancelled_during_close:
			raise asyncio.CancelledError

	def _shutdown_executor(self) -> None:
		if self._executor_shutdown:
			return
		self._executor.shutdown(wait=True, cancel_futures=True)
		self._executor_shutdown = True

	async def _worker(self) -> None:
		while True:
			items = await self._next_batch()
			if not items:
				return
			await self._run_batch(items)

	async def _next_batch(self) -> List[Dict[str, Any]]:
		loop = asyncio.get_running_loop()
		async with self._condition:
			while True:
				while not self._queue:
					if self._closed:
						return []
					await self._condition.wait()

				while (
					self._queue
					and len(self._queue) < self.batch_size
					and not self._closed
				):
					if self.timeout_s <= 0.0:
						break
					# Re-read the head after every notification. A caller can cancel
					# the previous oldest item while this worker is waiting.
					oldest = self._queue[0]
					if "enqueue_loop_s" in oldest:
						deadline = (
							float(oldest["enqueue_loop_s"]) + self.timeout_s
						)
					else:
						# Compatibility for direct service tests/legacy callers that
						# only provide the perf-counter timestamp.
						age = max(
							0.0,
							time.perf_counter() - float(oldest["enqueue_s"]),
						)
						deadline = loop.time() + max(0.0, self.timeout_s - age)
					remaining = deadline - loop.time()
					if remaining <= 0.0:
						break
					try:
						await asyncio.wait_for(
							self._condition.wait(), timeout=remaining,
						)
					except asyncio.TimeoutError:
						break

				# Cancellation can empty the queue while the worker is waiting.
				# Stay alive for the next submission unless the service was closed.
				if not self._queue:
					continue
				n = min(len(self._queue), self.batch_size)
				items = self._queue[:n]
				del self._queue[:n]
				dequeue_s = time.perf_counter()
				for item in items:
					item["dequeue_s"] = dequeue_s
				return items

	async def _run_batch(self, items: List[Dict[str, Any]]) -> None:
		node = self.stage["node"]
		params = dict(self.stage["params"])
		instance = self.stage["instance"]

		def _work():
			# All pandas assembly runs in the worker thread: the concat /
			# merge / per-item splits used to run on the EVENT LOOP, where
			# they blocked HTTP pumping for every co-resident stage (part of
			# the ~6% driver-loop residue the r11 profile attributed to
			# loop-side concat + diffuse pandas). The service interval covers
			# the complete stage adapter (concat through split); pure and each
			# adapter segment are exported separately for calibration diagnosis.
			work_start = time.perf_counter()
			previous = pd.concat(
				[i["previous"] for i in items], ignore_index=True,
			)
			pure_start = time.perf_counter()
			result = instance.pure(previous.copy(deep=True), **params)
			pure_done = time.perf_counter()
			merged = self.owner._merge_service_node_result(
				node, previous, result)
			parts = []
			offset = 0
			for item in items:
				n_rows = int(item["n_rows"])
				parts.append(
					merged.iloc[offset:offset + n_rows]
					.copy().reset_index(drop=True))
				offset += n_rows
			service_done = time.perf_counter()
			# Trace recording joins the full passage texts (~19 KB/query) and
			# counts bytes — pure harness work that used to run on the EVENT
			# LOOP after the await, stalling HTTP pumping for every
			# co-resident stage (r20; every measured-window request records).
			# _run_batch explicitly propagates contextvars into the private
			# executor, so the recorder bound to the run is visible here.
			for item, part in zip(items, parts):
				if item["state"].record_trace:
					_record_pure_stage(node, params, part)
			work_done = time.perf_counter()
			return (
				work_start,
				pure_start,
				pure_done,
				service_done,
				work_done,
				parts,
			)

		try:
			submit_s = time.perf_counter()
			# run_in_executor does not propagate contextvars. Preserve the
			# run-scoped TraceRecorder exactly as asyncio.to_thread did.
			ctx = contextvars.copy_context()
			loop = asyncio.get_running_loop()
			(
				work_start,
				pure_start,
				pure_done,
				service_done,
				work_done,
				parts,
			) = await loop.run_in_executor(self._executor, ctx.run, _work)
			resume_s = time.perf_counter()

			self.batches += 1
			self.rows += len(items)
			self.max_batch_observed = max(self.max_batch_observed, len(items))
			self.win_max_batch_observed = max(
				self.win_max_batch_observed, len(items))
			if len(items) < self.batch_size:
				self.partial_batches += 1
			self.total_service_s += service_done - work_start
			self.total_executor_wait_s += work_start - submit_s
			self.total_prepare_s += pure_start - work_start
			self.total_pure_service_s += pure_done - pure_start
			self.total_postprocess_s += service_done - pure_done
			self.total_trace_s += work_done - service_done
			self.total_event_loop_resume_s += resume_s - work_done

			for item, part in zip(items, parts):
				enqueue_s = float(item["enqueue_s"])
				queue_wait = work_start - enqueue_s
				elapsed = resume_s - enqueue_s
				self.total_queue_wait_s += queue_wait
				self.total_scheduler_wait_s += (
					float(item["dequeue_s"]) - enqueue_s
				)
				fut = item["future"]
				if not fut.cancelled():
					fut.set_result(
						StageResult(
							df=part,
							gen_perf=None,
							elapsed_s=elapsed,
							queue_wait_s=queue_wait,
							service_s=service_done - work_start,
							batch_size=len(items),
						)
					)
		except Exception as exc:  # noqa: BLE001
			for item in items:
				fut = item["future"]
				if not fut.cancelled():
					fut.set_exception(exc)

	def stats(self) -> Dict[str, Any]:
		# Windowed deltas when the measured window was marked (same
		# semantics as qps); lifetime totals otherwise.
		c = self._windowed_counters()
		batches, rows = c["batches"], c["rows"]
		max_obs = (self.win_max_batch_observed
		           if self._win_start_counters is not None
		           else self.max_batch_observed)
		return {
			"batch_size": self.batch_size,
			"timeout_s": self.timeout_s,
			"batches": batches,
			"rows": rows,
			"partial_batches": c["partial_batches"],
			"max_batch_observed": max_obs,
			"avg_batch_size": rows / batches if batches else 0.0,
			"avg_queue_wait_s": (
				c["total_queue_wait_s"] / rows if rows else 0.0
			),
			"avg_service_s": (
				c["total_service_s"] / batches if batches else 0.0
			),
			"avg_scheduler_wait_s": (
				c["total_scheduler_wait_s"] / rows if rows else 0.0
			),
			"avg_executor_wait_s": (
				c["total_executor_wait_s"] / batches if batches else 0.0
			),
			"avg_prepare_s": (
				c["total_prepare_s"] / batches if batches else 0.0
			),
			"avg_pure_service_s": (
				c["total_pure_service_s"] / batches if batches else 0.0
			),
			"avg_postprocess_s": (
				c["total_postprocess_s"] / batches if batches else 0.0
			),
			"avg_trace_s": (
				c["total_trace_s"] / batches if batches else 0.0
			),
			"avg_event_loop_resume_s": (
				c["total_event_loop_resume_s"] / batches if batches else 0.0
			),
		}


class LLMContinuousStageService(BaseStageService):
	"""vLLM continuous-batching stage.

	The Python runtime submits independent async requests; vLLM performs the actual
	token-level continuous batching inside the server.
	"""

	def __init__(
		self,
		owner: Any,
		stage: Dict[str, Any],
		*,
		max_inflight: int,
		name: Optional[str] = None,
	):
		super().__init__(owner, stage, name=name)
		self.max_inflight = max(1, int(max_inflight))
		self._sem = asyncio.Semaphore(self.max_inflight)
		self.submitted = 0
		self.completed = 0
		# Explicit outer-semaphore waiters.  Do not infer this from
		# submitted-completed-inflight: a failed/cancelled request would leave
		# that arithmetic permanently non-zero and create false saturation.
		self.waiting = 0
		self.inflight = 0
		self.max_inflight_observed = 0
		self.win_max_inflight_observed = 0
		self.max_waiting_observed = 0
		self.win_max_waiting_observed = 0
		self.total_queue_wait_s = 0.0
		self.total_service_s = 0.0

	def _counter_snapshot(self) -> Dict[str, float]:
		return {
			"submitted": float(self.submitted),
			"completed": float(self.completed),
			"total_queue_wait_s": float(self.total_queue_wait_s),
			"total_service_s": float(self.total_service_s),
		}

	def mark_window_start(self) -> None:
		super().mark_window_start()
		self.win_max_inflight_observed = self.inflight
		self.win_max_waiting_observed = self.waiting
		marker = getattr(
			self._subprocess(), "mark_role_admission_window_start", None,
		)
		if callable(marker):
			marker()

	def mark_window_end(self) -> None:
		super().mark_window_end()
		marker = getattr(
			self._subprocess(), "mark_role_admission_window_end", None,
		)
		if callable(marker):
			marker()

	def _subprocess(self) -> Any:
		return getattr(self.stage["instance"], "_subprocess", None)

	# --- engine-occupancy probe -------------------------------------------
	# 1Hz sampler of the vLLM server's own gauges during the measured
	# window. num_requests_running is the ONLY observable that separates
	# "engine batch drains while the client loop stalls" from "engine full
	# but CPU-starved" (r12 driver-feed investigation) — the client-side
	# semaphore reads 100% busy in both regimes.

	def start_engine_probe(self, interval_s: float = 1.0) -> None:
		if getattr(self, "_probe_task", None) is not None:
			return
		self._probe_samples: List[Dict[str, Any]] = []
		self._probe_window_idx = 0
		# GPU clock sampling (variance forensics: engine token rate swung 34%
		# across identical 0046-A replays — clocks vs prefix-cache phase are
		# the two candidates). pynvml is cheap C calls; absent → skip.
		self._nvml_handles = []
		try:
			import pynvml
			pynvml.nvmlInit()
			self._nvml = pynvml
			self._nvml_handles = [
				pynvml.nvmlDeviceGetHandleByIndex(i)
				for i in range(pynvml.nvmlDeviceGetCount())
			]
		except Exception:  # noqa: BLE001 — clocks are optional forensics
			self._nvml = None
		self._probe_task = asyncio.create_task(self._probe_loop(interval_s))

	def stop_engine_probe(self) -> None:
		task = getattr(self, "_probe_task", None)
		if task is not None:
			task.cancel()
			self._probe_task = None
		if getattr(self, "_nvml", None) is not None:
			try:
				self._nvml.nvmlShutdown()
			except Exception:  # noqa: BLE001
				pass
			self._nvml = None
			self._nvml_handles = []

	def mark_probe_window(self) -> None:
		"""Anchor exported probe stats to the measured window (the probe now
		runs from serving start so the population adapter can read engine
		backlog during warmup)."""
		self._probe_window_idx = len(getattr(self, "_probe_samples", []) or [])

	def recent_engine_waiting(self, n: int = 5) -> List[float]:
		samples = getattr(self, "_probe_samples", None) or []
		return [float(s["wait"]) for s in samples[-n:]]

	async def _probe_loop(self, interval_s: float) -> None:
		import aiohttp

		sub = self._subprocess()
		base = getattr(sub, "base_url", None)
		if not base:
			return
		root = base.rsplit("/v1", 1)[0]
		try:
			async with aiohttp.ClientSession() as sess:
				while True:
					try:
						async with sess.get(
							f"{root}/metrics",
							timeout=aiohttp.ClientTimeout(total=2.0),
						) as resp:
							text = await resp.text()
						running = waiting = 0.0
						pc_queries = pc_hits = pc_rate = None
						seen = False
						for ln in text.splitlines():
							if ln.startswith("vllm:num_requests_running"):
								running += float(ln.rsplit(" ", 1)[1])
								seen = True
							elif ln.startswith("vllm:num_requests_waiting"):
								waiting += float(ln.rsplit(" ", 1)[1])
							elif ln.startswith((
								"vllm:gpu_prefix_cache_queries_total",
								"vllm:prefix_cache_queries_total",
							)):
								pc_queries = (pc_queries or 0.0) + float(
									ln.rsplit(" ", 1)[1])
							elif ln.startswith((
								"vllm:gpu_prefix_cache_hits_total",
								"vllm:prefix_cache_hits_total",
							)):
								pc_hits = (pc_hits or 0.0) + float(
									ln.rsplit(" ", 1)[1])
							elif ln.startswith("vllm:gpu_prefix_cache_hit_rate"):
								pc_rate = float(ln.rsplit(" ", 1)[1])
						clk = None
						if self._nvml_handles:
							try:
								clks = [
									self._nvml.nvmlDeviceGetClockInfo(
										h, self._nvml.NVML_CLOCK_SM)
									for h in self._nvml_handles
								]
								clk = sum(clks) / len(clks)
							except Exception:  # noqa: BLE001
								clk = None
						if seen:
							self._probe_samples.append({
								"run": running, "wait": waiting,
								"pcq": pc_queries, "pch": pc_hits,
								"pcr": pc_rate, "clk": clk,
							})
					except asyncio.CancelledError:
						raise
					except Exception:  # noqa: BLE001 — probe is best-effort
						pass
					await asyncio.sleep(interval_s)
		except asyncio.CancelledError:
			pass

	def engine_probe_series(self) -> Dict[str, List[Any]]:
		"""Windowed 1 Hz dashboard series — the stationarity evidence a
		record needs for post-hoc inspection (aggregates alone cannot show
		regime switches inside the window)."""
		all_samples = getattr(self, "_probe_samples", None)
		if not all_samples:
			return {}
		idx = int(getattr(self, "_probe_window_idx", 0) or 0)
		samples = all_samples[idx:] or all_samples
		return {
			"running": [s["run"] for s in samples],
			"waiting": [s["wait"] for s in samples],
			"gpu_sm_clock_mhz": [s["clk"] for s in samples],
		}

	def engine_probe_stats(self) -> Dict[str, Any]:
		all_samples = getattr(self, "_probe_samples", None)
		if not all_samples:
			return {}
		idx = int(getattr(self, "_probe_window_idx", 0) or 0)
		samples = all_samples[idx:] or all_samples
		runs = [s["run"] for s in samples]
		waits = [s["wait"] for s in samples]
		out: Dict[str, Any] = {
			"engine_running_min": min(runs),
			"engine_running_mean": sum(runs) / len(runs),
			"engine_running_max": max(runs),
			"engine_waiting_mean": sum(waits) / len(waits),
			"engine_probe_samples": len(samples),
		}
		# Prefix-cache hit rate over the window: prefer counter deltas
		# (exact), fall back to the gauge's mean.
		pcq = [s["pcq"] for s in samples if s["pcq"] is not None]
		pch = [s["pch"] for s in samples if s["pch"] is not None]
		if len(pcq) >= 2 and len(pch) >= 2 and (pcq[-1] - pcq[0]) > 0:
			out["engine_prefix_cache_hit_rate"] = (
				(pch[-1] - pch[0]) / (pcq[-1] - pcq[0])
			)
		else:
			rates = [s["pcr"] for s in samples if s["pcr"] is not None]
			if rates:
				out["engine_prefix_cache_hit_rate"] = sum(rates) / len(rates)
		clks = [s["clk"] for s in samples if s["clk"] is not None]
		if clks:
			out["engine_gpu_sm_clock_mean_mhz"] = sum(clks) / len(clks)
			out["engine_gpu_sm_clock_min_mhz"] = min(clks)
		return out

	async def close(self) -> None:
		"""Close the engine's raw HTTP session before the event loop ends
		(runtime.close() runs in the loop's finally). The vLLM subprocess
		itself is torn down later, outside the loop, by evict_vllm.
		QueryExpansionContinuousStageService inherits this (its _subprocess
		override resolves the QE engine's subprocess)."""
		self.stop_engine_probe()
		sub = self._subprocess()
		aclose = getattr(sub, "aclose_http", None)
		if aclose is not None:
			try:
				await aclose()
			except Exception:  # noqa: BLE001 — teardown is best-effort
				pass

	async def run(self, state: RequestState, previous: pd.DataFrame) -> StageResult:
		if "prompts" not in previous.columns:
			raise ValueError(f"{self.name} stage requires a prompts column")
		prompt = previous["prompts"].iloc[0]
		text, gen_perf, queue_wait, service_s, active_batch = await self.generate_text(
			state, prompt
		)
		n_tokens = max(int(gen_perf.get("n_output_tokens", 1) or 1), 1)
		output = {
			"generated_texts": [text],
			"generated_tokens": [[0] * n_tokens],
			"generated_log_probs": [[0.0] * n_tokens],
		}
		# The serving path owns a one-row frame and normally adds fresh output
		# columns. Avoid a second DataFrame + two deep copies + concat in that hot
		# case. Unusual multi-row/overlapping inputs retain the generic node merge
		# contract exactly (including duplicate overlapping columns).
		if len(previous) == 1 and not set(output).intersection(previous.columns):
			merged = previous.copy(deep=False)
			merged.index = pd.RangeIndex(len(merged))
			for column, values in output.items():
				merged[column] = values
			result = None
		else:
			result = pd.DataFrame(output)
			merged = self.owner._merge_service_node_result(
				self.stage["node"], previous, result,
			)
		if state.record_trace:
			if result is None:
				result = pd.DataFrame(output)
			_record_generate(previous, result, self.stage["params"])
		return StageResult(
			df=merged,
			gen_perf=gen_perf,
			elapsed_s=queue_wait + service_s,
			queue_wait_s=queue_wait,
			service_s=service_s,
			batch_size=active_batch,
		)

	async def generate_text(
		self,
		state: RequestState,
		prompt: Any,
		sampling_params: Optional[Dict[str, Any]] = None,
	) -> tuple[str, Dict[str, Any], float, float, int]:
		subprocess = self._subprocess()
		if subprocess is None or not hasattr(subprocess, "generate_one"):
			raise RuntimeError(
				f"{self.name} measured stage requires a deployed vLLM subprocess"
			)
		params = sampling_params or self.owner._sampling_params(dict(self.stage["params"]))
		enqueue = time.perf_counter()
		self.submitted += 1
		self.waiting += 1
		self.max_waiting_observed = max(self.max_waiting_observed, self.waiting)
		self.win_max_waiting_observed = max(
			self.win_max_waiting_observed, self.waiting
		)
		try:
			await self._sem.acquire()
		finally:
			self.waiting -= 1
		try:
			start = time.perf_counter()
			queue_wait = start - enqueue
			self.total_queue_wait_s += queue_wait
			self.inflight += 1
			active_batch = self.inflight
			self.max_inflight_observed = max(self.max_inflight_observed, self.inflight)
			self.win_max_inflight_observed = max(
				self.win_max_inflight_observed, self.inflight)
			try:
				text, gen_perf = await subprocess.generate_one(prompt, params)
			finally:
				self.inflight -= 1
			done = time.perf_counter()
			service_s = done - start
			self.completed += 1
			self.total_service_s += service_s
			return text, gen_perf, queue_wait, service_s, active_batch
		finally:
			self._sem.release()

	def stats(self) -> Dict[str, Any]:
		c = self._windowed_counters()
		completed = c["completed"]
		max_obs = (self.win_max_inflight_observed
		           if self._win_start_counters is not None
		           else self.max_inflight_observed)
		out = {
			"max_inflight": self.max_inflight,
			"current_inflight": self.inflight,
			"current_waiting": self.waiting,
			"submitted": c["submitted"],
			"completed": completed,
			"max_inflight_observed": max_obs,
			"max_waiting_observed": (
				self.win_max_waiting_observed
				if self._win_start_counters is not None
				else self.max_waiting_observed
			),
			"avg_queue_wait_s": (
				c["total_queue_wait_s"] / completed if completed else 0.0
			),
			"avg_service_s": (
				c["total_service_s"] / completed if completed else 0.0
			),
		}
		role_stats = getattr(self._subprocess(), "role_admission_stats", None)
		if callable(role_stats):
			out["pd_role_admission"] = role_stats()
		return out


class QueryExpansionContinuousStageService(LLMContinuousStageService):
	"""vLLM-backed query expansion without falling back to instance.pure()."""

	def _subprocess(self) -> Any:
		generator = getattr(self.stage["instance"], "generator", None)
		return getattr(generator, "_subprocess", None)

	async def run(self, state: RequestState, previous: pd.DataFrame) -> StageResult:
		prompt, parser = _query_expansion_prompt_and_parser(
			self.stage["instance"],
			dict(self.stage["params"]),
			previous,
		)
		text, gen_perf, queue_wait, service_s, active_batch = await self.generate_text(
			state, prompt
		)
		queries = parser(text)
		if len(previous) == 1 and "queries" not in previous.columns:
			merged = previous.copy(deep=False)
			merged.index = pd.RangeIndex(len(merged))
			merged["queries"] = [queries]
		else:
			# The historical query-expansion merge keeps a pre-existing queries
			# column. Preserve that edge-case behavior outside the one-row fast path.
			result = pd.DataFrame({"queries": [queries]})
			merged = self.owner._merge_service_node_result(
				self.stage["node"], previous, result,
			)
		if state.record_trace:
			_record_query_expansion(previous, prompt, text, gen_perf, self.stage["params"])
		return StageResult(
			df=merged,
			gen_perf=gen_perf,
			elapsed_s=queue_wait + service_s,
			queue_wait_s=queue_wait,
			service_s=service_s,
			batch_size=active_batch,
		)


_STACKDUMP_STARTED = False


def _maybe_start_stack_dump_watchdog() -> None:
	"""Periodic all-thread Python stack dumps (RAG_STACK_STACKDUMP_EVERY_S).

	Stall forensics from INSIDE the process: shared boxes lock ptrace down
	(yama scope 2), so py-spy cannot attach even as a parent — faulthandler
	writes every thread's Python stack to a file instead, and its C-level
	dumper keeps working while the GIL is contended (exactly the condition
	under investigation). Off unless the env var is set."""
	global _STACKDUMP_STARTED
	period = float(os.environ.get("RAG_STACK_STACKDUMP_EVERY_S", "0") or 0)
	if period <= 0 or _STACKDUMP_STARTED:
		return
	_STACKDUMP_STARTED = True
	path = os.environ.get(
		"RAG_STACK_STACKDUMP_FILE",
		os.path.join(
			tempfile.gettempdir(), f"rag_stack_stacks_{os.getpid()}.txt"),
	)
	fh = open(path, "a")

	def _loop() -> None:
		while True:
			time.sleep(period)
			fh.write(
				f"\n===== stack dump @ {time.strftime('%H:%M:%S')} =====\n")
			faulthandler.dump_traceback(file=fh, all_threads=True)
			fh.flush()

	threading.Thread(
		target=_loop, name="stackdump-watchdog", daemon=True).start()
	logger.info(
		f"[measured service] stack-dump watchdog every {period:.0f}s -> {path}"
	)


class _CpuProbe:
	"""1 Hz host-CPU sampler over the serving PROCESS FAMILIES (r29).

	The measured box time-multiplexes its logical cores between the vLLM
	engine family (API server + EngineCore + one worker per GPU), the aux
	worker child processes (retrieval / rerank / compress, post-r20), and
	the harness main process (event loop + worker threads). The engine's
	per-step host segments stretching under that contention is the part of
	the single-stage-to-e2e loss the closed-loop simulation's CPU resource
	models — this probe is the direct evidence for it: per-family CPU
	core-seconds, sampled once per second from cumulative psutil cpu_times
	(so window aggregates are exact deltas, not accumulated tick error).

	Runs in a daemon thread (the event loop is saturated during the window;
	a thread keeps the cadence honest). psutil missing → probe disabled.
	"""

	def __init__(self, families: Dict[str, tuple]):
		# families: name -> (root pid, include_descendants). Engine/worker
		# families include descendants (vLLM's API server spawns EngineCore
		# and the TP workers); the harness family is the main process ALONE
		# — its children are exactly the other families and would otherwise
		# be double-counted.
		self._families = dict(families)
		self._samples: List[Dict[str, Any]] = []  # {"t", <family>: cpu_s}
		self._window_idx = 0
		self._stop: Optional[Any] = None
		self._thread: Optional[Any] = None
		self._psutil = None

	def start(self, interval_s: float = 1.0) -> None:
		if self._thread is not None:
			return
		try:
			import psutil
		except ImportError:
			logger.info("[cpu probe] psutil not installed; CPU sampling skipped")
			return
		self._psutil = psutil
		import threading
		self._stop = threading.Event()
		self._thread = threading.Thread(
			target=self._loop, args=(interval_s,), daemon=True,
			name="measured-cpu-probe")
		self._thread.start()

	def _family_cpu_s(self, root_pid: int, include_children: bool) -> Optional[float]:
		psutil = self._psutil
		try:
			root = psutil.Process(root_pid)
			procs = [root] + (root.children(recursive=True)
			                  if include_children else [])
		except (psutil.NoSuchProcess, psutil.AccessDenied):
			return None
		total = 0.0
		for p in procs:
			try:
				t = p.cpu_times()
				total += t.user + t.system
			except (psutil.NoSuchProcess, psutil.AccessDenied):
				continue
		return total

	def _loop(self, interval_s: float) -> None:
		while not self._stop.is_set():
			sample: Dict[str, Any] = {"t": time.monotonic()}
			for name, (pid, with_children) in self._families.items():
				cpu = self._family_cpu_s(pid, with_children)
				if cpu is not None:
					sample[name] = cpu
			self._samples.append(sample)
			self._stop.wait(interval_s)

	def mark_window(self) -> None:
		self._window_idx = len(self._samples)

	def stop(self) -> None:
		if self._stop is not None:
			self._stop.set()
		if self._thread is not None:
			self._thread.join(timeout=2.0)
			self._thread = None

	def series(self) -> Dict[str, List[float]]:
		"""Windowed per-second core usage per family (Δcpu/Δt between
		consecutive samples) — the dashboard series for post-hoc forensics."""
		samples = self._samples[max(0, self._window_idx - 1):]
		if len(samples) < 2:
			return {}
		out: Dict[str, List[float]] = {}
		for name in self._families:
			vals: List[float] = []
			for a, b in zip(samples, samples[1:]):
				dt = b["t"] - a["t"]
				if name in a and name in b and dt > 0:
					vals.append(round((b[name] - a[name]) / dt, 3))
			if vals:
				out[name] = vals
		return out

	def stats(self) -> Dict[str, Any]:
		"""Window-mean cores per family (exact cumulative delta over the
		window) — the summary numbers CM forensics read."""
		samples = self._samples[self._window_idx:]
		if len(samples) < 2:
			return {}
		a, b = samples[0], samples[-1]
		span = b["t"] - a["t"]
		if span <= 0:
			return {}
		out: Dict[str, Any] = {"cpu_probe_samples": len(samples)}
		for name in self._families:
			if name in a and name in b:
				out[f"cpu_cores_{name}_mean"] = (b[name] - a[name]) / span
		if self._psutil is not None:
			out["cpu_logical_cores"] = self._psutil.cpu_count(logical=True)
		return out


class MeasuredServingRuntime:
	def __init__(
		self,
		*,
		owner: Any,
		stages: List[Dict[str, Any]],
		qa_data: pd.DataFrame,
		node_lines: Dict[str, List[Any]],
		system_config: Dict[str, Any],
		config: dict,
		trace_recorder: Optional[Any] = None,
	):
		self.owner = owner
		self.trace_recorder = trace_recorder
		self.stages = stages
		self.qa_data = qa_data.copy().reset_index(drop=True)
		self.node_lines = node_lines
		self.system_config = system_config
		self.config = config
		if "__qid__" not in self.qa_data.columns:
			self.qa_data["__qid__"] = range(len(self.qa_data))
		self.n_rows = len(self.qa_data)
		self.batch_size_request = request_batch(system_config)
		self.load_concurrency = _measured_load_concurrency(
			system_config, self.batch_size_request
		)
		self.load_concurrency_initial = self.load_concurrency
		# Saturation verdict, written by the population adapter; the default
		# marks runs whose window opened (wall-cap) before any verdict.
		self._saturation: Dict[str, Any] = {
			"evidence": "window_before_verdict", "saturated": None}
		self.warmup_queries = max(
			0,
			int(system_config.get("measured_warmup_queries", self.load_concurrency) or 0),
		)
		self.measured_target_explicit = "measured_queries" in system_config
		if self.measured_target_explicit:
			measured_target = int(system_config["measured_queries"] or 0)
		else:
			# react phase-locks into waves whose period is ~one population
			# turnover (population/qps); 5 turnovers average only ~5
			# oscillation periods and carried ±10-20% window-sampling
			# variance on qps (0046-A quadruplicate, 07-10). 10 turnovers
			# thicken the steady-state average; sequential pipelines keep
			# the cheaper 5 (no wave structure).
			_mode = str(
				(config.get("pipeline_runtime") or {}).get("rag_dataflow")
				or (config.get("pipeline_runtime") or {}).get("mode", "sequential")
			).lower()
			_turns = 10 if _mode == "react" else 5
			measured_target = max(self.n_rows, self.load_concurrency * _turns)
		self.measured_target = max(1, measured_target)
		self.mode = str(
			(config.get("pipeline_runtime") or {}).get("rag_dataflow")
			or (config.get("pipeline_runtime") or {}).get("mode", "sequential")
		).lower()
		self.load_concurrency_hard_cap = _measured_population_hard_cap(
			self.system_config,
			self.load_concurrency_initial,
			self.mode,
		)
		self.services: List[BaseStageService] = []
		# Frozen exactly once when every dataset row has a first completion.
		# Scoring consumes this same object only after run() has closed the
		# measured services, so host/API scorer work cannot perturb timing.
		self._quality_snapshot: Optional[pd.DataFrame] = None
		self._cpu_probe: Optional[_CpuProbe] = None

	async def run(self) -> tuple[pd.DataFrame, Dict[str, Any]]:
		if self.n_rows <= 0:
			return self.qa_data, perf_mod.summarize([], total_wall_clock_s=0.0, n_chips=1)
		if not self.services:
			self.services = self._build_services()
		if self.mode == "react":
			return await self._run_closed_loop_saturated(self._run_react_request)
		return await self._run_closed_loop_saturated(self._run_sequential_request)

	async def close(self) -> None:
		if self._cpu_probe is not None:
			self._cpu_probe.stop()
		for service in self.services:
			await service.close()

	def _freeze_first_completion_snapshot(
		self,
		first_by_idx: Dict[int, pd.DataFrame],
	) -> pd.DataFrame:
		"""Freeze dataset-ordered first-completion winners exactly once."""
		if self._quality_snapshot is None:
			if len(first_by_idx) != self.n_rows:
				raise RuntimeError(
					"cannot freeze a partial measured quality snapshot: "
					f"{len(first_by_idx)}/{self.n_rows} rows"
				)
			self._quality_snapshot = pd.concat(
				[first_by_idx[i] for i in range(self.n_rows)],
				ignore_index=True,
			)
		return self._quality_snapshot

	def _cpu_probe_families(self) -> Dict[str, tuple]:
		"""Process families for the host-CPU probe: each vLLM engine's
		subprocess tree (API server + EngineCore + TP workers), each aux
		worker child process (r20 processization), and the harness main
		process (event loop + its worker threads; children excluded — they
		ARE the other families)."""
		import os
		families: Dict[str, tuple] = {"harness": (os.getpid(), False)}
		for svc in self.services:
			name = str(svc.name)
			if isinstance(svc, LLMContinuousStageService):
				sub = svc._subprocess()
				pid = getattr(getattr(sub, "proc", None), "pid", None)
				if pid:
					families[name] = (int(pid), True)
			else:
				inst = svc.stage.get("instance")
				pid = getattr(getattr(inst, "_proc", None), "pid", None)
				if pid:
					families[name] = (int(pid), True)
		return families

	def _build_services(self) -> List[BaseStageService]:
		services: List[BaseStageService] = []
		dynamic_timeout = _dynamic_batch_timeout_s(self.system_config)
		for stage in self.stages:
			nt = str(stage["stage"])
			if nt == "generator":
				services.append(
					LLMContinuousStageService(
						self.owner,
						stage,
						max_inflight=_stage_max_inflight(
							stage,
							self.system_config,
							"generator",
							self.load_concurrency_hard_cap,
						),
					)
				)
			elif nt == "query_expansion":
				services.append(
					QueryExpansionContinuousStageService(
						self.owner,
						stage,
						max_inflight=_stage_max_inflight(
							stage,
							self.system_config,
							"query_expansion",
							self.load_concurrency_hard_cap,
						),
					)
				)
			elif nt == "semantic_retrieval":
				# ``batch_size_request`` is the per-encoder-replica cap. Collect
				# one such shard per declared GPU before dispatch so B=1/N=2
				# actually exercises both replicas. The retrieval node flattens
				# MQE sub-queries, encodes them in ordered DP, and performs one
				# FAISS search over the combined matrix.
				replica_count = max(
					1,
					len(stage_devices(self.system_config, "semantic_retrieval")),
				)
				services.append(
					BatchedPureStageService(
						self.owner,
						stage,
						batch_size=self.batch_size_request * replica_count,
						timeout_s=dynamic_timeout,
					)
				)
			elif nt in {"hybrid_retrieval", "lexical_retrieval"}:
				services.append(
					BatchedPureStageService(
						self.owner,
						stage,
						batch_size=self.batch_size_request,
						timeout_s=dynamic_timeout,
					)
				)
			elif nt in {"passage_reranker", "passage_compressor"}:
				module = getattr(stage.get("node"), "module", None)
				component = (
					module.get("component")
					if isinstance(module, dict)
					else getattr(module, "component", None)
				)
				# Rerankers admitted by the runtime validator and LLMLingua2 are
				# independent one-model-per-device DP stages.  Keep B as the local
				# replica cap and collect N*B requests before their ordered shard
				# dispatcher.  LongLLMLingua is deliberately excluded: its devices
				# shard one model rather than replicate it.
				is_data_parallel = (
					nt == "passage_reranker"
					or (nt == "passage_compressor" and component == "llmlingua2")
				)
				replica_count = (
					max(1, len(stage_devices(self.system_config, nt)))
					if is_data_parallel else 1
				)
				services.append(
					BatchedPureStageService(
						self.owner,
						stage,
						batch_size=self.batch_size_request * replica_count,
						timeout_s=dynamic_timeout,
					)
				)
			elif nt == "prompt_maker":
				services.append(
					BatchedPureStageService(
						self.owner,
						stage,
						batch_size=self.batch_size_request,
						timeout_s=dynamic_timeout,
					)
				)
			else:
				services.append(DirectPureStageService(self.owner, stage))
		return services

	async def _run_closed_loop_saturated(
		self,
		run_request: Callable[[RequestState], Any],
	) -> tuple[pd.DataFrame, Dict[str, Any]]:
		_raise_nofile_limit()
		recorder = self.trace_recorder
		if recorder is None:
			# Backward-compatible direct-runtime fallback. Production passes the
			# recorder explicitly across the defensive executor boundary.
			from rag_stack.static_rag_evaluator.recording import get_current_recorder

			recorder = get_current_recorder()
		state_lock = asyncio.Lock()
		next_seq = 0
		admission_order = _BalancedDatasetAdmissionOrder(self.n_rows)
		first_by_idx: Dict[int, pd.DataFrame] = {}
		first_qid_by_idx: Dict[int, Any] = {}
		quality_trace_keep_qids: set[Any] = set()
		quality_trace_candidate_admissions = 0
		performance_trace_reservoir = _DeterministicTraceReservoir(
			_PERFORMANCE_TRACE_SAMPLE_CAPACITY
		)
		performance_trace_selected_qids: List[Any] = []
		performance_trace_calls: List[List[dict]] = []
		performance_execution_trace: Optional[Dict[str, Any]] = None
		quality_complete_event = asyncio.Event()

		def _discard_trace_qids(qids: List[Any]) -> None:
			if recorder is None or not qids:
				return
			recorder.discard_qids(qids, require_all=False)

		def _maybe_freeze_snapshot() -> None:
			if len(first_by_idx) < self.n_rows:
				return
			if self._quality_snapshot is None:
				self._freeze_first_completion_snapshot(first_by_idx)
				logger.info(
					"[measured service] quality snapshot frozen "
					f"({self.n_rows} first-completion QA rows); scoring is "
					"deferred until the measured runtime has ended"
				)
			quality_complete_event.set()

		def make_fill_state(idx: int) -> RequestState:
			nonlocal next_seq
			row = self.qa_data.iloc[[idx]].copy().reset_index(drop=True)
			seq = next_seq
			next_seq += 1
			qid = f"final-{seq}"
			row["__qid__"] = qid
			return RequestState(
				seq=seq,
				idx=idx,
				qid=qid,
				df=row,
				is_measured=False,
				# Fill is a real quality invocation. If it wins the missing
				# dataset row it must provide that row's execution trace.
				record_trace=True,
			)

		warmup_cap = _phase_wall_cap_s(self.system_config, "warmup")
		measured_cap = _phase_wall_cap_s(self.system_config, "measured")
		warmup_start = time.perf_counter()
		warmup_completed = 0
		measurement_started = self.warmup_queries <= 0
		measurement_start: Optional[float] = warmup_start if measurement_started else None
		measurement_end: Optional[float] = None
		measurement_start_reason = "no warmup" if measurement_started else ""
		measurement_start_gate = "no_warmup" if measurement_started else "pending"
		measurement_initial_target_queries: Optional[int] = (
			self.measured_target if measurement_started else None
		)
		measurement_stationarity_extensions = 0
		measurement_stop_reason = ""
		stop_admission = False
		perfs: List[perf_mod.QueryPerf] = []
		# Steady-state / closed-loop forensics (r22): warmup completion
		# timestamps feed the rate-stability warmup gate; window completion
		# timestamps feed the sub-window qps CV (the per-record stationarity
		# proof); driver error counters catch the closed-population leak
		# (an uncaught per-request error used to kill its driver: N -> N-1
		# silently mid-window).
		warmup_done_ts: List[float] = []
		window_done_ts: List[float] = []
		# A local completion-rate plateau is not sufficient while the finite
		# workload is still revealing unseen rows.  In a long-latency closed loop,
		# that cold cohort can finish much later as one large catch-up wave.  Anchor
		# the final two-span proof after every dataset row has completed once.  This
		# is workload-support coverage, not a population-turnover requirement: no
		# driver identity or outstanding-request count participates in the gate.
		workload_support_complete_ts: Optional[float] = None
		workload_support_complete_warmup_completed: Optional[int] = None
		saturation_stability_proof_start_completion: Optional[int] = None
		fresh_rate_completions_at_measurement_start: Optional[int] = None
		stability_span_dimension_preserved_across_support = False
		driver_errors_total = 0
		driver_errors_window = 0
		measurement_started_event = asyncio.Event()
		stop_event = asyncio.Event()
		driver_tasks: List[asyncio.Task] = []
		population_ramps: List[_PopulationRamp] = []
		latest_population_ramp: Optional[_PopulationRamp] = None
		active_driver_count = 0
		max_active_driver_count = 0
		active_drivers_at_measurement_start: Optional[int] = None
		population_ramp_pending_at_measurement_start: Optional[int] = None
		# Private, read-only audit state remains available when an attempt aborts
		# before it can publish a summary.
		self._population_ramps = population_ramps
		# Completions-triggered warmup exit defers to the population adapter:
		# current sustained saturation and completion-rate stability must BOTH
		# be established before the window opens.  The wall cap is only a
		# fail-closed runtime budget; it never bypasses either gate.
		adapter_done = self.warmup_queries <= 0
		saturation_stability_span: Optional[int] = None
		warmup_failure_reason: Optional[str] = None
		load_concurrency_initial = self.load_concurrency
		if measurement_started:
			measurement_started_event.set()

		def start_measurement_locked(now: float, reason: str, gate: str) -> None:
			nonlocal measurement_started, measurement_start, measurement_start_reason
			nonlocal measurement_start_gate, measurement_initial_target_queries
			nonlocal active_drivers_at_measurement_start
			nonlocal population_ramp_pending_at_measurement_start
			nonlocal saturation_stability_proof_start_completion
			nonlocal fresh_rate_completions_at_measurement_start
			if measurement_started:
				return
			if self.warmup_queries > 0:
				if workload_support_complete_warmup_completed is None:
					raise RuntimeError(
						"measurement cannot start before workload support is complete"
					)
				if saturation_stability_proof_start_completion is None:
					# The no-LLM path has an intrinsically current saturation
					# candidate and therefore needs no adapter-owned candidate epoch.
					saturation_stability_proof_start_completion = (
						workload_support_complete_warmup_completed
					)
				fresh_rate_completions_at_measurement_start = max(
					0,
					warmup_completed - saturation_stability_proof_start_completion,
				)
			# Keep an automatic target large enough for two copies of the same
			# bounded stage-rate span that proved warmup stability. The adaptive
			# client population is control state, not a workload shape, so growing
			# it must not inflate this evidence floor.
			if (
				not self.measured_target_explicit
				and saturation_stability_span is not None
			):
				# The low-QPS selector budgets four spans in the finite measured
				# window so completion-aligned stationarity has four boundaries.
				# Normal stage-cycle runs retain the existing two-wave target.
				target_waves = (
					4.0
					if self._saturation.get("stability_span_source") == "wall_budget"
					else _GT_MIN_RATE_STABILITY_WAVES
				)
				self.measured_target = max(
					self.measured_target,
					int(math.ceil(
						target_waves
						* saturation_stability_span
					)),
				)
			measurement_started = True
			measurement_start = now
			measurement_initial_target_queries = self.measured_target
			measurement_start_reason = reason
			measurement_start_gate = gate
			active_drivers_at_measurement_start = active_driver_count
			population_ramp_pending_at_measurement_start = (
				latest_population_ramp.pending_drivers
				if latest_population_ramp is not None else 0
			)
			measurement_started_event.set()
			# Anchor every stage service's stats to the measured window (the
			# exported avg_service_s etc. must not include warmup's cold
			# batches — same window semantics as qps).
			for _svc in self.services:
				_svc.mark_window_start()
				if isinstance(_svc, LLMContinuousStageService):
					# Probe runs from serving start (the population adapter
					# reads engine backlog during warmup); here we only pin
					# the window so exported stats stay windowed.
					_svc.mark_probe_window()
			if self._cpu_probe is not None:
				self._cpu_probe.mark_window()
			logger.info(
				"[measured service] measured window starts: "
				f"{reason}; warmup_completed={warmup_completed}"
			)

		def finish_measurement_locked(now: float, reason: str) -> None:
			nonlocal measurement_end, measurement_stop_reason, stop_admission
			if measurement_end is not None:
				return
			measurement_end = now
			measurement_stop_reason = reason
			stop_admission = True
			stop_event.set()
			# Freeze the stats window (the drain tail after this point is
			# protocol overhead, not steady-state service).
			for _svc in self.services:
				_svc.mark_window_end()
				if isinstance(_svc, LLMContinuousStageService):
					_svc.stop_engine_probe()
			if self._cpu_probe is not None:
				self._cpu_probe.stop()
			logger.info(
				"[measured service] measured window stops: "
				f"{reason}; completed={len(perfs)}"
			)

		def fail_warmup_locked(reason: str) -> None:
			"""Stop an attempt whose warmup budget expired without proof.

			No measured window is created: returning a low-CV rate from a window
			that began before saturation/steady-state would be a plausible-looking
			but invalid performance number.  The main task converts this terminal
			state into ``TrialInvalid`` after cancelling the closed population.
			"""
			nonlocal warmup_failure_reason, stop_admission
			if measurement_started or measurement_end is not None:
				return
			if warmup_failure_reason is not None:
				return
			warmup_failure_reason = reason
			stop_admission = True
			stop_event.set()
			logger.warning(f"[measured service] warmup rejected: {reason}")

		async def next_state() -> Optional[RequestState]:
			nonlocal next_seq, quality_trace_candidate_admissions
			async with state_lock:
				if stop_admission:
					return None
				now = time.perf_counter()
				if (
					measurement_started
					and measurement_end is None
					and measurement_start is not None
					and now - measurement_start >= measured_cap
				):
					finish_measurement_locked(now, f"measured cap {measured_cap:.0f}s")
					return None
				idx = admission_order.idx_for_seq(next_seq)
				seq = next_seq
				next_seq += 1
				is_measured_admission = measurement_started
				# Quality still considers every admission until global dataset
				# coverage. Performance recording continues for every active request
				# because membership in the completion-window reservoir is knowable
				# only after the whole invocation (and all of its stage calls) ends.
				# Completed non-winners are pruned immediately, bounding retained
				# state by active population + quality rows + reservoir capacity.
				record_quality_candidate = not quality_complete_event.is_set()
				if record_quality_candidate:
					quality_trace_candidate_admissions += 1
			row = self.qa_data.iloc[[idx]].copy().reset_index(drop=True)
			qid = f"measured-{seq}" if is_measured_admission else f"warmup-{seq}"
			row["__qid__"] = qid
			return RequestState(
				seq=seq,
				idx=idx,
				qid=qid,
				df=row,
				is_measured=is_measured_admission,
				record_trace=True,
			)

		def _rate_stable(w: int, *, start_index: int = 0) -> bool:
			"""Durations of the two most recent spans of w completions within
			10 % of each other. A span that finishes in <0.5 s has no
			seconds-scale settling dynamics to wait out (also keeps unit-test
			loops deterministic)."""
			return _completion_rate_stable(
				warmup_done_ts,
				w,
				start_index=start_index,
			)

		def _bounded_stage_stability_span(batch_cap: int) -> int:
			"""Completion span used to validate one candidate bottleneck.

			Four local batch cycles suppress batch-boundary noise, with a floor
			for tiny stages and a finite ceiling for very large continuous-LLM
			batches.  Unlike ``load_concurrency``, this is stage-local service
			shape and stays reachable when thousands of closed-loop clients queue
			behind one slow full-batch worker.
			"""
			return min(
				_SATURATION_STABILITY_MAX_SPAN,
				max(
					_SATURATION_STABILITY_MIN_SPAN,
					_SATURATION_STABILITY_BATCH_CYCLES
					* max(1, int(batch_cap)),
				),
			)

		def _warmup_rate_stable() -> bool:
			proof_start = saturation_stability_proof_start_completion
			if proof_start is None:
				proof_start = workload_support_complete_warmup_completed
			return bool(
				saturation_stability_span is not None
					and workload_support_complete_warmup_completed is not None
					and proof_start is not None
					and _rate_stable(
						saturation_stability_span,
						start_index=proof_start,
					)
			)

		def _warmup_gate_ready() -> bool:
			return bool(
				adapter_done
					and self._saturation.get("saturated") is True
					and _warmup_rate_stable()
					and latest_population_ramp is not None
					and latest_population_ramp.complete_event.is_set()
					and latest_population_ramp.cancelled_before_activation == 0
					and active_driver_count == len(driver_tasks)
			)

		def _maybe_start_after_warmup_locked(now: float) -> bool:
			if (
				self._saturation.get("candidate_identity") == "no_llm_stage"
				and self._saturation.get("saturated") is not True
				and _warmup_rate_stable()
			):
				# The no-LLM candidate is intrinsic, but its formal saturation
				# verdict is still withheld until support + fresh 2W is proven.
				self._saturation["saturated"] = True
			if not _warmup_gate_ready():
				return False
			start_measurement_locked(
				now,
				"workload support complete + "
				f"two stable spans of {saturation_stability_span} completions "
				"collected fresh after support + "
				f"current {self._saturation.get('evidence')}",
				"warmup_completion_rate_stable",
			)
			return True

		async def driver(
			start_delay_s: float,
			population_ramp: _PopulationRamp,
		) -> None:
			nonlocal warmup_completed
			nonlocal driver_errors_total, driver_errors_window
			nonlocal measurement_stationarity_extensions
			nonlocal active_driver_count, max_active_driver_count
			nonlocal workload_support_complete_ts
			nonlocal workload_support_complete_warmup_completed
			activated = False
			try:
				if start_delay_s > 0.0:
					# Staggered start breaks the phase-locked convoy of a
					# simultaneously-created population (all requests hitting
					# retrieval, then all hitting the generator, leaving each
					# resource idle half the time — r12 Regime-A finding).
					await asyncio.sleep(start_delay_s)
				activated = True
				active_driver_count += 1
				max_active_driver_count = max(
					max_active_driver_count,
					active_driver_count,
				)
				# Publish ramp completion only after this driver is part of the
				# active-population count observed by the waiting adapter.
				population_ramp.mark_activated(time.perf_counter())
				while True:
					state = await next_state()
					if state is None:
						return
					state.admit_ts = time.perf_counter()
					try:
						await run_request(state)
					except asyncio.CancelledError:
						_discard_trace_qids([state.qid])
						raise
					except Exception as exc:  # noqa: BLE001
						# A failed invocation is in neither completed workload. Drop its
						# partial call list immediately so trace memory stays bounded.
						_discard_trace_qids([state.qid])
						error_ts = time.perf_counter()
						msg = str(exc)
						if stop_admission and isinstance(exc, RuntimeError) and (
							("stage service" in msg and "closed" in msg)
							or "no running event loop" in msg
						):
							return  # benign shutdown race during drain
						if _request_failure_is_fatal(exc):
							logger.error(
								"[measured service] request "
								f"{state.qid} hit a fatal deployment error: "
								f"{type(exc).__name__}: {exc}"
							)
							raise
						# Closed-system invariant: a failed request must not
						# kill its driver (the population would silently run
						# at N-1 for the rest of the window). Count + recycle.
						async with state_lock:
							driver_errors_total += 1
							if (
								measurement_start is not None
								and error_ts >= measurement_start
								and (
									measurement_end is None
									or error_ts <= measurement_end
								)
							):
								driver_errors_window += 1
						logger.warning(
							"[measured service] request "
							f"{state.qid} failed and its slot was recycled: "
							f"{type(exc).__name__}: {exc}"
						)
						continue
					state.done_ts = time.perf_counter()
					perf = state.to_perf()
					async with state_lock:
						if state.idx not in first_by_idx:
							first_by_idx[state.idx] = state.df.copy().reset_index(drop=True)
							first_qid_by_idx[state.idx] = state.qid
							quality_trace_keep_qids.add(state.qid)
							_maybe_freeze_snapshot()
						if not measurement_started:
							warmup_completed += 1
							warmup_done_ts.append(state.done_ts)
							if (
								quality_complete_event.is_set()
								and workload_support_complete_warmup_completed is None
							):
								workload_support_complete_ts = state.done_ts
								workload_support_complete_warmup_completed = (
									warmup_completed
								)
								logger.info(
									"[measured service] workload support complete: "
									f"{self.n_rows}/{self.n_rows} rows; "
									"collecting a fresh two-span rate proof"
								)
							if _maybe_start_after_warmup_locked(state.done_ts):
								if state.qid not in quality_trace_keep_qids:
									_discard_trace_qids([state.qid])
								continue
							if state.done_ts - warmup_start >= warmup_cap:
								# Defensive duplicate of warmup_timer: if the
								# event loop delayed that task, a late completion
								# still cannot force-open the window.
								fail_warmup_locked(
									"measured_warmup_gate_timeout: "
									f"cap={warmup_cap:.3f}s "
									f"completions={warmup_completed}/"
									f"{self.warmup_queries} "
									"workload_support_complete="
									f"{workload_support_complete_warmup_completed is not None} "
									f"rate_stable={_warmup_rate_stable()} "
									f"saturation={self._saturation.get('saturated')} "
									f"evidence={self._saturation.get('evidence')}"
								)
							if state.qid not in quality_trace_keep_qids:
								_discard_trace_qids([state.qid])
							continue
						# Performance uses completion-window semantics. Admission
						# phase is deliberately irrelevant: warmup-admitted work
						# completed inside the measured phase contributes, while a
						# measured admission completed after the phase does not.
						# Quality trace selection is independent and happens by
						# dataset-row winner after all quality rows are covered.
						if measurement_start is None or state.done_ts < measurement_start:
							if state.qid not in quality_trace_keep_qids:
								_discard_trace_qids([state.qid])
							continue
						if measurement_end is not None and state.done_ts > measurement_end:
							if state.qid not in quality_trace_keep_qids:
								_discard_trace_qids([state.qid])
							continue
						perfs.append(perf)
						window_done_ts.append(state.done_ts)
						evicted_qid = performance_trace_reservoir.offer(state.qid)
						if (
							evicted_qid is not None
							and evicted_qid not in quality_trace_keep_qids
						):
							_discard_trace_qids([evicted_qid])
						if measurement_end is None and len(perfs) >= self.measured_target:
							current_stationarity = _qps_stationarity_evidence(
								window_done_ts,
								measurement_start,
								state.done_ts,
								completion_span_queries=saturation_stability_span,
							)
							current_cv = current_stationarity["selected_cv"]
							# Automatic measured runs may reach their nominal count while
							# the five-subwindow throughput proof still shows a ramp. Keep
							# collecting inside the existing wall budget instead of stopping
							# early and emitting a predictably inadmissible record. An
							# explicit caller target retains exact-count semantics.
							if (
								not self.measured_target_explicit
								and (
									current_cv is None
									or current_cv > _GT_MAX_QPS_SUBWINDOW_CV
								)
							):
								base_target = max(
									1,
									int(measurement_initial_target_queries or self.measured_target),
								)
								extension = max(
									int(saturation_stability_span or 1),
									max(32, base_target // 5),
								)
								self.measured_target = len(perfs) + extension
								measurement_stationarity_extensions += 1
								logger.info(
									"[measured service] nominal target reached but "
									f"qps_subwindow_cv={current_cv!r} "
									f"method={current_stationarity['method']}; extending by "
									f"{extension} completions to {self.measured_target}"
								)
							else:
								finish_measurement_locked(
									state.done_ts,
									f"target {self.measured_target} completions",
								)
						elif (
							measurement_end is None
							and
							measurement_start is not None
							and state.done_ts - measurement_start >= measured_cap
						):
							finish_measurement_locked(
								state.done_ts,
								f"measured cap {measured_cap:.0f}s",
							)
			except asyncio.CancelledError:
				raise
			except RuntimeError as exc:
				msg = str(exc)
				if stop_admission and (
					"stage service" in msg and "closed" in msg
					or "no running event loop" in msg
				):
					return
				raise
			finally:
				if activated:
					active_driver_count = max(0, active_driver_count - 1)
				elif population_ramp.pending_drivers > 0:
					population_ramp.mark_cancelled_before_activation(
						time.perf_counter()
					)

		def spawn_driver_ramp(
			count: int,
			*,
			spread_s: float,
			kind: str,
		) -> _PopulationRamp:
			"""Schedule one population transition without counting sleepers active."""
			nonlocal latest_population_ramp
			n = max(0, int(count))
			if n <= 0:
				raise ValueError("population ramp requires at least one driver")
			before = len(driver_tasks)
			ramp = _PopulationRamp(
				kind=kind,
				scheduled_drivers=n,
				population_before=before,
				population_after=before + n,
				spread_s=max(0.0, float(spread_s)),
				started_ts=time.perf_counter(),
			)
			population_ramps.append(ramp)
			latest_population_ramp = ramp
			for i in range(n):
				delay = (i / max(1, n)) * ramp.spread_s
				driver_tasks.append(
					asyncio.create_task(
						driver(delay, ramp),
						name=f"measured-driver-{kind}-{i}",
					)
				)
			return ramp

		async def wait_for_population_ramp(ramp: _PopulationRamp) -> bool:
			await ramp.complete_event.wait()
			return bool(
				not stop_event.is_set()
				and ramp.cancelled_before_activation == 0
				and active_driver_count == ramp.population_after
			)

		async def population_adapter() -> None:
			"""Warmup-phase saturation controller (FROZEN at the measured
			window start, so windowed stats stay stationary).

			The static population 4 x max(request, decode) provably cannot
			fill the LLM's slots when the request cycle parks time in other
			stages (react: Little's law gives generator inflight =
			population x holding/cycle — r12 Regime-A). Grow the population
			until a resource shows a REAL standing backlog — an LLM engine's
			server-side waiting queue, or a worker stage's submit queue
			holding more than a full batch — or the population cap is hit. A
			backlog is accepted only AFTER the completion rate is stable; this
			excludes the one-time burst produced when every sequential driver
			enters the first stage during startup.
			Client-side inflight fill alone cannot distinguish
			"engine full with backlog" from "engine barely fed" (r12), and
			0045's dashboard (running 246/256, waiting 0, token rate 51% of
			solo) showed a run the old criterion wrongly accepted as
			saturated. The stop verdict lands in the summary
			(saturation_evidence / population_saturated) so every record
			carries its own admissibility proof as CM truth.
			"""
			nonlocal adapter_done, saturation_stability_span
			nonlocal stability_span_dimension_preserved_across_support
			try:
				initial_ramp = latest_population_ramp
				if initial_ramp is None:
					raise RuntimeError(
						"population adapter started before initial driver ramp"
					)
				# Do not sample a half-created initial population. In the eval9
				# failure, the five-second adapter epoch ended halfway through the
				# existing ten-second stagger and doubled 256 sleeping/active drivers
				# to 512 from a meaningless low-fill snapshot.
				if not await wait_for_population_ramp(initial_ramp):
					return
				llm = [
					s for s in self.services
					if isinstance(s, LLMContinuousStageService)
				]
				workers = [
					s for s in self.services
					if isinstance(s, BatchedPureStageService)
				]
				if not llm:
					local_cap = max(
						[self.batch_size_request]
						+ [w.batch_size for w in workers]
					)
					preferred_span = _bounded_stage_stability_span(local_cap)
					saturation_stability_span = min(
						max(1, self.warmup_queries),
						preferred_span,
					)
					self._saturation = {
						"evidence": "no_llm_stage",
						"saturated": None,
						"candidate_identity": "no_llm_stage",
						"stability_span_source": "stage_cycles",
						"stability_preferred_span_queries": preferred_span,
						"stability_effective_span_queries": (
							saturation_stability_span
						),
					}
					adapter_done = True
					async with state_lock:
						_maybe_start_after_warmup_locked(time.perf_counter())
					return
				# This bounds the finite driver population, not a stage's HTTP
				# residency. A shallower LLM semaphore intentionally leaves excess
				# population queued before the frontend instead of opening thousands
				# of sockets / vLLM waiting requests.
				hard_cap = self.load_concurrency_hard_cap

				async def _verdict(
					evidence: str,
					saturated: bool,
					stability_span: int,
					*,
					stability_start_completion: Optional[int] = None,
					preferred_span: Optional[int] = None,
					span_source: str = "stage_cycles",
					candidate_identity: Optional[str] = None,
					wall_budget: Optional[Dict[str, Any]] = None,
				) -> None:
					nonlocal adapter_done, saturation_stability_span
					nonlocal saturation_stability_proof_start_completion
					if stop_event.is_set():
						return
					if (
						latest_population_ramp is None
						or not latest_population_ramp.complete_event.is_set()
						or latest_population_ramp.cancelled_before_activation
						or active_driver_count != len(driver_tasks)
					):
						raise RuntimeError(
							"population adapter attempted a saturation verdict "
							"during an incomplete driver ramp"
						)
					saturation_stability_span = max(1, int(stability_span))
					if saturated:
						if workload_support_complete_warmup_completed is None:
							raise RuntimeError(
								"saturated verdict requires complete workload support"
							)
						if stability_start_completion is None:
							raise RuntimeError(
								"saturated verdict requires a fresh-rate proof epoch"
							)
						saturation_stability_proof_start_completion = max(
							0, int(stability_start_completion),
						)
						if (
							saturation_stability_proof_start_completion
							< workload_support_complete_warmup_completed
						):
							raise RuntimeError(
								"saturated verdict rate proof predates workload support"
							)
						if not _rate_stable(
							saturation_stability_span,
							start_index=saturation_stability_proof_start_completion,
						):
							raise RuntimeError(
								"saturated verdict lacks a fresh two-span rate proof"
							)
					preferred = max(
						1,
						int(preferred_span or saturation_stability_span),
					)
					self._saturation = {
						"evidence": evidence,
						"saturated": saturated,
						"candidate_identity": candidate_identity or evidence,
						"stability_span_source": span_source,
						"stability_preferred_span_queries": preferred,
						"stability_effective_span_queries": (
							saturation_stability_span
						),
					}
					if wall_budget is not None:
						self._saturation.update({
							"candidate_age_s_at_span_freeze": wall_budget[
								"candidate_age_s"
							],
							"candidate_completions_at_span_freeze": wall_budget[
								"candidate_completions"
							],
							"candidate_rate_qps_at_span_freeze": wall_budget[
								"candidate_rate_qps"
							],
							"wall_budget_available_span_queries": wall_budget[
								"available_span_queries"
							],
							"wall_budget_measurement_span_queries": wall_budget[
								"measurement_span_queries"
							],
							"wall_budget_minimum_span_queries": wall_budget[
								"minimum_span_queries"
							],
						})
					adapter_done = True
					logger.info(
						"[measured service] population adapter stop: "
						f"{evidence} (saturated={saturated}, "
						f"population={active_driver_count}, cap {hard_cap}, "
						f"stability_span={saturation_stability_span}, "
						f"span_source={span_source})"
					)
					async with state_lock:
						if saturated:
							_maybe_start_after_warmup_locked(time.perf_counter())
						else:
							fail_warmup_locked(
								"measured_population_not_saturated: "
								f"evidence={evidence} population={active_driver_count} "
								f"cap={hard_cap} "
								f"stability_span={saturation_stability_span}"
							)

				tracked_candidate_identity: Optional[str] = None
				tracked_candidate_since: Optional[float] = None
				tracked_candidate_start_completion = 0
				tracked_candidate_preferred_span: Optional[int] = None
				tracked_candidate_effective_span: Optional[int] = None
				tracked_candidate_wall_budget: Optional[Dict[str, Any]] = None
				tracked_support_epoch: Optional[int] = None

				def _fresh_candidate_rate_stable(span: int) -> bool:
					"""Prove the current candidate after workload support is known.

					The start is also bounded by the candidate's own epoch, so a
					candidate switch cannot reuse completions collected under the
					previous bottleneck identity.
					"""
					if workload_support_complete_warmup_completed is None:
						return False
					proof_start = max(
						workload_support_complete_warmup_completed,
						tracked_candidate_start_completion,
					)
					return _rate_stable(span, start_index=proof_start)

				def _fresh_support_rate_stable(span: int) -> bool:
					"""Rate proof for a no-backlog cap verdict after full support.

					A cap-hit rejection before the complete workload has appeared can
					misclassify a transiently quiet prefix.  It is fail-closed, but it
					must still observe the same fresh two-span workload epoch as an
					accepted saturation verdict.
					"""
					if workload_support_complete_warmup_completed is None:
						return False
					return _rate_stable(
						span,
						start_index=workload_support_complete_warmup_completed,
					)

				def _fresh_candidate_proof_start() -> int:
					if workload_support_complete_warmup_completed is None:
						raise RuntimeError(
							"fresh candidate proof requested before workload support"
						)
					return max(
						workload_support_complete_warmup_completed,
						tracked_candidate_start_completion,
					)

				while not measurement_started_event.is_set():
					# A candidate accepted below was present throughout this complete
					# telemetry epoch.  Its fresh-rate suffix may therefore start at the
					# epoch boundary (still clamped to workload support), rather than at
					# the end of the five samples.  Starting at the end discards the very
					# post-support epoch that established the candidate and needlessly
					# requires a third epoch before a verdict can be published.
					sample_epoch_start_completion = len(warmup_done_ts)
					fills = []
					backlog_hits: Dict[str, int] = {}
					pd_stage_backlog_hits: Dict[str, Dict[str, Any]] = {}
					llm_stage_backlog_hits: Dict[str, Dict[str, Any]] = {}
					for _ in range(5):
						if measurement_started_event.is_set():
							return
						# Client occupancy sizes the next population increment
						# only. It is not engine-saturation evidence: requests can
						# be parked in the HTTP/frontend path while vLLM has neither
						# a full running batch nor a scheduler waiting queue.
						fills.append(max(
							float(s.inflight) / float(max(1, s.max_inflight))
							for s in llm
						))
						for w in workers:
							if w.backlog() > w.batch_size:
								backlog_hits[w.name] = (
									backlog_hits.get(w.name, 0) + 1)
						# A disaggregated P/D pair deliberately bounds outer client
						# residency to sum(role admission): P-engine-cap + feeder slots,
						# plus strict D cap.  The feeder/backpressure can keep vLLM's
						# instantaneous server waiting gauge at zero.  For this role-aware
						# case only, a sustained explicit queue outside a FULL, unclamped
						# P/D stage gate is the corresponding saturation proof.  This is
						# stronger than generic client fill: it requires standing demand
						# beyond every configured P/D admission slot.
						for s in llm:
							role_stats = getattr(
								s._subprocess(), "role_admission_stats", None,
							)
							if not callable(role_stats):
								continue
							try:
								roles = role_stats()
							except Exception:  # noqa: BLE001 — advisory telemetry
								continue
							role_rows = [
								stats for stats in (roles or {}).values()
								if isinstance(stats, dict)
							]
							if not role_rows:
								continue
							admission_sum = sum(
								max(1, int(stats.get("admission_limit") or 1))
								for stats in role_rows
							)
							# A population clamp below the real P/D admission sum would
							# make this an artificial outer-gate bottleneck, not engine
							# saturation, so fail closed in that case.
							if s.max_inflight != admission_sum:
								continue
							if s.waiting <= 0 or s.inflight < s.max_inflight:
								continue
							engine_cap = max(
								max(1, int(stats.get("engine_max_num_seqs") or 1))
								for stats in role_rows
							)
							hit = pd_stage_backlog_hits.setdefault(
								s.name,
								{"hits": 0, "engine_cap": engine_cap},
							)
							hit["hits"] += 1
						# A collocated engine has one explicit outer stage gate.  A
						# standing waiter outside that gate is valid saturation evidence
						# only when the gate is the deployed engine cap plus the existing
						# admission slack.  A population-clamped gate would merely prove
						# that the benchmark's artificial cap is full.  PD pairs are
						# excluded by their role-admission rows and retain the independent
						# role-aware proof above.
						for s in llm:
							engine_cap = _collocated_outer_backlog_engine_cap(
								s,
							)
							if engine_cap is None:
								continue
							hit = llm_stage_backlog_hits.setdefault(
								s.name,
								{"hits": 0, "engine_cap": engine_cap},
							)
							hit["hits"] += 1
						await asyncio.sleep(_POPULATION_ADAPTER_SAMPLE_S)
					if warmup_completed <= 0:
						continue  # nothing has flowed yet — keep observing
					# A persistent CURRENT candidate freezes population growth while
					# its bounded stage-cycle rate span settles. If the candidate was
					# only the startup wave it clears, and adaptation resumes. Candidate
					# identity is explicit so low-QPS evidence can never splice two
					# different bottlenecks together.
					current_candidates: List[tuple[str, int]] = []
					# 1) True engine backlog: the server's own waiting queue held
					# requests in >=3 of the CURRENT last 5 samples.
					for s in llm:
						waits = s.recent_engine_waiting(5)
						if len(waits) < 3 or sum(1 for w in waits if w > 0) < 3:
							continue
						candidate_identity = f"engine_backlog:{s.name}"
						span = _bounded_stage_stability_span(
							vllm_max_num_seqs(
								self.system_config, str(s.stage.get("stage") or s.name)
							)
						)
						current_candidates.append((candidate_identity, span))
					# 2) Bounded disaggregated-P/D outer-stage backlog, checked
					# before generic worker queues.  The stability span uses the
					# larger role engine cap (P=2/D=256 therefore uses 256).
					for stage_name, hit in pd_stage_backlog_hits.items():
						if int(hit.get("hits") or 0) < 3:
							continue
						candidate_identity = f"pd_stage_backlog:{stage_name}"
						span = _bounded_stage_stability_span(
							int(hit.get("engine_cap") or 1)
						)
						current_candidates.append((candidate_identity, span))
					# 3) Worker-stage backlog (rerank-/retrieval-bound
					# deployments: the LLM dashboards stay empty forever).
					for w in workers:
						if backlog_hits.get(w.name, 0) < 3:
							continue
						candidate_identity = f"worker_backlog:{w.name}"
						span = _bounded_stage_stability_span(w.batch_size)
						current_candidates.append((candidate_identity, span))
					# 4) Collocated outer-stage backlog. Keep this after true engine
					# and worker queues so the new proof cannot change their candidate
					# selection when multiple bottlenecks are simultaneously visible.
					for stage_name, hit in llm_stage_backlog_hits.items():
						if int(hit.get("hits") or 0) < 3:
							continue
						candidate_identity = f"llm_stage_backlog:{stage_name}"
						span = _bounded_stage_stability_span(
							int(hit.get("engine_cap") or 1)
						)
						current_candidates.append((candidate_identity, span))
					if (
						workload_support_complete_warmup_completed is not None
						and tracked_support_epoch
						!= workload_support_complete_warmup_completed
					):
						# Workload support starts a fresh RATE-proof epoch. If the same
						# candidate is still present in the current sampled epoch,
						# an already frozen low-QPS W may survive only as the proof's
						# dimension; no pre-support timestamp can satisfy the final 2W
						# check. If W is not frozen, its age/rate sizing also restarts at
						# support. Candidate disappearance/switch keeps the ordinary full
						# reset below. No driver/request turnover is required.
						tracked_support_epoch = (
							workload_support_complete_warmup_completed
						)
						candidate_by_identity = dict(current_candidates)
						candidate_still_current = (
							tracked_candidate_identity is not None
							and tracked_candidate_identity in candidate_by_identity
						)
						if candidate_still_current:
							tracked_candidate_start_completion = tracked_support_epoch
							if tracked_candidate_effective_span is None:
								tracked_candidate_since = (
									workload_support_complete_ts
									or time.perf_counter()
								)
								tracked_candidate_wall_budget = None
							else:
								stability_span_dimension_preserved_across_support = True
							logger.info(
								"[measured service] population adapter: workload support "
								"complete; retained current candidate"
								+ (
									" and frozen W as dimension only"
									if tracked_candidate_effective_span is not None else ""
								)
								+ "; reset fresh-rate proof epoch"
							)
						else:
							tracked_candidate_identity = None
							tracked_candidate_since = None
							tracked_candidate_start_completion = tracked_support_epoch
							tracked_candidate_preferred_span = None
							tracked_candidate_effective_span = None
							tracked_candidate_wall_budget = None
							logger.info(
								"[measured service] population adapter: workload support "
								"complete; reset absent candidate proof epoch"
							)
						# Support may have arrived halfway through the just-finished
						# telemetry epoch. Preserve only the candidate/W scalar state
						# above, then require the next complete five-sample epoch to
						# prove that the candidate is still current post-support.
						continue
					if current_candidates:
						candidate_by_identity = dict(current_candidates)
						if tracked_candidate_identity not in candidate_by_identity:
							tracked_candidate_identity, tracked_candidate_preferred_span = (
								current_candidates[0]
							)
							tracked_candidate_since = time.perf_counter()
							tracked_candidate_start_completion = (
								sample_epoch_start_completion
							)
							tracked_candidate_effective_span = None
							tracked_candidate_wall_budget = None
						if (
							tracked_candidate_identity is not None
							and tracked_candidate_preferred_span is not None
							and tracked_candidate_effective_span is None
							and _fresh_candidate_rate_stable(
								tracked_candidate_preferred_span
							)
						):
							await _verdict(
								tracked_candidate_identity,
								True,
								tracked_candidate_preferred_span,
								stability_start_completion=(
									_fresh_candidate_proof_start()
								),
								preferred_span=tracked_candidate_preferred_span,
								candidate_identity=tracked_candidate_identity,
							)
							return
						if (
							tracked_candidate_effective_span is None
							and tracked_candidate_since is not None
							and tracked_candidate_preferred_span is not None
							and (
								time.perf_counter() - warmup_start
								>= warmup_cap / 2.0
							)
						):
							now = time.perf_counter()
							candidate_completions = max(
								0,
								len(warmup_done_ts)
								- tracked_candidate_start_completion,
							)
							budget = _wall_budget_stability_span(
								preferred_span=tracked_candidate_preferred_span,
								candidate_completions=candidate_completions,
								candidate_age_s=now - tracked_candidate_since,
								measured_wall_cap_s=measured_cap,
							)
							effective_span = budget["effective_span_queries"]
							if effective_span is not None:
								# Freeze once for this persistent candidate. A later rate
								# change must fail the fixed proof, not select a friendlier W.
								# If W was frozen before support, it remains a dimension only;
								# _fresh_candidate_rate_stable still validates a disjoint 2W
								# timestamp suffix anchored at workload support.
								tracked_candidate_effective_span = int(effective_span)
								tracked_candidate_wall_budget = budget
								logger.info(
									"[measured service] low-QPS stability span frozen: "
									f"candidate={tracked_candidate_identity} "
									f"preferred={tracked_candidate_preferred_span} "
									f"effective={tracked_candidate_effective_span} "
									f"completions={candidate_completions} "
									f"age={budget['candidate_age_s']:.3f}s "
									f"rate={budget['candidate_rate_qps']:.6f}qps"
								)
						if (
							tracked_candidate_effective_span is not None
							and tracked_candidate_identity is not None
							and tracked_candidate_preferred_span is not None
							and tracked_candidate_wall_budget is not None
							and _fresh_candidate_rate_stable(
								tracked_candidate_effective_span
							)
						):
							await _verdict(
								tracked_candidate_identity,
								True,
								tracked_candidate_effective_span,
								stability_start_completion=(
									_fresh_candidate_proof_start()
								),
								preferred_span=tracked_candidate_preferred_span,
								span_source="wall_budget",
								candidate_identity=tracked_candidate_identity,
								wall_budget=tracked_candidate_wall_budget,
							)
							return
						continue
					tracked_candidate_identity = None
					tracked_candidate_since = None
					tracked_candidate_start_completion = len(warmup_done_ts)
					tracked_candidate_preferred_span = None
					tracked_candidate_effective_span = None
					tracked_candidate_wall_budget = None
					mean_fill = sum(fills) / len(fills)
					add = _next_population_increment(
						active_driver_count, hard_cap, mean_fill,
					)
					if add <= 0:
						# The population has reached its finite cap while the
						# startup wave is still settling. Keep observing until a
						# bounded local-rate span proves that there is STILL no
						# standing backlog, then fail closed at the warmup cap.
						fallback_span = _bounded_stage_stability_span(max(
							[self.batch_size_request]
							+ [
								vllm_max_num_seqs(
									self.system_config,
									str(s.stage.get("stage") or s.name),
								)
								for s in llm
							]
							+ [w.batch_size for w in workers]
						))
						if _fresh_support_rate_stable(fallback_span):
							await _verdict("cap_hit", False, fallback_span)
							return
						continue
					logger.info(
						"[measured service] population adapter: "
						f"max LLM client fill={mean_fill:.2f}, adding {add} drivers "
						f"({active_driver_count} -> {active_driver_count + add}, "
						f"cap {hard_cap})"
					)
					ramp = spawn_driver_ramp(
						add,
						# Preserve the paper benchmark's existing adaptive-ramp policy.
						# The bug was missing synchronization, not this 2 s schedule.
						spread_s=2.0,
						kind="adaptive_increment",
					)
					self.load_concurrency = ramp.population_after
					# The controller owns the only mutation path. Waiting here means no
					# saturation samples, candidate tracking, or span freezing can use
					# sleeping drivers as active population evidence.
					if not await wait_for_population_ramp(ramp):
						return
					tracked_candidate_identity = None
					tracked_candidate_since = None
					tracked_candidate_start_completion = len(warmup_done_ts)
					tracked_candidate_preferred_span = None
					tracked_candidate_effective_span = None
					tracked_candidate_wall_budget = None
			except Exception:
				logger.exception("[measured service] population adapter failed")
				raise
			finally:
				adapter_done = True

		async def warmup_timer() -> None:
			if self.warmup_queries <= 0:
				return
			await asyncio.sleep(warmup_cap)
			async with state_lock:
				if measurement_started or measurement_end is not None:
					return
				if _maybe_start_after_warmup_locked(time.perf_counter()):
					return
				fail_warmup_locked(
					"measured_warmup_gate_timeout: "
					f"cap={warmup_cap:.3f}s "
					f"completions={warmup_completed}/{self.warmup_queries} "
					"workload_support_complete="
					f"{workload_support_complete_warmup_completed is not None} "
					f"rate_stable={_warmup_rate_stable()} "
					f"adapter_done={adapter_done} "
					f"saturation={self._saturation.get('saturated')} "
					f"evidence={self._saturation.get('evidence')}"
				)

		async def measured_timer() -> None:
			await measurement_started_event.wait()
			await asyncio.sleep(measured_cap)
			async with state_lock:
				if measurement_end is None:
					finish_measurement_locked(
						time.perf_counter(),
						f"measured cap {measured_cap:.0f}s",
					)

		def throughput_rates() -> Dict[str, Any]:
			"""Return authoritative phase-window and diagnostic delta rates."""
			phase_wall = (
				max(0.0, measurement_end - measurement_start)
				if measurement_start is not None and measurement_end is not None
				else 0.0
			)
			phase_queries = len(perfs)
			phase_tokens = sum(p.n_output_tokens for p in perfs)
			out: Dict[str, Any] = {
				"phase_window_s": phase_wall,
				"phase_window_queries": phase_queries,
				"phase_window_output_tokens": phase_tokens,
				"phase_window_qps": (
					float(phase_queries) / phase_wall if phase_wall > 0.0 else 0.0
				),
				"phase_window_output_tokens_per_s": (
					float(phase_tokens) / phase_wall if phase_wall > 0.0 else 0.0
				),
				"completion_delta_span_s": 0.0,
				"completion_delta_queries": 0,
				"completion_delta_output_tokens": 0,
				"completion_delta_qps": 0.0,
				"completion_delta_output_tokens_per_s": 0.0,
			}
			if len(perfs) >= 2:
				first_done = float(perfs[0].last_token_ts)
				last_done = float(perfs[-1].last_token_ts)
				delta_span = last_done - first_done
				if delta_span > 0.0:
					delta_queries = len(perfs) - 1
					delta_tokens = sum(p.n_output_tokens for p in perfs[1:])
					out.update({
						"completion_delta_span_s": delta_span,
						"completion_delta_queries": delta_queries,
						"completion_delta_output_tokens": delta_tokens,
						"completion_delta_qps": float(delta_queries) / delta_span,
						"completion_delta_output_tokens_per_s": (
							float(delta_tokens) / delta_span
						),
					})
			return out

		async def finish_missing_quality_rows() -> None:
			missing = [idx for idx in range(self.n_rows) if idx not in first_by_idx]
			if not missing:
				return
			logger.info(
				"[measured service] quality fill starts: running "
				f"{len(missing)} missing QA row(s) with real inference"
			)
			sem = asyncio.Semaphore(max(1, min(self.load_concurrency, len(missing))))

			async def fill_one(idx: int) -> None:
				async with sem:
					state = make_fill_state(idx)
					state.admit_ts = time.perf_counter()
					await run_request(state)
					state.done_ts = time.perf_counter()
					async with state_lock:
						row = state.df.copy().reset_index(drop=True)
						if idx not in first_by_idx:
							first_by_idx[idx] = row
							first_qid_by_idx[idx] = state.qid
							_maybe_freeze_snapshot()

			await asyncio.gather(*(fill_one(idx) for idx in missing))

		async def drain_driver_completions(
			driver_tasks: List[asyncio.Task],
			*,
			timeout_s: Optional[float],
		) -> None:
			deadline = (
				None if timeout_s is None else time.perf_counter() + max(0.0, timeout_s)
			)
			while not quality_complete_event.is_set():
				live_drivers = [task for task in driver_tasks if not task.done()]
				if not live_drivers:
					break
				timeout: Optional[float] = None
				if deadline is not None:
					timeout = deadline - time.perf_counter()
					if timeout <= 0.0:
						break
				done, _pending = await asyncio.wait(
					live_drivers,
					timeout=timeout,
					return_when=asyncio.FIRST_COMPLETED,
				)
				if not done:
					break
				for task in done:
					await task

		async def cancel_drivers(
			driver_tasks: List[asyncio.Task],
			*,
			reason: str,
		) -> None:
			def consume_exception(task: asyncio.Task) -> None:
				try:
					task.exception()
				except (asyncio.CancelledError, asyncio.InvalidStateError):
					return

			pending_drivers = [task for task in driver_tasks if not task.done()]
			for task in pending_drivers:
				task.cancel()
			if pending_drivers:
				done_cancelled, still_pending = await asyncio.wait(
					pending_drivers,
					timeout=_PHASE_CANCEL_GRACE_S,
				)
				results = await asyncio.gather(*done_cancelled, return_exceptions=True)
				for result in results:
					if isinstance(result, BaseException) and not isinstance(
						result,
						asyncio.CancelledError,
					):
						raise result
			else:
				still_pending = set()
			for task in driver_tasks:
				if not task.done() or task.cancelled():
					continue
				exc = task.exception()
				if exc is not None:
					raise exc
			if still_pending:
				logger.warning(
					f"[measured service] {reason}; "
					f"{len(still_pending)} duplicate driver(s) still cancelling "
					f"after {_PHASE_CANCEL_GRACE_S:.0f}s; measured_completed={len(perfs)}"
				)
			for task in still_pending:
				task.add_done_callback(consume_exception)
				task.cancel()

		quality_trace_expected_qids: List[Any] = []
		quality_trace_selected_qids: List[Any] = []
		quality_trace_recorded_before_projection = 0
		quality_trace_exact = False

		logger.info(
			"[measured service] start: "
			f"mode={self.mode} batch_size_request={self.batch_size_request} "
			f"load_concurrency={self.load_concurrency} "
			f"warmup={self.warmup_queries} measured={self.measured_target} "
			f"qa_rows={self.n_rows} "
			f"wall_cap(warmup/measured)={warmup_cap:.0f}/{measured_cap:.0f}s"
		)
		try:
			# Convoy-breaking stagger only makes sense at real population
			# scale (waves of batch-quantized requests); tiny closed loops
			# must deploy instantly so short runs still reach full
			# concurrency.
			_spread = (
				min(10.0, warmup_cap * 0.25)
				if self.warmup_queries > 0 and self.load_concurrency >= 64
				else 0.0
			)
			# Engine dashboards from t=0: the population adapter's saturation
			# evidence (engine waiting > 0) needs probe samples during warmup.
			for _svc in self.services:
				if isinstance(_svc, LLMContinuousStageService):
					_svc.start_engine_probe()
			# Host-CPU dashboards from t=0 (windowed at measurement start):
			# per-process-family core usage — the direct evidence for the
			# CPU-axis contention the CM's cpu capacity / the closed-loop
			# sim's CPU resource model.
			self._cpu_probe = _CpuProbe(self._cpu_probe_families())
			self._cpu_probe.start()
			_maybe_start_stack_dump_watchdog()
			spawn_driver_ramp(
				self.load_concurrency,
				spread_s=_spread,
				kind="initial_population",
			)
			stop_waiter = asyncio.create_task(stop_event.wait())
			monitors = [
				asyncio.create_task(warmup_timer()),
				asyncio.create_task(measured_timer()),
				asyncio.create_task(population_adapter()),
			]
			try:
				while not stop_event.is_set():
					live_drivers = [task for task in driver_tasks if not task.done()]
					if not live_drivers:
						break
					done, _pending = await asyncio.wait(
						[*live_drivers, stop_waiter],
						return_when=asyncio.FIRST_COMPLETED,
					)
					if stop_waiter in done:
						break
					for task in done:
						await task
				if warmup_failure_reason is not None:
					# Fail closed before quality drain/fill or summary creation:
					# this attempt has no admissible performance window.
					from rag_stack.static_rag_evaluator.measured.vllm_deployment import (
						TrialInvalid,
					)
					raise TrialInvalid(warmup_failure_reason)
				if not stop_event.is_set():
					async with state_lock:
						if measurement_end is None:
							finish_measurement_locked(
								time.perf_counter(),
								"drivers completed before measured stop",
							)
				if not quality_complete_event.is_set():
					logger.info(
						"[measured service] measured window ended; waiting for "
						f"real QA coverage ({len(first_by_idx)}/{self.n_rows})"
					)
					await drain_driver_completions(
						driver_tasks,
						timeout_s=_QUALITY_DRAIN_GRACE_S,
					)
				if not quality_complete_event.is_set():
					logger.info(
						"[measured service] quality coverage still incomplete after "
						f"{_QUALITY_DRAIN_GRACE_S:.0f}s drain "
						f"({len(first_by_idx)}/{self.n_rows}); cancelling duplicate "
						"closed-loop drivers before fill"
					)
					await cancel_drivers(
						driver_tasks,
						reason="quality fill starting",
					)
					await finish_missing_quality_rows()
				if len(first_by_idx) < self.n_rows:
					raise RuntimeError(
						"measured service could not complete all QA rows for quality "
						f"({len(first_by_idx)}/{self.n_rows})"
					)
				await cancel_drivers(driver_tasks, reason="quality complete")

				# Performance is a done_ts-window cohort. Sort it independently
				# from the quality workload trace; the two have intentionally
				# different cardinality and semantics.
				if len(perfs) != len(window_done_ts):
					raise RuntimeError(
						"measured performance cohort lost perf/timestamp alignment"
					)
				cohort = sorted(
					zip(window_done_ts, perfs),
					key=lambda item: (item[0], item[1].request_send_ts),
				)
				window_done_ts[:] = [item[0] for item in cohort]
				perfs[:] = [item[1] for item in cohort]
				performance_trace_selected_qids = list(
					performance_trace_reservoir.qids
				)
				expected_performance_sample = min(
					_PERFORMANCE_TRACE_SAMPLE_CAPACITY,
					len(perfs),
				)
				if len(performance_trace_selected_qids) != expected_performance_sample:
					raise RuntimeError(
						"performance trace reservoir lost completion-window members: "
						f"expected {expected_performance_sample}, got "
						f"{len(performance_trace_selected_qids)}"
					)

				# Quality trace is one concrete invocation per dataset QA row,
				# matching the exact first-completion dataframe returned for
				# quality. Projection order is dataset idx, never completion or
				# performance order.
				if len(first_qid_by_idx) != self.n_rows:
					raise RuntimeError(
						"quality trace winners do not cover every dataset row "
						f"({len(first_qid_by_idx)}/{self.n_rows})"
					)
				quality_trace_expected_qids = [
					first_qid_by_idx[idx] for idx in range(self.n_rows)
				]

				if recorder is not None:
					# This counter retains the historical quality-candidate
					# diagnostic even though duplicate completed traces are now
					# streamed out of the recorder instead of retained to the end.
					quality_trace_recorded_before_projection = (
						quality_trace_candidate_admissions
					)
					quality_trace_selected_qids = list(
						recorder.select_qids(
							quality_trace_expected_qids,
							require_all=True,
						)
					)
					if quality_trace_selected_qids != quality_trace_expected_qids:
						raise RuntimeError(
							"quality trace projection did not preserve dataset winner order"
						)
					quality_trace_exact = True
				rates = throughput_rates()
			except BaseException:
				# Abort path (e.g. a driver failed the attempt with an engine
				# error): cancel and CONSUME every remaining driver before the
				# caller tears services down. Un-retrieved stragglers otherwise
				# hit the closed services and asyncio prints their tracebacks
				# minutes later inside the NEXT eval's log section.
				for task in driver_tasks:
					if not task.done():
						task.cancel()
				done_now, still_pending = await asyncio.wait(
					driver_tasks, timeout=_PHASE_CANCEL_GRACE_S
				)
				for task in done_now:
					if not task.cancelled():
						task.exception()  # retrieve, don't re-raise
				for task in still_pending:
					task.add_done_callback(
						lambda t: t.cancelled() or t.exception()
					)
				raise
			finally:
				stop_waiter.cancel()
				for task in monitors:
					task.cancel()
				await asyncio.gather(stop_waiter, *monitors, return_exceptions=True)
			# Return the exact object frozen at dataset coverage. The shared
			# pipeline scores it only after this runtime has fully ended.
			if self._quality_snapshot is None:
				raise RuntimeError(
					"measured service completed without freezing a quality snapshot"
				)
			final_result = self._quality_snapshot
		finally:
			await self.close()

		# Deferred tokenization and the second cohort readout happen only after
		# every measured service has closed. ``select_qids`` above froze writers
		# but intentionally defers pruning, so this non-destructive read cannot
		# alter the quality winners returned by the shared pipeline core.
		if recorder is not None:
			performance_trace_calls = recorder.trace_for_qids(
				performance_trace_selected_qids,
				require_all=True,
			)
			performance_execution_trace = make_performance_trace_envelope(
				performance_trace_calls,
				invocation_ids=performance_trace_selected_qids,
				capacity=_PERFORMANCE_TRACE_SAMPLE_CAPACITY,
				population_queries=len(perfs),
			)

		summary = perf_mod.summarize(
			perfs,
			total_wall_clock_s=rates["phase_window_s"],
			n_chips=self.owner._generator_chip_count(self.system_config),
			throughput_queries=rates["phase_window_queries"],
			throughput_output_tokens=rates["phase_window_output_tokens"],
		)
		self._annotate_summary(summary, len(perfs), rates["phase_window_s"])
		# ``qps`` is authoritative phase-window throughput.  Keep the old
		# generic names as explicit aliases and publish the completion-delta
		# estimate under names that cannot be mistaken for the GT rate.
		summary["measurement_rate_mode"] = "phase_window"
		summary["measurement_rate_queries"] = rates["phase_window_queries"]
		summary["measurement_phase_window_s"] = rates["phase_window_s"]
		summary["measurement_phase_window_queries"] = rates["phase_window_queries"]
		summary["measurement_phase_window_output_tokens"] = rates[
			"phase_window_output_tokens"
		]
		summary["measurement_phase_window_qps"] = rates["phase_window_qps"]
		summary["measurement_phase_window_output_tokens_per_s"] = rates[
			"phase_window_output_tokens_per_s"
		]
		summary["measurement_completion_delta_span_s"] = rates[
			"completion_delta_span_s"
		]
		summary["measurement_completion_delta_queries"] = rates[
			"completion_delta_queries"
		]
		summary["measurement_completion_delta_output_tokens"] = rates[
			"completion_delta_output_tokens"
		]
		summary["measurement_completion_delta_qps"] = rates[
			"completion_delta_qps"
		]
		summary["measurement_completion_delta_output_tokens_per_s"] = rates[
			"completion_delta_output_tokens_per_s"
		]
		summary["warmup_completed"] = warmup_completed
		summary["warmup_wall_cap_s"] = warmup_cap
		summary["measured_wall_cap_s"] = measured_cap
		summary["workload_support_gate_required"] = self.warmup_queries > 0
		summary["workload_support_expected_rows"] = self.n_rows
		summary["workload_support_complete_before_measurement"] = bool(
			self.warmup_queries <= 0
			or workload_support_complete_warmup_completed is not None
		)
		summary["workload_support_complete_warmup_completed"] = (
			int(workload_support_complete_warmup_completed)
			if workload_support_complete_warmup_completed is not None else None
		)
		summary["workload_support_complete_offset_s"] = (
			max(0.0, workload_support_complete_ts - warmup_start)
			if workload_support_complete_ts is not None else None
		)
		summary["warmup_fresh_rate_proof_start_completion"] = (
			int(saturation_stability_proof_start_completion)
			if saturation_stability_proof_start_completion is not None else None
		)
		summary["warmup_fresh_rate_required_completions"] = (
			2 * int(saturation_stability_span)
			if self.warmup_queries > 0 and saturation_stability_span is not None
			else 0
		)
		summary["warmup_fresh_rate_observed_completions"] = (
			int(fresh_rate_completions_at_measurement_start)
			if fresh_rate_completions_at_measurement_start is not None else 0
		)
		summary[
			"warmup_stability_span_dimension_preserved_across_workload_support"
		] = stability_span_dimension_preserved_across_support
		summary["population_ramp_count"] = len(population_ramps)
		summary["population_ramp_added_drivers"] = sum(
			ramp.scheduled_drivers
			for ramp in population_ramps
			if ramp.kind == "adaptive_increment"
		)
		summary["population_ramp_initial_spread_s"] = (
			population_ramps[0].spread_s if population_ramps else 0.0
		)
		summary["population_ramp_max_pending_drivers"] = max(
			(ramp.scheduled_drivers for ramp in population_ramps),
			default=0,
		)
		summary["population_ramp_max_active_drivers"] = max_active_driver_count
		summary["population_active_drivers_at_measurement_start"] = (
			active_drivers_at_measurement_start
		)
		summary["population_ramp_pending_drivers_at_measurement_start"] = (
			population_ramp_pending_at_measurement_start
		)
		summary["population_ramp_complete_before_measurement"] = bool(
			population_ramp_pending_at_measurement_start == 0
			and all(
				ramp.complete_event.is_set()
				and ramp.cancelled_before_activation == 0
				for ramp in population_ramps
			)
		)
		summary["population_ramps"] = [
			ramp.diagnostic(warmup_start) for ramp in population_ramps
		]
		summary["saturation_stability_span_queries"] = (
			int(saturation_stability_span)
			if saturation_stability_span is not None else None
		)
		summary["measurement_rate_stability_waves"] = (
			float(len(perfs)) / float(saturation_stability_span)
			if saturation_stability_span else 0.0
		)
		summary["measurement_initial_target_queries"] = (
			int(measurement_initial_target_queries)
			if measurement_initial_target_queries is not None else None
		)
		summary["measurement_stationarity_extensions"] = (
			measurement_stationarity_extensions
		)
		summary["quality_rows_completed"] = len(first_by_idx)
		summary["measurement_start_reason"] = measurement_start_reason
		summary["measurement_start_gate"] = measurement_start_gate
		summary["measurement_stop_reason"] = measurement_stop_reason
		# Compatibility field: cap-forced stall recovery was removed because
		# it could open an unproved measurement window.
		summary["warmup_stall_recovery"] = False
		# Closed-population integrity: window errors mean the loop ran below
		# its declared population for part of the window.
		summary["driver_errors_total"] = driver_errors_total
		summary["driver_errors_window"] = driver_errors_window
		summary["quality_trace_expected_queries"] = self.n_rows
		summary["quality_trace_recorded_queries"] = len(
			quality_trace_selected_qids
		)
		summary["quality_trace_provisional_queries"] = (
			quality_trace_recorded_before_projection
		)
		summary["quality_trace_discarded_duplicate_queries"] = max(
			0,
			quality_trace_recorded_before_projection
			- len(quality_trace_selected_qids),
		)
		summary["quality_trace_fill_queries"] = sum(
			1 for qid in quality_trace_expected_qids if str(qid).startswith("final-")
		)
		summary["quality_trace_warmup_winner_queries"] = sum(
			1 for qid in quality_trace_expected_qids if str(qid).startswith("warmup-")
		)
		summary["quality_trace_measured_winner_queries"] = sum(
			1 for qid in quality_trace_expected_qids if str(qid).startswith("measured-")
		)
		summary["quality_trace_exact"] = bool(
			quality_trace_exact
			and len(quality_trace_selected_qids) == self.n_rows
		)
		summary["quality_trace_dataset_order"] = bool(quality_trace_exact)
		summary["quality_trace_matches_final_result"] = bool(quality_trace_exact)
		summary["performance_trace_population_queries"] = len(perfs)
		summary["performance_trace_sample_queries"] = len(
			performance_trace_selected_qids
		)
		summary["performance_trace_sample_capacity"] = (
			_PERFORMANCE_TRACE_SAMPLE_CAPACITY
		)
		summary["performance_trace_sampling"] = "sha256_bottom_k"
		summary["performance_trace_completion_window_only"] = True
		if performance_execution_trace is not None:
			# Private transport key: MeasuredEvaluator removes it from the timing
			# summary and sends it down the independent persistence path.  It can
			# therefore never be mistaken for a measured metric or fed to CM as
			# runtime_stats.
			summary["__performance_execution_trace__"] = (
				performance_execution_trace
			)
		# Per-record stationarity proof. Sparse dynamic-batch waves use aligned
		# fixed-completion spans to avoid wall-bin aliasing. Dense completion
		# streams use five equal wall-time bins because individual span durations
		# amplify harmless micro-jitter. Missing span evidence and terminal stalls
		# fail closed through the wall-time proof as well.
		summary["qps_subwindow_stationary"] = False
		if (
			measurement_start is not None
			and measurement_end is not None
			and window_done_ts
		):
			# A wall-budget span is an affordability bound for low-QPS warmup,
			# not a stage-cycle dimension. Reusing that reduced W for the final
			# completion-aligned proof can cut through a larger dynamic-batch
			# wave (for example W=41 for a preferred W=256) and turn stable
			# batch-boundary jitter into a false stationarity failure. Keep the
			# preferred stage-cycle span here; when the finite measured window
			# cannot supply four such boundaries, the existing equal-wall-time
			# proof remains the fail-closed fallback.
			stationarity_completion_span = saturation_stability_span
			if self._saturation.get("stability_span_source") == "wall_budget":
				preferred_span = self._saturation.get(
					"stability_preferred_span_queries"
				)
				if (
					isinstance(preferred_span, int)
					and not isinstance(preferred_span, bool)
					and preferred_span > 0
				):
					stationarity_completion_span = preferred_span
			summary["qps_stationarity_completion_span_queries_requested"] = (
				stationarity_completion_span
			)
			_stationarity = _qps_stationarity_evidence(
				window_done_ts,
				measurement_start,
				measurement_end,
				completion_span_queries=stationarity_completion_span,
			)
			summary["qps_stationarity_method"] = _stationarity["method"]
			summary["qps_stationarity_selection_reason"] = _stationarity[
				"selection_reason"
			]
			summary["qps_wall_subwindow_cv"] = _stationarity[
				"wall_subwindow_cv"
			]
			summary["qps_wall_subwindow_completion_spans"] = _stationarity[
				"wall_subwindow_completion_spans"
			]
			summary["qps_wall_time_dense_completion_stream"] = _stationarity[
				"wall_time_dense_completion_stream"
			]
			summary["qps_completion_span_cv"] = _stationarity[
				"completion_span_cv"
			]
			summary["qps_completion_span_tail_change_cv"] = _stationarity[
				"completion_span_tail_change_cv"
			]
			summary["qps_completion_span_tail_change_suffix_spans"] = (
				_stationarity["completion_span_tail_change_suffix_spans"]
			)
			summary["qps_completion_span_queries"] = _stationarity[
				"completion_span_queries"
			]
			summary["qps_completion_span_rates_qps"] = _stationarity[
				"completion_span_rates_qps"
			]
			summary["qps_completion_span_durations_s"] = _stationarity[
				"completion_span_durations_s"
			]
			summary["qps_completion_span_tail_s"] = _stationarity[
				"completion_span_tail_s"
			]
			summary.update(_wall_subwindow_workload_diagnostics(
				window_done_ts,
				[perf.n_output_tokens for perf in perfs],
				measurement_start,
				measurement_end,
			))
			_selected_cv = _stationarity["selected_cv"]
			if _selected_cv is not None:
				summary["qps_subwindow_cv"] = _selected_cv
				summary["qps_subwindow_stationary"] = bool(
					_selected_cv <= _GT_MAX_QPS_SUBWINDOW_CV
				)
		admissibility = build_measured_gt_admissibility(summary)
		summary["measured_gt_admissibility"] = admissibility
		# Future artifacts keep the legacy convenience field, but it now means
		# the same strict five-condition performance proof rather than CV alone.
		summary["steady_state_ok"] = bool(admissibility["admissible"])
		return final_result, summary

	async def _run_sequential_request(self, state: RequestState) -> None:
		current = state.df
		for service in self.services:
			result = await service.run(state, current)
			current = result.df
			state.add_stage(service.name, result)
		state.df = current

	async def _run_react_request(self, state: RequestState) -> None:
		from rag_stack.static_rag_evaluator.agentic_react import (
			_FINISH_RE,
			_REACT_PROMPT,
			_SEARCH_RE,
			_STOP_TOKENS,
			_docs_to_observation,
		)

		gen_service = self._service_for("generator")
		retr_service = self._service_for("semantic_retrieval")
		rerank_service = self._service_for("passage_reranker")
		if not isinstance(gen_service, LLMContinuousStageService):
			raise RuntimeError("react measured mode requires a generator vLLM service")
		if retr_service is None:
			raise RuntimeError("react measured mode requires semantic_retrieval")
		gen_stage = gen_service.stage
		gen_params = self.owner._sampling_params(dict(gen_stage["params"]))
		gen_params["stop"] = _STOP_TOKENS
		# Same semantics as the quality loop (run_react): max_iter is THE
		# react round cap (react-only sub-config, REQUIRED) — it doubles as
		# the dead-loop guard; there is no separate safety constant.
		_mi = (self.config.get("pipeline_runtime") or {}).get("max_iter")
		if _mi is None:
			raise RuntimeError(
				"react measured mode requires pipeline.max_iter "
				"(max_agent_steps_safety was removed)"
			)
		cap = max(1, int(_mi))
		query = str(state.df["query"].iloc[0])
		prompt = _REACT_PROMPT.format(question=query)
		# Per-round retrieval frames are stamped from ONE template built
		# here instead of copy(deep)+reset_index+4 column assigns per round
		# — that per-round pandas ran on the event loop and stalled
		# completion reads at high dispatch rates (r12 driver-feed
		# investigation). The template carries state.df's full column set so
		# retrieval variants keep whatever context they read today.
		retr_template = state.df.copy(deep=True).reset_index(drop=True)
		retr_template["__qid__"] = [state.qid]
		if "qid" not in retr_template.columns:
			retr_template["qid"] = [state.qid]
		pred = ""
		last_semantic_ids: List[str] = []
		last_semantic_contents: List[str] = []
		last_semantic_scores: List[float] = []
		last_final_ids: List[str] = []
		last_final_contents: List[str] = []
		last_final_scores: List[float] = []
		round_idx = 1
		generate_calls = 0
		retrieval_calls = 0
		truncated = True
		for _round in range(cap):
			text, gen_perf, qwait, service_s, active_batch = await gen_service.generate_text(
				state, prompt, gen_params
			)
			generate_calls += 1
			gen_result = StageResult(
				df=state.df,
				gen_perf=gen_perf,
				elapsed_s=qwait + service_s,
				queue_wait_s=qwait,
				service_s=service_s,
				batch_size=active_batch,
			)
			state.add_stage("generator", gen_result)
			if state.record_trace:
				_record_react_generate(
					state.qid,
					prompt,
					text,
					gen_perf,
					gen_stage["params"],
				)
			out = str(text).strip()
			prompt = prompt + " " + out
			finish_m = _FINISH_RE.findall(out)
			search_m = _SEARCH_RE.findall(out)
			if finish_m:
				pred = finish_m[-1].strip()
				truncated = False
				break
			if not search_m:
				pred = out
				truncated = False
				break
			if _round == cap - 1:
				# Cap reached — a search's observation could never be consumed
				# by a next generate. Exit truncated, exactly like run_react.
				break
			search_query = search_m[-1].strip()
			retr_df = retr_template.copy()
			retr_df["query"] = [search_query]
			retr_df["queries"] = [[search_query]]
			retr_res = await retr_service.run(state, retr_df)
			retrieval_calls += 1
			state.add_stage(retr_service.name, retr_res)
			rdf = retr_res.df
			last_semantic_contents = _first_list(rdf, "retrieved_contents_semantic")
			last_semantic_ids = _first_list(rdf, "retrieved_ids_semantic")
			last_semantic_scores = _first_list(rdf, "retrieve_scores_semantic")
			if rerank_service is not None:
				rerank_res = await rerank_service.run(state, rdf)
				state.add_stage(rerank_service.name, rerank_res)
				rdf = rerank_res.df
			last_final_contents = (
				_first_list(rdf, "retrieved_contents")
				or _first_list(rdf, "retrieved_contents_semantic")
			)
			last_final_ids = (
				_first_list(rdf, "retrieved_ids")
				or _first_list(rdf, "retrieved_ids_semantic")
			)
			last_final_scores = (
				_first_list(rdf, "retrieve_scores")
				or _first_list(rdf, "retrieve_scores_semantic")
			)
			prompt += (
				f"\nObservation {round_idx}: {_docs_to_observation(last_final_contents)}"
				f"\nThought {round_idx + 1}:"
			)
			round_idx += 1
		if not pred:
			pred = "No valid answer found"
		state.agent_generate_calls = generate_calls
		state.agent_retrieval_calls = retrieval_calls
		state.agent_truncated = truncated
		# Single-constructor assembly: copy()+reset_index+8-11 column
		# assigns per request ran ~1ms of pandas on the event loop; one
		# row-dict -> DataFrame build keeps identical columns/values, and
		# the build itself runs in a worker thread (per-request pandas is
		# pure harness work — the loop only awaits it).
		def _assemble_final_row() -> pd.DataFrame:
			row = state.df.iloc[0].to_dict()
			row.update({
				"generated_texts": pred,
				"agent_generate_calls": generate_calls,
				"agent_retrieval_calls": retrieval_calls,
				"agent_truncated": truncated,
				"retrieved_contents_semantic": last_semantic_contents,
				"retrieved_ids_semantic": last_semantic_ids,
				"retrieve_scores_semantic": last_semantic_scores,
			})
			if rerank_service is not None:
				row.update({
					"retrieved_contents": last_final_contents,
					"retrieved_ids": last_final_ids,
					"retrieve_scores": last_final_scores,
				})
			return pd.DataFrame([row])

		state.df = await asyncio.to_thread(_assemble_final_row)

	def _service_for(self, stage: str) -> Optional[BaseStageService]:
		for svc in self.services:
			if str(svc.stage["stage"]) == stage:
				return svc
		return None

	def _annotate_summary(
		self,
		summary: Dict[str, Any],
		n_measured: int,
		total_wall: float,
	) -> None:
		summary["qps_batch"] = summary["qps"]
		summary["rag_dataflow"] = (
			"react_service" if self.mode == "react" else "sequential_service"
		)
		gen = engine_info(self.system_config, "generator")
		summary["serving_mode"] = str(gen.get("pd_serving", "collocated_pd"))
		summary["scheduling_policy"] = "closed-loop-saturated-stage-services"
		summary["workload_admission_order"] = "balanced_epoch_permutation_v1"
		summary["batch_size_request"] = self.batch_size_request
		summary["batch_size_decode"] = decode_batch(self.system_config)
		summary["stage_batch_cap"] = self.batch_size_request
		summary["load_concurrency"] = self.load_concurrency
		summary["load_concurrency_initial"] = self.load_concurrency_initial
		summary["load_concurrency_hard_cap"] = self.load_concurrency_hard_cap
		# Admissibility proof: was the system demonstrably saturated when the
		# window opened, and by whose backlog? False/None records must not be
		# used as CM truth (the deployment's capacity was not reached).
		summary["saturation_evidence"] = self._saturation.get("evidence")
		summary["population_saturated"] = self._saturation.get("saturated")
		summary["saturation_candidate_identity"] = self._saturation.get(
			"candidate_identity"
		)
		summary["saturation_stability_span_source"] = self._saturation.get(
			"stability_span_source"
		)
		summary["saturation_stability_preferred_span_queries"] = (
			self._saturation.get("stability_preferred_span_queries")
		)
		summary["saturation_stability_effective_span_queries"] = (
			self._saturation.get("stability_effective_span_queries")
		)
		summary["saturation_candidate_age_s_at_span_freeze"] = (
			self._saturation.get("candidate_age_s_at_span_freeze")
		)
		summary["saturation_candidate_completions_at_span_freeze"] = (
			self._saturation.get("candidate_completions_at_span_freeze")
		)
		summary["saturation_candidate_rate_qps_at_span_freeze"] = (
			self._saturation.get("candidate_rate_qps_at_span_freeze")
		)
		summary["saturation_wall_budget_available_span_queries"] = (
			self._saturation.get("wall_budget_available_span_queries")
		)
		summary["saturation_wall_budget_measurement_span_queries"] = (
			self._saturation.get("wall_budget_measurement_span_queries")
		)
		summary["saturation_wall_budget_minimum_span_queries"] = (
			self._saturation.get("wall_budget_minimum_span_queries")
		)
		# The initial/effective concurrency values are saturation-controller
		# audit metadata only. They must never enter CM workload shape, curve
		# selection, or deployment repricing; capacity is read from the
		# demonstrably saturated measurement window.
		summary["load_concurrency_initial"] = getattr(
			self, "load_concurrency_initial", self.load_concurrency
		)
		# Backward-compatible metric name for downstream readers; it now means
		# adaptive saturation-control concurrency, not a workload population or
		# per-stage service batch cap.
		summary["service_concurrency"] = self.load_concurrency
		summary["warmup_queries"] = self.warmup_queries
		summary["measured_target_queries"] = self.measured_target
		summary["measured_target_explicit"] = self.measured_target_explicit
		summary["measured_queries"] = n_measured
		summary["measurement_wall_s"] = total_wall
		summary["measurement_waves"] = (
			float(n_measured) / float(self.load_concurrency)
			if self.load_concurrency else 0.0
		)
		summary["measurement_request_batch_waves"] = (
			float(n_measured) / float(self.batch_size_request)
			if self.batch_size_request else 0.0
		)
		# The canonical quality-trace envelope is self-describing. These historical
		# keys remain quality-only capture diagnostics; performance sampling has
		# its own explicit diagnostics and independent envelope.
		# All admissions are captured until global QA coverage so every dataset
		# row's first completion has a recorded trace to select. Projection
		# keeps one winner per row in dataset order and is independent of
		# measured_queries.
		summary["trace_capture_scope"] = (
			"all_admissions_until_global_qa_coverage"
		)
		summary["trace_projection_basis"] = (
			"first_completed_invocation_per_dataset_idx"
		)
		summary["performance_trace_projection_basis"] = (
			"deterministic_bottom_k_of_measurement_phase_completions"
		)
		for service in self.services:
			stats = service.stats()
			if not stats:
				continue
			prefix = str(service.name)
			if isinstance(service, LLMContinuousStageService):
				summary[f"{prefix}_max_inflight"] = stats.get("max_inflight")
				summary[f"{prefix}_max_inflight_observed"] = stats.get(
					"max_inflight_observed"
				)
				summary[f"{prefix}_avg_queue_wait_s"] = stats.get("avg_queue_wait_s")
				# Per-CALL service time (driver semaphore wait excluded;
				# vLLM-internal queueing included) — the generator-side
				# calibration observable that was never exported (the 0039
				# autopsy had to reverse it from output tok/s).
				summary[f"{prefix}_avg_service_s"] = stats.get("avg_service_s")
				summary[f"{prefix}_completed_calls"] = stats.get("completed")
				for role, role_stats in (
					stats.get("pd_role_admission") or {}
				).items():
					for metric, value in role_stats.items():
						summary[f"{prefix}_pd_{role}_{metric}"] = value
				for pk, pv in service.engine_probe_stats().items():
					summary[f"{prefix}_{pk}"] = pv
				_series = service.engine_probe_series()
				if _series:
					summary[f"{prefix}_engine_series"] = _series
			else:
				summary[f"{prefix}_dynamic_batch"] = stats.get("batch_size")
				summary[f"{prefix}_dynamic_batch_timeout_s"] = stats.get("timeout_s")
				summary[f"{prefix}_dynamic_batch_avg_size"] = stats.get("avg_batch_size")
				summary[f"{prefix}_dynamic_batch_max_observed"] = stats.get(
					"max_batch_observed"
				)
				summary[f"{prefix}_dynamic_batch_avg_queue_wait_s"] = stats.get(
					"avg_queue_wait_s"
				)
				# Full per-batch stage-adapter service (queue + trace excluded).
				# The pure backend and control-plane segments are exported next
				# so a hardware×stage curve never absorbs executor/harness delay.
				summary[f"{prefix}_dynamic_batch_avg_service_s"] = stats.get(
					"avg_service_s"
				)
				for metric in (
					"scheduler_wait_s",
					"executor_wait_s",
					"prepare_s",
					"pure_service_s",
					"postprocess_s",
					"trace_s",
					"event_loop_resume_s",
				):
					summary[f"{prefix}_dynamic_batch_avg_{metric}"] = stats.get(
						f"avg_{metric}"
					)
		# Host-CPU forensics (r29): window-mean cores per process family +
		# the 1 Hz per-second series (same dashboard pattern as the engine
		# probe) — the measured side of the CM's CPU resource axis.
		if self._cpu_probe is not None:
			for ck, cv in self._cpu_probe.stats().items():
				summary[ck] = cv
			_cpu_series = self._cpu_probe.series()
			if _cpu_series:
				summary["cpu_probe_series"] = _cpu_series
		self.owner._add_deployment_metadata(summary, self.node_lines, self.system_config)


def _admission_slack(max_num_seqs: int) -> int:
	"""Small frontend queue behind one vLLM engine-resident batch."""
	cap = max(1, int(max_num_seqs))
	return max(1, (cap + 4) // 5)


def _collocated_outer_backlog_engine_cap(
	service: LLMContinuousStageService,
) -> Optional[int]:
	"""Return the deployed engine cap for one valid collocated outer backlog.

	The outer semaphore is a saturation proof only when it is exactly the
	unclamped collocated stage gate: deployed ``max_num_seqs`` plus the shared
	frontend admission slack.  A smaller population-clamped semaphore is an
	artificial benchmark bottleneck.  Role-aware P/D stages are deliberately
	excluded so they remain governed solely by their per-role admission proof.
	"""
	subprocess = service._subprocess()
	role_stats = getattr(subprocess, "role_admission_stats", None)
	if callable(role_stats):
		try:
			roles = role_stats()
		except Exception:  # noqa: BLE001 — missing telemetry must fail closed
			return None
		role_rows = [
			stats for stats in (roles or {}).values()
			if isinstance(stats, dict)
		]
		if role_rows:
			return None
	# A P/D pair can have no loop-local role rows during startup.  Its explicit
	# stage admission contract still distinguishes it from a collocated engine.
	if callable(getattr(subprocess, "stage_admission_limit", None)):
		return None
	key = getattr(subprocess, "key", None)
	deployed_max_num_seqs = getattr(key, "max_num_seqs", None)
	if deployed_max_num_seqs is None:
		# Saturation evidence must bind to the actual launched engine, not merely
		# a possibly stale configuration fallback.
		return None
	engine_cap = max(1, int(deployed_max_num_seqs))
	unclamped_stage_gate = engine_cap + _admission_slack(engine_cap)
	if service.max_inflight != unclamped_stage_gate:
		return None
	if service.inflight != service.max_inflight or service.waiting <= 0:
		return None
	return engine_cap


def _next_population_increment(
	population: int,
	hard_cap: int,
	mean_fill: float,
) -> int:
	"""Grow below the finite cap even when proportional rounding is zero."""
	remaining = max(0, int(hard_cap) - int(population))
	if remaining <= 0:
		return 0
	deficit = max(0.0, 1.0 / max(float(mean_fill), 0.05) - 1.0)
	proposed = int(int(population) * min(1.0, deficit))
	return min(max(1, proposed), remaining)


def _stage_max_inflight(
	stage_service: Dict[str, Any],
	system_config: Dict[str, Any],
	stage_name: str,
	population_hard_cap: Optional[int] = None,
) -> int:
	"""Bound HTTP residency independently from the closed-loop population.

	The deployed subprocess key is authoritative because it contains the
	launch-time engine capacity. Multiple API frontends feed the same vLLM
	EngineCore, so they increase host-side request handling capacity but do not
	multiply engine admission. During partially initialized/unit-test paths, fall
	back to the resolved per-engine system config. Excess driver population waits
	on the stage semaphore before consuming frontend sockets or vLLM queue entries.
	"""
	instance = stage_service.get("instance")
	if stage_name == "query_expansion":
		subprocess = getattr(getattr(instance, "generator", None), "_subprocess", None)
	else:
		subprocess = getattr(instance, "_subprocess", None)
	key = getattr(subprocess, "key", None)
	# A disaggregated pair has two independent vLLM roles. Its outer stage
	# residency is the sum of the pair's role-local client gates: engine P plus
	# one bounded feeder, and strict engine D. Reusing unified-engine
	# ``max_num_seqs + 20%`` here either underfeeds P or over-admits decode-side
	# NIXL handles.
	role_aware_stage_limit = getattr(subprocess, "stage_admission_limit", None)
	if callable(role_aware_stage_limit):
		stage_cap = max(1, int(role_aware_stage_limit()))
		if population_hard_cap is None:
			return stage_cap
		return min(max(1, int(population_hard_cap)), stage_cap)

	key_max_num_seqs = getattr(key, "max_num_seqs", None)
	max_num_seqs = (
		max(1, int(key_max_num_seqs))
		if key_max_num_seqs is not None
		else vllm_max_num_seqs(system_config, stage_name)
	)
	stage_cap = max_num_seqs + _admission_slack(max_num_seqs)
	if population_hard_cap is None:
		return stage_cap
	return min(
		max(1, int(population_hard_cap)),
		stage_cap,
	)


def _measured_population_hard_cap(
	system_config: Dict[str, Any],
	initial_population: int,
	mode: str,
) -> int:
	"""Maximum finite closed-loop driver population.

	``max_num_seqs`` is deliberately absent: each LLM service applies its own
	HTTP admission gate, while this deeper population can wait before that gate
	to keep the whole RAG pipeline under load.
	"""
	initial = max(1, int(initial_population))
	batching = system_config.get("batching")
	batching = batching if isinstance(batching, dict) else {}
	explicit = system_config.get("measured_max_load_concurrency")
	if explicit is None:
		explicit = batching.get("max_load_concurrency")
	if explicit is not None:
		cap = max(1, int(explicit))
	else:
		# ReAct parks each request between LLM rounds and can require a much
		# deeper population than a single-pass sequential pipeline.
		cap = (16 if str(mode).lower() == "react" else 4) * initial
	# A user can explicitly choose an initial population above the adapter cap.
	# Never set the driver cap below the population already admitted.
	return max(initial, cap)


def _dynamic_batch_timeout_s(system_config: Dict[str, Any]) -> float:
	return dynamic_batch_timeout_s(system_config)


def _measured_load_concurrency(
	system_config: Dict[str, Any],
	batch_size_request: int,
) -> int:
	"""Closed-loop client population used to saturate the measured service.

	``batch_size_request`` is the per-stage service batch cap. The load
	population is deliberately larger so non-bottleneck stage queues can observe
	real arrival pressure without redefining the system design knob.
	"""
	batching = system_config.get("batching")
	batching = batching if isinstance(batching, dict) else {}
	explicit = system_config.get("measured_load_concurrency")
	if explicit is None:
		explicit = batching.get("load_concurrency")
	if explicit is not None:
		return max(1, int(explicit))

	# Population derives from the LARGEST deployed batch (user decision
	# 07-08): decode concurrency (max_num_seqs = batching.decode) above the
	# request batch can only saturate when the client population covers it.
	# Mirrors engines._closed_loop_load_concurrency; decode ≤ request keeps
	# the exact previous request-based population.
	try:
		batch_size_request = max(
			1, int(batch_size_request), int(batching.get("decode") or 0),
		)
	except (TypeError, ValueError):
		batch_size_request = max(1, int(batch_size_request))

	factor = system_config.get("measured_load_factor")
	if factor is None:
		factor = batching.get("load_factor", 4)
	load = int(max(1, batch_size_request) * max(1.0, float(factor)))
	max_load = system_config.get("measured_max_load_concurrency")
	if max_load is None:
		max_load = batching.get("max_load_concurrency")
	if max_load is not None:
		load = min(load, max(1, int(max_load)))
	return max(max(1, batch_size_request), load)


def _qids(df: pd.DataFrame) -> List[Any]:
	if "__qid__" not in df.columns:
		return list(range(len(df)))
	return df["__qid__"].tolist()


def _model_id(params: Dict[str, Any]) -> Any:
	return params.get("model") or params.get("model_type") or params.get("llm")


def _first_list(df: pd.DataFrame, column: str) -> List[Any]:
	if column not in df.columns or len(df) == 0:
		return []
	value = df[column].iloc[0]
	if isinstance(value, list):
		return list(value)
	if isinstance(value, tuple):
		return list(value)
	if value is None:
		return []
	return [value]


def _record_generate(previous: pd.DataFrame, result: pd.DataFrame, params: Dict[str, Any]) -> None:
	from rag_stack.static_rag_evaluator import recording as rec

	qids = _qids(previous)
	prompts = previous["prompts"].tolist() if "prompts" in previous.columns else [""] * len(qids)
	out_texts = result["generated_texts"].tolist() if "generated_texts" in result.columns else None
	out_tokens = (
		result["generated_tokens"].tolist()
		if "generated_tokens" in result.columns else None
	)
	rec.record_io(
		"generator",
		qids,
		prompts,
		out_texts=out_texts,
		out_token_ids=out_tokens,
		model_id=_model_id(params),
	)


def _record_react_generate(
	qid: Any,
	prompt: str,
	text: str,
	gen_perf: Dict[str, Any],
	params: Dict[str, Any],
) -> None:
	from rag_stack.static_rag_evaluator import recording as rec

	n = max(int(gen_perf.get("n_output_tokens", 1) or 1), 1)
	rec.record_io(
		"generator",
		[qid],
		[prompt],
		out_texts=[text],
		out_token_ids=[[0] * n],
		model_id=_model_id(params),
	)


def _record_query_expansion(
	previous: pd.DataFrame,
	prompt: str,
	text: str,
	gen_perf: Dict[str, Any],
	params: Dict[str, Any],
) -> None:
	from rag_stack.static_rag_evaluator import recording as rec

	n = max(int(gen_perf.get("n_output_tokens", 1) or 1), 1)
	rec.record_io(
		"query_expansion",
		_qids(previous),
		[prompt],
		out_texts=[text],
		out_token_ids=[[0] * n],
		model_id=_model_id(params),
	)


def _record_pure_stage(
	node: Any,
	params: Dict[str, Any],
	best_result: pd.DataFrame,
) -> None:
	from rag_stack.static_rag_evaluator import recording as rec

	stage = str(getattr(node, "stage", node))
	if "__qid__" not in best_result.columns:
		return
	qids = _qids(best_result)
	queries = (
		best_result["query"].astype(str).tolist()
		if "query" in best_result.columns else [""] * len(qids)
	)
	if stage in {"semantic_retrieval", "lexical_retrieval", "hybrid_retrieval"}:
		rc_col = next(
			(c for c in ("retrieved_contents_semantic", "retrieved_contents") if c in best_result.columns),
			None,
		)
		contents = best_result[rc_col].tolist() if rc_col else [[]] * len(qids)
		rec.record_io("semantic_retrieval_encode", qids, queries, model_id=params.get("embedding_model"))
		rec.record_io(
			"semantic_retrieval_vectorsearch",
			qids,
			queries,
			out_texts=contents,
			model_id=params.get("vectordb") or params.get("model"),
		)
	elif stage == "passage_reranker":
		rc_col = next(
			(c for c in ("retrieved_contents_semantic", "retrieved_contents") if c in best_result.columns),
			None,
		)
		contents = best_result[rc_col].tolist() if rc_col else [[]] * len(qids)
		in_texts = [
			[q] + (rc if isinstance(rc, (list, tuple)) else [rc])
			for q, rc in zip(queries, contents)
		]
		# The trace carries the mean-pair shape only (query + passages).
		# Padding excess is CM-internal: it depends on the batch the CM is
		# pricing, derived from chunk_token_quantiles + the shared
		# cost_model.reranker_policy table — never from the measured run.
		rec.record_io(
			"passage_reranker", qids, in_texts, model_id=_model_id(params),
		)
	elif stage == "passage_compressor":
		rc_col = next(
			(c for c in ("retrieved_contents", "retrieved_contents_semantic") if c in best_result.columns),
			None,
		)
		contents = best_result[rc_col].tolist() if rc_col else [[]] * len(qids)
		rec.record_io("passage_compressor", qids, contents, model_id=_model_id(params))


def _query_expansion_prompt_and_parser(
	instance: Any,
	params: Dict[str, Any],
	previous: pd.DataFrame,
) -> tuple[str, Callable[[str], List[str]]]:
	queries = previous["query"].astype(str).tolist()
	if len(queries) != 1:
		raise ValueError("measured query_expansion service expects one logical request")
	query = queries[0]
	module_name = instance.__class__.__name__
	if module_name == "HyDE":
		from rag_stack.static_rag_evaluator.nodes.queryexpansion.hyde import hyde_prompt

		prompt_cfg = params.get("prompt", hyde_prompt)
		prefix = prompt_cfg if not bool(prompt_cfg) else hyde_prompt
		prompt = prefix + f"\nQuestion: {query}\nPassage:"
		return prompt, lambda text: instance._check_expanded_query(
			[query], [[text]]
		)[0]
	if module_name == "MultiQueryExpansion":
		from rag_stack.static_rag_evaluator.nodes.queryexpansion.multi_query_expansion import (
			get_multi_query_expansion,
			multi_query_expansion_prompt,
		)

		prompt = params.get("prompt", multi_query_expansion_prompt).format(query=query)
		return prompt, lambda text: instance._check_expanded_query(
			[query], [get_multi_query_expansion(query, text)]
		)[0]
	if module_name == "QueryDecompose":
		from rag_stack.static_rag_evaluator.nodes.queryexpansion.query_decompose import (
			decompose_prompt,
			get_query_decompose,
		)

		prompt_cfg = params.get("prompt", decompose_prompt)
		if bool(prompt_cfg):
			prompt = f"prompt: {prompt_cfg}\n\n question: {query}"
		else:
			prompt = decompose_prompt.format(question=query)
		return prompt, lambda text: instance._check_expanded_query(
			[query], [get_query_decompose(query, text)]
		)[0]
	raise RuntimeError(
		f"measured query_expansion module {module_name!r} has no continuous vLLM adapter"
	)
