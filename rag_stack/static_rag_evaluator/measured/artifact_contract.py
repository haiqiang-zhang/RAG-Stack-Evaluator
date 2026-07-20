"""Structural validation for persisted measured-result control artifacts."""

from __future__ import annotations

import json
from typing import Any


_ADMISSIBILITY_FIELDS = frozenset({"admissible", "checks", "reasons"})
_ADMISSIBILITY_CHECKS = frozenset({
    "driver_errors_window_zero",
    "measurement_rate_stability_waves_ge_2",
    "population_saturated",
    "qps_subwindow_cv_le_0_15",
    "warmup_rate_stable_not_cap_forced",
})
_CHECK_FIELDS = frozenset({"observed", "passed", "required"})


class MeasuredGTInadmissibleError(RuntimeError):
    """A measured window exists but must not be published as optimizer GT.

    This is intentionally a retryable runtime failure, not ``TrialInvalid``:
    an unstable measurement window says nothing about whether the sampled
    deployment is feasible, and a fresh run can produce an admissible window.
    """


def validate_measured_gt_admissibility(value: Any) -> dict:
    """Return the verdict only when it has the exact current shape."""

    if not isinstance(value, dict) or set(value) != _ADMISSIBILITY_FIELDS:
        raise ValueError("measured GT admissibility has invalid fields")
    if type(value["admissible"]) is not bool:
        raise ValueError("measured GT admissibility must be boolean")
    reasons = value["reasons"]
    if not isinstance(reasons, list) or any(
        not isinstance(reason, str) or not reason for reason in reasons
    ):
        raise ValueError("measured GT admissibility reasons must be strings")
    if value["admissible"] != (not reasons):
        raise ValueError("measured GT admissibility conflicts with reasons")
    checks = value["checks"]
    if not isinstance(checks, dict) or set(checks) != _ADMISSIBILITY_CHECKS:
        raise ValueError("measured GT admissibility has invalid checks")
    for name, check in checks.items():
        if not isinstance(check, dict) or set(check) != _CHECK_FIELDS:
            raise ValueError(f"measured GT admissibility check {name} is malformed")
        if type(check["passed"]) is not bool:
            raise ValueError(f"measured GT admissibility check {name} has invalid result")
    return value


def require_measured_gt_admissible(performance: Any) -> dict:
    """Return the exact verdict or raise before a measured result is published.

    ``performance`` is the raw measured-performance mapping, not the outer
    ``performance.json`` envelope. Missing/malformed verdicts fail closed just
    like an explicit ``admissible=false`` verdict.
    """

    if not isinstance(performance, dict):
        raise MeasuredGTInadmissibleError(
            "measured performance payload is not a mapping"
        )
    try:
        verdict = validate_measured_gt_admissibility(
            performance.get("measured_gt_admissibility")
        )
    except ValueError as exc:
        raise MeasuredGTInadmissibleError(
            f"invalid measured_gt_admissibility verdict: {exc}"
        ) from exc
    if verdict["admissible"] is not True:
        reasons = ", ".join(verdict["reasons"]) or "unspecified"
        failed_checks = {
            name: {
                "observed": check.get("observed"),
                "required": check.get("required"),
            }
            for name, check in verdict["checks"].items()
            if check.get("passed") is not True
        }
        stationarity_keys = (
            "qps_subwindow_cv",
            "qps_stationarity_method",
            "qps_stationarity_selection_reason",
            "qps_wall_subwindow_cv",
            "qps_wall_subwindow_completion_spans",
            "qps_completion_span_cv",
            "qps_completion_span_tail_change_cv",
            "qps_completion_span_tail_change_suffix_spans",
            "qps_completion_span_queries",
            "qps_completion_span_rates_qps",
            "qps_completion_span_durations_s",
            "qps_completion_span_tail_s",
            "window_subwindow_completions",
            "window_subwindow_output_tokens",
            "window_subwindow_mean_output_tokens",
            "window_subwindow_output_tokens_per_s",
            "measurement_phase_window_queries",
            "measurement_phase_window_s",
        )
        evidence = {
            "failed_checks": failed_checks,
            "stationarity": {
                key: performance[key]
                for key in stationarity_keys
                if key in performance
            },
        }
        raise MeasuredGTInadmissibleError(
            "measured GT inadmissible: "
            f"{reasons}; evidence="
            f"{json.dumps(evidence, sort_keys=True, default=str)}"
        )
    return verdict


__all__ = [
    "MeasuredGTInadmissibleError",
    "require_measured_gt_admissible",
    "validate_measured_gt_admissibility",
]
