"""Shared runtime identity for independently scoped vLLM telemetry."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass


CALIBRATION_RUN_ID_ENV = "RAG_STACK_VLLM_CALIBRATION_RUN_ID"
LEGACY_COMPONENT_RUN_ID_ENV = "RAG_STACK_LLM_COMPONENT_TELEMETRY_RUN_ID"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def validate_calibration_run_id(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("vLLM calibration run id cannot be empty")
    result = value.strip()
    if len(result) > 128 or any(char in result for char in ("\n", "\r", ":")):
        raise ValueError("vLLM calibration run id is invalid")
    return result


def resolve_calibration_run_id(
    environ: Mapping[str, str] | None = None,
    *,
    explicit: str | None = None,
) -> str:
    if explicit is not None:
        return validate_calibration_run_id(explicit)
    values = os.environ if environ is None else environ
    value = values.get(CALIBRATION_RUN_ID_ENV)
    if value is None:
        value = values.get(LEGACY_COMPONENT_RUN_ID_ENV)
    return validate_calibration_run_id(value)


def scheduler_shape_sha256(shape: object) -> str:
    if is_dataclass(shape) and not isinstance(shape, type):
        payload: object = asdict(shape)
    elif isinstance(shape, Mapping):
        payload = dict(shape)
    else:
        raise TypeError("scheduler shape must be a dataclass or mapping")
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def scheduler_cycle_id(
    run_id: str, work_sequence: int, shape_digest: str
) -> str:
    run_id = validate_calibration_run_id(run_id)
    if isinstance(work_sequence, bool) or not isinstance(work_sequence, int):
        raise ValueError("work_sequence must be a non-negative integer")
    if work_sequence < 0:
        raise ValueError("work_sequence must be a non-negative integer")
    if (
        not isinstance(shape_digest, str)
        or len(shape_digest) != 64
        or any(char not in "0123456789abcdef" for char in shape_digest)
    ):
        raise ValueError("shape_digest must be a lowercase SHA-256")
    return f"{run_id}:{work_sequence}:{shape_digest[:16]}"


def vllm_model_identity(vllm_config: object) -> dict[str, object]:
    """Return the model graph identity shared by both calibration sensors.

    The component sensor runs in a GPU worker while the stage sensor runs in
    the API frontend.  Deriving this value from the common ``model_config``
    contract is what lets an offline publisher prove that two *independent*
    runs used the same model without joining their timestamps or request IDs.
    No vLLM import is needed here, which keeps publication and CPU tests safe.
    """

    model_config = getattr(vllm_config, "model_config", vllm_config)
    compute_hash = getattr(model_config, "compute_hash", None)
    if not callable(compute_hash):
        raise ValueError("vLLM model config has no compute_hash")
    graph_hash = compute_hash()
    if not isinstance(graph_hash, str) or not graph_hash:
        raise ValueError("vLLM model compute hash is invalid")
    model = getattr(model_config, "model", None)
    if not isinstance(model, str) or not model:
        raise ValueError("vLLM model identity is missing")
    return {
        "model": model,
        "revision": getattr(model_config, "revision", None),
        "dtype": str(getattr(model_config, "dtype", None)),
        "quantization": getattr(model_config, "quantization", None),
        "model_compute_hash": graph_hash,
    }


def identity_sha256(value: Mapping[str, object]) -> str:
    """Hash one canonical runtime identity mapping."""

    if not isinstance(value, Mapping) or not value:
        raise ValueError("runtime identity must be a non-empty mapping")
    return hashlib.sha256(_canonical_json(dict(value))).hexdigest()


__all__ = [
    "CALIBRATION_RUN_ID_ENV",
    "LEGACY_COMPONENT_RUN_ID_ENV",
    "resolve_calibration_run_id",
    "scheduler_cycle_id",
    "scheduler_shape_sha256",
    "validate_calibration_run_id",
    "identity_sha256",
    "vllm_model_identity",
]
