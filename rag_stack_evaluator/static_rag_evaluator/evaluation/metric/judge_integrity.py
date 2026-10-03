"""Validation, token budgets, and durable evidence for structured judges.

This module does not load a model or contact a service. A configured tokenizer
is loaded lazily from local files solely to count chat-template tokens.
"""

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import uuid

from pydantic import Field, create_model


GUARD_VERSION = "context-precision-cardinality-1"


def json_value(value):
    if hasattr(value, "model_dump"):
        return json_value(value.model_dump())
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if hasattr(value, "item"):
        return json_value(value.item())
    return value


def json_hash(value):
    payload = json.dumps(json_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x") as handle:
            json.dump(json_value(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class JudgeGuardConfig:
    context_window: int | None = None
    tokenizer_path: str | None = None
    max_output_tokens: int = 16384
    max_attempts: int = 4

    @classmethod
    def pop_from(cls, kwargs):
        values = {
            "context_window": kwargs.pop("judge_context_window", None),
            "tokenizer_path": kwargs.pop("judge_tokenizer_path", None),
            "max_output_tokens": kwargs.pop("judge_max_output_tokens", 16384),
            "max_attempts": kwargs.pop("judge_max_attempts", 4),
        }
        for key in ("max_output_tokens", "max_attempts"):
            positive_integer(f"judge_{key}", values[key])
        if values["context_window"] is not None:
            positive_integer("judge_context_window", values["context_window"])
            if not values["tokenizer_path"]:
                raise ValueError("judge_context_window requires judge_tokenizer_path for exact local token counting")
        if values["tokenizer_path"] and values["context_window"] is None:
            raise ValueError("judge_tokenizer_path requires judge_context_window")
        return cls(**values)


@lru_cache(maxsize=8)
def _local_tokenizer(path):
    from transformers import AutoTokenizer

    path = os.path.expandvars(os.path.expanduser(path))
    if not Path(path).is_dir():
        raise ValueError(f"judge_tokenizer_path must name an existing local directory: {path}")
    return AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)


def count_chat_tokens(path, messages, chat_template_kwargs=None):
    tokenizer = _local_tokenizer(str(path))
    tokens = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        **dict(chat_template_kwargs or {}),
    )
    return len(tokens)


def output_budgets(initial, cap, attempts):
    """Bounded growth; repeated capped attempts can recover malformed output."""
    initial = positive_integer("initial judge output budget", initial)
    cap = positive_integer("available judge output budget", cap)
    attempts = positive_integer("judge attempts", attempts)
    current = min(initial, cap)
    budgets = []
    for index in range(attempts):
        budgets.append(current)
        current = min(cap, max(4096, current * (4 if index == 0 else 2)))
    return budgets


@lru_cache(maxsize=256)
def exact_verdict_schema(verdict_type, count):
    positive_integer("expected verdict count", count)
    return create_model(
        f"ExactContextualPrecisionVerdicts{count}",
        verdicts=(list[verdict_type], Field(..., min_length=count, max_length=count)),
    )


def validate_verdict_count(verdicts, expected):
    if not isinstance(verdicts, (list, tuple)) or len(verdicts) != expected:
        count = len(verdicts) if isinstance(verdicts, (list, tuple)) else None
        raise ValueError(f"Contextual precision requires exactly {expected} verdicts; received {count}")


def exception_evidence(error):
    """Capture output/usage when exposed by the SDK, never request headers."""
    evidence = {"type": type(error).__name__, "message": str(error)}
    completion = getattr(error, "completion", None)
    if completion is not None and hasattr(completion, "model_dump"):
        evidence["completion"] = completion.model_dump()
    return evidence


def receipt_path(directory, metric_name, case_index, qid):
    identifier = hashlib.sha256(str(qid).encode()).hexdigest()[:16]
    return Path(directory) / metric_name / f"case_{case_index:05d}_{identifier}.json"
