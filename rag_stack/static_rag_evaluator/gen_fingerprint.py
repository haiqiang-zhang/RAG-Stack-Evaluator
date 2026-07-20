"""Generation-input fingerprint store (M3, 07-07): dedup by CONTENT, not config.

Quality's causal chain is ``final retrieved contexts + prompts + generator
config → answer distribution → judge distribution``. When every input on that
chain is byte-identical to an earlier eval's, re-running generation + judging
re-samples the SAME random variable — zero new information for a full GT
price. This store lets the generator node recognize such evals and inherit
the donor's answers and quality wholesale (the CM still prices THIS config's
performance: retrieval-side knobs like nprobe/ef_search legitimately differ
in cost while retrieving identical content — the exact phantom-knob triple
measured on ax_s44: token metrics byte-identical, three GT seats spent on
judge re-rolls of 0.483/0.453/0.468).

Equivalence is judged on CONTENT (queries + contexts + prompts), so sampled
upstream stages self-invalidate: query expansion at temperature > 0 yields
different expansions → different contexts → fingerprint miss → normal eval.

Store layout: ``<project_dir>/fingerprints/gen/<fp>.json`` with two-phase
records — the donor eval writes ``answers``/``token_counts`` at generation
time and the evaluator attaches ``quality`` after judging completes. A record
without ``quality`` is not yet usable (donor still mid-eval); hits require a
complete record.

Everything is gated on ``RAG_STACK_GEN_FP_DEDUP=1`` (exported by the
Controller from the config flag ``global: gen_fingerprint_dedup`` — default
OFF, so baselines and existing runs are byte-identical to before).


EQUIVALENCE BOUNDARY (07-08, verified on production hit/donor pairs incl. a
QE-on inheriting from a QE-off donor): the inheritance proof requires every
quality metric to be a pure function of (query, retrieved contents, prompts,
answer, ground truth) — the exact content the fingerprint hashes. All current
metrics satisfy this. A future metric that reads pipeline INTERMEDIATES the
fingerprint does not cover (e.g. the QE rewrite text itself, per-hop react
traces) breaks the equivalence and must re-audit this module before shipping.
React never enters this path at all (the agentic engine bypasses the
sequential generator node).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger("RAG-Stack")

_ENV_FLAG = "RAG_STACK_GEN_FP_DEDUP"
# Transport/deployment knobs that cannot change the generated text.  This is
# public because hardware-independent quality identities must apply the same
# semantic boundary when canonicalising complete nested algorithm configs.
TRANSPORT_ONLY_PARAMS = frozenset({
    "uri", "batch", "request_timeout", "project_dir", "gpu_memory_utilization",
})


def enabled() -> bool:
    return os.environ.get(_ENV_FLAG, "") == "1"


def _canon(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _canon(v[k]) for k in sorted(v)}
    if isinstance(v, (list, tuple)):
        return [_canon(x) for x in v]
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def fingerprint(
    queries: List[str],
    contexts: List[Any],
    prompts: List[str],
    module_name: str,
    module_param: Dict[str, Any],
) -> str:
    """sha256 over the full generation-input content + generator semantics."""
    sem_param = {
        k: _canon(v) for k, v in (module_param or {}).items()
        if k not in TRANSPORT_ONLY_PARAMS
    }
    payload = json.dumps(
        {
            "queries": [str(q) for q in queries],
            "contexts": _canon(contexts),
            "prompts": [str(p) for p in prompts],
            "module": module_name,
            "param": sem_param,
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _store_dir(project_dir: str) -> str:
    return os.path.join(project_dir, "fingerprints", "gen")


def record_path(project_dir: str, fp: str) -> str:
    return os.path.join(_store_dir(project_dir), f"{fp}.json")


def load_complete_record(project_dir: str, fp: str) -> Optional[dict]:
    """The donor record, ONLY when it already carries quality (a record still
    missing quality belongs to an eval that never finished judging — unusable)."""
    path = record_path(project_dir, fp)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as fh:
            rec = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if not rec.get("answers") or not isinstance(rec.get("quality"), dict):
        return None
    return rec


def save_answers(
    project_dir: str,
    fp: str,
    answers: List[str],
    token_counts: List[int],
    donor: str,
) -> Optional[str]:
    """Phase-1 write at donor generation time (atomic; never overwrites a
    complete record)."""
    path = record_path(project_dir, fp)
    if os.path.isfile(path):
        return path
    try:
        os.makedirs(_store_dir(project_dir), exist_ok=True)
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w") as fh:
            json.dump(
                {"fp": fp, "donor": donor, "answers": list(answers),
                 "token_counts": [int(t) for t in token_counts]},
                fh,
            )
        os.replace(tmp, path)
        return path
    except OSError as exc:
        logger.warning(f"[gen-fp] answer record write failed ({exc}); dedup off for this eval")
        return None


def attach_quality(project_dir: str, fp: str, quality: Dict[str, Any]) -> None:
    """Phase-2 write after the donor's judging completes."""
    path = record_path(project_dir, fp)
    if not os.path.isfile(path):
        return
    try:
        with open(path) as fh:
            rec = json.load(fh)
        if isinstance(rec.get("quality"), dict):
            return  # first complete donor wins
        rec["quality"] = {
            k: v for k, v in quality.items()
            if not str(k).startswith("__") and isinstance(v, (int, float))
        }
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w") as fh:
            json.dump(rec, fh)
        os.replace(tmp, path)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"[gen-fp] quality attach failed ({exc})")


# ── run-dir marker files (node ↔ evaluator channel, one eval's run root) ────

def write_marker(run_root: str, name: str, payload: dict) -> None:
    try:
        with open(os.path.join(run_root, name), "w") as fh:
            json.dump(payload, fh)
    except OSError as exc:
        logger.warning(f"[gen-fp] marker write failed ({exc})")


def read_marker(run_root: str, name: str) -> Optional[dict]:
    path = os.path.join(run_root, name)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
