"""MT-RAG generation-evaluator bridge; importing starts no model runtime.

An external command can run the full official generation evaluator or a
selected judge. Its scored records are joined by task ID, never row position.
These are generation records, not retrieval submissions: chunks receive no
invented benchmark document IDs or retrieval scores.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Sequence


def _index_tasks(tasks: Sequence[dict]) -> dict[str, dict]:
    if not tasks:
        raise ValueError("MT-RAG tasks must be nonempty")
    indexed = {}
    for task in tasks:
        qid = task.get("task_id") if isinstance(task, dict) else None
        if not isinstance(qid, str) or not qid.strip() or qid in indexed:
            raise ValueError(f"Missing, invalid, or duplicate MT-RAG task_id: {qid!r}")
        turns, targets = task.get("input"), task.get("targets")
        if not isinstance(turns, list) or not turns or any(
            not isinstance(turn, dict)
            or not isinstance(turn.get("speaker"), str) or not turn["speaker"]
            or not isinstance(turn.get("text"), str)
            for turn in turns
        ):
            raise ValueError(f"Invalid conversation input for {qid!r}")
        if not isinstance(targets, list) or not targets or any(
            not isinstance(target, dict) or not isinstance(target.get("text"), str)
            for target in targets
        ):
            raise ValueError(f"Invalid reference targets for {qid!r}")
        indexed[qid] = task
    return indexed


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _write_jsonl(path: Path, records: Sequence[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                            for row in records), encoding="utf-8")


def load_tasks(path: str | Path, expected_sha256: str) -> list[dict]:
    """Load an immutable JSONL task sidecar and verify original task identities."""
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ValueError("expected_sha256 must be a full SHA-256 digest")
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256.lower():
        raise ValueError("MT-RAG task sidecar SHA-256 mismatch")
    tasks = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    _index_tasks(tasks)
    return tasks


def to_qa_rows(tasks: Sequence[dict]) -> list[dict]:
    """Map fixed benchmark history to one RAG-Stack invocation per task.

    Flattening matches upstream ``extract_conversation``. Reference targets
    remain in ``generation_gt``; they are never appended to the query.
    """
    return [{
        "qid": qid,
        "query": "\n".join(f"{turn['speaker']}: {turn['text']}" for turn in task["input"]),
        "generation_gt": [target["text"] for target in task["targets"]],
    } for qid, task in _index_tasks(tasks).items()]


def build_judge_records(metric_inputs: Sequence[Any], tasks: Sequence[dict]) -> list[dict]:
    """Join actual answers and final contexts to original tasks by stable qid.

    Query text is not a join key: host preprocessing legitimately normalizes
    Unicode/emoji and whitespace. The hash-verified sidecar owns judge history.
    Empty emitted answers/contexts are valid; structurally absent fields fail.
    """
    indexed, seen, records = _index_tasks(tasks), set(), []
    if not metric_inputs:
        raise ValueError("MT-RAG metric input batch must be nonempty")
    for item in metric_inputs:
        qid = getattr(item, "qid", None)
        if not isinstance(qid, str) or qid not in indexed or qid in seen:
            raise ValueError(f"Missing, unknown, or duplicate metric qid: {qid!r}")
        answer = getattr(item, "generated_texts", None)
        contexts = getattr(item, "retrieved_contents", None)
        if not isinstance(answer, str):
            raise ValueError(f"Missing or invalid generated_texts for {qid!r}")
        if not isinstance(contexts, list) or any(not isinstance(text, str) for text in contexts):
            raise ValueError(f"Missing or invalid retrieved_contents for {qid!r}")
        record = deepcopy(indexed[qid])
        record["contexts"] = [{"text": text} for text in contexts]
        record["predictions"] = [{"text": answer}]
        record.pop("metrics", None)
        records.append(record)
        seen.add(qid)
    return records


def _selected_scores(
    records: Sequence[dict], output: Sequence[dict], metric_keys: dict[str, str],
    score_bounds: dict[str, tuple[float, float]],
) -> dict[str, list[float]]:
    expected = {record["task_id"]: record for record in records}
    scores = {}
    for row in output:
        qid = row.get("task_id") if isinstance(row, dict) else None
        if not isinstance(qid, str) or qid not in expected or qid in scores:
            raise ValueError(f"Missing, extra, or duplicate scorer task_id: {qid!r}")
        for key in ("input", "contexts", "predictions", "targets"):
            if row.get(key) != expected[qid][key]:
                raise ValueError(f"Scorer changed {key} for {qid!r}")
        metrics = row.get("metrics")
        selected = {}
        for name, key in metric_keys.items():
            value = metrics.get(key) if isinstance(metrics, dict) else None
            if not isinstance(value, list) or len(value) != 1:
                raise ValueError(f"Expected metrics.{key}=[score] for {qid!r}")
            score = value[0]
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError(f"Invalid MT-RAG {key} score for {qid!r}: {score!r}")
            if name in score_bounds and not score_bounds[name][0] <= score <= score_bounds[name][1]:
                raise ValueError(f"Out-of-range MT-RAG {key} score for {qid!r}: {score!r}")
            selected[name] = float(score)
        scores[qid] = selected
    if scores.keys() != expected.keys():
        raise ValueError("Scorer output is missing MT-RAG task IDs")
    return {name: [scores[record["task_id"]][name] for record in records]
            for name in metric_keys}


def score_with_mtrag(
    metric_inputs: Sequence[Any], *, tasks_path: str, tasks_sha256: str,
    scorer_command: list[str], scorer_identity: str, upstream_revision: str,
    metric_keys: dict[str, str],
    score_bounds: dict[str, tuple[float, float]] | None = None,
    timeout_seconds: float = 3600, audit_dir: str | None = None,
) -> dict[str, list[float]]:
    """Run MT-RAG once and return selected per-task scores in input order.

    ``metric_keys`` maps public names to upstream ``metrics`` keys. Each
    selected key must contain exactly one finite numeric score per task.
    Multi-reference/passage arrays need an explicit upstream aggregation;
    this bridge never silently picks an element or invents an aggregation.
    Generic scores need not lie in [0, 1]; bounds can be supplied per metric.

    ``scorer_command`` is argv (no shell), with exactly one standalone
    ``{input}`` and ``{output}`` argument. It runs in the caller's working
    directory; use absolute paths for the scorer's interpreter and script.
    Secrets belong in the scorer environment, never in this configuration.
    Provenance fields must describe the actual external scorer installation;
    this bridge cannot verify arbitrary executable contents from a label.
    With ``audit_dir``, a unique child preserves inputs, outputs, provenance,
    and subprocess logs, including failures. No scores are imputed or dropped.
    """
    if not isinstance(scorer_identity, str) or not scorer_identity.strip():
        raise ValueError("scorer_identity must identify the scorer/model configuration")
    if not isinstance(upstream_revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", upstream_revision):
        raise ValueError("upstream_revision must be a full 40-hex commit")
    if not isinstance(scorer_command, list) or not scorer_command or any(
        not isinstance(arg, str) or not arg for arg in scorer_command
    ):
        raise ValueError("scorer_command must be a nonempty argv list")
    for marker in ("{input}", "{output}"):
        if scorer_command.count(marker) != 1 or any(marker in arg and arg != marker for arg in scorer_command):
            raise ValueError(f"scorer_command requires exactly one standalone {marker}")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    if not isinstance(metric_keys, dict) or not metric_keys or any(
        not isinstance(name, str) or not name.strip() or name.startswith("__")
        or not isinstance(key, str) or not key.strip()
        for name, key in metric_keys.items()
    ):
        raise ValueError("metric_keys must map nonempty public names to upstream keys")
    score_bounds = {} if score_bounds is None else score_bounds
    if not isinstance(score_bounds, dict) or any(
        name not in metric_keys or not isinstance(bounds, (list, tuple)) or len(bounds) != 2
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
               for value in bounds)
        or bounds[0] > bounds[1]
        for name, bounds in score_bounds.items()
    ):
        raise ValueError("score_bounds must contain finite ordered bounds for selected metrics")
    records = build_judge_records(metric_inputs, load_tasks(tasks_path, tasks_sha256))
    provenance = {"metric_keys": dict(metric_keys), "score_bounds": dict(score_bounds),
                  "tasks_sha256": tasks_sha256.lower(), "scorer_identity": scorer_identity,
                  "upstream_revision": upstream_revision.lower(), "status": "failed",
                  "scorer_command": list(scorer_command), "timeout_seconds": timeout_seconds,
                  "task_ids": [record["task_id"] for record in records]}
    with tempfile.TemporaryDirectory(prefix="rag_stack_mtrag_") as temporary:
        work = Path(temporary)
        source, output = work / "input.jsonl", work / "output.jsonl"
        _write_jsonl(source, records)
        replacements = {"{input}": str(source), "{output}": str(output)}
        command = [replacements.get(arg, arg) for arg in scorer_command]
        try:
            with (work / "stdout.log").open("w") as stdout, (work / "stderr.log").open("w") as stderr:
                subprocess.run(command, check=True, shell=False, timeout=timeout_seconds,
                               stdout=stdout, stderr=stderr)
            scores = _selected_scores(records, _read_jsonl(output), metric_keys, score_bounds)
            provenance.update(status="complete", scores=scores)
            return scores
        finally:
            (work / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
            if audit_dir is not None:
                Path(audit_dir).mkdir(parents=True, exist_ok=True)
                destination = tempfile.mkdtemp(prefix="mt_rag_", dir=audit_dir)
                shutil.copytree(work, destination, dirs_exist_ok=True)


def mt_rag_faithfulness(
    metric_inputs: Sequence[Any], *, tasks_path: str, tasks_sha256: str,
    scorer_command: list[str], scorer_identity: str, upstream_revision: str,
    timeout_seconds: float = 3600, audit_dir: str | None = None,
) -> list[float]:
    """Backward-compatible static-metric bridge for MT-RAG's ``RL_F`` judge."""
    return score_with_mtrag(
        metric_inputs, tasks_path=tasks_path, tasks_sha256=tasks_sha256,
        scorer_command=scorer_command, scorer_identity=scorer_identity,
        upstream_revision=upstream_revision,
        metric_keys={"mt_rag_faithfulness": "RL_F"},
        score_bounds={"mt_rag_faithfulness": (0.0, 1.0)},
        timeout_seconds=timeout_seconds, audit_dir=audit_dir,
    )["mt_rag_faithfulness"]
