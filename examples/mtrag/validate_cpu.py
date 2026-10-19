"""Check the MT-RAG adapter against pinned public data, without running judges.

Requires pandas/tqdm for upstream judge_utils, but imports no model libraries.
Input is a checkout (or the three pinned files below) of IBM/mt-rag-benchmark.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from rag_stack_mtrag.scoring import (
    build_judge_records, load_tasks, to_qa_rows,
)


REVISION = "53fef6f5917c7e10c436f697e69a955dbc795d81"
FILES = {
    "mtrag-human/generation_tasks/reference.jsonl":
        "b3dde4076f9f692a99513e424a0a39f57591dbe45600a4c1017aaa7a7edcfaac",
    "scripts/evaluation/sample_data/responses-10.jsonl":
        "8bfc4901e27dfdb1433bb56861a76a901f0b3470dc4cf12c2c26de4f174ce343",
    "scripts/evaluation/judge_utils.py":
        "222bd11dc06fcf63144aa09a23ecd7ce8ef2f2bab90bf2943ecabc3e4fd23669",
}


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    for relative, digest in FILES.items():
        actual = hashlib.sha256((args.upstream / relative).read_bytes()).hexdigest()
        check(actual == digest, f"Upstream content changed: {relative}")

    spec = importlib.util.spec_from_file_location(
        "mtrag_upstream_judge_utils", args.upstream / "scripts/evaluation/judge_utils.py"
    )
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)

    task_file = "mtrag-human/generation_tasks/reference.jsonl"
    tasks = load_tasks(args.upstream / task_file, FILES[task_file])
    before = deepcopy(tasks)
    qa_rows = to_qa_rows(tasks)
    check(len(tasks) == len(qa_rows) == 842, "Unexpected task count")
    for task, qa in zip(tasks, qa_rows):
        check(qa["qid"] == task["task_id"], "Task identity changed")
        check(qa["query"] == upstream.extract_conversation(task["input"]),
              "History does not match upstream serialization")
        check(qa["generation_gt"] == [target["text"] for target in task["targets"]],
              "Reference mapping changed")

    # Explicit synthetic fields test all task identities without treating targets
    # as generated answers or reference contexts as retrieval results.
    fixtures = [SimpleNamespace(
        qid=row["qid"], query=row["query"],
        generated_texts="CPU CONTRACT FIXTURE; not a model prediction",
        retrieved_contents=["CPU CONTRACT FIXTURE; not retrieved evidence"],
    ) for row in reversed(qa_rows)]
    records = build_judge_records(fixtures, tasks)
    check(tasks == before, "Adapter mutated the source benchmark")
    task_by_id = {task["task_id"]: task for task in tasks}
    for fixture, record in zip(fixtures, records):
        check(record["task_id"] == fixture.qid, "Adapter relied on row order")
        check(record["input"] == task_by_id[fixture.qid]["input"], "History changed")
        check(upstream.extract_texts(record["predictions"]) == fixture.generated_texts,
              "Generated answer mapping changed")
        check(upstream.extract_document_texts(record["contexts"]) == fixture.retrieved_contents,
              "Actual context mapping changed")

    # Replay public saved responses through the same mapping, without rescoring.
    sample_file = "scripts/evaluation/sample_data/responses-10.jsonl"
    samples = load_tasks(args.upstream / sample_file, FILES[sample_file])
    sample_qa = to_qa_rows(samples)
    replay_inputs = [SimpleNamespace(
        qid=row["qid"], query=row["query"],
        generated_texts=upstream.extract_texts(task["predictions"]),
        retrieved_contents=upstream.extract_document_texts(task["contexts"]),
    ) for row, task in zip(sample_qa, samples)]
    replay_records = build_judge_records(replay_inputs, samples)
    for original, replay in zip(samples, replay_records):
        for key, extract in (("input", upstream.extract_conversation),
                             ("predictions", upstream.extract_texts),
                             ("contexts", upstream.extract_document_texts)):
            check(extract(original[key]) == extract(replay[key]), f"Replay changed {key}")

    report = {
        "scope": "CPU data/interface validation; no generation or judge execution",
        "upstream_revision": REVISION,
        "sha256": FILES,
        "mapped_tasks": len(tasks),
        "conversations": len({task["conversation_id"] for task in tasks}),
        "tasks_by_collection": dict(sorted(Counter(task["Collection"] for task in tasks).items())),
        "public_saved_responses_replayed": len(replay_records),
        "checks": ["all histories match actual upstream extraction",
                   "reference mapping", "reversed-order identity join",
                   "actual output/context mapping", "source immutability",
                   "public saved-response extraction round trip"],
        "judge_calls": 0,
        "quality_scores": None,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
