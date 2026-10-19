"""Exercise plugin registration, tracing, scoring transport, and result normalization.

Run with the evaluator and host packages available on PYTHONPATH. The external
scorer is an explicitly synthetic stdlib subprocess; no judge/model is loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from rag_stack_evaluator.api import normalize_evaluation_result
from rag_stack_evaluator.factory import available_backends, create_quality_evaluator, register_backend
from rag_stack_mtrag.backend import BACKEND
from rag_stack_mtrag.fixture import fixture_config, fixture_tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    task_path = args.output / "tasks.jsonl"
    task_path.write_text("".join(json.dumps(task) + "\n" for task in fixture_tasks()))
    options = {
        "tasks_path": str(task_path.resolve()),
        "tasks_sha256": hashlib.sha256(task_path.read_bytes()).hexdigest(),
        "pipeline": "rag_stack_mtrag.fixture:run_pipeline",
        "pipeline_identity": "synthetic-cpu-contract-fixture-v1",
        "scorer_command": [sys.executable, str(Path(__file__).parent / "rag_stack_mtrag/fixture.py"),
                           "--input", "{input}", "--output", "{output}"],
        "scorer_identity": "synthetic-cpu-scores-v1; not official MT-RAG results",
        "upstream_revision": "53fef6f5917c7e10c436f697e69a955dbc795d81",
    }
    if "mtrag" not in available_backends():
        register_backend(BACKEND)
    evaluator = create_quality_evaluator(backend="mtrag", backend_options=options)
    callbacks = []
    result = evaluator.evaluate(fixture_config(), run_dir=str(args.output), on_trace_ready=callbacks.append)
    assert len(callbacks) == 1 and callbacks[0] is result.trace
    flat = normalize_evaluation_result(result)
    report = {
        "scope": "Synthetic CPU interface fixture; not MT-RAG benchmark measurements",
        "official_judge_calls": 0, "model_loads": 0,
        "tasks": len(fixture_tasks()), "fixture_metrics": result.metrics,
        "trace_published_before_scoring": True,
        "result": flat,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
