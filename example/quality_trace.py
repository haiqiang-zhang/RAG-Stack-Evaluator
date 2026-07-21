"""Evaluate one pipeline and consume its canonical trace callback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rag_stack_evaluator.static_rag_evaluator import (
    DatasetEvalManager,
    StaticRAGEvaluatorQualityOnly,
)


def load_json_dict(path: str, *, label: str) -> dict[str, Any]:
    value = json.loads(Path(path).expanduser().read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def evaluate_with_trace(
    *,
    pipeline_config: dict[str, Any],
    qa_parquet: str,
    corpus_parquet: str,
    project_dir: str,
    run_dir: str,
    on_trace_ready,
) -> dict[str, Any]:
    dataset = DatasetEvalManager(
        project_dir=project_dir,
        qa_data_path=qa_parquet,
        corpus_data_path=corpus_parquet,
    )
    evaluator = StaticRAGEvaluatorQualityOnly(
        dataset_manager=dataset,
        project_dir=project_dir,
    )
    return evaluator.evaluate(
        pipeline_config,
        run_dir=run_dir,
        on_trace_ready=on_trace_ready,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-config", required=True)
    parser.add_argument("--qa", required=True, help="QA Parquet path")
    parser.add_argument("--corpus", required=True, help="Pre-chunked corpus Parquet path")
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--run-dir")
    parser.add_argument("--quality-output")
    parser.add_argument("--trace-output")
    args = parser.parse_args()

    project_dir = Path(args.project_dir).expanduser().resolve()
    run_dir = (
        Path(args.run_dir).expanduser().resolve()
        if args.run_dir
        else project_dir / "runs" / "quality-trace"
    )
    quality_output = (
        Path(args.quality_output).expanduser().resolve()
        if args.quality_output
        else run_dir / "quality.json"
    )
    trace_output = (
        Path(args.trace_output).expanduser().resolve()
        if args.trace_output
        else run_dir / "trace.json"
    )
    callback_traces: list[dict[str, Any]] = []

    def on_trace_ready(trace: dict[str, Any]) -> None:
        callback_traces.append(trace)
        write_json(trace_output, trace)

    pipeline_config = load_json_dict(args.pipeline_config, label="pipeline config")
    result = evaluate_with_trace(
        pipeline_config=pipeline_config,
        qa_parquet=str(Path(args.qa).expanduser().resolve()),
        corpus_parquet=str(Path(args.corpus).expanduser().resolve()),
        project_dir=str(project_dir),
        run_dir=str(run_dir),
        on_trace_ready=on_trace_ready,
    )
    write_json(quality_output, result)
    print(
        json.dumps(
            {
                "quality_result": str(quality_output),
                "trace_result": str(trace_output),
                "callback_count": len(callback_traces),
                "callback_is_returned_trace": bool(callback_traces)
                and callback_traces[0] is result.get("__execution_dag__"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
