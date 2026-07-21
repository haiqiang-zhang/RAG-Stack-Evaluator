"""Evaluate one pipeline with a caller-selected metric subset."""

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


def evaluate_selected_metrics(
    *,
    pipeline_config: dict[str, Any],
    metrics: list[str],
    qa_parquet: str,
    corpus_parquet: str,
    project_dir: str,
    run_dir: str,
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
        metrics_override=metrics,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-config", required=True)
    parser.add_argument("--qa", required=True, help="QA Parquet path")
    parser.add_argument("--corpus", required=True, help="Pre-chunked corpus Parquet path")
    parser.add_argument("--project-dir", required=True)
    parser.add_argument(
        "--metric",
        action="append",
        required=True,
        help="Metric name; repeat this option to request multiple metrics",
    )
    parser.add_argument("--run-dir")
    parser.add_argument("--output")
    args = parser.parse_args()

    project_dir = Path(args.project_dir).expanduser().resolve()
    run_dir = (
        Path(args.run_dir).expanduser().resolve()
        if args.run_dir
        else project_dir / "runs" / "quality-metrics-override"
    )
    output = Path(args.output).expanduser().resolve() if args.output else run_dir / "result.json"

    pipeline_config = load_json_dict(args.pipeline_config, label="pipeline config")
    result = evaluate_selected_metrics(
        pipeline_config=pipeline_config,
        metrics=list(args.metric),
        qa_parquet=str(Path(args.qa).expanduser().resolve()),
        corpus_parquet=str(Path(args.corpus).expanduser().resolve()),
        project_dir=str(project_dir),
        run_dir=str(run_dir),
    )
    write_json(output, result)
    print(json.dumps(result, indent=2, default=str))
    print(f"result: {output}")


if __name__ == "__main__":
    main()
