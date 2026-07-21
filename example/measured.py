"""Run one resolved pipeline deployment and write its measured result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rag_stack_evaluator.static_rag_evaluator import (
    DatasetEvalManager,
    MeasuredProvider,
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


def evaluate_measured(
    *,
    pipeline_config: dict[str, Any],
    system_config: dict[str, Any],
    available_gpus: list[str],
    qa_parquet: str,
    corpus_parquet: str,
    project_dir: str,
    run_dir: str,
    n_queries: int | None,
    selection: str,
    force_disagg: bool,
    require_admissible: bool,
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
    with MeasuredProvider(
        evaluator,
        available_gpus,
        n_queries=n_queries,
        selection=selection,
    ) as provider:
        result = provider.evaluate(
            pipeline_config,
            system_config,
            run_dir=run_dir,
            force_disagg=force_disagg,
            require_admissible=require_admissible,
        )
    return {
        "performance_score": result.performance_score,
        "quality": result.quality,
        "raw_performance": result.raw_performance,
        "performance_execution_trace": result.performance_execution_trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-config", required=True)
    parser.add_argument("--system-config", required=True)
    parser.add_argument("--qa", required=True, help="QA Parquet path")
    parser.add_argument("--corpus", required=True, help="Pre-chunked corpus Parquet path")
    parser.add_argument("--project-dir", required=True)
    parser.add_argument(
        "--gpu",
        action="append",
        required=True,
        help="Owned device such as cuda:0; repeat for multiple devices",
    )
    parser.add_argument("--n-queries", type=int)
    parser.add_argument(
        "--selection",
        choices=("max_throughput", "min_latency"),
        default="max_throughput",
    )
    parser.add_argument("--force-disagg", action="store_true")
    parser.add_argument("--allow-inadmissible", action="store_true")
    parser.add_argument("--run-dir")
    parser.add_argument("--output")
    args = parser.parse_args()

    project_dir = Path(args.project_dir).expanduser().resolve()
    run_dir = (
        Path(args.run_dir).expanduser().resolve()
        if args.run_dir
        else project_dir / "runs" / "measured"
    )
    output = Path(args.output).expanduser().resolve() if args.output else run_dir / "result.json"

    pipeline_config = load_json_dict(args.pipeline_config, label="pipeline config")
    system_config = load_json_dict(args.system_config, label="system config")
    result = evaluate_measured(
        pipeline_config=pipeline_config,
        system_config=system_config,
        available_gpus=list(args.gpu),
        qa_parquet=str(Path(args.qa).expanduser().resolve()),
        corpus_parquet=str(Path(args.corpus).expanduser().resolve()),
        project_dir=str(project_dir),
        run_dir=str(run_dir),
        n_queries=args.n_queries,
        selection=args.selection,
        force_disagg=args.force_disagg,
        require_admissible=not args.allow_inadmissible,
    )
    write_json(output, result)
    print(json.dumps(result, indent=2, default=str))
    print(f"result: {output}")


if __name__ == "__main__":
    main()
