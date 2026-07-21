"""Receive the canonical execution trace through ``on_trace_ready``."""

from __future__ import annotations

import argparse
import json

from _offline_fixture import (
    build_evaluator,
    prepare_project,
    resolved_pipeline_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--work-dir",
        help="Optional persistent project directory; defaults to a new /tmp directory.",
    )
    args = parser.parse_args()

    project_dir = prepare_project(args.work_dir, prefix="trace")
    evaluator = build_evaluator(project_dir)
    callback_traces: list[dict] = []
    quality = evaluator.evaluate(
        resolved_pipeline_config(),
        run_dir=str(project_dir / "runs" / "trace"),
        on_trace_ready=callback_traces.append,
    )

    returned_trace = quality["__execution_dag__"]
    trace_path = project_dir / "trace.json"
    trace_path.write_text(json.dumps(returned_trace, indent=2) + "\n")

    print(
        json.dumps(
            {
                "callback_count": len(callback_traces),
                "callback_is_returned_trace": callback_traces[0] is returned_trace,
                "quality_metrics": {
                    name: value
                    for name, value in quality.items()
                    if not name.startswith("__")
                },
                "trace_path": str(trace_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"artifacts: {project_dir}")


if __name__ == "__main__":
    main()
