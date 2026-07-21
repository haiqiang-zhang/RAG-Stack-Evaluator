"""Run a complete offline retrieval-quality evaluation."""

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

    project_dir = prepare_project(args.work_dir, prefix="quality")
    evaluator = build_evaluator(project_dir)
    quality = evaluator.evaluate(
        resolved_pipeline_config(),
        run_dir=str(project_dir / "runs" / "quality"),
    )

    public_metrics = {
        name: value for name, value in quality.items() if not name.startswith("__")
    }
    print(json.dumps(public_metrics, indent=2, sort_keys=True))
    print(f"artifacts: {project_dir}")


if __name__ == "__main__":
    main()
