"""Run IBM's RL_F judge in its own environment; never used by CPU validation."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--provider", choices=("hf", "vllm"), required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    repo = args.repo.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != args.revision:
        parser.error(f"Expected upstream revision {args.revision}, found {revision}")
    subprocess.run(
        ["git", "-C", str(repo), "diff", "--exit-code", "HEAD", "--", "scripts/evaluation"],
        check=True, stdout=subprocess.DEVNULL,
    )
    input_path, output_path = args.input.resolve(), args.output.resolve()
    if input_path == output_path:
        parser.error("Use separate input and output files to preserve source evidence")

    # Import only when explicitly running a judge. This is IBM's implementation,
    # including its provider setup and dependency requirements, not a reimplementation.
    sys.path.insert(0, str(repo / "scripts" / "evaluation"))
    from judge_wrapper import run_ragas_judges_local

    run_ragas_judges_local(
        args.provider, args.judge_model, str(input_path), str(output_path)
    )


if __name__ == "__main__":
    main()
