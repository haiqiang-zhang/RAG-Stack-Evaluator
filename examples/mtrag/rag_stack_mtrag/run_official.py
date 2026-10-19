"""Invoke IBM's full generation evaluator in its separately provisioned environment.

This module imports only the standard library. Calling main explicitly starts
the official algorithmic and model-based judges; CPU validation never calls it.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--provider", required=True, choices=("hf", "vllm"))
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--algorithmic-config", type=Path)
    parser.add_argument("--algorithmic-config-sha256", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be a full lowercase 40-hex commit")
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if revision != args.revision:
        parser.error(f"Expected upstream revision {args.revision}, found {revision}")
    subprocess.run(["git", "-C", str(repo), "diff", "--exit-code", "HEAD", "--", "scripts/evaluation"],
                   check=True, stdout=subprocess.DEVNULL)
    evaluator_config = (args.algorithmic_config or repo / "scripts/evaluation/config.yaml").resolve()
    if hashlib.sha256(evaluator_config.read_bytes()).hexdigest() != args.algorithmic_config_sha256:
        parser.error("Algorithmic evaluator configuration SHA-256 mismatch")
    source, destination = args.input.resolve(), args.output.resolve()
    if source == destination:
        parser.error("Use separate source and scored-output files")
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(repo / "scripts/evaluation/run_generation_eval.py"),
         "--input", str(source), "--output", str(destination),
         "--algorithmic_evaluators", str(evaluator_config),
         "--provider", args.provider, "--judge_model", args.judge_model]
    # Replace this wrapper so the bridge's timeout owns the judge process.
    os.chdir(repo)
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
