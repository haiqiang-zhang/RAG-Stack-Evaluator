"""Launch the vLLM CLI with RAG-Stack's validated CUDA environment."""

from __future__ import annotations

import os
import sys

from rag_stack_evaluator.vllm_env import configure_vllm_worker_env


def main() -> None:
    env = dict(os.environ)
    configure_vllm_worker_env(env=env)
    os.execvpe("vllm", ["vllm", *sys.argv[1:]], env)


if __name__ == "__main__":
    main()
