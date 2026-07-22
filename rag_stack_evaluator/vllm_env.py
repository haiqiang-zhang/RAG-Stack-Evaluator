"""Lightweight environment configuration shared by every vLLM entrypoint."""

from __future__ import annotations

from logging import Logger
import os
import sys
from typing import MutableMapping, Optional

from rag_stack_evaluator.cuda_toolkit import configure_cuda_toolkit_env


def _prepend_env_path(
    target: MutableMapping[str, str], name: str, path: str
) -> bool:
    current = target.get(name, "")
    parts = [part for part in current.split(os.pathsep) if part]
    if path in parts:
        return False
    target[name] = path if not current else f"{path}{os.pathsep}{current}"
    return True


def _is_external_ucx_path(path: str) -> bool:
    abs_path = os.path.abspath(path)
    env_prefix = os.path.abspath(sys.prefix)
    if abs_path == env_prefix or abs_path.startswith(f"{env_prefix}{os.sep}"):
        return False
    parts = {part.lower() for part in abs_path.split(os.sep) if part}
    return "ucx" in parts


def _remove_external_ucx_paths(
    target: MutableMapping[str, str],
    name: str,
) -> list[str]:
    current = target.get(name, "")
    if not current:
        return []
    parts = [part for part in current.split(os.pathsep) if part]
    removed = [part for part in parts if _is_external_ucx_path(part)]
    if not removed:
        return []
    target[name] = os.pathsep.join(
        part for part in parts if not _is_external_ucx_path(part)
    )
    return removed


def ensure_python_env_bin_on_path(
    env: Optional[MutableMapping[str, str]] = None,
    logger: Optional[Logger] = None,
) -> None:
    """Prepend ``sys.prefix/bin`` so worker JIT tools are discoverable."""

    bin_dir = os.path.join(sys.prefix, "bin")
    if not os.path.isdir(bin_dir):
        return

    target = os.environ if env is None else env
    changed = _prepend_env_path(target, "PATH", bin_dir)
    if changed and logger is not None:
        logger.info(
            "vLLM: prepended %s to PATH for worker subprocesses",
            bin_dir,
        )


def remove_external_ucx_from_ld_library_path(
    env: Optional[MutableMapping[str, str]] = None,
    logger: Optional[Logger] = None,
) -> None:
    """Avoid mixing external UCX module paths with NIXL's bundled UCX libs."""

    target = os.environ if env is None else env
    removed = _remove_external_ucx_paths(target, "LD_LIBRARY_PATH")
    if removed and logger is not None:
        logger.info(
            "vLLM: removed %s external UCX path(s) from LD_LIBRARY_PATH for "
            "worker subprocesses",
            len(removed),
        )


def ensure_python_env_lib_in_ld_library_path(
    env: Optional[MutableMapping[str, str]] = None,
    logger: Optional[Logger] = None,
) -> None:
    """Prepend ``sys.prefix/lib`` so spawned vLLM workers use this env's libs."""

    lib_dir = os.path.join(sys.prefix, "lib")
    if not os.path.isdir(lib_dir):
        return

    target = os.environ if env is None else env
    changed = _prepend_env_path(target, "LD_LIBRARY_PATH", lib_dir)
    if changed and logger is not None:
        logger.info(
            "vLLM: prepended %s to LD_LIBRARY_PATH for worker subprocesses",
            lib_dir,
        )


def configure_vllm_worker_env(
    env: Optional[MutableMapping[str, str]] = None,
    logger: Optional[Logger] = None,
) -> None:
    """Apply environment fixes before importing or spawning vLLM."""

    remove_external_ucx_from_ld_library_path(env=env, logger=logger)
    ensure_python_env_bin_on_path(env=env, logger=logger)
    ensure_python_env_lib_in_ld_library_path(env=env, logger=logger)
    configure_cuda_toolkit_env(env=env, logger=logger)


__all__ = [
    "configure_vllm_worker_env",
    "ensure_python_env_bin_on_path",
    "ensure_python_env_lib_in_ld_library_path",
    "remove_external_ucx_from_ld_library_path",
]
