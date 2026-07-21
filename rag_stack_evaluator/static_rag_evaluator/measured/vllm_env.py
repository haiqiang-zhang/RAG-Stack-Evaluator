"""Environment helpers for vLLM worker subprocesses."""

from __future__ import annotations

import os
import sys
from logging import Logger
from typing import MutableMapping, Optional


def _prepend_env_path(target: MutableMapping[str, str], name: str, path: str) -> bool:
	current = target.get(name, "")
	parts = [p for p in current.split(os.pathsep) if p]
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
	parts = [p for p in current.split(os.pathsep) if p]
	removed = [p for p in parts if _is_external_ucx_path(p)]
	if not removed:
		return []
	target[name] = os.pathsep.join(p for p in parts if not _is_external_ucx_path(p))
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


def ensure_cuda_wheel_toolkit_in_env(
	env: Optional[MutableMapping[str, str]] = None,
	logger: Optional[Logger] = None,
) -> None:
	"""Expose the CUDA toolkit wheel shim to vLLM/FlashInfer JIT subprocesses."""
	target = os.environ if env is None else env
	shim_home = os.path.join(sys.prefix, ".rag_stack", "cuda", "cu13")
	shim_nvcc = os.path.join(shim_home, "bin", "nvcc")
	if not os.path.isfile(shim_nvcc) or not os.access(shim_nvcc, os.X_OK):
		return

	current_cuda_home = target.get("CUDA_HOME") or target.get("CUDA_PATH")
	if current_cuda_home:
		current_nvcc = os.path.join(current_cuda_home, "bin", "nvcc")
		if os.path.isfile(current_nvcc) and os.access(current_nvcc, os.X_OK):
			shim_home = current_cuda_home

	target.setdefault("RAG_STACK_CUDA_WHEEL_HOME", shim_home)
	target["CUDA_HOME"] = shim_home
	target["CUDA_PATH"] = shim_home

	bin_dir = os.path.join(shim_home, "bin")
	lib64_dir = os.path.join(shim_home, "lib64")
	path_changed = _prepend_env_path(target, "PATH", bin_dir)
	ld_changed = False
	if os.path.isdir(lib64_dir):
		ld_changed = _prepend_env_path(target, "LD_LIBRARY_PATH", lib64_dir)

	if logger is not None and (path_changed or ld_changed):
		logger.info(
			"vLLM: exposed CUDA toolkit shim %s for worker subprocesses",
			shim_home,
		)


def configure_vllm_worker_env(
	env: Optional[MutableMapping[str, str]] = None,
	logger: Optional[Logger] = None,
) -> None:
	"""Apply environment fixes needed by spawned vLLM worker subprocesses."""
	remove_external_ucx_from_ld_library_path(env=env, logger=logger)
	ensure_python_env_bin_on_path(env=env, logger=logger)
	ensure_python_env_lib_in_ld_library_path(env=env, logger=logger)
	ensure_cuda_wheel_toolkit_in_env(env=env, logger=logger)
