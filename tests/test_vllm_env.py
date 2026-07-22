import os
from pathlib import Path
import sys

import pytest

from rag_stack_evaluator import cuda_toolkit
from rag_stack_evaluator.cuda_toolkit import (
	CudaToolkitConfigurationError,
	configure_cuda_toolkit_env,
)
from rag_stack_evaluator.vllm_env import configure_vllm_worker_env


def _fake_toolkit(tmp_path: Path, major: int, name: str) -> Path:
	home = tmp_path / name
	(home / "bin").mkdir(parents=True)
	(home / "include").mkdir()
	(home / "lib").mkdir()
	(home / "include" / "cuda.h").write_text("/* fake CUDA */", encoding="utf-8")
	(home / "lib" / "libcudart.so").write_bytes(b"fake runtime")
	nvcc = home / "bin" / "nvcc"
	nvcc.write_text(
		"#!/bin/sh\n"
		f"echo 'Cuda compilation tools, release {major}.0, V{major}.0.0'\n",
		encoding="utf-8",
	)
	nvcc.chmod(0o755)
	return home


def test_configure_vllm_worker_env_adds_env_bin_and_removes_hpcx_ucx():
	env = {
		"PATH": os.pathsep.join(["/usr/bin", "/bin"]),
		"LD_LIBRARY_PATH": os.pathsep.join(
			[
				"/opt/hpcx/ucx/lib",
				"/usr/local/ucx/lib",
				"/usr/lib",
				"/opt/hpcx/ompi/lib",
			]
		),
	}

	configure_vllm_worker_env(env=env)

	path_parts = env["PATH"].split(os.pathsep)
	assert os.path.join(sys.prefix, "bin") in path_parts[:3]
	assert "/opt/hpcx/ucx/lib" not in env["LD_LIBRARY_PATH"].split(os.pathsep)
	assert "/usr/local/ucx/lib" not in env["LD_LIBRARY_PATH"].split(os.pathsep)
	assert "/opt/hpcx/ompi/lib" in env["LD_LIBRARY_PATH"].split(os.pathsep)


def test_cuda_major_is_derived_from_torch_wheel_suffix():
	assert cuda_toolkit._cuda_major_from_local_version("2.10.0+cu128") == 12
	assert cuda_toolkit._cuda_major_from_local_version("2.11.0+cu130") == 13
	assert cuda_toolkit._cuda_major_from_local_version("2.10.0") is None


def test_cu13_unified_wheel_toolkit_is_discovered_without_shell(
	tmp_path, monkeypatch
):
	home = _fake_toolkit(tmp_path, 13, "cu13")
	(home / "lib" / "libcudart.so").unlink()
	(home / "lib" / "libcudart.so.13").write_bytes(b"fake runtime")
	driver = tmp_path / "driver" / "libcuda.so.1"
	driver.parent.mkdir()
	driver.write_bytes(b"fake driver")
	monkeypatch.setattr(cuda_toolkit, "installed_torch_cuda_major", lambda: 13)
	monkeypatch.setattr(
		cuda_toolkit,
		"_automatic_toolkit_roots",
		lambda _env, _major: [home],
	)
	monkeypatch.setattr(
		cuda_toolkit,
		"_overlay_base_roots",
		lambda: [tmp_path / "overlays"],
	)
	monkeypatch.setattr(
		cuda_toolkit,
		"_find_cuda_driver_library",
		lambda _env: driver,
	)
	env = {"PATH": "/usr/bin", "LD_LIBRARY_PATH": "/usr/lib"}

	selected = configure_cuda_toolkit_env(env=env)
	assert selected is not None
	assert selected.parent == tmp_path / "overlays"
	assert env["CUDA_HOME"] == str(selected)
	assert env["CUDA_PATH"] == str(selected)
	assert env["PATH"].split(os.pathsep)[0] == str(selected / "bin")
	assert env["LD_LIBRARY_PATH"].split(os.pathsep)[0] == str(
		selected / "lib64"
	)
	assert (selected / "lib64" / "libcudart.so").resolve() == (
		home / "lib" / "libcudart.so.13"
	).resolve()
	assert (selected / "lib64" / "libcuda.so").resolve() == driver.resolve()
	assert (selected / "lib64" / "stubs" / "libcuda.so").resolve() == (
		driver.resolve()
	)

	# Repeated vLLM construction must not grow inherited path variables.
	configure_cuda_toolkit_env(env=env)
	assert env["PATH"].split(os.pathsep).count(str(selected / "bin")) == 1
	assert env["LD_LIBRARY_PATH"].split(os.pathsep).count(
		str(selected / "lib64")
	) == 1


def test_matching_explicit_cu12_toolkit_is_preserved(tmp_path, monkeypatch):
	home = _fake_toolkit(tmp_path, 12, "cuda-12")
	monkeypatch.setattr(cuda_toolkit, "installed_torch_cuda_major", lambda: 12)
	monkeypatch.setattr(
		cuda_toolkit,
		"_automatic_toolkit_roots",
		lambda *_args: pytest.fail("explicit CUDA_HOME must win"),
	)
	env = {"CUDA_HOME": str(home), "PATH": "/usr/bin"}

	assert configure_cuda_toolkit_env(env=env) == home.resolve()
	assert env["CUDA_HOME"] == str(home.resolve())
	assert env["CUDA_PATH"] == str(home.resolve())


def test_conda_targets_layout_is_canonicalized_for_cu12(tmp_path, monkeypatch):
	home = _fake_toolkit(tmp_path, 12, "conda-prefix")
	(home / "include" / "cuda.h").unlink()
	(home / "lib" / "libcudart.so").unlink()
	target = home / "targets" / "x86_64-linux"
	(target / "include").mkdir(parents=True)
	(target / "lib").mkdir()
	(target / "include" / "cuda.h").write_text(
		"/* target CUDA */", encoding="utf-8"
	)
	(target / "lib" / "libcudart.so").write_bytes(b"target runtime")
	monkeypatch.setattr(cuda_toolkit, "installed_torch_cuda_major", lambda: 12)
	monkeypatch.setattr(
		cuda_toolkit,
		"_overlay_base_roots",
		lambda: [tmp_path / "overlays"],
	)
	monkeypatch.setattr(
		cuda_toolkit,
		"_find_cuda_driver_library",
		lambda _env: None,
	)
	env = {"CUDA_HOME": str(home), "PATH": "/usr/bin"}

	selected = configure_cuda_toolkit_env(env=env)
	assert selected is not None and selected != home
	assert (selected / "include").resolve() == (target / "include").resolve()
	assert (selected / "lib").resolve() == (target / "lib").resolve()
	assert env["CUDA_HOME"] == str(selected)


def test_explicit_wrong_major_toolkit_fails_closed(tmp_path, monkeypatch):
	home = _fake_toolkit(tmp_path, 13, "cuda-13")
	monkeypatch.setattr(cuda_toolkit, "installed_torch_cuda_major", lambda: 12)
	env = {"CUDA_HOME": str(home), "PATH": "/usr/bin"}

	with pytest.raises(CudaToolkitConfigurationError, match="built for CUDA 12"):
		configure_cuda_toolkit_env(env=env)


def test_incomplete_cu12_wheel_is_not_presented_as_cuda_home(
	tmp_path, monkeypatch
):
	# CUDA 12's split nvidia-cuda-nvcc wheel has ptxas but no nvcc driver.
	home = tmp_path / "split-cu12"
	(home / "bin").mkdir(parents=True)
	(home / "bin" / "ptxas").write_text("not nvcc", encoding="utf-8")
	(home / "include").mkdir()
	(home / "lib").mkdir()
	monkeypatch.setattr(cuda_toolkit, "installed_torch_cuda_major", lambda: 12)
	monkeypatch.setattr(
		cuda_toolkit,
		"_automatic_toolkit_roots",
		lambda _env, _major: [home],
	)
	env = {"PATH": "/usr/bin"}

	assert configure_cuda_toolkit_env(env=env) is None
	assert "CUDA_HOME" not in env
	assert "CUDA_PATH" not in env


def test_conflicting_explicit_cuda_variables_are_rejected(tmp_path, monkeypatch):
	cu12 = _fake_toolkit(tmp_path, 12, "cuda-12")
	cu13 = _fake_toolkit(tmp_path, 13, "cuda-13")
	monkeypatch.setattr(cuda_toolkit, "installed_torch_cuda_major", lambda: 12)

	with pytest.raises(CudaToolkitConfigurationError, match="different toolkits"):
		configure_cuda_toolkit_env(
			env={
				"CUDA_HOME": str(cu12),
				"CUDA_PATH": str(cu13),
				"PATH": "/usr/bin",
			}
		)


def test_cuda_driver_discovery_ignores_linker_stubs(tmp_path, monkeypatch):
	stub_dir = tmp_path / "cuda" / "lib64" / "stubs"
	real_dir = tmp_path / "driver"
	stub_dir.mkdir(parents=True)
	real_dir.mkdir()
	(stub_dir / "libcuda.so").write_bytes(b"stub")
	real_driver = real_dir / "libcuda.so.1"
	real_driver.write_bytes(b"driver")
	monkeypatch.setattr(cuda_toolkit.shutil, "which", lambda _name: None)
	monkeypatch.setattr(
		cuda_toolkit, "_standard_driver_library_dirs", lambda: []
	)

	selected = cuda_toolkit._find_cuda_driver_library(
		{
			"LD_LIBRARY_PATH": os.pathsep.join(
				[str(stub_dir), str(real_dir)]
			)
		}
	)
	assert selected == real_driver.resolve()


def test_vllm_cli_launcher_configures_env_before_exec(monkeypatch):
	from rag_stack_evaluator import vllm_launcher

	seen = {}

	def _configure(*, env):
		env["CUDA_HOME"] = "/validated/cuda"

	def _exec(file, argv, env):
		seen.update(file=file, argv=argv, env=env)
		raise RuntimeError("exec intercepted")

	monkeypatch.setattr(vllm_launcher, "configure_vllm_worker_env", _configure)
	monkeypatch.setattr(vllm_launcher.os, "execvpe", _exec)
	monkeypatch.setattr(vllm_launcher.sys, "argv", ["launcher", "serve", "model"])

	with pytest.raises(RuntimeError, match="exec intercepted"):
		vllm_launcher.main()
	assert seen["file"] == "vllm"
	assert seen["argv"] == ["vllm", "serve", "model"]
	assert seen["env"]["CUDA_HOME"] == "/validated/cuda"
