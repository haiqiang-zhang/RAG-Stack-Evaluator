"""Discover a CUDA toolkit that matches the installed PyTorch build.

This module is intentionally CUDA-runtime neutral: discovery reads package
metadata and runs ``nvcc --version`` but never queries a GPU.  It is used at
every vLLM boundary so users do not need Conda activation hooks or repository
shell scripts to expose a toolkit to FlashInfer JIT workers.
"""

from __future__ import annotations

import ctypes
import hashlib
from importlib import metadata, util
import os
import platform
from pathlib import Path
import re
import shutil
import subprocess
import sys
from logging import Logger
from threading import Lock
from typing import MutableMapping, Optional


class CudaToolkitConfigurationError(RuntimeError):
    """The configured CUDA toolkit is incomplete or mismatches PyTorch."""


_CUDA_LOCAL_VERSION = re.compile(r"(?:\+|-)cu(?P<digits>\d{2,3})(?:\D|$)")
_TORCH_CUDA_ASSIGNMENT = re.compile(
    r"^cuda(?:\s*:\s*[^=]+)?\s*=\s*['\"](?P<version>\d+(?:\.\d+)*)['\"]",
    re.MULTILINE,
)
_NVCC_RELEASE = re.compile(r"\brelease\s+(?P<major>\d+)(?:\.\d+)?\b")
_PRELOADED_RUNTIMES: set[Path] = set()
_PRELOAD_LOCK = Lock()
_VERSIONED_SHARED_LIBRARY = re.compile(
    r"^(?P<unversioned>lib.+\.so)\.(?P<version>\d.*)$"
)


def _cuda_major_from_local_version(version: str) -> int | None:
    match = _CUDA_LOCAL_VERSION.search(version)
    if match is None:
        return None
    digits = match.group("digits")
    # PyTorch wheel suffixes are cu118, cu128, cu130, and so on.  Retain
    # support for the shorter, major-only spelling as well.
    return int(digits) // 10 if len(digits) == 3 else int(digits)


def installed_torch_cuda_major() -> int | None:
    """Return PyTorch's compiled CUDA major without initializing CUDA."""

    loaded_torch = sys.modules.get("torch")
    loaded_cuda = getattr(getattr(loaded_torch, "version", None), "cuda", None)
    if loaded_cuda:
        return int(str(loaded_cuda).split(".", 1)[0])

    try:
        torch_dist = metadata.distribution("torch")
    except metadata.PackageNotFoundError:
        return None

    major = _cuda_major_from_local_version(torch_dist.version)
    if major is not None:
        return major

    # Some indexes strip the wheel's local version suffix.  Reading the
    # generated version.py is still metadata-only and avoids importing torch.
    version_file = Path(torch_dist.locate_file("torch/version.py"))
    try:
        version_text = version_file.read_text(encoding="utf-8")
    except OSError:
        return None
    match = _TORCH_CUDA_ASSIGNMENT.search(version_text)
    if match is None:
        return None
    return int(match.group("version").split(".", 1)[0])


def _nvcc_major(home: Path) -> int:
    nvcc = home / "bin" / "nvcc"
    if not nvcc.is_file() or not os.access(nvcc, os.X_OK):
        raise CudaToolkitConfigurationError(
            f"CUDA toolkit {home} has no executable bin/nvcc"
        )
    try:
        completed = subprocess.run(
            [str(nvcc), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CudaToolkitConfigurationError(
            f"cannot execute {nvcc}: {exc}"
        ) from exc
    output = f"{completed.stdout}\n{completed.stderr}"
    match = _NVCC_RELEASE.search(output)
    if completed.returncode != 0 or match is None:
        raise CudaToolkitConfigurationError(
            f"cannot determine CUDA version from {nvcc} --version"
        )
    return int(match.group("major"))


def _toolkit_library_dir(home: Path) -> Path | None:
    for name in ("lib64", "lib"):
        candidate = home / name
        if candidate.is_dir():
            return candidate
    return None


def _contains_cuda_runtime(library_dir: Path) -> bool:
    return any(library_dir.glob("libcudart.so*"))


def _toolkit_layout(home: Path) -> tuple[Path, Path]:
    """Resolve headers and libraries for system, wheel, or Conda layouts."""

    include_dir = home / "include"
    library_dir = _toolkit_library_dir(home)
    if (
        (include_dir / "cuda.h").is_file()
        and library_dir is not None
        and _contains_cuda_runtime(library_dir)
    ):
        return include_dir, library_dir

    targets_dir = home / "targets"
    machine = platform.machine().lower()
    preferred_names = {
        "x86_64": "x86_64-linux",
        "amd64": "x86_64-linux",
        "aarch64": "aarch64-linux",
        "arm64": "aarch64-linux",
    }
    target_roots = sorted(path for path in targets_dir.glob("*") if path.is_dir())
    preferred_name = preferred_names.get(machine)
    if preferred_name:
        target_roots.sort(key=lambda path: path.name != preferred_name)
    for target in target_roots:
        target_include = target / "include"
        target_library = _toolkit_library_dir(target)
        if (
            (target_include / "cuda.h").is_file()
            and target_library is not None
            and _contains_cuda_runtime(target_library)
        ):
            return target_include, target_library
    raise CudaToolkitConfigurationError(
        f"CUDA toolkit {home} has no complete CUDA headers/runtime layout"
    )


def _needs_linker_overlay(library_dir: Path) -> bool:
    """Return whether versioned wheel libraries lack linker-facing names."""

    for entry in library_dir.iterdir():
        match = _VERSIONED_SHARED_LIBRARY.match(entry.name)
        if match is None:
            continue
        if not (library_dir / match.group("unversioned")).exists():
            return True
    return False


def _overlay_base_roots() -> list[Path]:
    roots = [Path(sys.prefix) / ".rag_stack" / "cuda"]
    cache_home = os.environ.get("XDG_CACHE_HOME")
    user_cache = (
        Path(cache_home).expanduser()
        if cache_home
        else Path.home() / ".cache"
    )
    roots.append(user_cache / "rag-stack" / "cuda")
    return roots


def _ensure_directory_symlink(link: Path, target: Path) -> None:
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        raise CudaToolkitConfigurationError(
            f"CUDA linker overlay path is not a symlink: {link}"
        )
    try:
        link.symlink_to(target, target_is_directory=True)
    except FileExistsError:
        if not link.is_symlink() or link.resolve() != target.resolve():
            raise


def _ensure_file_symlink(link: Path, target: Path) -> None:
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        return
    try:
        link.symlink_to(target)
    except FileExistsError:
        if not link.exists():
            raise


def _standard_driver_library_dirs() -> list[Path]:
    return [
        Path("/usr/lib/x86_64-linux-gnu"),
        Path("/lib/x86_64-linux-gnu"),
        Path("/usr/lib/aarch64-linux-gnu"),
        Path("/lib/aarch64-linux-gnu"),
        Path("/usr/lib64"),
        Path("/lib64"),
    ]


def _find_cuda_driver_library(
    target: MutableMapping[str, str],
) -> Path | None:
    ldconfig = shutil.which("ldconfig")
    if ldconfig is not None:
        try:
            completed = subprocess.run(
                [ldconfig, "-p"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            completed = None
        if completed is not None and completed.returncode == 0:
            versioned: Path | None = None
            for line in completed.stdout.splitlines():
                match = re.match(
                    r"^\s*(?P<name>libcuda\.so(?:\.1)?)\s+.*=>\s+"
                    r"(?P<path>/\S+)\s*$",
                    line,
                )
                if match is None:
                    continue
                candidate = Path(match.group("path"))
                if "stubs" in candidate.parts or not candidate.is_file():
                    continue
                if match.group("name") == "libcuda.so":
                    return candidate.resolve()
                versioned = candidate.resolve()
            if versioned is not None:
                return versioned

    standard_dirs = _standard_driver_library_dirs()
    inherited_dirs = [
        Path(part)
        for part in target.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if part and "stubs" not in Path(part).parts
    ]
    for name in ("libcuda.so", "libcuda.so.1"):
        for directory in [*standard_dirs, *inherited_dirs]:
            candidate = directory / name
            if candidate.is_file() and "stubs" not in candidate.parts:
                return candidate.resolve()
    return None


def _materialize_linker_overlay(
    home: Path,
    cuda_major: int,
    *,
    include_dir: Path,
    source_lib: Path,
    target: MutableMapping[str, str],
) -> Path:
    """Build an environment-local CUDA_HOME overlay for wheel toolkits.

    NVIDIA's unified CUDA 13 wheels place libraries in ``lib`` and publish
    versioned runtime names such as ``libcudart.so.13``.  CUDA build frontends
    commonly link through ``$CUDA_HOME/lib64 -lcudart``.  The overlay supplies
    those conventional names with symlinks and never mutates site-packages.
    """

    nvcc_stat = (home / "bin" / "nvcc").stat()
    driver_library = _find_cuda_driver_library(target)
    identity = (
        f"{home.resolve()}:{include_dir.resolve()}:{source_lib.resolve()}:"
        f"{nvcc_stat.st_size}:{nvcc_stat.st_mtime_ns}:{driver_library}"
    ).encode("utf-8")
    digest = hashlib.sha256(identity).hexdigest()[:12]
    directory_name = f"toolkit-cu{cuda_major}-{digest}"

    last_error: OSError | None = None
    for base in _overlay_base_roots():
        overlay = base / directory_name
        try:
            overlay.mkdir(parents=True, exist_ok=True)
            _ensure_directory_symlink(overlay / "bin", home / "bin")
            _ensure_directory_symlink(overlay / "include", include_dir)
            _ensure_directory_symlink(overlay / "lib", source_lib)
            layout_root = include_dir.parent
            for name in ("nvvm", "cccl", "targets"):
                source = home / name
                if not source.exists():
                    source = layout_root / name
                if source.exists():
                    _ensure_directory_symlink(overlay / name, source)

            overlay_lib = overlay / "lib64"
            overlay_lib.mkdir(exist_ok=True)
            (overlay_lib / "stubs").mkdir(exist_ok=True)
            for source in sorted(source_lib.iterdir()):
                if not (source.is_file() or source.is_symlink()):
                    continue
                _ensure_file_symlink(overlay_lib / source.name, source)

            for versioned in sorted(overlay_lib.iterdir()):
                match = _VERSIONED_SHARED_LIBRARY.match(versioned.name)
                if match is None:
                    continue
                unversioned = overlay_lib / match.group("unversioned")
                _ensure_file_symlink(unversioned, versioned)
            if driver_library is not None:
                _ensure_file_symlink(
                    overlay_lib / "libcuda.so", driver_library
                )
                _ensure_file_symlink(
                    overlay_lib / "stubs" / "libcuda.so", driver_library
                )
            return overlay
        except OSError as exc:
            last_error = exc
            continue
    raise CudaToolkitConfigurationError(
        f"cannot create CUDA linker overlay for {home}: {last_error}"
    )


def _validate_toolkit(home: Path, expected_major: int | None) -> int:
    home = home.expanduser().resolve()
    actual_major = _nvcc_major(home)
    if expected_major is not None and actual_major != expected_major:
        raise CudaToolkitConfigurationError(
            f"CUDA toolkit {home} is CUDA {actual_major}, but installed PyTorch "
            f"was built for CUDA {expected_major}"
        )
    _toolkit_layout(home)
    return actual_major


def _wheel_toolkit_roots(cuda_major: int) -> list[Path]:
    """Return complete, unified CUDA wheel roots for one major version."""

    try:
        spec = util.find_spec(f"nvidia.cu{cuda_major}")
    except (ImportError, ModuleNotFoundError, ValueError):
        return []
    if spec is None or not spec.submodule_search_locations:
        return []
    return [Path(location) for location in spec.submodule_search_locations]


def _automatic_toolkit_roots(
    target: MutableMapping[str, str], cuda_major: int
) -> list[Path]:
    candidates: list[Path] = []
    nvcc = shutil.which("nvcc", path=target.get("PATH"))
    if nvcc:
        candidates.append(Path(nvcc).resolve().parent.parent)
    versioned_system_roots = sorted(Path("/usr/local").glob(f"cuda-{cuda_major}*"))
    candidates.extend(
        [
            Path(sys.prefix),
            *versioned_system_roots,
            Path("/usr/local/cuda"),
            *_wheel_toolkit_roots(cuda_major),
        ]
    )

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _prepend_env_path(
    target: MutableMapping[str, str], name: str, path: Path
) -> bool:
    value = str(path)
    current = target.get(name, "")
    parts = [part for part in current.split(os.pathsep) if part]
    if value in parts:
        return False
    target[name] = value if not current else f"{value}{os.pathsep}{current}"
    return True


def _preload_cuda_runtime(home: Path, cuda_major: int) -> None:
    """Make wheel libcudart visible to in-process vLLM imports on Linux."""

    library_dir = _toolkit_library_dir(home)
    if library_dir is None:
        return
    exact = library_dir / f"libcudart.so.{cuda_major}"
    candidates = [exact] if exact.is_file() else []
    candidates.extend(sorted(library_dir.glob(f"libcudart.so.{cuda_major}.*")))
    if not candidates:
        return
    runtime = candidates[0].resolve()
    with _PRELOAD_LOCK:
        if runtime in _PRELOADED_RUNTIMES:
            return
        try:
            ctypes.CDLL(str(runtime), mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            raise CudaToolkitConfigurationError(
                f"cannot load CUDA runtime {runtime}: {exc}"
            ) from exc
        _PRELOADED_RUNTIMES.add(runtime)


def configure_cuda_toolkit_env(
    env: Optional[MutableMapping[str, str]] = None,
    logger: Optional[Logger] = None,
) -> Path | None:
    """Expose a major-matched toolkit to vLLM and FlashInfer.

    A valid explicit ``CUDA_HOME``/``CUDA_PATH`` wins.  Otherwise a matching
    Conda/system toolkit is discovered, followed by a unified NVIDIA wheel
    toolkit (currently provided by the cu13 extra).  CUDA 12's split PyPI
    packages do not contain an ``nvcc`` driver, so they are deliberately not
    presented as a fake ``CUDA_HOME``.

    No toolkit is required for prebuilt kernels.  When none is available this
    function leaves the environment unchanged; a workload that genuinely
    needs source JIT then receives FlashInfer's normal missing-compiler error.
    A user-specified incomplete or wrong-major toolkit is always rejected.
    """

    target = os.environ if env is None else env
    expected_major = installed_torch_cuda_major()

    explicit_homes = {
        Path(str(value).strip()).expanduser().resolve()
        for value in (target.get("CUDA_HOME"), target.get("CUDA_PATH"))
        if value and str(value).strip()
    }
    if len(explicit_homes) > 1:
        raise CudaToolkitConfigurationError(
            "CUDA_HOME and CUDA_PATH name different toolkits"
        )

    selected: Path | None = None
    actual_major: int | None = None
    if explicit_homes:
        selected = explicit_homes.pop()
        actual_major = _validate_toolkit(selected, expected_major)
    elif expected_major is not None:
        rejected: list[str] = []
        for candidate in _automatic_toolkit_roots(target, expected_major):
            try:
                actual_major = _validate_toolkit(candidate, expected_major)
            except CudaToolkitConfigurationError as exc:
                rejected.append(str(exc))
                continue
            selected = candidate
            break
        if selected is None and logger is not None and rejected:
            logger.debug(
                "vLLM: no complete CUDA %s toolkit discovered (%s)",
                expected_major,
                "; ".join(rejected),
            )

    if selected is None or actual_major is None:
        return None

    include_dir, library_dir = _toolkit_layout(selected)
    has_root_layout = (
        include_dir == selected / "include"
        and library_dir in (selected / "lib64", selected / "lib")
    )
    if not has_root_layout or _needs_linker_overlay(library_dir):
        selected = _materialize_linker_overlay(
            selected,
            actual_major,
            include_dir=include_dir,
            source_lib=library_dir,
            target=target,
        )

    target["CUDA_HOME"] = str(selected)
    target["CUDA_PATH"] = str(selected)
    path_changed = _prepend_env_path(target, "PATH", selected / "bin")
    library_dir = _toolkit_library_dir(selected)
    ld_changed = False
    if library_dir is not None:
        ld_changed = _prepend_env_path(target, "LD_LIBRARY_PATH", library_dir)

    # A copied child environment is applied by exec/spawn.  For an in-process
    # vLLM import, LD_LIBRARY_PATH was not present at interpreter startup, so
    # preload the CUDA runtime by absolute path before importing the extension.
    if env is None:
        _preload_cuda_runtime(selected, actual_major)

    if logger is not None and (path_changed or ld_changed):
        logger.info(
            "vLLM: using CUDA %s toolkit at %s",
            actual_major,
            selected,
        )
    return selected


__all__ = [
    "CudaToolkitConfigurationError",
    "configure_cuda_toolkit_env",
    "installed_torch_cuda_major",
]
