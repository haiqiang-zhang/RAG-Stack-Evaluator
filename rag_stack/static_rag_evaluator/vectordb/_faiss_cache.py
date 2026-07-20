"""Cross-process safety helpers for the global FAISS cache."""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import shutil
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import faiss


READY_FORMAT_VERSION = 1
LOCAL_READ_CACHE_ENV = "RAG_STACK_FAISS_LOCAL_READ_CACHE_DIR"
PROCESS_READ_CACHE_MAX_BYTES_ENV = "RAG_STACK_FAISS_PROCESS_CACHE_MAX_BYTES"
DEFAULT_PROCESS_READ_CACHE_MAX_BYTES = 64 * 1024**3

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaissReadyIdentity:
    """Immutable identity of one atomically published FAISS pair."""

    index_path: str
    meta_path: str
    index_size: int
    meta_size: int
    index_mtime_ns: int
    meta_mtime_ns: int
    next_idx: int


@dataclass
class ReadOnlyFaissCacheEntry:
    """One process-local, immutable FAISS reader payload.

    Only read-only stores receive these entries. The index and normalized id
    maps are shared; the lock serializes each index's mutable search knobs with
    the search that consumes them.
    """

    identity: FaissReadyIdentity
    metadata: dict
    id_to_idx: dict
    idx_to_id: dict
    next_idx: int
    index: Any = None
    search_lock: Any = field(default_factory=threading.RLock)

    @property
    def resident_file_bytes(self) -> int:
        return self.identity.meta_size + (
            self.identity.index_size if self.index is not None else 0
        )


# FAISS's OpenMP thread count is process-global. Per-index locks alone cannot
# protect two different cached indexes from changing it underneath each other.
FAISS_OMP_SEARCH_LOCK = threading.RLock()

_PROCESS_READ_CACHE_LOCK = threading.RLock()
_PROCESS_READ_CACHE: "OrderedDict[FaissReadyIdentity, ReadOnlyFaissCacheEntry]" = (
    OrderedDict()
)
_PROCESS_READ_CACHE_BYTES = 0


def _process_read_cache_budget_bytes() -> int:
    raw = os.environ.get(PROCESS_READ_CACHE_MAX_BYTES_ENV, "").strip()
    if not raw:
        return DEFAULT_PROCESS_READ_CACHE_MAX_BYTES
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Ignoring invalid %s=%r; using %d bytes",
            PROCESS_READ_CACHE_MAX_BYTES_ENV,
            raw,
            DEFAULT_PROCESS_READ_CACHE_MAX_BYTES,
        )
        return DEFAULT_PROCESS_READ_CACHE_MAX_BYTES
    return max(0, value)


def _remove_process_entry_locked(identity: FaissReadyIdentity) -> None:
    global _PROCESS_READ_CACHE_BYTES
    entry = _PROCESS_READ_CACHE.pop(identity, None)
    if entry is not None:
        _PROCESS_READ_CACHE_BYTES -= entry.resident_file_bytes


def _discard_stale_process_entries_locked(identity: FaissReadyIdentity) -> None:
    for candidate in list(_PROCESS_READ_CACHE):
        if (
            candidate.index_path == identity.index_path
            and candidate.meta_path == identity.meta_path
            and candidate != identity
        ):
            _remove_process_entry_locked(candidate)


def _enforce_process_read_budget_locked(
    protected: FaissReadyIdentity | None = None,
) -> None:
    budget = _process_read_cache_budget_bytes()
    while _PROCESS_READ_CACHE_BYTES > budget and _PROCESS_READ_CACHE:
        victim = next(iter(_PROCESS_READ_CACHE))
        if victim == protected:
            if len(_PROCESS_READ_CACHE) == 1:
                break
            _PROCESS_READ_CACHE.move_to_end(victim)
            continue
        _remove_process_entry_locked(victim)
    if (
        protected is not None
        and _PROCESS_READ_CACHE_BYTES > budget
        and protected in _PROCESS_READ_CACHE
    ):
        # An entry larger than the whole budget remains usable by its current
        # store, but is not retained for a later evaluation.
        _remove_process_entry_locked(protected)


def _clear_read_only_faiss_process_cache_for_tests() -> None:
    """Reset process-local state. Test-only; production eviction is LRU."""
    global _PROCESS_READ_CACHE_BYTES
    with _PROCESS_READ_CACHE_LOCK:
        _PROCESS_READ_CACHE.clear()
        _PROCESS_READ_CACHE_BYTES = 0


def stage_faiss_read_file(source_path: str) -> str:
	"""Copy one immutable FAISS cache file to node-local storage for reading.

	The global cache remains authoritative and writable. When
	``RAG_STACK_FAISS_LOCAL_READ_CACHE_DIR`` is unset, this is a no-op. When it
	is set, readers share one atomic, versioned local copy keyed by the source
	path, size, and nanosecond mtime. Builders never call this helper.
	"""
	cache_root = os.environ.get(LOCAL_READ_CACHE_ENV, "").strip()
	if not cache_root:
		return source_path

	source_path = os.path.realpath(source_path)
	path_key = hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:20]
	key_dir = Path(cache_root).expanduser() / path_key
	key_dir.mkdir(parents=True, exist_ok=True)
	lock_path = key_dir / ".stage.lock"
	fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o664)
	try:
		fcntl.flock(fd, fcntl.LOCK_EX)
		for _attempt in range(2):
			source_stat = os.stat(source_path)
			version = f"{source_stat.st_size}-{source_stat.st_mtime_ns}"
			destination_dir = key_dir / version
			destination_dir.mkdir(parents=True, exist_ok=True)
			destination = destination_dir / Path(source_path).name
			for orphan in destination_dir.glob(f"{destination.name}.tmp.*"):
				try:
					orphan.unlink()
				except FileNotFoundError:
					pass
			try:
				if destination.stat().st_size == source_stat.st_size:
					return str(destination)
			except FileNotFoundError:
				pass

			token = f"tmp.{os.getpid()}.{uuid.uuid4().hex}"
			temporary = destination_dir / f"{destination.name}.{token}"
			started = time.monotonic()
			logger.info(
				"Staging FAISS read cache %s -> %s (%d bytes)",
				source_path,
				destination,
				source_stat.st_size,
			)
			try:
				shutil.copyfile(source_path, temporary)
				after_stat = os.stat(source_path)
				if (
					after_stat.st_size != source_stat.st_size
					or after_stat.st_mtime_ns != source_stat.st_mtime_ns
				):
					continue
				os.chmod(temporary, 0o444)
				os.replace(temporary, destination)
				logger.info(
					"Staged FAISS read cache in %.1fs: %s",
					time.monotonic() - started,
					destination,
				)
				return str(destination)
			finally:
				try:
					temporary.unlink()
				except FileNotFoundError:
					pass
		raise RuntimeError(
			f"FAISS cache file changed repeatedly while staging: {source_path}"
		)
	finally:
		fcntl.flock(fd, fcntl.LOCK_UN)
		os.close(fd)


@contextmanager
def faiss_cache_build_lock(cache_path: str) -> Iterator[None]:
	"""Serialize builders for one content-addressed index across NFS clients."""
	if not cache_path:
		yield
		return
	lock_path = f"{cache_path.rstrip(os.sep)}.build.lock"
	Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
	fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o664)
	try:
		fcntl.flock(fd, fcntl.LOCK_EX)
		yield
	finally:
		fcntl.flock(fd, fcntl.LOCK_UN)
		os.close(fd)


def faiss_cache_ready_path(meta_path: str) -> str:
	"""Return the completion-manifest path paired with ``meta_path``."""
	if meta_path.endswith(".meta.json"):
		return f"{meta_path[:-len('.meta.json')]}.ready.json"
	return f"{meta_path}.ready.json"


def invalidate_faiss_ready_marker(meta_path: str) -> None:
	"""Make a cache entry unavailable before mutating either persisted half."""
	try:
		os.unlink(faiss_cache_ready_path(meta_path))
	except FileNotFoundError:
		pass


def cleanup_orphan_faiss_temps(index_path: str, meta_path: str) -> None:
	"""Remove abandoned same-entry temp files while holding the build lock."""
	ready_path = faiss_cache_ready_path(meta_path)
	for final_path in (index_path, meta_path, ready_path):
		parent = Path(final_path).parent
		for candidate in parent.glob(f"{Path(final_path).name}.tmp.*"):
			try:
				candidate.unlink()
			except FileNotFoundError:
				pass


def _ready_payload(index_path: str, meta_path: str, next_idx: int) -> dict:
	index_stat = os.stat(index_path)
	meta_stat = os.stat(meta_path)
	return {
		"format_version": READY_FORMAT_VERSION,
		"index_file": os.path.basename(index_path),
		"meta_file": os.path.basename(meta_path),
		"index_size": index_stat.st_size,
		"meta_size": meta_stat.st_size,
		"index_mtime_ns": index_stat.st_mtime_ns,
		"meta_mtime_ns": meta_stat.st_mtime_ns,
		"next_idx": int(next_idx),
	}


def publish_faiss_ready_marker(
	index_path: str, meta_path: str, next_idx: int,
) -> None:
	"""Atomically publish a completion manifest after both pair files exist."""
	ready_path = faiss_cache_ready_path(meta_path)
	ready_tmp = f"{ready_path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
	try:
		with open(ready_tmp, "w", encoding="utf-8") as handle:
			json.dump(_ready_payload(index_path, meta_path, next_idx), handle)
			handle.flush()
			os.fsync(handle.fileno())
		os.replace(ready_tmp, ready_path)
	finally:
		try:
			os.unlink(ready_tmp)
		except FileNotFoundError:
			pass


def _faiss_ready_identity_if_complete(
    index_path: str,
    meta_path: str,
    *,
    expected_rows: int | None = None,
) -> FaissReadyIdentity | None:
    """Validate the small ready marker without parsing the large id-map JSON."""
    index_path = os.path.realpath(index_path)
    meta_path = os.path.realpath(meta_path)
    ready_path = faiss_cache_ready_path(meta_path)
    try:
        with open(ready_path, encoding="utf-8") as handle:
            ready = json.load(handle)
        index_stat = os.stat(index_path)
        meta_stat = os.stat(meta_path)
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(ready, dict):
        return None
    try:
        next_idx = int(ready["next_idx"])
        valid = (
            int(ready.get("format_version", -1)) == READY_FORMAT_VERSION
            and ready.get("index_file") == os.path.basename(index_path)
            and ready.get("meta_file") == os.path.basename(meta_path)
            and int(ready.get("index_size", -1)) == index_stat.st_size > 0
            and int(ready.get("meta_size", -1)) == meta_stat.st_size > 0
            and int(ready.get("index_mtime_ns", -1)) == index_stat.st_mtime_ns
            and int(ready.get("meta_mtime_ns", -1)) == meta_stat.st_mtime_ns
            and (expected_rows is None or next_idx == int(expected_rows))
        )
    except (KeyError, TypeError, ValueError):
        return None
    if not valid:
        return None
    return FaissReadyIdentity(
        index_path=index_path,
        meta_path=meta_path,
        index_size=index_stat.st_size,
        meta_size=meta_stat.st_size,
        index_mtime_ns=index_stat.st_mtime_ns,
        meta_mtime_ns=meta_stat.st_mtime_ns,
        next_idx=next_idx,
    )


def _parse_valid_metadata(identity: FaissReadyIdentity) -> dict | None:
    try:
        with open(identity.meta_path, encoding="utf-8") as handle:
            meta = json.load(handle)
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(meta, dict):
        return None
    try:
        next_idx = int(meta["next_idx"])
        id_to_idx = meta["id_to_idx"]
        idx_to_id_raw = meta["idx_to_id"]
        valid = (
            next_idx == identity.next_idx
            and isinstance(id_to_idx, dict)
            and isinstance(idx_to_id_raw, dict)
            and len(id_to_idx) == len(idx_to_id_raw) == next_idx
        )
        if not valid:
            return None
    except (KeyError, TypeError, ValueError):
        return None
    return meta


def _parse_ready_metadata(
    identity: FaissReadyIdentity,
) -> ReadOnlyFaissCacheEntry | None:
    meta = _parse_valid_metadata(identity)
    if meta is None:
        return None
    next_idx = int(meta["next_idx"])
    id_to_idx = meta["id_to_idx"]
    # JSON object keys are strings. Normalize once per resident entry;
    # rebuilding this 8.8M-row dict on every eval was itself material work.
    try:
        idx_to_id = {int(key): value for key, value in meta["idx_to_id"].items()}
    except (TypeError, ValueError):
        return None
    # Drop the raw string-key dict instead of retaining both 8.8M-row maps.
    # The process-cached metadata is explicitly read-only and normalized.
    meta["idx_to_id"] = idx_to_id
    return ReadOnlyFaissCacheEntry(
        identity=identity,
        metadata=meta,
        id_to_idx=id_to_idx,
        idx_to_id=idx_to_id,
        next_idx=next_idx,
    )


def _read_only_metadata_entry(
    identity: FaissReadyIdentity,
) -> ReadOnlyFaissCacheEntry | None:
    """Return one cached parsed/normalized metadata payload."""
    global _PROCESS_READ_CACHE_BYTES
    with _PROCESS_READ_CACHE_LOCK:
        _discard_stale_process_entries_locked(identity)
        entry = _PROCESS_READ_CACHE.get(identity)
        if entry is not None:
            _PROCESS_READ_CACHE.move_to_end(identity)
            return entry
        entry = _parse_ready_metadata(identity)
        if entry is None:
            return None
        _PROCESS_READ_CACHE[identity] = entry
        _PROCESS_READ_CACHE_BYTES += entry.resident_file_bytes
        _enforce_process_read_budget_locked(protected=identity)
        return entry


def load_read_only_faiss_pair(
    index_path: str,
    meta_path: str,
    *,
    reader: Callable[[str], Any],
    validator: Callable[[Any, dict], None] | None = None,
    expected_rows: int | None = None,
) -> ReadOnlyFaissCacheEntry | None:
    """Load or reuse one complete immutable FAISS pair in this process.

    The key is the canonical pair path plus the atomic ready marker's file size
    and nanosecond-mtime identity. A replacement at the same path therefore
    invalidates the resident object. Builders never call this function.
    """
    global _PROCESS_READ_CACHE_BYTES
    identity = _faiss_ready_identity_if_complete(
        index_path, meta_path, expected_rows=expected_rows,
    )
    if identity is None:
        return None
    # Serialize first-load work. Quality evaluations are synchronous; keeping
    # this path simple also prevents two readers from materializing the same
    # multi-dozen-GB HNSW object concurrently.
    with _PROCESS_READ_CACHE_LOCK:
        _discard_stale_process_entries_locked(identity)
        entry = _PROCESS_READ_CACHE.get(identity)
        if entry is None:
            entry = _parse_ready_metadata(identity)
            if entry is None:
                return None
            _PROCESS_READ_CACHE[identity] = entry
            _PROCESS_READ_CACHE_BYTES += entry.resident_file_bytes
        else:
            _PROCESS_READ_CACHE.move_to_end(identity)
        if entry.index is not None:
            if validator is not None:
                validator(entry.index, entry.metadata)
            return entry

        try:
            index = reader(identity.index_path)
            if int(getattr(index, "ntotal", -1)) != entry.next_idx:
                raise ValueError(
                    "FAISS cache cardinality mismatch: "
                    f"index={getattr(index, 'ntotal', None)}, "
                    f"metadata={entry.next_idx}"
                )
            if validator is not None:
                validator(index, entry.metadata)
        except Exception:
            # Parsed metadata can still be useful to a later ready check, but an
            # index that failed read/shape validation must never become resident.
            raise

        entry.index = index
        _PROCESS_READ_CACHE_BYTES += identity.index_size
        _PROCESS_READ_CACHE.move_to_end(identity)
        _enforce_process_read_budget_locked(protected=identity)
        return entry


def faiss_cache_metadata_if_ready(
    index_path: str,
    meta_path: str,
    *,
    expected_rows: int | None = None,
    process_cache: bool = False,
) -> dict | None:
    """Validate a published pair and return its already-parsed metadata.

    This deliberately avoids loading a multi-gigabyte FAISS binary. The writer
    publishes the manifest last while holding the per-key lock; callers that use
    this as a fast ingest check take that same lock around this function.
    """
    identity = _faiss_ready_identity_if_complete(
        index_path, meta_path, expected_rows=expected_rows,
    )
    if identity is None:
        return None
    if process_cache:
        entry = _read_only_metadata_entry(identity)
        return entry.metadata if entry is not None else None
    return _parse_valid_metadata(identity)


def faiss_cache_pair_ready(
	index_path: str,
	meta_path: str,
	*,
	expected_rows: int | None = None,
) -> bool:
	"""Return whether a complete, validated FAISS cache pair is available."""
	return faiss_cache_metadata_if_ready(
		index_path, meta_path, expected_rows=expected_rows,
	) is not None


def remove_incomplete_pair(index_path: str, meta_path: str) -> None:
	"""Remove a pair rejected while the caller holds its build lock."""
	for path in (index_path, meta_path, faiss_cache_ready_path(meta_path)):
		try:
			os.unlink(path)
		except FileNotFoundError:
			pass


def atomic_save_faiss_pair(index, index_path: str, meta_path: str, meta: dict) -> None:
	"""Write a FAISS binary and metadata via same-directory atomic renames.

	Callers hold :func:`faiss_cache_build_lock`, so readers that miss during the
	brief binary-before-metadata publish window re-check after acquiring the
	lock instead of starting a duplicate build.
	"""
	Path(index_path).parent.mkdir(parents=True, exist_ok=True)
	invalidate_faiss_ready_marker(meta_path)
	cleanup_orphan_faiss_temps(index_path, meta_path)
	token = f"tmp.{os.getpid()}.{uuid.uuid4().hex}"
	index_tmp = f"{index_path}.{token}"
	meta_tmp = f"{meta_path}.{token}"
	try:
		faiss.write_index(index, index_tmp)
		with open(index_tmp, "rb") as handle:
			os.fsync(handle.fileno())
		with open(meta_tmp, "w", encoding="utf-8") as handle:
			json.dump(meta, handle)
			handle.flush()
			os.fsync(handle.fileno())
		os.replace(index_tmp, index_path)
		os.replace(meta_tmp, meta_path)
		publish_faiss_ready_marker(
			index_path, meta_path, int(meta["next_idx"]),
		)
	finally:
		for path in (index_tmp, meta_tmp):
			try:
				os.unlink(path)
			except FileNotFoundError:
				pass
