from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import time

import faiss
import pytest

from rag_stack_evaluator.static_rag_evaluator.vectordb._faiss_cache import (
	atomic_save_faiss_pair,
	faiss_cache_build_lock,
	faiss_cache_pair_ready,
	faiss_cache_ready_path,
)


def _hold_lock(path: str, acquired, release) -> None:
	with faiss_cache_build_lock(path):
		acquired.set()
		release.wait(10)


def _measure_lock_wait(path: str, output) -> None:
	started = time.monotonic()
	with faiss_cache_build_lock(path):
		output.put(time.monotonic() - started)


def test_build_lock_serializes_processes(tmp_path):
	# This helper is POSIX-only (it exercises fcntl.flock), so use fork. Spawn
	# re-imports the full evaluator/test graph in each child and can spend over
	# a minute on shared-NFS imports before it reaches the lock under load.
	ctx = mp.get_context("fork")
	acquired = ctx.Event()
	release = ctx.Event()
	output = ctx.Queue()
	cache_path = str(tmp_path / "hash" / "params")
	owner = ctx.Process(target=_hold_lock, args=(cache_path, acquired, release))
	waiter = ctx.Process(target=_measure_lock_wait, args=(cache_path, output))
	owner.start()
	assert acquired.wait(10)
	waiter.start()
	time.sleep(0.4)
	assert waiter.is_alive()
	release.set()
	owner.join(10)
	waiter.join(10)
	assert owner.exitcode == 0
	assert waiter.exitcode == 0
	assert output.get(timeout=2) >= 0.3


def test_atomic_save_publishes_complete_pair_and_cleans_temps(tmp_path):
	index = faiss.IndexFlatL2(2)
	index_path = tmp_path / "cache" / "my_index.ivf.faiss"
	meta_path = tmp_path / "cache" / "my_index.ivf.meta.json"
	atomic_save_faiss_pair(
		index,
		str(index_path),
		str(meta_path),
		{"id_to_idx": {}, "idx_to_id": {}, "next_idx": 0},
	)
	assert faiss.read_index(str(index_path)).d == 2
	assert meta_path.is_file()
	assert faiss_cache_pair_ready(str(index_path), str(meta_path), expected_rows=0)
	assert not faiss_cache_pair_ready(str(index_path), str(meta_path), expected_rows=1)
	assert not [name for name in os.listdir(index_path.parent) if ".tmp." in name]


def test_nonempty_pair_without_ready_marker_is_not_complete(tmp_path):
	index_path = tmp_path / "cache" / "my_index.ivf.faiss"
	meta_path = tmp_path / "cache" / "my_index.ivf.meta.json"
	index_path.parent.mkdir()
	faiss.write_index(faiss.IndexFlatL2(2), str(index_path))
	meta_path.write_text('{"id_to_idx": {}, "idx_to_id": {}, "next_idx": 0}')
	assert not faiss_cache_pair_ready(str(index_path), str(meta_path))


def test_atomic_save_cleans_orphan_temps(tmp_path):
	index = faiss.IndexFlatL2(2)
	index_path = tmp_path / "cache" / "my_index.ivf.faiss"
	meta_path = tmp_path / "cache" / "my_index.ivf.meta.json"
	index_path.parent.mkdir()
	for final_path in (
		index_path, meta_path, faiss_cache_ready_path(str(meta_path)),
	):
		orphan = f"{final_path}.tmp.dead"
		open(orphan, "wb").close()
	atomic_save_faiss_pair(
		index, str(index_path), str(meta_path),
		{"id_to_idx": {}, "idx_to_id": {}, "next_idx": 0},
	)
	assert not [name for name in os.listdir(index_path.parent) if ".tmp." in name]


def test_failed_publish_never_leaves_ready_marker(monkeypatch, tmp_path):
	index = faiss.IndexFlatL2(2)
	index_path = tmp_path / "cache" / "my_index.ivf.faiss"
	meta_path = tmp_path / "cache" / "my_index.ivf.meta.json"
	meta = {"id_to_idx": {}, "idx_to_id": {}, "next_idx": 0}
	atomic_save_faiss_pair(index, str(index_path), str(meta_path), meta)

	real_replace = os.replace

	def fail_meta_replace(source, destination):
		if destination == str(meta_path):
			raise OSError("injected publish failure")
		return real_replace(source, destination)

	monkeypatch.setattr(os, "replace", fail_meta_replace)
	with pytest.raises(OSError, match="injected publish failure"):
		atomic_save_faiss_pair(index, str(index_path), str(meta_path), meta)
	assert not os.path.exists(faiss_cache_ready_path(str(meta_path)))
	assert not faiss_cache_pair_ready(str(index_path), str(meta_path))


@pytest.mark.parametrize("store_name", ["ivf", "hnsw"])
def test_shared_faiss_delete_is_explicitly_unsupported(store_name):
	if store_name == "ivf":
		from rag_stack_evaluator.static_rag_evaluator.vectordb.faiss_ivf import FaissIVF
		store_type = FaissIVF
	else:
		from rag_stack_evaluator.static_rag_evaluator.vectordb.faiss_hnsw import FaissHNSW
		store_type = FaissHNSW
	store = object.__new__(store_type)
	with pytest.raises(NotImplementedError, match="immutable shared FAISS"):
		asyncio.run(store.delete(["doc-1"]))


def test_hnsw_failed_save_rolls_back_in_memory_state(monkeypatch, tmp_path):
	from rag_stack_evaluator.static_rag_evaluator.vectordb.faiss_hnsw import FaissHNSW

	store = object.__new__(FaissHNSW)
	store.path = str(tmp_path / "cache")
	store.collection_name = "default"
	store.similarity_metric = "l2"
	store.d = 2
	store.M = 2
	store.ef_construction = 8
	store.ef_search = 4
	store._faiss_indexing_thread = 1
	store.index = None
	store._id_to_idx = {}
	store._idx_to_id = {}
	store._next_idx = 0
	monkeypatch.setattr(store, "_load_if_exists", lambda: False)
	monkeypatch.setattr(
		store, "_save", lambda: (_ for _ in ()).throw(RuntimeError("save failed")),
	)
	with pytest.raises(RuntimeError, match="save failed"):
		store.add_embedding(["doc-1"], [[1.0, 2.0]])
	assert store.index is None
	assert store._id_to_idx == {}
	assert store._idx_to_id == {}
	assert store._next_idx == 0


def test_failed_append_preserves_previous_complete_generation(monkeypatch, tmp_path):
	import numpy as np

	from rag_stack_evaluator.static_rag_evaluator.vectordb.faiss_hnsw import FaissHNSW

	store = object.__new__(FaissHNSW)
	store.path = str(tmp_path / "cache")
	store.collection_name = "default"
	store.similarity_metric = "l2"
	store.d = 2
	store.M = 2
	store.ef_construction = 8
	store.ef_search = 4
	store._faiss_indexing_thread = 1
	store.index = faiss.IndexHNSWFlat(2, 2)
	store.index.add(np.asarray([[1.0, 2.0]], dtype="float32"))
	store._id_to_idx = {"doc-1": 0}
	store._idx_to_id = {0: "doc-1"}
	store._next_idx = 1
	store._save()
	index_path = store._index_path()
	meta_path = store._meta_path()
	assert faiss_cache_pair_ready(index_path, meta_path, expected_rows=1)

	monkeypatch.setattr(store, "_load_if_exists", lambda: True)
	monkeypatch.setattr(
		store, "_save", lambda: (_ for _ in ()).throw(RuntimeError("save failed")),
	)
	with pytest.raises(RuntimeError, match="save failed"):
		store.add_embedding(["doc-2"], [[3.0, 4.0]])
	assert faiss_cache_pair_ready(index_path, meta_path, expected_rows=1)
