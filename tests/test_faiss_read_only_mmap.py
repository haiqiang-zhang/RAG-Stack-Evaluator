from __future__ import annotations

import asyncio
import copy
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import faiss
import numpy as np
import pytest

from rag_stack.static_rag_evaluator.vectordb._faiss_cache import (
    LOCAL_READ_CACHE_ENV,
    PROCESS_READ_CACHE_MAX_BYTES_ENV,
    _clear_read_only_faiss_process_cache_for_tests,
    atomic_save_faiss_pair,
    faiss_cache_metadata_if_ready,
    load_read_only_faiss_pair,
    publish_faiss_ready_marker,
    stage_faiss_read_file,
)
from rag_stack.static_rag_evaluator.vectordb.faiss_hnsw import FaissHNSW
from rag_stack.static_rag_evaluator.vectordb.faiss_ivf import FaissIVF


@pytest.fixture(autouse=True)
def _reset_process_faiss_cache(monkeypatch):
    _clear_read_only_faiss_process_cache_for_tests()
    monkeypatch.delenv(PROCESS_READ_CACHE_MAX_BYTES_ENV, raising=False)
    yield
    _clear_read_only_faiss_process_cache_for_tests()


def _metadata(count: int, **extra):
    result = {
        "id_to_idx": {f"doc-{i}": i for i in range(count)},
        "idx_to_id": {str(i): f"doc-{i}" for i in range(count)},
        "next_idx": count,
    }
    result.update(extra)
    return result


class _FakeIndex:
    def __init__(self, ntotal: int, label: str):
        self.ntotal = ntotal
        self.label = label


class _DummyEmbedding:
    def get_text_embedding_batch(self, queries):
        return [[0.0, 0.0] for _ in queries]


class _ConcurrentSearchIndex:
    def __init__(self, kind: str):
        self.ntotal = 1
        self.kind = kind
        self.hnsw = SimpleNamespace(efSearch=0)
        self.nprobe = 0
        self.parallel_mode = 0
        self._state_lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.observed = []

    def search(self, queries, top_k):
        with self._state_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            knobs = (
                self.hnsw.efSearch
                if self.kind == "hnsw"
                else (self.nprobe, self.parallel_mode)
            )
            self.observed.append(knobs)
        time.sleep(0.03)
        with self._state_lock:
            self.active -= 1
        rows = len(queries)
        return (
            np.zeros((rows, top_k), dtype="float32"),
            np.zeros((rows, top_k), dtype="int64"),
        )


def _write_mock_pair(path: Path, *, label: str, count: int = 2):
    path.mkdir(parents=True, exist_ok=True)
    index_path = path / "mock.faiss"
    meta_path = path / "mock.meta.json"
    index_path.write_bytes((label * 17).encode("utf-8"))
    meta_path.write_text(json.dumps(_metadata(count, label=label)))
    publish_faiss_ready_marker(str(index_path), str(meta_path), count)
    return index_path, meta_path


def _hnsw_store(path: Path, read_only: bool) -> FaissHNSW:
    store = object.__new__(FaissHNSW)
    store.path = str(path)
    store.collection_name = "test"
    store.M = 8
    store.ef_construction = 40
    store.ef_search = 16
    store._read_only = read_only
    store.index = None
    store._id_to_idx = {}
    store._idx_to_id = {}
    store._next_idx = 0
    return store


def _ivf_store(path: Path, index_type: str, read_only: bool) -> FaissIVF:
    store = object.__new__(FaissIVF)
    store.path = str(path)
    store.collection_name = "test"
    store.d = 16
    store.index_type = index_type
    store.nlist = 8
    store.M = 4
    store.nbits = 4
    store.nprobe = 4
    store._read_only = read_only
    store.index = None
    store._is_trained = False
    store._id_to_idx = {}
    store._idx_to_id = {}
    store._next_idx = 0
    return store


def test_local_read_staging_is_opt_in(monkeypatch, tmp_path):
    source = tmp_path / "source.faiss"
    source.write_bytes(b"index bytes")
    monkeypatch.delenv(LOCAL_READ_CACHE_ENV, raising=False)
    assert stage_faiss_read_file(str(source)) == str(source)


def test_local_read_staging_reuses_and_versions_atomic_copy(
    monkeypatch, tmp_path
):
    import rag_stack.static_rag_evaluator.vectordb._faiss_cache as cache_module

    source = tmp_path / "source" / "test.faiss"
    source.parent.mkdir()
    source.write_bytes(b"first version")
    local_cache = tmp_path / "local"
    monkeypatch.setenv(LOCAL_READ_CACHE_ENV, str(local_cache))

    copy_calls = []
    real_copyfile = cache_module.shutil.copyfile

    def tracked_copyfile(source_path, destination_path):
        copy_calls.append((source_path, destination_path))
        return real_copyfile(source_path, destination_path)

    monkeypatch.setattr(cache_module.shutil, "copyfile", tracked_copyfile)
    first = Path(stage_faiss_read_file(str(source)))
    orphan = first.parent / f"{first.name}.tmp.abandoned"
    orphan.write_bytes(b"partial")
    again = Path(stage_faiss_read_file(str(source)))
    assert first == again
    assert first.read_bytes() == b"first version"
    assert not orphan.exists()
    assert len(copy_calls) == 1

    previous_mtime_ns = source.stat().st_mtime_ns
    source.write_bytes(b"newer version")
    source.touch()
    if source.stat().st_mtime_ns == previous_mtime_ns:
        source_stat = source.stat()
        os.utime(
            source,
            ns=(source_stat.st_atime_ns, previous_mtime_ns + 1),
        )
    second = Path(stage_faiss_read_file(str(source)))
    assert second != first
    assert second.read_bytes() == b"newer version"
    assert len(copy_calls) == 2
    assert not list(local_cache.rglob("*.tmp.*"))


def test_concurrent_local_staging_copies_once(monkeypatch, tmp_path):
    import rag_stack.static_rag_evaluator.vectordb._faiss_cache as cache_module

    source = tmp_path / "source.faiss"
    source.write_bytes(b"x" * 1024 * 1024)
    monkeypatch.setenv(LOCAL_READ_CACHE_ENV, str(tmp_path / "local"))
    real_copyfile = cache_module.shutil.copyfile
    copy_count = 0
    count_lock = threading.Lock()

    def slow_copyfile(source_path, destination_path):
        nonlocal copy_count
        with count_lock:
            copy_count += 1
        time.sleep(0.05)
        return real_copyfile(source_path, destination_path)

    monkeypatch.setattr(cache_module.shutil, "copyfile", slow_copyfile)
    with ThreadPoolExecutor(max_workers=4) as executor:
        destinations = list(
            executor.map(lambda _: stage_faiss_read_file(str(source)), range(4))
        )
    assert len(set(destinations)) == 1
    assert copy_count == 1


def test_process_read_cache_reuses_index_and_normalized_id_maps(tmp_path):
    index_path, meta_path = _write_mock_pair(tmp_path / "pair", label="a")
    reads = []

    def reader(path):
        reads.append(path)
        return _FakeIndex(2, "loaded")

    primed = faiss_cache_metadata_if_ready(
        str(index_path), str(meta_path), process_cache=True,
    )
    first = load_read_only_faiss_pair(
        str(index_path), str(meta_path), reader=reader,
    )
    second = load_read_only_faiss_pair(
        str(index_path), str(meta_path), reader=reader,
    )

    assert first is second
    assert first.index is second.index
    assert first.id_to_idx is second.id_to_idx
    assert first.idx_to_id is second.idx_to_id
    assert first.idx_to_id == {0: "doc-0", 1: "doc-1"}
    assert primed is first.metadata
    assert first.metadata["idx_to_id"] is first.idx_to_id
    assert reads == [str(index_path.resolve())]


def test_process_read_cache_revalidates_resident_index_for_each_caller(tmp_path):
    index_path, meta_path = _write_mock_pair(tmp_path / "pair", label="a")
    index = _FakeIndex(2, "shape-a")

    def accept_shape_a(loaded, _meta):
        assert loaded.label == "shape-a"

    first = load_read_only_faiss_pair(
        str(index_path),
        str(meta_path),
        reader=lambda _path: index,
        validator=accept_shape_a,
    )
    assert first.index is index

    def reject_resident(loaded, _meta):
        assert loaded is index
        raise ValueError("caller expects a different shape")

    with pytest.raises(ValueError, match="different shape"):
        load_read_only_faiss_pair(
            str(index_path),
            str(meta_path),
            reader=lambda _path: pytest.fail("resident hit must not reread"),
            validator=reject_resident,
        )


def test_process_read_cache_invalidates_replaced_pair(tmp_path):
    index_path, meta_path = _write_mock_pair(tmp_path / "pair", label="a")
    reads = []

    def reader(_path):
        label = chr(ord("a") + len(reads))
        reads.append(label)
        return _FakeIndex(2, label)

    first = load_read_only_faiss_pair(
        str(index_path), str(meta_path), reader=reader,
    )
    old_mtime = index_path.stat().st_mtime_ns
    index_path.write_bytes(b"replacement-index-version")
    if index_path.stat().st_mtime_ns == old_mtime:
        stat = index_path.stat()
        os.utime(index_path, ns=(stat.st_atime_ns, old_mtime + 1))
    publish_faiss_ready_marker(str(index_path), str(meta_path), 2)
    second = load_read_only_faiss_pair(
        str(index_path), str(meta_path), reader=reader,
    )

    assert first is not second
    assert first.index is not second.index
    assert reads == ["a", "b"]


def test_process_read_cache_evicts_lru_by_pair_file_bytes(
    monkeypatch, tmp_path,
):
    pair_a = _write_mock_pair(tmp_path / "a", label="a")
    pair_b = _write_mock_pair(tmp_path / "b", label="b")
    weights = [
        index_path.stat().st_size + meta_path.stat().st_size
        for index_path, meta_path in (pair_a, pair_b)
    ]
    assert max(weights) < sum(weights)
    monkeypatch.setenv(
        PROCESS_READ_CACHE_MAX_BYTES_ENV,
        str(max(weights)),
    )
    reads = []

    def load(pair, label):
        index_path, meta_path = pair
        return load_read_only_faiss_pair(
            str(index_path),
            str(meta_path),
            reader=lambda _path: (
                reads.append(label) or _FakeIndex(2, label)
            ),
        )

    load(pair_a, "a")
    load(pair_b, "b")
    load(pair_a, "a")
    assert reads == ["a", "b", "a"]


def _query_store(store, **kwargs):
    return asyncio.run(store.query(["question"], 1, num_threads=1, **kwargs))


def test_hnsw_shared_entry_serializes_ef_search_with_search(tmp_path):
    index = _ConcurrentSearchIndex("hnsw")
    shared_lock = threading.RLock()
    stores = []
    for _ in range(2):
        store = _hnsw_store(tmp_path / "unused", read_only=True)
        store.embedding = _DummyEmbedding()
        store.similarity_metric = "l2"
        store.index = index
        store._idx_to_id = {0: "doc-0"}
        store._search_lock = shared_lock
        stores.append(store)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(_query_store, stores[0], ef_search=8),
            executor.submit(_query_store, stores[1], ef_search=32),
        ]
        [future.result() for future in futures]

    assert index.max_active == 1
    assert set(index.observed) == {8, 32}


def test_ivf_shared_entry_serializes_runtime_knobs_with_search(tmp_path):
    index = _ConcurrentSearchIndex("ivf")
    shared_lock = threading.RLock()
    stores = []
    for _ in range(2):
        store = _ivf_store(tmp_path / "unused", "flat", read_only=True)
        store.embedding = _DummyEmbedding()
        store.similarity_metric = "l2"
        store.index = index
        store._idx_to_id = {0: "doc-0"}
        store._search_lock = shared_lock
        stores.append(store)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                _query_store, stores[0], nprobe=1, parallel_mode=0,
            ),
            executor.submit(
                _query_store, stores[1], nprobe=8, parallel_mode=1,
            ),
        ]
        [future.result() for future in futures]

    assert index.max_active == 1
    assert set(index.observed) == {(1, 0), (8, 1)}


def test_hnsw_read_only_staged_load_matches_writable(monkeypatch, tmp_path):
    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(512, 16)).astype("float32")
    queries = rng.normal(size=(16, 16)).astype("float32")

    index = faiss.IndexHNSWFlat(16, 8, faiss.METRIC_L2)
    index.hnsw.efConstruction = 40
    index.hnsw.efSearch = 16
    index.add(vectors)

    cache_path = tmp_path / "hnsw"
    writable = _hnsw_store(cache_path, read_only=False)
    atomic_save_faiss_pair(
        index,
        writable._index_path(),
        writable._meta_path(),
        _metadata(
            len(vectors),
            M=8,
            ef_construction=40,
            ef_search=16,
        ),
    )

    observed_flags = []
    observed_paths = []
    real_read_index = faiss.read_index

    def tracked_read_index(path, flags=0):
        observed_paths.append(path)
        observed_flags.append(flags)
        return real_read_index(path, flags)

    monkeypatch.setattr(faiss, "read_index", tracked_read_index)
    assert writable._load_if_exists()
    local_cache = tmp_path / "local-read-cache"
    monkeypatch.setenv(LOCAL_READ_CACHE_ENV, str(local_cache))
    mapped = _hnsw_store(cache_path, read_only=True)
    assert mapped._load_if_exists()
    mapped_again = _hnsw_store(cache_path, read_only=True)
    assert mapped_again._load_if_exists()
    assert mapped_again.index is mapped.index
    assert mapped_again._id_to_idx is mapped._id_to_idx
    assert mapped_again._idx_to_id is mapped._idx_to_id
    assert mapped_again._search_lock is mapped._search_lock

    for ef_search in (8, 32):
        writable.index.hnsw.efSearch = ef_search
        mapped.index.hnsw.efSearch = ef_search
        expected_distances, expected_ids = writable.index.search(queries, 10)
        actual_distances, actual_ids = mapped.index.search(queries, 10)
        np.testing.assert_array_equal(actual_ids, expected_ids)
        np.testing.assert_allclose(actual_distances, expected_distances)

    # The available Conda FAISS 1.14.1 and 1.14.2 builds both segfault when
    # IndexHNSWFlat is read with IO_FLAG_MMAP_IFC. Never select that flag;
    # read-only speedup comes from the tested node-local staging path.
    assert observed_flags == [0, 0]
    assert Path(mapped._index_path()) != Path(observed_paths[1])
    assert Path(observed_paths[1]).is_relative_to(local_cache)


@pytest.mark.parametrize("index_type", ["flat", "pq"])
def test_ivf_read_only_mmap_matches_writable_load(
    monkeypatch, tmp_path, index_type
):
    rng = np.random.default_rng(11)
    vectors = rng.normal(size=(2048, 16)).astype("float32")
    queries = rng.normal(size=(16, 16)).astype("float32")
    factory = "IVF8,Flat" if index_type == "flat" else "IVF8,PQ4x4"
    index = faiss.index_factory(16, factory, faiss.METRIC_L2)
    index.train(vectors)
    index.make_direct_map()
    index.add(vectors)
    index.nprobe = 4

    cache_path = tmp_path / index_type
    writable = _ivf_store(cache_path, index_type, read_only=False)
    atomic_save_faiss_pair(
        index,
        writable._index_path(),
        writable._meta_path(),
        _metadata(
            len(vectors),
            is_trained=True,
            index_type=index_type,
            nlist=8,
            nprobe=4,
        ),
    )

    observed_flags = []
    observed_paths = []
    real_read_index = faiss.read_index

    def tracked_read_index(path, flags=0):
        observed_paths.append(path)
        observed_flags.append(flags)
        return real_read_index(path, flags)

    monkeypatch.setattr(faiss, "read_index", tracked_read_index)
    assert writable._load_if_exists()
    local_cache = tmp_path / "local-read-cache"
    monkeypatch.setenv(LOCAL_READ_CACHE_ENV, str(local_cache))
    mapped = _ivf_store(cache_path, index_type, read_only=True)
    assert mapped._load_if_exists()
    mapped_again = _ivf_store(cache_path, index_type, read_only=True)
    assert mapped_again._load_if_exists()
    assert mapped_again.index is mapped.index
    assert mapped_again._id_to_idx is mapped._id_to_idx
    assert mapped_again._idx_to_id is mapped._idx_to_id
    assert mapped_again._search_lock is mapped._search_lock

    for nprobe in (1, 8):
        writable.index.nprobe = nprobe
        mapped.index.nprobe = nprobe
        expected_distances, expected_ids = writable.index.search(queries, 10)
        actual_distances, actual_ids = mapped.index.search(queries, 10)
        np.testing.assert_array_equal(actual_ids, expected_ids)
        np.testing.assert_allclose(actual_distances, expected_distances)

    assert observed_flags == [0, faiss.IO_FLAG_MMAP]
    assert Path(mapped._index_path()) != Path(observed_paths[1])
    assert Path(observed_paths[1]).is_relative_to(local_cache)


@pytest.mark.parametrize(
    ("module_name", "store_factory"),
    [
        (
            "rag_stack.static_rag_evaluator.vectordb.faiss_hnsw",
            lambda path: _hnsw_store(path, read_only=True),
        ),
        (
            "rag_stack.static_rag_evaluator.vectordb.faiss_ivf",
            lambda path: _ivf_store(path, "flat", read_only=True),
        ),
    ],
)
def test_read_only_stage_failure_is_fatal(
    monkeypatch, tmp_path, module_name, store_factory
):
    import importlib

    module = importlib.import_module(module_name)
    store = store_factory(tmp_path / "cache")
    monkeypatch.setattr(
        module,
        "load_read_only_faiss_pair",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("staging failed")
        ),
    )
    with pytest.raises(RuntimeError, match="required read-only FAISS"):
        store._load_if_exists()


@pytest.mark.parametrize("store_type", [FaissHNSW, FaissIVF])
def test_read_only_constructor_requires_complete_cache(
    monkeypatch, tmp_path, store_type
):
    from rag_stack.static_rag_evaluator.vectordb.base import BaseVectorStore

    def fake_base_init(self, _model, _metric, _batch, embedding_dim):
        self.embedding_dim = embedding_dim

    monkeypatch.setattr(BaseVectorStore, "__init__", fake_base_init)
    with pytest.raises(RuntimeError, match="cache is unavailable"):
        store_type(
            None,
            path=str(tmp_path / "missing"),
            embedding_dim=16,
            read_only=True,
        )


@pytest.mark.parametrize("store_type", [FaissHNSW, FaissIVF])
def test_read_only_faiss_store_rejects_add(store_type, tmp_path):
    if store_type is FaissHNSW:
        store = _hnsw_store(tmp_path / "hnsw", read_only=True)
    else:
        store = _ivf_store(tmp_path / "ivf", "flat", read_only=True)

    with pytest.raises(RuntimeError, match="read-only FAISS"):
        store.add_embedding(["doc-0"], [[0.0] * 16])


@pytest.mark.parametrize("db_type", ["faiss_hnsw", "FaissIVF"])
def test_yaml_loader_marks_only_faiss_retrieval_read_only(
    monkeypatch, db_type
):
    import rag_stack.static_rag_evaluator.vectordb as vectordb_module

    config = {
        "vectordb": [
            {
                "name": "candidate",
                "db_type": db_type,
                "embedding_model": "test-embedding",
            }
        ]
    }
    monkeypatch.setattr(
        vectordb_module,
        "load_yaml_config",
        lambda _path: copy.deepcopy(config),
    )
    monkeypatch.setattr(
        vectordb_module,
        "load_vectordb",
        lambda name, **kwargs: (name, kwargs),
    )

    _, writable_kwargs = vectordb_module.load_vectordb_from_yaml(
        "unused.yaml", "candidate", "unused-project"
    )
    _, read_only_kwargs = vectordb_module.load_vectordb_from_yaml(
        "unused.yaml", "candidate", "unused-project", read_only=True
    )
    assert writable_kwargs["read_only"] is False
    assert read_only_kwargs["read_only"] is True


def test_yaml_loader_does_not_inject_read_only_into_non_faiss(
    monkeypatch,
):
    import rag_stack.static_rag_evaluator.vectordb as vectordb_module

    config = {
        "vectordb": [
            {
                "name": "candidate",
                "db_type": "chroma",
                "embedding_model": "test-embedding",
            }
        ]
    }
    monkeypatch.setattr(
        vectordb_module,
        "load_yaml_config",
        lambda _path: copy.deepcopy(config),
    )
    monkeypatch.setattr(
        vectordb_module,
        "load_vectordb",
        lambda name, **kwargs: (name, kwargs),
    )

    _, kwargs = vectordb_module.load_vectordb_from_yaml(
        "unused.yaml", "candidate", "unused-project", read_only=True
    )
    assert "read_only" not in kwargs


def test_cached_ingest_check_reuses_validated_metadata(monkeypatch, tmp_path):
    import rag_stack.static_rag_evaluator.vectordb._faiss_cache as cache_module
    from rag_stack.static_rag_evaluator.static_rag_evaluator import (
        StaticRAGEvaluatorQualityOnly,
    )

    calls = []

    def ready_metadata(
        index_path,
        meta_path,
        *,
        expected_rows=None,
        process_cache=False,
    ):
        calls.append((index_path, meta_path, expected_rows, process_cache))
        return {
            "M": 8,
            "ef_construction": 40,
            "next_idx": expected_rows,
            "id_to_idx": {},
            "idx_to_id": {},
        }

    monkeypatch.setattr(
        cache_module,
        "faiss_cache_metadata_if_ready",
        ready_metadata,
    )
    config = {
        "db_type": "faiss_hnsw",
        "path": str(tmp_path / "cache"),
        "collection_name": "test",
        "M": 8,
        "ef_construction": 40,
    }
    assert StaticRAGEvaluatorQualityOnly._cached_faiss_vectordb_ready(config, 123)
    assert len(calls) == 1
    assert calls[0][2] == 123
    assert calls[0][3] is True
