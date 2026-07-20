"""Tests for Faiss IVF-PQ vector store.

Run with: python tests/test_faiss_ivf.py
(pytest may segfault due to faiss + pytest interaction on some platforms)
"""

import asyncio
import tempfile
import shutil
import sys

import faiss  # import first to avoid segfault
import numpy as np

from rag_stack.static_rag_evaluator.vectordb.faiss_ivf import FaissIVF


def random_embeddings(n, d=64):
    return np.random.rand(n, d).astype(np.float32).tolist()


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


N_VECTORS = 2000  # >= 624 (16 centroids * 39) to avoid FAISS training warnings


def make_store(tmp_dir, embedding_dim=64, nlist=4, M=8, nbits=4, similarity_metric="l2"):
    return FaissIVF(
        embedding_model="mock",
        similarity_metric=similarity_metric,
        collection_name="test",
        path=tmp_dir,
        embedding_dim=embedding_dim,
        index_type="pq",
        nlist=nlist,
        M=M,
        nbits=nbits,
        nprobe=nlist,
    )


def test_add_embedding():
    tmp = tempfile.mkdtemp()
    try:
        store = make_store(tmp)
        ids = [f"id_{i}" for i in range(N_VECTORS)]
        embeddings = random_embeddings(N_VECTORS)
        store.add_embedding(ids, embeddings)
        assert store.index.ntotal == N_VECTORS
        print("PASS: add_embedding")
    finally:
        shutil.rmtree(tmp)


def test_skip_duplicates():
    tmp = tempfile.mkdtemp()
    try:
        store = make_store(tmp)
        ids = [f"id_{i}" for i in range(N_VECTORS)]
        embeddings = random_embeddings(N_VECTORS)
        store.add_embedding(ids, embeddings)
        store.add_embedding(ids[:10], embeddings[:10])
        assert store.index.ntotal == N_VECTORS  # no new vectors added
        print("PASS: skip_duplicates")
    finally:
        shutil.rmtree(tmp)


def test_search():
    tmp = tempfile.mkdtemp()
    try:
        store = make_store(tmp)
        embeddings = random_embeddings(N_VECTORS)
        ids = [f"id_{i}" for i in range(N_VECTORS)]
        store.add_embedding(ids, embeddings)

        query_vec = np.array([embeddings[0]], dtype=np.float32)
        D, I = store.index.search(query_vec, 5)
        assert I[0][0] == 0
        print("PASS: search")
    finally:
        shutil.rmtree(tmp)


def test_fetch():
    tmp = tempfile.mkdtemp()
    try:
        store = make_store(tmp)
        embeddings = random_embeddings(N_VECTORS)
        ids = [f"id_{i}" for i in range(N_VECTORS)]
        store.add_embedding(ids, embeddings)

        results = run(store.fetch(["id_0", "id_1", "nonexistent"]))
        assert len(results[0]) == 64
        assert len(results[1]) == 64
        assert results[2] == []
        print("PASS: fetch")
    finally:
        shutil.rmtree(tmp)


def test_is_exist():
    tmp = tempfile.mkdtemp()
    try:
        store = make_store(tmp)
        embeddings = random_embeddings(N_VECTORS)
        ids = [f"id_{i}" for i in range(N_VECTORS)]
        store.add_embedding(ids, embeddings)

        results = run(store.is_exist(["id_0", "nonexistent", "id_5"]))
        assert results == [True, False, True]
        print("PASS: is_exist")
    finally:
        shutil.rmtree(tmp)


def test_delete():
    # Immutable shared FAISS caches (cache-management change): delete is
    # deliberately unsupported — the store must refuse loudly instead of
    # mutating an index that other trials share.
    tmp = tempfile.mkdtemp()
    try:
        store = make_store(tmp)
        embeddings = random_embeddings(N_VECTORS)
        ids = [f"id_{i}" for i in range(N_VECTORS)]
        store.add_embedding(ids, embeddings)

        try:
            run(store.delete(["id_0", "id_1"]))
            raise AssertionError("delete must raise on immutable shared caches")
        except NotImplementedError:
            pass

        print("PASS: delete")
    finally:
        shutil.rmtree(tmp)


def test_persistence():
    tmp = tempfile.mkdtemp()
    try:
        store = make_store(tmp)
        embeddings = random_embeddings(N_VECTORS)
        ids = [f"id_{i}" for i in range(N_VECTORS)]
        store.add_embedding(ids, embeddings)

        # Reload from disk (delete is unsupported on immutable shared caches
        # — persistence is exercised without it)
        store2 = make_store(tmp)
        assert store2.index.ntotal == N_VECTORS
        results = run(store2.is_exist(["id_0", "id_2", f"id_{N_VECTORS - 1}"]))
        assert results == [True, True, True]

        # Fetch after reload
        results = run(store2.fetch(["id_2"]))
        assert len(results[0]) == 64
        print("PASS: persistence")
    finally:
        shutil.rmtree(tmp)


def test_cosine_metric():
    tmp = tempfile.mkdtemp()
    try:
        store = make_store(tmp, similarity_metric="cosine")
        store.add_embedding(
            [f"c_{i}" for i in range(N_VECTORS)], random_embeddings(N_VECTORS)
        )
        assert store.index.ntotal == N_VECTORS
        print("PASS: cosine_metric")
    finally:
        shutil.rmtree(tmp)


def test_ip_metric():
    tmp = tempfile.mkdtemp()
    try:
        store = make_store(tmp, similarity_metric="ip")
        store.add_embedding(
            [f"ip_{i}" for i in range(N_VECTORS)], random_embeddings(N_VECTORS)
        )
        assert store.index.ntotal == N_VECTORS
        print("PASS: ip_metric")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    tests = [
        test_add_embedding,
        test_skip_duplicates,
        test_search,
        test_fetch,
        test_is_exist,
        test_delete,
        test_persistence,
        test_cosine_metric,
        test_ip_metric,
    ]

    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"FAIL: {test.__name__} - {e}")
            failed += 1

    print(f"\n{'ALL TESTS PASSED' if failed == 0 else f'{failed} test(s) FAILED'}")
    sys.exit(failed)
