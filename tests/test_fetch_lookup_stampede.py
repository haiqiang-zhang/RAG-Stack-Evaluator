"""fetch_contents corpus lookup: exactly ONE build under concurrency.

The lazy per-corpus lookup used to be built with no lock: at serving start
hundreds of executor threads saw the empty cache simultaneously and each
built its own copy of the corpus-sized dict — GIL-serialized minutes of
pandas that stalled the whole closed loop (msmarco s44/eval_0009)."""

import threading
import time

import pandas as pd

from rag_stack_evaluator.static_rag_evaluator.utils import util as u


def _corpus(n=100):
    return pd.DataFrame({
        "doc_id": [f"d{i}" for i in range(n)],
        "contents": [f"text {i}" for i in range(n)],
    })


def test_concurrent_first_access_builds_once(monkeypatch):
    corpus = _corpus()
    builds = []
    real_build = u._build_fetch_lookup

    def counting_build(df, id_col, col):
        builds.append(threading.get_ident())
        time.sleep(0.05)  # widen the race window
        return real_build(df, id_col, col)

    monkeypatch.setattr(u, "_build_fetch_lookup", counting_build)

    results = []
    barrier = threading.Barrier(16)

    def worker():
        barrier.wait()
        results.append(u._fetch_lookup(corpus, "doc_id", "contents"))

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(builds) == 1
    assert all(r is results[0] for r in results)
    assert results[0]["d3"] == "text 3"


def test_prewarm_pays_the_build_up_front(monkeypatch):
    corpus = _corpus()
    builds = []
    real_build = u._build_fetch_lookup
    monkeypatch.setattr(
        u, "_build_fetch_lookup",
        lambda *a: (builds.append(1), real_build(*a))[1])

    u.prewarm_fetch_lookup(corpus, "doc_id", "contents")
    assert len(builds) == 1
    out = u.fetch_contents(corpus, [["d1", "d2"]])
    assert out == [["text 1", "text 2"]]
    assert len(builds) == 1  # serving-time call did not rebuild
