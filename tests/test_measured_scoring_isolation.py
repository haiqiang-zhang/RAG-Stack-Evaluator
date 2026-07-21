from types import SimpleNamespace

import pandas as pd

from rag_stack_evaluator.static_rag_evaluator.measured.serving_runtime import (
    MeasuredServingRuntime,
)
from rag_stack_evaluator.static_rag_evaluator.static_rag_evaluator import (
    StaticRAGEvaluatorQualityOnly,
)


def _pipeline_harness(tmp_path):
    evaluator = object.__new__(StaticRAGEvaluatorQualityOnly)
    evaluator.project_dir = str(tmp_path)
    evaluator.qa_data = pd.DataFrame({"query": ["q0", "q1"]})
    corpus = pd.DataFrame({"contents": ["document"]})
    corpus_view = SimpleNamespace(
        chunk_hash="self-contained",
        n_vectors=len(corpus),
        token_stats=None,
    )
    evaluator._dataset = SimpleNamespace(
        corpus_data=corpus,
        resolve_corpus=lambda _params: corpus_view,
        resolve_nlist_factor=lambda _configs, _n_vectors: None,
        activate=lambda _view: None,
    )
    evaluator.corpus_data = corpus
    evaluator._parse_node_lines = lambda _config, _metrics: {
        "main": [SimpleNamespace(stage="generator")]
    }
    evaluator._ingest_bm25 = lambda _node_lines: None
    evaluator._ingest_vectordb = lambda _configs, _node_lines: None
    evaluator._resolve_vectordb_paths = (
        lambda configs, *, chunk_hash: configs
    )
    evaluator._release_gpu_memory = lambda: None

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "corpus.parquet").write_bytes(b"test sentinel")
    return evaluator


def _config(*, performance_only=False):
    return {
        "pipeline_runtime": {"mode": "sequential"},
        "vectordb": [],
        "eval_backend_setting": {
            "metrics": [{"metric_name": "blocking_test_metric"}],
            "performance_only": performance_only,
        },
    }


def test_measured_scorer_starts_after_runtime_and_gets_same_snapshot(tmp_path):
    evaluator = _pipeline_harness(tmp_path)
    snapshot = pd.DataFrame({
        "query": ["q0", "q1"],
        "generated_texts": ["a0", "a1"],
        "__qid__": ["warmup-7", "measured-12"],
    })
    events = []
    measurement_ended = False

    def measured_runner(**kwargs):
        nonlocal measurement_ended
        events.append("runtime_started")
        # A scorer callback here would reintroduce host/API work into the
        # measured window. The runtime contract no longer exposes one.
        assert "on_quality_snapshot" not in kwargs
        assert events == ["runtime_started"]
        measurement_ended = True
        events.append("measurement_ended")
        return snapshot, {"measurement_end": 1.0}

    def blocking_scorer(final_result, _evaluation_config, _last_node_type):
        assert measurement_ended
        assert final_result is snapshot
        events.append("scorer_started")
        return {"blocking_test_metric": 0.75}

    evaluator._evaluate_final_result = blocking_scorer
    run = evaluator._run_pipeline(
        _config(),
        run_dir=str(tmp_path / "run"),
        sequential_runner=measured_runner,
    )

    assert events == ["runtime_started", "measurement_ended", "scorer_started"]
    assert run.previous_result is snapshot
    assert run.quality == {"blocking_test_metric": 0.75}


def test_first_completion_snapshot_is_frozen_once_by_identity():
    runtime = object.__new__(MeasuredServingRuntime)
    runtime.n_rows = 2
    runtime._quality_snapshot = None
    winners = {
        0: pd.DataFrame({"__qid__": ["warmup-3"], "answer": ["first-0"]}),
        1: pd.DataFrame({"__qid__": ["measured-8"], "answer": ["first-1"]}),
    }

    frozen = runtime._freeze_first_completion_snapshot(winners)
    winners[0] = pd.DataFrame({"__qid__": ["late-99"], "answer": ["late"]})
    frozen_again = runtime._freeze_first_completion_snapshot(winners)

    assert frozen_again is frozen
    assert frozen["__qid__"].tolist() == ["warmup-3", "measured-8"]
    assert frozen["answer"].tolist() == ["first-0", "first-1"]


def test_performance_only_never_invokes_final_scorer(tmp_path):
    evaluator = _pipeline_harness(tmp_path)
    snapshot = pd.DataFrame({
        "query": ["q0", "q1"],
        "generated_texts": ["a0", "a1"],
        "__qid__": ["warmup-1", "warmup-2"],
    })
    scorer_calls = 0

    def measured_runner(**_kwargs):
        return snapshot, {"measurement_end": 1.0}

    def scorer_must_not_run(*_args, **_kwargs):
        nonlocal scorer_calls
        scorer_calls += 1
        raise AssertionError("performance_only invoked the quality scorer")

    evaluator._evaluate_final_result = scorer_must_not_run
    run = evaluator._run_pipeline(
        _config(performance_only=True),
        run_dir=str(tmp_path / "run"),
        sequential_runner=measured_runner,
    )

    assert scorer_calls == 0
    assert run.previous_result is snapshot
    assert run.quality == {}
