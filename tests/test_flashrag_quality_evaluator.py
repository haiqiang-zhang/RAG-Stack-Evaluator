"""CPU tests for evaluator-owned FlashRAG adapters and monitoring."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest


def test_content_hash_handles_container_columns():
    """Corpus/QA parquets carry ndarray / dict / list object columns (e.g.
    dragonball ``start_end_idx`` + ``metadata``); hashing must not raise and
    must stay row-order insensitive."""
    import numpy as np
    import pandas as pd
    from rag_stack_evaluator.flashrag_quality_evaluator import dataset_adapter as da

    df = pd.DataFrame({
        "doc_id": ["b", "a"],
        "contents": ["second doc", "first doc"],
        "start_end_idx": [np.array([10, 20]), np.array([0, 10])],
        "metadata": [{"k": 2}, {"k": 1}],
        "tags": [["x", "y"], ["z"]],
    })
    h = da.corpus_hash(df)
    assert isinstance(h, str) and len(h) == 16
    # Row order must not matter.
    assert da.corpus_hash(df.iloc[::-1]) == h
    # Content changes must change the hash.
    changed = df.copy()
    changed.loc[0, "contents"] = "second doc EDITED"
    assert da.corpus_hash(changed) != h



def test_method_knob_forwarding():
    import numpy as np
    from rag_stack_evaluator.flashrag_quality_evaluator.flashrag_quality_evaluator import (
        _method_kwargs_for,
    )

    class DummyIrcot:
        def __init__(self, config, prompt_template=None, max_iter=2,
                     retriever=None, generator=None):
            pass

    cfg = {"pipeline_runtime": {
        "mode": "ircot",
        "max_iter": np.int64(3),       # numpy scalar from the decoded config
        "bogus_knob": 1,               # must be dropped (with a warning)
    }}
    kw = _method_kwargs_for(DummyIrcot, cfg)
    assert kw == {"max_iter": 3}
    assert type(kw["max_iter"]) is int
    # mode itself is never forwarded; empty runtime → no kwargs.
    assert _method_kwargs_for(DummyIrcot, {"pipeline_runtime": {"mode": "x"}}) == {}
    assert _method_kwargs_for(DummyIrcot, {}) == {}


def test_metric_alias_is_discoverable_and_translated():
    """Backend metadata and runtime translation agree on accepted aliases."""
    from rag_stack_evaluator.factory import supported_metrics
    from rag_stack_evaluator.flashrag_quality_evaluator.config_translator import _resolve_metrics

    assert "rouge_l" in supported_metrics("flashrag")
    assert _resolve_metrics({"metrics": [{"metric_name": "rouge_l"}]}) == ["rouge-l"]



def test_monitor_records_events():
    """Feed synthetic events to the Monitor and verify the DAG structure."""
    from rag_stack_evaluator.flashrag_quality_evaluator.monitor import (
        ExecutionDAG, ExecutionNode, Monitor,
    )

    mon = Monitor()

    # One generate batch over 2 queries.
    mon.record_generate_batch(
        query_ids=["q1", "q2"],
        step_idx=0,
        model_id="test-model",
        prompts=["prompt 1", "prompt 2"],
        outputs=["out 1", "out 2"],
        input_token_counts=[5, 6],
        output_token_counts=[3, 4],
        latency_ms=12.5,
    )
    # One retrieve batch — only q1 (q2 finished early in the imagined pipeline).
    mon.record_retrieve_batch(
        query_ids=["q1"],
        step_idx=1,
        model_id="e5",
        queries=["query text 1"],
        doc_lists=[[{"contents": "doc1"}, {"contents": "doc2"}]],
        input_token_counts=[3],
        output_token_counts=[7],
        latency_ms=4.0,
    )
    # Terminate event for q2.
    mon.record_terminate(query_id="q2", step_idx=1, reason="found_answer")

    dags = mon.build_dags()
    by_qid = {d.query_id: d for d in dags}

    # q1: generate(step 0) + retrieve(step 1) → 2 nodes
    q1 = by_qid["q1"]
    assert len(q1.nodes) == 2
    assert q1.nodes[0].node_type == "generate"
    assert q1.nodes[0].input_tokens == 5
    assert q1.nodes[0].output_tokens == 3
    assert q1.nodes[1].node_type == "retrieve"
    assert q1.nodes[1].input_tokens == 3
    assert q1.nodes[1].output_tokens == 7
    assert q1.total_input_tokens == 8
    assert q1.total_output_tokens == 10

    # q2: generate(step 0) + terminate(step 1) → 2 nodes
    q2 = by_qid["q2"]
    assert len(q2.nodes) == 2
    assert q2.nodes[0].node_type == "generate"
    assert q2.nodes[-1].node_type == "terminate"
    assert q2.nodes[-1].extras["reason"] == "found_answer"



def test_flashrag_producer_ships_v2_envelope_without_bytes():
    """P1 producer contract: monitor DAGs wrap into THE v2 quality-trace
    envelope (question_id = row position, dense retrieve split into the
    encode + vectorsearch pair) and FlashRAG's byte-less events stay
    byte-less — absent fields, never zero-filled."""
    from rag_stack_evaluator.flashrag_quality_evaluator.flashrag_quality_evaluator import (
        _dag_to_dict,
        _quality_trace_envelope,
    )
    from rag_stack_evaluator.flashrag_quality_evaluator.monitor import Monitor
    from rag_stack.rag_ir import validate_quality_trace_envelope

    mon = Monitor()
    mon.record_retrieve_batch(
        query_ids=["q1", "q2"],
        step_idx=0,
        model_id="e5",
        queries=["query 1", "query 2"],
        doc_lists=[[{"contents": "doc1"}], [{"contents": "doc2"}]],
        input_token_counts=[3, 4],
        output_token_counts=[7, 8],
        latency_ms=4.0,
    )
    mon.record_generate_batch(
        query_ids=["q1", "q2"],
        step_idx=1,
        model_id="test-model",
        prompts=["prompt 1", "prompt 2"],
        outputs=["out 1", "out 2"],
        input_token_counts=[5, 6],
        output_token_counts=[3, 4],
        latency_ms=12.5,
    )
    mon.record_terminate(query_id="q2", step_idx=2, reason="found_answer")

    envelope = _quality_trace_envelope(
        [_dag_to_dict(d) for d in mon.build_dags()]
    )

    validate_quality_trace_envelope(envelope)
    assert [q["question_id"] for q in envelope["queries"]] == ["0", "1"]
    q1_stages = [c["stage"] for c in envelope["queries"][0]["calls"]]
    assert q1_stages == [
        "semantic_retrieval_encode",
        "semantic_retrieval_vectorsearch",
        "generator",
    ]
    for query in envelope["queries"]:
        for call in query["calls"]:
            assert "input_bytes" not in call  # absent, NOT zero-filled
            assert "output_bytes" not in call
            assert "latency_ms" not in call



def test_query_context_attribution():
    """The thread-local query_context propagates to inner records.

    Inner-layer FlashRAG patches (reranker.rerank, _batch_search) read
    :func:`flashrag.monitor_hook.current_query_context` to attribute their
    events to the right query / step. This exercises that path without
    needing to actually invoke a reranker or FAISS index.
    """
    pytest.importorskip("flashrag.monitor_hook")
    from flashrag.monitor_hook import (
        current_query_context, query_context, set_monitor, get_monitor,
    )
    # No context → returns None.
    assert current_query_context() is None
    # Push two levels — the stack should be honored LIFO.
    with query_context(["a", "b"], step_idx=1):
        assert current_query_context() == (["a", "b"], 1)
        with query_context(["c"], step_idx=2):
            assert current_query_context() == (["c"], 2)
        assert current_query_context() == (["a", "b"], 1)
    assert current_query_context() is None

    # set_monitor / get_monitor roundtrip.
    set_monitor("dummy")
    assert get_monitor() == "dummy"
    set_monitor(None)
    assert get_monitor() is None



def test_vectordb_and_rerank_imports():
    """Sanity: the patched modules still import and expose their helpers."""
    pytest.importorskip("flashrag")
    from flashrag.retriever.retriever import (
        DenseRetriever, BM25Retriever, _record_vectordb_batch,
    )
    from flashrag.retriever.reranker import BaseReranker
    assert callable(_record_vectordb_batch)
    # BaseReranker.rerank should still be defined and accept the standard signature.
    assert hasattr(BaseReranker, "rerank")



def test_normalize_drops_vectordb_and_terminate_orders_by_step():
    from rag_stack.rag_ir import make_quality_trace_envelope, normalize_quality_trace
    from rag_stack_evaluator.flashrag_quality_evaluator.trace_adapter import normalize_traces

    # Monitor DAG events keep the FlashRAG adapter vocabulary (node_type).
    dags = [{
        "query_id": "q1",
        "nodes": [
            {"node_type": "generate", "step_idx": 1, "input_tokens": 800, "output_tokens": 50},
            {"node_type": "retrieve", "step_idx": 0, "input_tokens": 20, "output_tokens": 0},
            {"node_type": "vectordb", "step_idx": 0, "input_tokens": 0, "output_tokens": 0},
            {"node_type": "terminate", "step_idx": 2},
        ],
    }]
    traces = normalize_traces(dags)
    assert len(traces) == 1
    stages = [c["stage"] for c in traces[0]]
    # sorted by step_idx; vectordb+terminate dropped; the dense retrieve event
    # expands into its encode + vector-search call pair.
    assert stages == ["semantic_retrieval_encode", "semantic_retrieval_vectorsearch", "generator"]
    assert traces[0][2]["input_tokens"] == 800
    # The producer wraps the normalized traces in the canonical envelope; the CM
    # boundary consumes them unchanged.
    envelope = make_quality_trace_envelope(traces, question_ids=["q1"])
    assert normalize_quality_trace(envelope).traces == traces


def test_flashrag_public_imports_do_not_load_host_or_model_libraries():
    """Optional backend metadata stays usable without the host or a GPU stack."""
    code = r'''
import sys

class RejectRuntimeImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"rag_stack", "flashrag", "torch", "vllm"}:
            raise AssertionError("Unexpected runtime dependency: " + fullname)

sys.meta_path.insert(0, RejectRuntimeImports())
from rag_stack_evaluator.base import BaseEvaluator
from rag_stack_evaluator.factory import supported_metrics, supported_pipeline_modes
from rag_stack_evaluator.flashrag_quality_evaluator import FlashRAGQualityEvaluator
from rag_stack_evaluator.flashrag_quality_evaluator.trace_adapter import normalize_traces
assert issubclass(FlashRAGQualityEvaluator, BaseEvaluator)
assert "react" in supported_pipeline_modes("flashrag")
assert "rouge-l" in supported_metrics("flashrag")
assert normalize_traces([]) == []
'''
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def fake_flashrag_runtime(tmp_path, monkeypatch):
    """Replace model execution while exercising real translation and JSONL IO."""
    from rag_stack_evaluator.flashrag_quality_evaluator import (
        FlashRAGQualityEvaluator, dataset_adapter, pipeline_factory,
    )

    state = SimpleNamespace(monitor=None, failure=None, config=None, max_iter=None)
    module_names = ("flashrag", "flashrag.config", "flashrag.utils", "flashrag.monitor_hook")
    modules = {name: ModuleType(name) for name in module_names}
    modules["flashrag"].__path__ = []
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    def set_monitor(monitor):
        state.monitor = monitor

    class Config(dict):
        def __init__(self, *, config_dict):
            if state.failure == "config":
                raise RuntimeError("fixture config failure")
            super().__init__(config_dict)
            state.config = self

    def get_dataset(config):
        if state.failure == "dataset":
            raise RuntimeError("fixture dataset failure")
        path = Path(config["data_dir"]) / config["dataset_name"] / "test.jsonl"
        return {"test": [json.loads(line) for line in path.read_text().splitlines()]}

    class Pipeline:
        def __init__(self, config, max_iter=1):
            if state.failure == "constructor":
                raise RuntimeError("fixture constructor failure")
            state.max_iter = max_iter

        def run(self, dataset, *, do_eval):
            assert do_eval
            if state.failure == "run":
                raise RuntimeError("fixture run failure")
            if state.monitor is not None:
                state.monitor.record_generate_batch(
                    query_ids=[row["id"] for row in dataset], step_idx=0,
                    model_id="fixture-generator", prompts=["prompt"] * len(dataset),
                    outputs=["answer"] * len(dataset), input_token_counts=[4] * len(dataset),
                    output_token_counts=[2] * len(dataset), latency_ms=1.0,
                )
            return [SimpleNamespace(output={"metric_score": {"em": 1.0, "f1": 0.5}})]

    modules["flashrag.config"].Config = Config
    modules["flashrag.utils"].get_dataset = get_dataset
    modules["flashrag.monitor_hook"].set_monitor = set_monitor
    state.pipeline_loader = Mock(return_value=Pipeline)
    monkeypatch.setattr(pipeline_factory, "build_pipeline_class", state.pipeline_loader)
    state.ensure_index = Mock(return_value=str(tmp_path / "fixture.index"))
    monkeypatch.setattr(dataset_adapter, "ensure_index", state.ensure_index)
    state.manager = SimpleNamespace(
        project_dir=str(tmp_path / "project"),
        qa_data=pd.DataFrame({
            "qid": ["q0"], "query": ["Question?"], "generation_gt": [["answer"]],
        }),
        corpus_data=pd.DataFrame({"doc_id": ["d0"], "contents": ["answer passage"]}),
        resolve_corpus=Mock(return_value="corpus-view"), activate=Mock(),
    )
    state.evaluator = FlashRAGQualityEvaluator(dataset_manager=state.manager)
    state.pipeline_config = {
        "pipeline_runtime": {"mode": "ircot", "max_iter": 3},
        "corpus_runtime": {"chunker": {"chunk_size": 32}},
        "vectordb": [{"embedding_model": "fixture-encoder", "faiss_type": "Flat"}],
        "node_lines": [{"nodes": [
            {"stage": "semantic_retrieval", "top_k": 2},
            {"stage": "generator", "modules": [{
                "component": "openai", "model": "fixture-generator",
            }]},
        ]}],
        "eval_backend_setting": {"metrics": [{"metric_name": "em"}, {"metric_name": "f1"}]},
    }
    return state


def test_fake_pipeline_materializes_data_and_publishes_canonical_trace(fake_flashrag_runtime):
    state = fake_flashrag_runtime
    callback = Mock()
    result = state.evaluator.evaluate(
        state.pipeline_config, metrics_override=[{"metric_name": "em"}],
        on_trace_ready=callback,
    )

    assert result["em"] == 1.0
    assert "f1" not in result
    assert state.config["metrics"] == ["em"]
    assert state.pipeline_config["eval_backend_setting"]["metrics"] == [
        {"metric_name": "em"}, {"metric_name": "f1"},
    ]
    assert state.config["retrieval_topk"] == 2
    assert state.max_iter == 3
    state.pipeline_loader.assert_called_once_with("ircot")
    state.manager.resolve_corpus.assert_called_once_with({"chunk_size": 32})
    state.manager.activate.assert_called_once_with("corpus-view")
    state.ensure_index.assert_called_once()
    envelope = result["__execution_dag__"]
    assert envelope["queries"][0]["question_id"] == "0"
    assert envelope["queries"][0]["calls"][0]["stage"] == "generator"
    assert callback.call_args.args[0] is envelope
    assert state.monitor is None
    assert json.loads(Path(state.config["corpus_path"]).read_text()) == {
        "id": "d0", "contents": "answer passage",
    }


@pytest.mark.parametrize("phase", ["config", "dataset", "constructor", "run"])
def test_monitor_clears_after_runtime_failure(fake_flashrag_runtime, phase):
    state = fake_flashrag_runtime
    state.failure = phase
    with pytest.raises(RuntimeError, match=f"fixture {phase} failure"):
        state.evaluator.evaluate(state.pipeline_config)
    assert state.monitor is None


def test_monitor_can_be_disabled(fake_flashrag_runtime):
    state = fake_flashrag_runtime
    state.pipeline_config["eval_backend_setting"]["flashrag"] = {"monitor": "off"}
    result = state.evaluator.evaluate(state.pipeline_config)
    assert result == {"em": 1.0, "f1": 0.5}
    assert state.monitor is None


def test_trace_callback_failure_does_not_discard_metrics(fake_flashrag_runtime):
    state = fake_flashrag_runtime
    callback = Mock(side_effect=RuntimeError("advisory callback failure"))
    result = state.evaluator.evaluate(state.pipeline_config, on_trace_ready=callback)
    assert result["em"] == 1.0
    assert callback.call_args.args[0] is result["__execution_dag__"]
    assert state.monitor is None


def test_real_sequential_pipeline_scores_synthetic_queries_without_models(tmp_path, monkeypatch):
    """Run FlashRAG's real dataset, pipeline, scoring, and monitor on the CPU."""
    pytest.importorskip("flashrag")
    from flashrag.config import Config
    from flashrag import monitor_hook
    from flashrag.pipeline import pipeline as pipeline_module
    from rag_stack_evaluator.flashrag_quality_evaluator import (
        FlashRAGQualityEvaluator, dataset_adapter,
    )

    # Config normally probes GPUs and seeds CUDA even for a fake model. Keep
    # those machine-level effects outside this pipeline integration test.
    monkeypatch.setattr(
        Config, "_init_device",
        lambda config: config.final_config.update(device="cpu", gpu_num=0),
    )
    monkeypatch.setattr(Config, "_set_seed", lambda _config: None)

    class Retriever:
        retrieval_method = "fixture-encoder"

        def batch_search(self, queries):
            documents = [[{"contents": answer}] for answer in ("Paris", "Tokyo")]
            monitor_hook.record_retrieve_call(self, queries, documents, latency_ms=1.0)
            return documents

    class Generator:
        model_name = "fixture-generator"

        def generate(self, prompts):
            answers = ["Paris", "Tokyo"]
            monitor_hook.record_generate_call(self, prompts, answers, latency_ms=1.0)
            return answers

    class PromptTemplate:
        def __init__(self, config):
            pass

        def get_string(self, *, question, retrieval_result):
            return question + " " + retrieval_result[0]["contents"]

    monkeypatch.setattr(pipeline_module, "get_retriever", lambda _config: Retriever())
    monkeypatch.setattr(pipeline_module, "get_generator", lambda _config: Generator())
    monkeypatch.setattr(pipeline_module, "PromptTemplate", PromptTemplate)
    monkeypatch.setattr(dataset_adapter, "ensure_index", lambda **_: str(tmp_path / "fixture.index"))
    manager = SimpleNamespace(
        project_dir=str(tmp_path / "real-pipeline"),
        qa_data=pd.DataFrame({
            "qid": ["france", "japan"],
            "query": ["Capital of France?", "Capital of Japan?"],
            "generation_gt": [["Paris"], ["Tokyo"]],
        }),
        corpus_data=pd.DataFrame({"doc_id": ["fr", "jp"], "contents": ["Paris", "Tokyo"]}),
        resolve_corpus=lambda _params: "fixture", activate=lambda _view: None,
    )
    evaluator = FlashRAGQualityEvaluator(dataset_manager=manager)
    result = evaluator.evaluate({
        "pipeline_runtime": {"mode": "sequential"},
        "vectordb": [{"embedding_model": "fixture-encoder", "faiss_type": "Flat"}],
        "node_lines": [{"nodes": [
            {"stage": "semantic_retrieval", "top_k": 1},
            {"stage": "generator", "modules": [{
                "component": "openai", "model": "fixture-generator",
            }]},
        ]}],
        "eval_backend_setting": {"metrics": [{"metric_name": "em"}, {"metric_name": "f1"}]},
    })

    assert result["em"] == 1.0
    assert result["f1"] == 1.0
    queries = result["__execution_dag__"]["queries"]
    assert [query["question_id"] for query in queries] == ["0", "1"]
    for query in queries:
        assert [call["stage"] for call in query["calls"]] == [
            "semantic_retrieval_encode", "semantic_retrieval_vectorsearch", "generator",
        ]
    assert monitor_hook.get_monitor() is None
