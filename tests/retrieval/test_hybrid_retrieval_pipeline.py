"""CPU integration contracts for real BM25/RRF and neutral retrieval quality."""

import asyncio
from types import SimpleNamespace

import pandas as pd
import pytest

from rag_stack_evaluator.static_rag_evaluator.fusion_trace import FUSION_TRACE_COLUMN
from rag_stack_evaluator.static_rag_evaluator.retrieval_result import (
    snapshot_retrieval_result, validate_retrieval_capabilities,
)


class SelectLastPassage:
    """A deterministic truncating reranker fixture with observable input size."""
    def pure(self, previous_result, **_kwargs):
        return pd.DataFrame({name: previous_result[name].map(lambda values: list(values)[-1:])
                             for name in ("retrieved_contents", "retrieved_ids", "retrieve_scores")})

    @classmethod
    def run_evaluator(cls, *, previous_result, **kwargs):
        return cls().pure(previous_result, **kwargs)


class CompressPassages:
    def pure(self, previous_result, **_kwargs):
        return pd.DataFrame({"retrieved_contents": [["compressed"]] * len(previous_result)})

    @classmethod
    def run_evaluator(cls, *, previous_result, **kwargs):
        return cls().pure(previous_result, **kwargs)


def _prepare(tmp_path, monkeypatch, vector_node):
    from rag_stack_evaluator.static_rag_evaluator.nodes.lexicalretrieval import bm25

    monkeypatch.setitem(bm25.BM25_TOKENIZER, "unit_words", lambda texts: [text.split() for text in texts])
    (tmp_path / "data").mkdir()
    (tmp_path / "resources").mkdir()
    corpus = pd.DataFrame({
        "doc_id": list("abxyz"),
        "contents": ["irrelevant dense", "single target", "left", "right", "neither"],
        "metadata": [{"source": "test"}] * 5,
    })
    vector_node.corpus_df = corpus
    vector_node.embedding_model = SimpleNamespace(model_name="actual-encoder")
    vector_node._contents_by_doc_id = dict(zip(corpus.doc_id, corpus.contents))
    corpus.to_parquet(tmp_path / "data/corpus.parquet")
    qa = pd.DataFrame({
        "qid": ["q1", "q2"], "__qid__": ["q1", "q2"],
        "query": ["single", "left"], "queries": [["single"], ["left", "right"]],
        "retrieval_gt_contents": [[["single target"]], [["right"]]],
        "generation_gt": [["target"], ["right"]],
    })
    qa.to_parquet(tmp_path / "data/qa.parquet")
    bm25.bm25_ingest(str(tmp_path / "resources/bm25_unit_words.pkl"), corpus, "unit_words")
    monkeypatch.setenv("PROJECT_DIR", str(tmp_path))
    return qa, corpus


def test_sequential_real_bm25_rrf_traces_and_effective_quality(tmp_path, monkeypatch, vector_node, recorder):
    from rag_stack_evaluator.static_rag_evaluator.nodes.semanticretrieval.run import run_semantic_retrieval_node
    from rag_stack_evaluator.static_rag_evaluator.nodes.lexicalretrieval.run import run_lexical_retrieval_node
    from rag_stack_evaluator.static_rag_evaluator.nodes.hybridretrieval.run import run_hybrid_retrieval_node
    from rag_stack_evaluator.static_rag_evaluator.nodes.lexicalretrieval.bm25 import BM25
    from rag_stack_evaluator.static_rag_evaluator.nodes.hybridretrieval.hybrid_rrf import HybridRRF
    from rag_stack_evaluator.static_rag_evaluator.static_rag_evaluator import StaticRAGEvaluatorQualityOnly
    from rag_stack_evaluator.static_rag_evaluator.evaluation.metric.retrieval_contents import retrieval_token_recall
    from rag_stack.rag_ir import make_quality_trace_envelope
    from rag_stack_evaluator.static_rag_evaluator.nodes.passagereranker.run import run_passage_reranker_node
    from rag_stack_evaluator.static_rag_evaluator.nodes.passagecompressor.run import run_passage_compressor_node

    qa, _ = _prepare(tmp_path, monkeypatch, vector_node)

    class Dense:
        @staticmethod
        def run_evaluator(*, previous_result, **kwargs):
            return vector_node.pure(previous_result, top_k=1)

    output_dir = str(tmp_path / "run")
    dense = run_semantic_retrieval_node([Dense], [{}], qa, output_dir, {})
    lexical = run_lexical_retrieval_node([BM25], [{"top_k": 1, "bm25_tokenizer": "unit_words"}], dense, output_dir, {})
    result = run_hybrid_retrieval_node([HybridRRF], [{"top_k": 2, "weight": 60}], lexical, output_dir, {})
    assert result.retrieved_ids_semantic.iloc[0] == ["a"]
    assert result.retrieved_ids_lexical.iloc[0] == ["b"]
    assert set(result.retrieved_ids_retrieval.iloc[0]) == {"a", "b"}
    assert FUSION_TRACE_COLUMN not in result
    expected_inputs = [len(" ".join([query] + contents).encode("utf-8"))
                       for query, contents in zip(result["query"], result.retrieved_contents)]
    result = run_passage_reranker_node([SelectLastPassage], [{}], result, output_dir, {})
    expected_compressor_inputs = [len(" ".join(contents).encode("utf-8")) for contents in result.retrieved_contents]
    result = run_passage_compressor_node([CompressPassages], [{}], result, output_dir, {})
    evaluator = object.__new__(StaticRAGEvaluatorQualityOnly)
    evaluator.qa_data = qa
    inputs = evaluator._create_metric_inputs(result, {})
    assert inputs[0].retrieved_contents == ["compressed"]
    assert inputs[0].retrieved_contents_semantic == ["irrelevant dense"]
    assert retrieval_token_recall(inputs)[0] > retrieval_token_recall(
        evaluator._create_metric_inputs(dense, {})
    )[0]
    traces = recorder.to_trace_v1()
    for trace in traces:
        assert [call["stage"] for call in trace] == [
            "semantic_retrieval_encode", "semantic_retrieval_vectorsearch", "retrieval_fusion",
            "lexical_retrieval", "retrieval_fusion", "retrieval_fusion",
            "passage_reranker",
            "passage_compressor",
        ]
        assert [call["node_id"] for call in trace[3:6]] == [
            "lexical_retrieval", "lexical_retrieval", "hybrid_retrieval",
        ]
        assert trace[5]["input_refs"] == {
            "dense": [trace[2]["call_id"]], "lexical": [trace[4]["call_id"]],
        }
    assert [trace[6]["input_bytes"] for trace in traces] == expected_inputs
    assert [trace[7]["input_bytes"] for trace in traces] == expected_compressor_inputs
    assert all(trace[0]["model_id"] == "actual-encoder" for trace in traces)
    assert traces[1][0]["input_bytes"] == len("left right".encode("utf-8"))
    # The canonical envelope requires a terminal generation observation.
    # Generation is a fixture here; the separately run GPU smoke uses a real LLM.
    for qid in ("q1", "q2"):
        recorder.record(qid, "generator", model_id="fixture-generator", input_tokens=1, output_tokens=1)
    make_quality_trace_envelope(recorder.to_trace_v1(), question_ids=["q1", "q2"])


def test_measured_merge_and_record_use_real_lexical_and_fusion_stages(tmp_path, monkeypatch, vector_node, recorder):
    from rag_stack_evaluator.static_rag_evaluator.measured.evaluator import MeasuredEvaluator
    from rag_stack_evaluator.static_rag_evaluator.measured.serving_runtime import _record_pure_stage
    from rag_stack_evaluator.static_rag_evaluator.nodes.lexicalretrieval.bm25 import BM25
    from rag_stack_evaluator.static_rag_evaluator.nodes.hybridretrieval.hybrid_rrf import HybridRRF

    qa, _ = _prepare(tmp_path, monkeypatch, vector_node)
    frame = qa
    instances = [vector_node, BM25(str(tmp_path), bm25_tokenizer="unit_words"), HybridRRF(str(tmp_path))]
    for stage, instance in zip(("semantic_retrieval", "lexical_retrieval", "hybrid_retrieval"), instances):
        node = SimpleNamespace(stage=stage, strategy={})
        params = {"top_k": 2}
        pure = instance.pure(frame, **params)
        frame = MeasuredEvaluator._merge_service_node_result(node, frame, pure)
        _record_pure_stage(node, params, frame)
    assert frame.retrieved_ids_retrieval.tolist() == frame.retrieved_ids.tolist()
    for trace in recorder.to_trace_v1():
        assert [call["stage"] for call in trace].count("semantic_retrieval_encode") == 1
        assert [call["stage"] for call in trace].count("lexical_retrieval") == 1
        assert [call["stage"] for call in trace].count("retrieval_fusion") == 3


def test_bm25_cache_replaces_old_corpus_and_content(tmp_path, monkeypatch):
    from rag_stack_evaluator.static_rag_evaluator.nodes.lexicalretrieval import bm25
    monkeypatch.setitem(bm25.BM25_TOKENIZER, "unit_words", lambda texts: [text.split() for text in texts])
    path = str(tmp_path / "corpus.pkl")
    bm25.bm25_ingest(path, pd.DataFrame({"doc_id": ["old"], "contents": ["old words"], "metadata": [{}]}), "unit_words")
    bm25.bm25_ingest(path, pd.DataFrame({"doc_id": ["new"], "contents": ["new words"], "metadata": [{}]}), "unit_words")
    cached = bm25.load_bm25_corpus(path)
    assert cached["passage_id"] == ["new"]
    bm25.bm25_ingest(path, pd.DataFrame({"doc_id": ["new"], "contents": ["changed"], "metadata": [{}]}), "unit_words")
    assert bm25.load_bm25_corpus(path)["tokens"] == [["changed"]]


@pytest.mark.parametrize("postprocess", [False, True], ids=["fused-context", "compressed-context"])
def test_measured_scheduler_executes_all_hybrid_services(tmp_path, monkeypatch, vector_node, recorder, postprocess):
    from rag_stack_evaluator.static_rag_evaluator.measured.evaluator import MeasuredEvaluator
    from rag_stack_evaluator.static_rag_evaluator.measured.serving_runtime import (
        LLMContinuousStageService, MeasuredServingRuntime, RequestState,
    )
    from rag_stack_evaluator.static_rag_evaluator.nodes.lexicalretrieval.bm25 import BM25
    from rag_stack_evaluator.static_rag_evaluator.nodes.hybridretrieval.hybrid_rrf import HybridRRF
    from rag_stack_evaluator.static_rag_evaluator.nodes.promptmaker.fstring import Fstring
    from rag_stack.rag_ir import make_quality_trace_envelope

    qa, _ = _prepare(tmp_path, monkeypatch, vector_node)
    instances = [vector_node, BM25(str(tmp_path), bm25_tokenizer="unit_words"), HybridRRF(str(tmp_path))]
    received_prompts = []

    async def generate_one(prompt, params):
        # Stub only the model boundary: admission, prompt transport, output
        # merge, and terminal trace still use the production measured service.
        received_prompts.append(prompt)
        assert params["temperature"] == 0.0
        assert params["max_tokens"] == 2
        return "fixture answer", {"n_output_tokens": 2}

    generator = SimpleNamespace(_subprocess=SimpleNamespace(
        key=SimpleNamespace(max_num_seqs=2), generate_one=generate_one,
    ))

    async def scenario():
        runtime = object.__new__(MeasuredServingRuntime)
        runtime.owner = MeasuredEvaluator
        runtime.system_config = {"batching": {"dynamic_timeout_s": 0.01}, "layout": {"performance_stages": {}}}
        runtime.batch_size_request = 2
        runtime.load_concurrency_hard_cap = 2
        runtime.stages = [{"stage": stage, "node": SimpleNamespace(stage=stage, strategy={}),
                           "params": {"top_k": 2}, "instance": instance}
                          for stage, instance in zip(("semantic_retrieval", "lexical_retrieval", "hybrid_retrieval"), instances)]
        if postprocess:
            runtime.stages.append({"stage": "passage_reranker", "node": SimpleNamespace(stage="passage_reranker", strategy={}),
                                   "params": {}, "instance": SelectLastPassage()})
            runtime.stages.append({"stage": "passage_compressor", "node": SimpleNamespace(stage="passage_compressor", strategy={}),
                                   "params": {}, "instance": CompressPassages()})
        runtime.stages.append({"stage": "prompt_maker", "node": SimpleNamespace(stage="prompt_maker", strategy={}),
                               "params": {"prompt": "Question: {query}\nContext: {retrieved_contents}"},
                               "instance": Fstring(str(tmp_path))})
        runtime.stages.append({"stage": "generator", "node": SimpleNamespace(stage="generator", strategy={}),
                               "params": {"model": "fixture-generator", "temperature": 0.0, "max_tokens": 2},
                               "instance": generator})
        runtime.services = runtime._build_services()
        assert isinstance(runtime.services[-1], LLMContinuousStageService)
        states = [RequestState(seq=i, idx=i, qid=qid, df=qa.iloc[[i]].reset_index(drop=True),
                               is_measured=True, record_trace=True) for i, qid in enumerate(qa.qid)]
        try:
            await asyncio.wait_for(asyncio.gather(*(runtime._run_sequential_request(state) for state in states)), timeout=20)
            assert runtime.services[-1].completed == len(states)
            return states
        finally:
            for service in runtime.services:
                await service.close()

    states = asyncio.run(scenario())
    expected_prompts = []
    for state, trace in zip(states, recorder.to_trace_v1()):
        assert len(state.df.retrieved_ids_retrieval.iloc[0]) == 2
        assert trace[5]["node_id"] == "hybrid_retrieval"
        if postprocess:
            assert len(state.df.retrieved_ids.iloc[0]) == 1
            expected = " ".join([state.df["query"].iloc[0]] + state.df.retrieved_contents_retrieval.iloc[0])
            assert trace[6]["input_bytes"] == len(expected.encode("utf-8"))
            assert trace[7]["input_bytes"] == len(state.df.retrieved_contents_retrieval.iloc[0][-1].encode("utf-8"))
            assert state.df.retrieved_contents.iloc[0] == ["compressed"]
        else:
            assert state.df.retrieved_contents.iloc[0] == state.df.retrieved_contents_retrieval.iloc[0]
        assert trace[0]["input_bytes"] == len(" ".join(state.df.queries.iloc[0]).encode("utf-8"))
        assert trace[0]["model_id"] == "actual-encoder"
        context = "\n\n".join(state.df.retrieved_contents.iloc[0])
        prompt = f"Question: {state.df['query'].iloc[0]}\nContext: {context}"
        expected_prompts.append(prompt)
        assert state.df.prompts.iloc[0] == prompt
        assert state.df.generated_texts.iloc[0] == "fixture answer"
        assert trace[-1]["stage"] == "generator"
        assert trace[-1]["model_id"] == "fixture-generator"
        assert trace[-1]["input_bytes"] == len(prompt.encode("utf-8"))
        assert trace[-1]["output_tokens"] == 2
        assert len(trace) == (9 if postprocess else 7)
    assert sorted(received_prompts) == sorted(expected_prompts)
    # The single-query row differs from its dense arm, so a hidden dense-first
    # fallback cannot satisfy the exact prompt assertion above.
    assert states[0].df.retrieved_contents.iloc[0] != states[0].df.retrieved_contents_semantic.iloc[0]
    make_quality_trace_envelope(recorder.to_trace_v1(), question_ids=[state.qid for state in states])


def test_registry_keeps_canonical_stage_names():
    from rag_stack_evaluator.static_rag_evaluator.support import get_stage_components
    components = get_stage_components()
    assert "vectordb" in components["semantic_retrieval"]
    assert "bm25" in components["lexical_retrieval"]
    assert "hybrid_rrf" in components["hybrid_retrieval"]
    assert "dense_retrieval" not in components and "sparse_retrieval" not in components


@pytest.mark.parametrize("stage,component,mode", [
    ("hybrid_retrieval", "hybrid_rrf", "react"),
    ("lexical_retrieval", "bm25", "react"),
    ("hybrid_retrieval", "hybrid_cc", "sequential"),
])
def test_unsupported_retrieval_modes_fail_early(stage, component, mode):
    with pytest.raises(ValueError):
        validate_retrieval_capabilities({
            "pipeline_runtime": {"rag_dataflow": mode},
            "node_lines": [{"nodes": [{"stage": stage, "module": {"component": component}}]}],
        })
