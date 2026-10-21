"""CPU contracts for actual static RRF invocations and ordered trace transport."""

import asyncio
from types import SimpleNamespace

import pandas as pd

from rag_stack_evaluator.static_rag_evaluator.fusion_trace import (
    FUSION_TRACE_COLUMN,
    record_fusion_observations,
    take_fusion_observations,
)


def _previous():
    return pd.DataFrame({
        "__qid__": ["one", "two"],
        "query": ["original one", "original two"],
        "queries": [["single"], ["left", "right"]],
    })


def test_real_rrf_profiles_include_single_list_duplicates_and_union(vector_node):
    result = vector_node.pure(_previous(), top_k=2, embedding_batch=2)
    observations = take_fusion_observations(result)
    assert FUSION_TRACE_COLUMN not in result
    assert [len(row) for row in observations] == [1, 1]
    assert observations[0][0] == {
        "input_profile": {
            "items": 3, "list_sizes": [3],
            "list_unique_items": [2], "unique_items": 2,
        },
        "output_profile": {"items": 2, "unique_items": 2},
    }
    assert observations[1][0] == {
        "input_profile": {
            "items": 5, "list_sizes": [3, 2],
            "list_unique_items": [2, 2], "unique_items": 3,
        },
        "output_profile": {"items": 2, "unique_items": 2},
    }
    assert result["retrieved_contents_semantic"].tolist()[0] == ["passage a", "passage b"]


def test_quality_producer_emits_search_then_observed_fusion_without_persisting_metadata(
    vector_node, recorder, monkeypatch, tmp_path,
):
    from rag_stack_evaluator.static_rag_evaluator.nodes.semanticretrieval.run import run_semantic_retrieval_node

    class Module:
        @staticmethod
        def run_evaluator(*, previous_result, **_kwargs):
            return vector_node.pure(previous_result, top_k=2)

    monkeypatch.setenv("PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(pd, "read_parquet", lambda *_args, **_kwargs: pd.DataFrame({
        "retrieval_gt_contents": [[], []], "query": ["q1", "q2"],
        "generation_gt": [[], []],
    }))
    persisted_columns = []
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda frame, *_args, **_kwargs: persisted_columns.append(tuple(frame.columns)))
    result = run_semantic_retrieval_node(
        [Module], [{"top_k": 2}], _previous(), str(tmp_path / "node"), {},
    )
    traces = recorder.to_trace_v1()
    assert len(traces) == 2
    for trace in traces:
        assert [call["stage"] for call in trace] == [
            "semantic_retrieval_encode", "semantic_retrieval_vectorsearch", "retrieval_fusion",
        ]
        assert trace[-1]["model_id"] == "rrf_pure"
    assert traces[0][-1]["input_profile"]["list_sizes"] == [3]
    assert traces[1][-1]["input_profile"]["list_sizes"] == [3, 2]
    assert FUSION_TRACE_COLUMN not in result
    assert all(FUSION_TRACE_COLUMN not in columns for columns in persisted_columns)


def test_measured_batch_preserves_row_identity_and_removes_unretained_metadata(vector_node, recorder):
    from rag_stack_evaluator.static_rag_evaluator.measured import serving_runtime as sr

    class Owner:
        @staticmethod
        def _merge_service_node_result(_node, previous, result):
            return pd.concat([previous.reset_index(drop=True), result.reset_index(drop=True)], axis=1)

    async def scenario():
        stage = {
            "stage": "semantic_retrieval",
            "node": SimpleNamespace(stage="semantic_retrieval"),
            "params": {"top_k": 2}, "instance": vector_node,
        }
        service = sr.BatchedPureStageService(Owner(), stage, batch_size=2, timeout_s=0.01)
        states = [sr.RequestState(
            seq=i, idx=i, qid=qid, df=_previous().iloc[[i]].reset_index(drop=True),
            is_measured=True, record_trace=(i == 1),
        ) for i, qid in enumerate(("one", "two"))]
        try:
            return await asyncio.wait_for(
                asyncio.gather(*(service.run(state, state.df) for state in states)), timeout=15,
            )
        finally:
            await service.close()

    results = asyncio.run(scenario())
    assert recorder.qids == ("two",)
    trace = recorder.to_trace_v1()[0]
    assert [call["stage"] for call in trace] == [
        "semantic_retrieval_encode", "semantic_retrieval_vectorsearch", "retrieval_fusion",
    ]
    assert trace[-1]["input_profile"]["list_sizes"] == [3, 2]
    assert all(FUSION_TRACE_COLUMN not in result.df for result in results)


def test_ids_fetch_bypass_does_not_invent_fusion(vector_node, monkeypatch):
    monkeypatch.setattr(vector_node, "_VectorDB__get_ids_scores", lambda _queries, ids, _batch: (ids, [[1.0], [1.0]]))
    result = vector_node.pure(_previous(), top_k=2, ids=[["a"], ["x"]])
    assert take_fusion_observations(result) == [[], []]


def test_direct_public_hooks_capture_actual_rrf_once_per_request(vector_node):
    from rag_stack.evaluation import capture_trace, trace_batch, trace_stage

    with capture_trace() as session:
        with trace_batch(["one", "two"]):
            vector_node._pure(_previous()["queries"].tolist(), top_k=2)
            with trace_stage("generator", model_id="test-generator") as call:
                call.input(token_counts=[1, 1])
                call.output(token_counts=[1, 1])
    envelope = session.finish(expected_question_ids=["one", "two"])
    assert [[call["stage"] for call in query["calls"]] for query in envelope["queries"]] == [
        ["retrieval_fusion", "generator"], ["retrieval_fusion", "generator"],
    ]
    assert envelope["queries"][0]["calls"][0]["input_profile"]["list_sizes"] == [3]
    assert envelope["queries"][1]["calls"][0]["input_profile"]["list_sizes"] == [3, 2]


def test_deferred_fusion_profiles_are_independent_snapshots(vector_node, recorder):
    observations = take_fusion_observations(vector_node.pure(_previous(), top_k=2))
    for qid in ("one", "two"):
        recorder.record(qid, "semantic_retrieval_vectorsearch", input_tokens=1,
                        model_id="test", node_id="semantic_retrieval")
    record_fusion_observations(["one", "two"], observations)
    observations[0][0]["input_profile"]["list_sizes"][0] = 999
    trace = recorder.to_trace_v1()[0]
    assert trace[-1]["input_profile"]["list_sizes"] == [3]
    assert "input_bytes" not in trace[-1] and "output_bytes" not in trace[-1]


def test_quality_react_records_real_single_list_fusion_on_each_visit(vector_node, recorder):
    from rag_stack_evaluator.static_rag_evaluator.agentic_react import run_react

    class Generator:
        rounds = 0

        def _pure(self, prompts, **_kwargs):
            self.rounds += 1
            text = "Search[single]" if self.rounds < 3 else "Finish[answer]"
            return [text] * len(prompts), [[1]] * len(prompts), None

    qa = _previous().assign(qid=["one", "two"], generation_gt=[["answer"], ["answer"]])
    result = run_react(
        qa_data=qa, retriever=vector_node, generator=Generator(),
        generator_model="test-generator", embedding_model_id="test-embedding",
        top_k=2, gen_params={}, max_iter=3,
    )
    assert result["agent_retrieval_calls"].tolist() == [2, 2]
    for trace in recorder.to_trace_v1():
        assert [call["stage"] for call in trace] == [
            "generator", "semantic_retrieval_encode", "semantic_retrieval_vectorsearch", "retrieval_fusion",
            "generator", "semantic_retrieval_encode", "semantic_retrieval_vectorsearch", "retrieval_fusion",
            "generator",
        ]
        fusion = [call for call in trace if call["stage"] == "retrieval_fusion"]
        assert all(call["input_profile"]["list_sizes"] == [3] for call in fusion)
        assert all(call["output_profile"] == {"items": 2, "unique_items": 2} for call in fusion)
