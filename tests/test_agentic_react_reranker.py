import pandas as pd

from rag_stack_evaluator.static_rag_evaluator import recording
from rag_stack_evaluator.static_rag_evaluator.agentic_react import run_react


class _FakeGenerator:
    _subprocess = None

    def _pure(self, prompts, **kwargs):
        texts = []
        for prompt in prompts:
            if "second passage" in prompt:
                texts.append("Action 2: Finish[final answer]")
            else:
                texts.append("Action 1: Search[test entity]")
        return texts, [[1, 2] for _ in texts], None


class _FakeRetriever:
    def __init__(self):
        self.corpus_df = pd.DataFrame(
            {
                "doc_id": ["d1", "d2"],
                "contents": ["first passage", "second passage"],
            }
        )

    def _pure(self, queries, **kwargs):
        assert queries == [["test entity"]]
        return [["d1", "d2"]], [[0.1, 0.2]]


class _FakeReranker:
    def pure(self, previous_result, **kwargs):
        # The real reranker's cast_to_run runs validate_qa_dataset on its
        # input — the react loop must satisfy that node contract.
        for col in ("qid", "query", "generation_gt"):
            assert col in previous_result.columns, f"missing {col}"
        assert previous_result["retrieved_ids_semantic"].iloc[0] == ["d1", "d2"]
        assert kwargs["top_k"] == 1
        return pd.DataFrame(
            {
                "retrieved_contents": [["second passage"]],
                "retrieved_ids": [["d2"]],
                "retrieve_scores": [[0.9]],
            }
        )


def test_react_search_can_rerank_and_preserve_semantic_columns(monkeypatch):
    monkeypatch.setattr(
        recording,
        "count_tokens_batch",
        lambda texts, model_id=None: [1 for _ in texts],
    )

    rec = recording.TraceRecorder()
    recording.set_current_recorder(rec)
    try:
        result = run_react(
            qa_data=pd.DataFrame({
                "qid": ["qa-0"],
                "query": ["question"],
                "generation_gt": [["gold answer"]],
                "__qid__": ["q0"],
            }),
            retriever=_FakeRetriever(),
            generator=_FakeGenerator(),
            reranker=_FakeReranker(),
            generator_model="fake-generator",
            embedding_model_id="fake-embedding",
            top_k=2,
            gen_params={"max_tokens": 8},
            reranker_params={"top_k": 1},
            reranker_model_id="fake-reranker",
            max_iter=4,
        )
    finally:
        recording.clear_current_recorder()

    row = result.iloc[0]
    assert row["generated_texts"] == "final answer"
    assert row["agent_generate_calls"] == 2
    assert row["agent_retrieval_calls"] == 1
    assert bool(row["agent_truncated"]) is False
    assert row["retrieved_ids_semantic"] == ["d1", "d2"]
    assert row["retrieved_ids"] == ["d2"]

    trace = rec.to_trace_v1()
    assert [[call["stage"] for call in trace_i] for trace_i in trace] == [
        ["generator", "semantic_retrieval_encode", "semantic_retrieval_vectorsearch", "passage_reranker", "generator"]
    ]


class _NeverFinishGenerator:
    """Always emits a Search action — exercises the max_iter cap."""

    _subprocess = None

    def __init__(self):
        self.calls = 0

    def _pure(self, prompts, **kwargs):
        self.calls += 1
        return (
            [f"Action {self.calls}: Search[loop query]" for _ in prompts],
            [[1, 2] for _ in prompts],
            None,
        )


def test_react_max_iter_caps_rounds_and_skips_dangling_search(monkeypatch):
    monkeypatch.setattr(
        recording,
        "count_tokens_batch",
        lambda texts, model_id=None: [1 for _ in texts],
    )

    rec = recording.TraceRecorder()
    recording.set_current_recorder(rec)
    gen = _NeverFinishGenerator()
    retriever = _FakeRetriever()
    retriever._pure = lambda queries, **kw: (
        [["d1", "d2"] for _ in queries], [[0.1, 0.2] for _ in queries],
    )
    try:
        result = run_react(
            qa_data=pd.DataFrame({
                "qid": ["qa-0"],
                "query": ["question"],
                "generation_gt": [["gold answer"]],
                "__qid__": ["q0"],
            }),
            retriever=retriever,
            generator=gen,
            generator_model="fake-generator",
            embedding_model_id="fake-embedding",
            top_k=2,
            gen_params={"max_tokens": 8},
            max_iter=2,
        )
    finally:
        recording.clear_current_recorder()

    row = result.iloc[0]
    # max_iter=2 → exactly 2 generate rounds; the 2nd (last allowed) round's
    # Search is skipped (its Observation could never be consumed) → 1 retrieval.
    assert row["agent_generate_calls"] == 2
    assert row["agent_retrieval_calls"] == 1
    assert bool(row["agent_truncated"]) is True
    assert row["generated_texts"] == "No valid answer found"
    trace = rec.to_trace_v1()
    assert [[c["stage"] for c in q] for q in trace] == [
        ["generator", "semantic_retrieval_encode", "semantic_retrieval_vectorsearch", "generator"]
    ]


class _WhitespaceGenerator:
    _subprocess = None

    def _pure(self, prompts, **kwargs):
        return [" \n\t" for _ in prompts], [[1] for _ in prompts], None


def test_react_finished_whitespace_answers_keep_all_metric_rows(monkeypatch):
    monkeypatch.setattr(
        recording,
        "count_tokens_batch",
        lambda texts, model_id=None: [1 for _ in texts],
    )

    qa_count = 100
    qa_data = pd.DataFrame({
        "qid": [f"qa-{i}" for i in range(qa_count)],
        "query": [f"question {i}" for i in range(qa_count)],
        "generation_gt": [[f"gold {i}"] for i in range(qa_count)],
        "__qid__": [f"q{i}" for i in range(qa_count)],
    })
    rec = recording.TraceRecorder()
    recording.set_current_recorder(rec)
    try:
        result = run_react(
            qa_data=qa_data,
            retriever=_FakeRetriever(),
            generator=_WhitespaceGenerator(),
            generator_model="fake-generator",
            embedding_model_id="fake-embedding",
            top_k=2,
            gen_params={"max_tokens": 8},
            max_iter=2,
        )
    finally:
        recording.clear_current_recorder()

    assert len(result) == qa_count
    assert result["qid"].tolist() == qa_data["qid"].tolist()
    assert result["generated_texts"].tolist() == [
        "No valid answer found"
    ] * qa_count
    assert result["agent_generate_calls"].tolist() == [1] * qa_count
    assert result["agent_retrieval_calls"].tolist() == [0] * qa_count
    assert result["agent_truncated"].tolist() == [False] * qa_count
    assert len(rec.to_trace_v1()) == qa_count
    assert all(
        [call["stage"] for call in query_trace] == ["generator"]
        for query_trace in rec.to_trace_v1()
    )
