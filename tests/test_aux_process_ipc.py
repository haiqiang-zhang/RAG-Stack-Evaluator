"""AuxProcessStage IPC: column pruning, id-keyed reranker reply, gating.

Covers r20: retrieval joins the process-isolated set, the parent ships only
the columns a stage's pure() reads, and rerankers reply ids+scores only
(parent rebuilds contents from its own input copy, with a verified fallback
when a module rewrites text).
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from rag_stack_evaluator.static_rag_evaluator.measured.aux_process import (  # noqa: E402
    AUX_PROCESS_STAGES,
    RETRIEVAL_PROCESS_STAGES,
    AuxProcessStage,
    _pack_reranker_result,
    _unpack_reranker_result,
    process_isolated_stage,
)

import aux_process_fixtures as fx  # noqa: E402


class ReportingReranker(fx.FakeReranker):
    """Spawn-importable fake exposing the MonoT5 execution-report hook."""

    def __init__(self, project_dir, **kwargs):
        super().__init__(project_dir, **kwargs)
        self._last_forward_execution_report = None

    def pure(self, previous_result, *args, **kwargs):
        emit_report = kwargs.pop("emit_execution_report", True)
        result = super().pure(previous_result, *args, **kwargs)
        self._last_forward_execution_report = (
            {
                "schema": "monot5_forward_execution",
                "requested_forward_microbatch": 4,
                "successful_forward_microbatches": [4, 2],
                "actual_forward_microbatch": 4,
                "oom_fallback_count": 0,
                "failed_forward_microbatches": [],
            }
            if emit_report
            else None
        )
        return result

    def pop_last_forward_execution_report(self):
        report = self._last_forward_execution_report
        self._last_forward_execution_report = None
        return report


def _reranker_frame(n=3, fanout=4):
    return pd.DataFrame(
        {
            "qid": [f"q{i}" for i in range(n)],
            "query": [f"question {i}" for i in range(n)],
            "generation_gt": [[f"gt {i}"] for i in range(n)],
            "__qid__": [f"measured-{i}" for i in range(n)],
            "retrieved_contents": [
                [f"passage {i}-{j} " + "x" * 64 for j in range(fanout)]
                for i in range(n)
            ],
            "retrieved_ids": [
                [f"doc-{i}-{j}" for j in range(fanout)] for i in range(n)
            ],
            "retrieve_scores": [[1.0 - 0.1 * j for j in range(fanout)]] * n,
            "secret_big_column": ["B" * 4096] * n,
            "prompts": ["should not ship"] * n,
        }
    )


def _pruned(df):
    return df.drop(columns=["secret_big_column", "prompts"])


def test_pack_unpack_round_trip_exact():
    df = _pruned(_reranker_frame())
    result = fx.FakeReranker(".").pure(df)
    slim = _pack_reranker_result(df, result)
    assert slim is not None
    assert "retrieved_contents" not in slim.columns
    rebuilt = _unpack_reranker_result(df, slim)
    assert list(rebuilt.columns)[:3] == [
        "retrieved_contents", "retrieved_ids", "retrieve_scores",
    ]
    pd.testing.assert_frame_equal(
        rebuilt.reset_index(drop=True), result.reset_index(drop=True)
    )


def test_pack_refuses_rewritten_contents():
    df = _pruned(_reranker_frame())
    result = fx.FakeRewritingReranker(".").pure(df)
    assert _pack_reranker_result(df, result) is None


def test_pack_refuses_unknown_ids():
    df = _pruned(_reranker_frame())
    result = fx.FakeReranker(".").pure(df)
    result.at[1, "retrieved_ids"] = ["doc-elsewhere"]
    assert _pack_reranker_result(df, result) is None


@pytest.fixture()
def child_pythonpath():
    """Let the spawn child import the fixtures module."""
    tests_dir = os.path.dirname(__file__)
    prior = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = (
        tests_dir if not prior else tests_dir + os.pathsep + prior
    )
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = prior


def test_reranker_child_process_prunes_and_rebuilds(child_pythonpath):
    df = _reranker_frame()
    stage = AuxProcessStage(
        fx.FakeReranker, ".", {}, stage="passage_reranker"
    )
    try:
        out = stage.pure(df.copy(deep=True), top_k=2)
        expected = fx.FakeReranker(".").pure(_pruned(df))
        pd.testing.assert_frame_equal(
            out.reset_index(drop=True), expected.reset_index(drop=True)
        )
    finally:
        stage.close()


def test_rewriting_reranker_falls_back_to_full_reply(child_pythonpath):
    df = _reranker_frame()
    stage = AuxProcessStage(
        fx.FakeRewritingReranker, ".", {}, stage="passage_reranker"
    )
    try:
        out = stage.pure(df.copy(deep=True))
        assert all(
            c.endswith("REWRITTEN") for c in out["retrieved_contents"].iloc[0]
        )
    finally:
        stage.close()


def test_execution_report_crosses_aux_ipc_and_is_cleared_per_call(
    child_pythonpath,
):
    from rag_stack.cost_model.reranker_policy import (
        MONOT5_FORWARD_EXECUTION_SCHEMA,
    )

    stage = AuxProcessStage(
        ReportingReranker, ".", {}, stage="passage_reranker"
    )
    try:
        stage.pure(_reranker_frame(), emit_execution_report=True)
        assert stage.last_forward_execution_report == {
            "schema": MONOT5_FORWARD_EXECUTION_SCHEMA,
            "requested_forward_microbatch": 4,
            "successful_forward_microbatches": [4, 2],
            "actual_forward_microbatch": 4,
            "oom_fallback_count": 0,
            "failed_forward_microbatches": [],
        }

        stage.pure(_reranker_frame(), emit_execution_report=False)
        assert stage.last_forward_execution_report is None
    finally:
        stage.close()


def test_retrieval_child_process_prunes_input(child_pythonpath):
    df = _reranker_frame()  # carries contents columns retrieval must not see
    stage = AuxProcessStage(
        fx.FakeRetrieval, ".", {"device": "cpu"}, stage="semantic_retrieval"
    )
    try:
        out = stage.pure(df.copy(deep=True), top_k=1)
        assert len(out) == len(df)
        assert "retrieved_contents_semantic" in out.columns
    finally:
        stage.close()


def test_stage_gating_env(monkeypatch):
    for st in AUX_PROCESS_STAGES | RETRIEVAL_PROCESS_STAGES:
        assert process_isolated_stage(st)
    assert not process_isolated_stage("prompt_maker")
    monkeypatch.setenv("RAG_STACK_AUX_PROCESS", "0")
    assert not process_isolated_stage("passage_reranker")
    assert not process_isolated_stage("semantic_retrieval")
