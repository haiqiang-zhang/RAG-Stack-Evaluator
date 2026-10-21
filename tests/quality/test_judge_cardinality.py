"""CPU-only regressions for complete contextual-precision judging."""

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from deepeval.metrics.contextual_precision.template import ContextualPrecisionTemplate

from rag_stack_evaluator.static_rag_evaluator.evaluation.metric import deepeval_metrics as dm
from rag_stack_evaluator.static_rag_evaluator.evaluation.metric import judge_integrity as integrity
from rag_stack_evaluator.static_rag_evaluator.schema.metricinput import MetricInput


class FakeClient:
    model = "fake-judge"
    _default_request_kwargs = {"max_tokens": 1024}
    _request_extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

    def __init__(self, responder):
        self.responder = responder
        self.calls = []

    async def structured_output(self, messages, response_format, **kwargs):
        call = {"messages": messages, "schema": response_format, "kwargs": kwargs}
        self.calls.append(call)
        return self.responder(call, self.calls)

    async def aclose(self):
        pass


def verdicts(count, first_no=False):
    return {"verdicts": [
        {"verdict": "no" if first_no and i == 0 else "yes", "reason": f"Evidence {i}"}
        for i in range(count)
    ]}


def inputs(count=64, qid="qid-1", query="test query"):
    return MetricInput(qid=qid, query=query, generation_gt=["Expected answer."],
                       generated_texts="Generated answer.",
                       retrieved_contents=[f"Passage {i}" for i in range(count)])


def wire_client(monkeypatch, client):
    monkeypatch.setattr(dm, "_build_adapter", lambda model, kwargs: dm._AIClientDeepEvalAdapter(client))


def test_schema_rejects_short_and_long_lists():
    schema = integrity.exact_verdict_schema(dm.ContextualPrecisionVerdict, 64)
    array = schema.model_json_schema()["properties"]["verdicts"]
    assert array["minItems"] == array["maxItems"] == 64
    for count in (0, 6, 63, 65):
        with pytest.raises(ValidationError):
            schema.model_validate(verdicts(count))
    assert len(schema.model_validate(verdicts(64)).verdicts) == 64


def test_valid_case_keeps_original_prompt_and_score(monkeypatch, tmp_path):
    client = FakeClient(lambda call, calls: call["schema"].model_validate(verdicts(3, first_no=True)))
    wire_client(monkeypatch, client)
    mi = inputs(3)
    scores = dm.deepeval_context_precision([mi], model="vllm/fake", judge_audit_dir=str(tmp_path))
    assert scores == pytest.approx([(1 / 2 + 2 / 3) / 2])
    original = ContextualPrecisionTemplate.generate_verdicts(
        input=mi.query, expected_output="Expected answer.", retrieval_context=mi.retrieved_contents, multimodal=False,
    )
    assert client.calls[0]["messages"] == [{"role": "user", "content": original}]
    assert client.calls[0]["kwargs"] == {"max_tokens": 1024, "structured_output_mode": "parse"}
    receipt = json.loads(next(tmp_path.rglob("case_*.json")).read_text())
    assert receipt["qid"] == "qid-1"
    assert receipt["verdict_count"] == receipt["retrieval_context_count"] == 3
    assert len(receipt["raw_verdicts"]) == 3
    assert receipt["guard"]["attempts"][0]["status"] == "accepted"
    assert receipt["template_sha256"]


def test_retry_is_per_case_and_keeps_rejected_raw_evidence(monkeypatch, tmp_path):
    counts = {"good": 0, "bad": 0}
    def respond(call, calls):
        key = "bad" if "bad query" in call["messages"][0]["content"] else "good"
        counts[key] += 1
        return verdicts(6 if key == "bad" and counts[key] == 1 else 64)
    client = FakeClient(respond)
    wire_client(monkeypatch, client)
    scores = dm.deepeval_context_precision(
        [inputs(qid="good", query="good query"), inputs(qid="bad", query="bad query")],
        model="vllm/fake", judge_audit_dir=str(tmp_path),
    )
    assert scores == [1.0, 1.0]
    assert counts == {"good": 1, "bad": 2}
    receipt = next(json.loads(p.read_text()) for p in tmp_path.rglob("case_*.json") if json.loads(p.read_text())["qid"] == "bad")
    attempts = receipt["guard"]["attempts"]
    assert [a["max_tokens"] for a in attempts] == [1024, 4096]
    assert len(attempts[0]["raw_verdict_output"]["verdicts"]) == 6
    assert [a["status"] for a in attempts] == ["rejected", "accepted"]


def test_short_json_fallback_cannot_be_scored(monkeypatch):
    client = FakeClient(lambda call, calls: json.dumps(verdicts(6)))
    wire_client(monkeypatch, client)
    with pytest.raises(RuntimeError, match="partial metric results discarded"):
        dm.deepeval_context_precision([inputs()], model="vllm/fake", judge_max_attempts=3)
    assert len(client.calls) == 3
    assert [c["kwargs"]["max_tokens"] for c in client.calls] == [1024, 4096, 8192]


def test_context_budget_never_exceeds_remaining_window(monkeypatch):
    monkeypatch.setattr(dm, "count_chat_tokens", lambda *args: 110)
    client = FakeClient(lambda call, calls: verdicts(1))
    wire_client(monkeypatch, client)
    with pytest.raises(RuntimeError, match="partial metric results discarded"):
        dm.deepeval_context_precision(
            [inputs()], model="vllm/fake", judge_context_window=128,
            judge_tokenizer_path="/local/tokenizer", judge_max_attempts=3,
        )
    assert [c["kwargs"]["max_tokens"] for c in client.calls] == [18, 18, 18]


def test_oversize_prompt_fails_without_judge_call(monkeypatch):
    monkeypatch.setattr(dm, "count_chat_tokens", lambda *args: 129)
    client = FakeClient(lambda call, calls: verdicts(64))
    wire_client(monkeypatch, client)
    with pytest.raises(RuntimeError, match="no output tokens remain"):
        dm.deepeval_context_precision(
            [inputs()], model="vllm/fake", judge_context_window=128,
            judge_tokenizer_path="/local/tokenizer",
        )
    assert client.calls == []


def test_second_post_measure_guard_rejects_bypassed_generation(monkeypatch):
    class BadMetric:
        def __init__(self, **kwargs):
            self.score = 1.0
            self.error = None
            self.verdicts = [SimpleNamespace(verdict="yes", reason="partial")]
        async def a_measure(self, case, **kwargs):
            return self.score
    client = FakeClient(lambda call, calls: verdicts(64))
    wire_client(monkeypatch, client)
    monkeypatch.setattr(dm, "_GuardedContextualPrecisionMetric", BadMetric)
    with pytest.raises(RuntimeError, match="requires exactly 64 verdicts; received 1"):
        dm.deepeval_context_precision([inputs()], model="vllm/fake")


def test_empty_context_is_zero_with_receipt_and_no_judge(monkeypatch, tmp_path):
    monkeypatch.setattr(dm, "_build_adapter", lambda *args: pytest.fail("No judge for empty retrieval"))
    assert dm.deepeval_context_precision([inputs(0)], judge_audit_dir=str(tmp_path)) == [0.0]
    receipt = json.loads(next(tmp_path.rglob("case_*.json")).read_text())
    assert receipt["kind"] == "deterministic_empty_retrieval_zero"
    assert receipt["score"] == 0.0
    assert receipt["qid"] == "qid-1"


def test_receipts_keep_original_indices_when_empty_cases_are_not_judged(monkeypatch, tmp_path):
    client = FakeClient(lambda call, calls: call["schema"].model_validate(verdicts(1)))
    wire_client(monkeypatch, client)
    assert dm.deepeval_context_precision(
        [inputs(0, qid="empty"), inputs(1, qid="judged")], model="vllm/fake",
        judge_audit_dir=str(tmp_path),
    ) == [0.0, 1.0]
    receipts = [json.loads(p.read_text()) for p in tmp_path.rglob("case_*.json")]
    assert {r["qid"]: r["case_index"] for r in receipts} == {"empty": 0, "judged": 1}


def test_nested_numpy_metric_inputs_are_serialized_without_loss(tmp_path):
    import numpy as np
    value = {"qid": np.int64(17), "retrieved_contents": np.array(["first", "second"], dtype=object),
             "tokens": [np.array([1, 2], dtype=np.int32)], "score": np.float64(.5)}
    integrity.atomic_json(tmp_path / "inputs.json", value)
    assert json.loads((tmp_path / "inputs.json").read_text()) == {
        "qid": 17, "retrieved_contents": ["first", "second"], "tokens": [[1, 2]], "score": .5,
    }


def test_context_configuration_requires_exact_tokenizer():
    with pytest.raises(ValueError, match="requires judge_tokenizer_path"):
        integrity.JudgeGuardConfig.pop_from({"judge_context_window": 32768})
    with pytest.raises(ValueError, match="positive integer"):
        integrity.JudgeGuardConfig.pop_from({"judge_max_attempts": True})


def test_local_tokenizer_uses_local_files_and_expands_environment(monkeypatch, tmp_path):
    import sys
    calls = []
    fake = SimpleNamespace(from_pretrained=lambda path, **kw: calls.append((path, kw)) or "tokenizer")
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=fake))
    monkeypatch.setenv("TEST_JUDGE_TOKENIZER", str(tmp_path))
    integrity._local_tokenizer.cache_clear()
    assert integrity._local_tokenizer("${TEST_JUDGE_TOKENIZER}") == "tokenizer"
    assert calls == [(str(tmp_path), {"local_files_only": True, "trust_remote_code": False})]
    integrity._local_tokenizer.cache_clear()


def test_recall_uses_answer_sentence_verdicts_not_passage_count(monkeypatch, tmp_path):
    client = FakeClient(lambda call, calls: call["schema"].model_validate(verdicts(1)))
    wire_client(monkeypatch, client)
    assert dm.deepeval_context_recall([inputs(64)], model="vllm/fake", judge_audit_dir=str(tmp_path)) == [1.0]
    receipt = json.loads(next(tmp_path.rglob("case_*.json")).read_text())
    assert receipt["metric"] == "ContextualRecallMetric"
    assert receipt["retrieval_context_count"] == 64
    assert receipt["verdict_count"] == 1
    assert receipt["guard"] is None


def test_all_dataflow_final_inputs_are_saved_without_node_outputs(tmp_path):
    import pandas as pd
    from pathlib import Path
    from rag_stack_evaluator.static_rag_evaluator.static_rag_evaluator import StaticRAGEvaluatorQualityOnly
    evaluator = object.__new__(StaticRAGEvaluatorQualityOnly)
    evaluator.qa_data = pd.DataFrame({"qid": ["qa-0", "qa-1"], "query": ["q0", "q1"],
                                   "generation_gt": [["a0"], ["a1"]]})
    final = pd.DataFrame({"generated_texts": ["generated0", ""],
                          "retrieved_contents_semantic": [["raw0"], []],
                          "retrieved_contents": [["final0"], []]})
    first = evaluator._persist_final_metric_inputs(final, {}, str(tmp_path))
    second = evaluator._persist_final_metric_inputs(final, {}, str(tmp_path))
    assert first != second  # A retried evaluation preserves the previous evidence.
    record = json.loads((Path(first) / "metric_inputs.json").read_text())
    assert record["rows_sha256"] == integrity.json_hash(record["rows"])
    assert [row["qid"] for row in record["rows"]] == ["qa-0", "qa-1"]
    assert record["rows"][0]["retrieved_contents"] == ["final0"]
    assert record["rows"][0]["retrieved_contents_semantic"] == ["raw0"]
    assert record["rows"][1]["generated_texts"] == ""
    assert record["rows"][1]["retrieved_contents"] == []
    assert not list(tmp_path.rglob("*.parquet"))


def test_aux_sample_and_objective_coverage_are_preserved_in_saved_scores(tmp_path):
    import pandas as pd
    from rag_stack_evaluator.static_rag_evaluator.static_rag_evaluator import StaticRAGEvaluatorQualityOnly
    evaluator = object.__new__(StaticRAGEvaluatorQualityOnly)
    evaluator.qa_data = pd.DataFrame({"qid": [f"qa-{i}" for i in range(100)],
                                    "query": [f"query{i}" for i in range(100)],
                                    "generation_gt": [["answer"] for _ in range(100)]})
    final = pd.DataFrame({"generated_texts": ["answer"] * 100,
                          "retrieved_contents_semantic": [["context"] for _ in range(100)]})
    seen = {}
    def metric(name):
        def run(metric_inputs, **kwargs):
            seen[name] = [mi.qid for mi in metric_inputs]
            assert kwargs["judge_audit_dir"] == str(tmp_path)
            return [0.5] * len(metric_inputs)
        return run
    evaluator._ALL_METRIC_FUNC_DICT = {
        name: metric(name) for name in ("deepeval_context_precision", "deepeval_answer_correctness")
    }
    config = {"metrics": [{"metric_name": name} for name in evaluator._ALL_METRIC_FUNC_DICT],
              "combined_quality": {"metrics": ["deepeval_answer_correctness"]},
              "submetric_query_subset": 15, "_judge_audit_dir": str(tmp_path)}
    assert evaluator._evaluate_final_result(final, config) == {
        "deepeval_context_precision": 0.5, "deepeval_answer_correctness": 0.5,
    }
    assert seen["deepeval_context_precision"] == [f"qa-{i}" for i in range(15)]
    assert len(seen["deepeval_answer_correctness"]) == 100
    scores = json.loads((tmp_path / "deepeval_context_precision_scores.json").read_text())
    assert scores["scores"] == [0.5] * 15
