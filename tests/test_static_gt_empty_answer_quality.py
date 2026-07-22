import logging

import pandas as pd
import pytest

from rag_stack_evaluator.static_rag_evaluator.evaluation.metric import deepeval_metrics
from rag_stack_evaluator.static_rag_evaluator.schema.metricinput import MetricInput
from rag_stack_evaluator.static_rag_evaluator.static_rag_evaluator import (
	StaticRAGEvaluatorQualityOnly,
)


def _evaluator(n_rows: int = 2) -> StaticRAGEvaluatorQualityOnly:
	evaluator = object.__new__(StaticRAGEvaluatorQualityOnly)
	evaluator.qa_data = pd.DataFrame(
		{
			"query": [f"q{i}" for i in range(n_rows)],
			"generation_gt": [[f"gold-{i}"] for i in range(n_rows)],
		}
	)
	return evaluator


def _final_result(*, include_answers: bool = True) -> pd.DataFrame:
	data = {
		"retrieved_contents": [["context-0"], ["context-1"]],
		"retrieved_contents_semantic": [["context-0"], ["context-1"]],
	}
	if include_answers:
		data["generated_texts"] = ["   ", "non-empty answer"]
	return pd.DataFrame(data)


def _metrics(*names: str) -> dict:
	return {
		"strategy": "mean",
		"metrics": [
			{"metric_name": name, "model": "dummy-judge"}
			for name in names
		],
	}


def test_metric_input_preserves_emitted_empty_answer_only():
	inputs = MetricInput.from_dataframe(
		pd.DataFrame(
			{
				"query": ["   "],
				"generated_texts": ["   "],
			}
		)
	)

	assert inputs[0].query is None
	assert inputs[0].generated_texts == ""


def test_metric_input_distinguishes_empty_retrieval_from_missing_field():
	emitted_empty = MetricInput.from_dataframe(
		pd.DataFrame({"query": ["q"], "retrieved_contents": [[]]})
	)[0]
	missing = MetricInput.from_dataframe(pd.DataFrame({"query": ["q"]}))[0]

	assert emitted_empty.retrieved_contents == []
	assert missing.retrieved_contents is None


def test_empty_answers_are_zero_in_correctness_and_faithfulness_aggregates(
	monkeypatch, caplog,
):
	judged_batches = []

	def fake_measure_batch(
		metric_class, metric_inputs, build_case_fn, model, threshold,
		ai_client_kwargs,
	):
		judged_batches.append((metric_class, list(metric_inputs)))
		assert [mi.generated_texts for mi in metric_inputs] == ["non-empty answer"]
		return [0.8]

	monkeypatch.setattr(deepeval_metrics, "_measure_batch", fake_measure_batch)
	caplog.set_level(logging.INFO, logger="RAG-Stack")

	quality = _evaluator()._evaluate_final_result(
		_final_result(),
		_metrics("deepeval_answer_correctness", "deepeval_faithfulness"),
	)

	assert quality == {
		"deepeval_answer_correctness": pytest.approx(0.4),
		"deepeval_faithfulness": pytest.approx(0.4),
	}
	assert len(judged_batches) == 2
	coverage_logs = [
		record.getMessage() for record in caplog.records
		if "answer coverage" in record.getMessage()
	]
	assert len(coverage_logs) == 2
	assert all("scored=2/2" in message for message in coverage_logs)
	assert all("judged=1 empty_answer_zero=1" in message for message in coverage_logs)


@pytest.mark.parametrize(
	"metric_name",
	("deepeval_answer_correctness", "deepeval_faithfulness"),
)
def test_answer_metrics_fail_closed_when_generated_texts_field_is_missing(
	monkeypatch, metric_name,
):
	monkeypatch.setattr(
		deepeval_metrics,
		"_measure_batch",
		lambda *args, **kwargs: pytest.fail("judge must not run for malformed input"),
	)

	with pytest.raises(
		ValueError,
		match=r"structurally missing or invalid.*generated_texts",
	):
		_evaluator()._evaluate_final_result(
			_final_result(include_answers=False),
			_metrics(metric_name),
		)


def test_context_precision_and_recall_do_not_require_generated_texts(monkeypatch):
	def fake_measure_batch(
		metric_class, metric_inputs, build_case_fn, model, threshold,
		ai_client_kwargs,
	):
		assert len(metric_inputs) == 2
		return [0.25, 0.75]

	monkeypatch.setattr(deepeval_metrics, "_measure_batch", fake_measure_batch)

	quality = _evaluator()._evaluate_final_result(
		_final_result(include_answers=False),
		_metrics("deepeval_context_precision", "deepeval_context_recall"),
	)

	assert quality == {
		"deepeval_context_precision": pytest.approx(0.5),
		"deepeval_context_recall": pytest.approx(0.5),
	}


def test_no_retrieval_rows_are_zero_and_remain_in_all_context_means(
	monkeypatch, caplog,
):
	# A legal ReAct outcome: only 3 of the 15 sampled queries issued Search.
	# Every metric must aggregate over all 15 rows, not only the 3 survivors.
	contexts = [[] for _ in range(12)] + [[f"context-{i}"] for i in range(3)]
	final_result = pd.DataFrame(
		{
			"generated_texts": [f"answer-{i}" for i in range(15)],
			"retrieved_contents": contexts,
			"retrieved_contents_semantic": contexts,
		}
	)
	judged_batches = []

	def fake_measure_batch(
		metric_class, metric_inputs, build_case_fn, model, threshold,
		ai_client_kwargs,
	):
		judged_batches.append((metric_class.__name__, list(metric_inputs)))
		assert len(metric_inputs) == 3
		assert all(mi.retrieved_contents for mi in metric_inputs)
		return [0.6, 0.6, 0.6]

	monkeypatch.setattr(deepeval_metrics, "_measure_batch", fake_measure_batch)
	caplog.set_level(logging.INFO, logger="RAG-Stack")

	quality = _evaluator(15)._evaluate_final_result(
		final_result,
		_metrics(
			"deepeval_faithfulness",
			"deepeval_context_precision",
			"deepeval_context_recall",
		),
	)

	assert quality == {
		"deepeval_faithfulness": pytest.approx(0.12),
		"deepeval_context_precision": pytest.approx(0.12),
		"deepeval_context_recall": pytest.approx(0.12),
	}
	assert len(judged_batches) == 3
	coverage_logs = [
		record.getMessage() for record in caplog.records
		if "coverage" in record.getMessage()
	]
	assert len(coverage_logs) == 3
	assert all("scored=15/15" in message for message in coverage_logs)
	assert all("judged=3" in message for message in coverage_logs)
	assert all("empty_retrieval_zero=12" in message for message in coverage_logs)


@pytest.mark.parametrize(
	"metric_func",
	(
		deepeval_metrics.deepeval_faithfulness,
		deepeval_metrics.deepeval_context_precision,
		deepeval_metrics.deepeval_context_recall,
	),
)
def test_context_metrics_fail_closed_when_retrieval_field_is_missing(
	monkeypatch, metric_func,
):
	metric_inputs = MetricInput.from_dataframe(
		pd.DataFrame(
			{
				"query": ["q"],
				"generation_gt": [["gold"]],
				"generated_texts": ["answer"],
			}
		)
	)
	monkeypatch.setattr(
		deepeval_metrics,
		"_measure_batch",
		lambda *args, **kwargs: pytest.fail("judge must not run for malformed input"),
	)

	with pytest.raises(
		ValueError,
		match=r"structurally missing or invalid.*retrieved_contents",
	):
		metric_func(metric_inputs=metric_inputs, model="dummy-judge")
