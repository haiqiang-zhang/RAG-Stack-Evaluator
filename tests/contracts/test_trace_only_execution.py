"""Serving-only previews must not invoke explicit node-level judges."""

from copy import deepcopy

import pandas as pd
import pytest

from rag_stack_evaluator.static_rag_evaluator.static_rag_evaluator import (
    StaticRAGEvaluatorQualityOnly,
)


@pytest.mark.parametrize("serving_only", [False, True])
def test_serving_only_skips_node_judge_without_changing_retrieval(serving_only):
    config = {
        "eval_backend_setting": {"performance_only": serving_only},
        "node_lines": [{"node_line_name": "retrieval", "nodes": [{
            "stage": "lexical_retrieval", "top_k": 3,
            "strategy": {"metrics": ["deepeval_contextual_precision"], "strategy": "mean"},
            "modules": [{"component": "bm25", "bm25_tokenizer": "porter_stemmer"}],
        }]}],
    }
    original = deepcopy(config)
    node = StaticRAGEvaluatorQualityOnly._parse_node_lines(config, [])["retrieval"][0]
    assert node.strategy["metrics"] == ([] if serving_only else ["deepeval_contextual_precision"])
    assert node.strategy["strategy"] == "mean"
    assert node.get_param_combinations()[1] == [{"top_k": 3, "bm25_tokenizer": "porter_stemmer"}]
    assert config == original


def test_serving_only_skips_final_judge_even_if_metrics_are_present(monkeypatch):
    evaluator = object.__new__(StaticRAGEvaluatorQualityOnly)

    def forbid_metric_inputs(*args, **kwargs):
        raise AssertionError("Serving-only evaluation must not prepare judge requests")

    monkeypatch.setattr(evaluator, "_create_metric_inputs", forbid_metric_inputs)
    result = evaluator._evaluate_final_result(
        pd.DataFrame(),
        {"performance_only": True, "metrics": ["deepeval_answer_correctness"]},
        "generator",
    )
    assert result == {}
