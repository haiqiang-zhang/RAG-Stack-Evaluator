"""CPU-only checks for query-expansion fanout and rank-fusion semantics."""

from types import SimpleNamespace

import pandas as pd
import pytest

from rag_stack_evaluator.static_rag_evaluator.nodes.hybridretrieval.hybrid_rrf import rrf_pure
from rag_stack_evaluator.static_rag_evaluator.nodes.queryexpansion.base import (
    check_expanded_query,
)
from rag_stack_evaluator.static_rag_evaluator.nodes.queryexpansion.multi_query_expansion import (
    MultiQueryExpansion,
)


def test_blank_separators_do_not_duplicate_original_question():
    assert check_expanded_query("original", ["original", " first ", "", "  ", "second"]) == [
        "original", "first", "second",
    ]


def test_duplicate_expansions_keep_first_occurrence_and_order():
    assert check_expanded_query("original", ["b", "a", " b ", "a", "c"]) == ["b", "a", "c"]


@pytest.mark.parametrize("expansions", [[], [""], [" ", "\n"]])
def test_unusable_expansion_falls_back_once(expansions):
    assert check_expanded_query("original", expansions) == ["original"]


def test_hyde_text_is_retained_without_adding_original_question():
    paragraph = "A hypothetical passage.\nIts second sentence."
    assert check_expanded_query("original", [paragraph]) == [paragraph]


def test_multi_query_blank_line_format_produces_four_retrieval_queries():
    module = object.__new__(MultiQueryExpansion)
    module.generator = SimpleNamespace(pure=lambda **kwargs: pd.DataFrame({
        "generated_texts": ["alternative one\n\nalternative two\n\nalternative three"],
    }))
    raw = module._pure(["original"])
    assert module._check_expanded_query(["original"], raw) == [[
        "original", "alternative one", "alternative two", "alternative three",
    ]]


def test_formatting_does_not_increase_original_query_rank_fusion_weight():
    queries = check_expanded_query("original", ["original", "", "alternative", "original"])
    rankings = {"original": ["document_a"], "alternative": ["document_b"]}
    ids, scores = rrf_pure(tuple(rankings[q] for q in queries), tuple([1.0] for _ in queries), 60, 2)
    assert set(ids) == {"document_a", "document_b"}
    assert scores == pytest.approx([1 / 61, 1 / 61])
