import ast
import logging
from pathlib import Path

import rag_stack_evaluator

import pytest


ROOT = Path(rag_stack_evaluator.__file__).resolve().parent.parent


def _function_node(relative_path: str, class_name: str, function_name: str):
    tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    )


@pytest.mark.parametrize(
    ("relative_path", "class_name"),
    [
        (
            "rag_stack_evaluator/static_rag_evaluator/nodes/promptmaker/base.py",
            "BasePromptMaker",
        ),
        (
            "rag_stack_evaluator/static_rag_evaluator/nodes/retrieval/base.py",
            "BaseRetrieval",
        ),
        (
            "rag_stack_evaluator/static_rag_evaluator/nodes/passagereranker/base.py",
            "BasePassageReranker",
        ),
        (
            "rag_stack_evaluator/static_rag_evaluator/nodes/passagecompressor/base.py",
            "BasePassageCompressor",
        ),
    ],
)
def test_measured_hot_path_does_not_log_at_info(relative_path, class_name):
    function = _function_node(relative_path, class_name, "cast_to_run")
    logger_calls = {
        node.func.attr
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "logger"
    }

    assert "debug" in logger_calls
    assert "info" not in logger_calls


@pytest.mark.parametrize(
    ("relative_path", "class_name"),
    [
        (
            "rag_stack_evaluator/static_rag_evaluator/nodes/retrieval/base.py",
            "BaseRetrieval",
        ),
        (
            "rag_stack_evaluator/static_rag_evaluator/nodes/passagereranker/base.py",
            "BasePassageReranker",
        ),
        (
            "rag_stack_evaluator/static_rag_evaluator/nodes/passagecompressor/base.py",
            "BasePassageCompressor",
        ),
    ],
)
def test_aux_node_destructor_skips_logging_during_interpreter_shutdown(
    relative_path, class_name,
):
    function = _function_node(relative_path, class_name, "__del__")
    finalizing_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "sys"
        and node.func.attr == "is_finalizing"
    ]
    assert len(finalizing_calls) == 1


def test_measured_vllm_launchers_disable_uvicorn_access_logs():
    launchers = (
        (
            "rag_stack_evaluator/static_rag_evaluator/measured/vllm_subprocess.py",
            "VllmSubprocess",
            "__init__",
        ),
        (
            "rag_stack_evaluator/static_rag_evaluator/measured/vllm_pd_pair.py",
            "VllmPdPair",
            "_launch_engine",
        ),
    )

    for relative_path, class_name, function_name in launchers:
        function = _function_node(relative_path, class_name, function_name)
        string_literals = {
            node.value
            for node in ast.walk(function)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        assert "--disable-uvicorn-access-log" in string_literals


def test_long_context_reorder_logs_compressor_fallback_once(
    monkeypatch, caplog,
):
    from rag_stack_evaluator.static_rag_evaluator.nodes.promptmaker import (
        long_context_reorder as module,
    )

    monkeypatch.setattr(module, "_summarizer_fallback_logged", False)
    prompt_maker = module.LongContextReorder.__new__(module.LongContextReorder)
    prompt = "Question: {query}\nContext: {retrieved_contents}"
    queries = ["q0", "q1"]
    contents = [["a", "b"], ["c"]]
    mismatched_scores = [[1.0], [0.5, 0.2]]

    with caplog.at_level(logging.INFO, logger="RAG-Stack"):
        first = prompt_maker._pure(prompt, queries, contents, mismatched_scores)
        second = prompt_maker._pure(prompt, queries, contents, mismatched_scores)

    assert first == second == [
        "Question: q0\nContext: a\n\nb",
        "Question: q1\nContext: c",
    ]
    fallback_records = [
        record
        for record in caplog.records
        if "long_context_reorder passes contents through unchanged" in record.message
    ]
    assert len(fallback_records) == 1
