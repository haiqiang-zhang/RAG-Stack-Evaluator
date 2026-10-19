"""Compatibility imports for the optional, independently packaged MT-RAG bridge.

The implementation belongs to the example plugin rather than the static backend.
Install it with ``python -m pip install --no-deps -e examples/mtrag`` from a host
checkout to use the static ``mt_rag_faithfulness`` metric or these legacy imports.
"""

from importlib import import_module

__all__ = ["load_tasks", "to_qa_rows", "build_judge_records", "score_with_mtrag",
           "mt_rag_faithfulness"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    try:
        scoring = import_module("rag_stack_mtrag.scoring")
    except ModuleNotFoundError as error:
        if error.name not in {"rag_stack_mtrag", "rag_stack_mtrag.scoring"}:
            raise
        raise ImportError(
            "MT-RAG integration requires the optional example plugin. From the "
            "RAG-Stack checkout, run: python -m pip install --no-deps -e "
            "examples/mtrag"
        ) from error
    return getattr(scoring, name)
