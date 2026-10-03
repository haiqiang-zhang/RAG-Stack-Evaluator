"""CPU regression for sequence normalization at metric/client boundaries."""
import ast
from collections import UserDict
from collections.abc import Iterable, Mapping
from copy import deepcopy
import functools
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel as BM
from pydantic.v1 import BaseModel


def _production_functions():
    # Execute the actual small utilities without importing model backends.
    path = Path(__file__).resolve().parents[1] / (
        "rag_stack_evaluator/static_rag_evaluator/utils/util.py"
    )
    names = {"to_list", "convert_inputs_to_list"}
    nodes = [node for node in ast.parse(path.read_text()).body
             if getattr(node, "name", None) in names]
    assert {node.name for node in nodes} == names
    namespace = dict(np=np, pd=pd, Iterable=Iterable, Mapping=Mapping,
                     BaseModel=BaseModel, BM=BM, functools=functools)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def test_metric_decorator_preserves_nested_judge_request_settings():
    functions = _production_functions()
    options = {
        "default_request_kwargs": {"max_tokens": 1024},
        "request_extra_body": {"thinking": {"type": "disabled"}},
        "structured_output_mode": "json_object",
    }
    before = deepcopy(options)

    @functions["convert_inputs_to_list"]
    def metric(cases, **kwargs):
        # These operations match the client boundary that previously failed.
        return cases, dict(kwargs["default_request_kwargs"]), dict(kwargs["request_extra_body"])

    cases, defaults, extra = metric(pd.Series(["first", "second"]), **options)
    assert cases == ["first", "second"]
    assert defaults == {"max_tokens": 1024}
    assert extra == {"thinking": {"type": "disabled"}}
    extra["thinking"]["type"] = "changed"
    assert options == before


def test_mapping_values_normalize_sequences_without_losing_keys():
    convert = _production_functions()["to_list"]
    source = UserDict({7: {"values": np.array([[1, 2], [3, 4]])},
                       "series": pd.Series(["a", "b"]), "tuple": (1, 2)})
    assert convert(source) == {7: {"values": [[1, 2], [3, 4]]},
                               "series": ["a", "b"], "tuple": [1, 2]}
    assert isinstance(source[7]["values"], np.ndarray)


def test_scalar_strings_and_schema_models_remain_intact():
    convert = _production_functions()["to_list"]
    class Schema(BM):
        text: str
    item = Schema(text="unchanged")
    assert convert(item) is item
    assert convert(["text", b"bytes", None, 4]) == ["text", b"bytes", None, 4]
