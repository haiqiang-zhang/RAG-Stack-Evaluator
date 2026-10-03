"""CPU-only dispatch tests; execute production methods against fake engines.

Extracting definitions avoids the static evaluator's eager model imports. The
methods under test are compiled unchanged; only their GPU/network dependencies
and the DataFrame decorator are replaced.
"""
from __future__ import annotations

import abc
import ast
from collections.abc import Iterable, Mapping
from copy import deepcopy
import inspect
import logging
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from rag_stack_evaluator.generation_protocol import (
    CHAT_COMPLETIONS,
    LEGACY_COMPLETION,
    reject_legacy_generation_protocol,
    resolve_generation_protocol,
)

ROOT = Path(__file__).resolve().parents[1] / "rag_stack_evaluator"


def _definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec",
                 flags=__future_annotations()), namespace)


def __future_annotations():
    import __future__
    return __future__.annotations.compiler_flag


@pytest.fixture
def native(monkeypatch):
    calls = []

    class SamplingParams:
        def __init__(self, temperature=1.0, max_tokens=16, logprobs=None):
            self.temperature = temperature
            self.max_tokens = max_tokens
            self.logprobs = logprobs

        @classmethod
        def from_optional(cls, temperature=1.0, max_tokens=16, logprobs=None):
            return cls(temperature, max_tokens, logprobs)

    class LLM:
        def __init__(self, model, **kwargs):
            calls.append(("constructor", model, kwargs))

        def _result(self, kind, prompts, sampling_params, kwargs):
            calls.append((kind, prompts, sampling_params, kwargs))
            return [SimpleNamespace(outputs=[SimpleNamespace(
                text="answer", token_ids=[7],
                logprobs=[{7: SimpleNamespace(logprob=-0.2)}],
            )]) for _ in prompts]

        def generate(self, prompts, sampling_params, use_tqdm=True):
            return self._result("generate", prompts, sampling_params,
                                {"use_tqdm": use_tqdm})

        def chat(self, prompts, sampling_params, chat_template_kwargs=None, use_tqdm=True):
            return self._result("chat", prompts, sampling_params,
                                {"chat_template_kwargs": chat_template_kwargs})

    class BaseGenerator:
        def __init__(self, project_dir, model, **kwargs):
            self.model = model

    fake = ModuleType("vllm")
    fake.LLM, fake.SamplingParams = LLM, SamplingParams
    outputs = ModuleType("vllm.outputs")
    outputs.RequestOutput = object
    logprobs = ModuleType("vllm.logprobs")
    logprobs.SampleLogprobs = list
    cache_module = ModuleType("rag_stack_evaluator.static_rag_evaluator.measured.cache")
    cache_module.get_current = lambda: None
    for name, module in (("vllm", fake), ("vllm.outputs", outputs),
                         ("vllm.logprobs", logprobs), (cache_module.__name__, cache_module)):
        monkeypatch.setitem(sys.modules, name, module)

    ns = dict(
        BaseGenerator=BaseGenerator, deepcopy=deepcopy, inspect=inspect,
        logger=logging.getLogger(__name__),
        pd=SimpleNamespace(DataFrame=object, Series=type("Series", (), {})),
        np=SimpleNamespace(ndarray=type("ndarray", (), {})),
        Iterable=Iterable, Mapping=Mapping, BaseModel=type("BaseModel", (), {}), BM=type("BM", (), {}),
        result_to_dataframe=lambda _: lambda function: function,
        configure_vllm_worker_env=lambda **kw: calls.append(("worker_env",)),
        _adapt_engine_resources=lambda *args, **kw: None,
        _build_inprocess_engine=lambda owner, factory: factory(),
        CHAT_COMPLETIONS=CHAT_COMPLETIONS, LEGACY_COMPLETION=LEGACY_COMPLETION,
        resolve_generation_protocol=resolve_generation_protocol,
    )
    _definitions(ROOT / "static_rag_evaluator/utils/util.py",
                 {"pop_params", "is_chat_prompt", "to_list"}, ns)
    _definitions(ROOT / "static_rag_evaluator/nodes/generator/vllm.py",
                 {"_prepare_vllm_prompts", "Vllm"}, ns)
    node = ns["Vllm"]
    # The fake engine owns no process and requires no cleanup/placement.
    monkeypatch.setattr(node, "__del__", lambda self: None)
    monkeypatch.setattr(node, "_maybe_pin_cvd", lambda self, device: None)
    return SimpleNamespace(cls=node, calls=calls, ns=ns, cache=cache_module)


def _build(native, **kwargs):
    return native.cls("unused", "test-model", tensor_parallel_size=2, **kwargs)


def test_legacy_raw_generation_preserves_sampling_and_prompt_boundary(native):
    node = _build(native, generation_protocol=LEGACY_COMPLETION, max_model_len=32768)
    prompts = ["A raw prompt"]
    assert node._pure(prompts, temperature=0.7, max_tokens=512) == (
        ["answer"], [[7]], [[-0.2]],
    )
    kind, received, sampling, kwargs = native.calls[-1]
    assert kind == "generate"
    assert received is prompts
    assert sampling.temperature == 0.7 and sampling.max_tokens == 512
    assert sampling.logprobs == 1
    constructor = next(call for call in native.calls if call[0] == "constructor")
    assert constructor[2] == {"tensor_parallel_size": 2, "max_model_len": 32768}


def test_legacy_qe_keeps_native_implicit_16_token_default(native):
    node = _build(native, generation_protocol=LEGACY_COMPLETION)
    node._pure(["Expand the query"])
    assert native.calls[-1][0] == "generate"
    assert native.calls[-1][2].max_tokens == 16
    assert native.calls[-1][2].temperature == 1.0


@pytest.mark.parametrize("protocol", [CHAT_COMPLETIONS, LEGACY_COMPLETION])
def test_structured_messages_always_use_chat(native, protocol):
    messages = [[{"role": "system", "content": "Be brief"},
                 {"role": "user", "content": "Question"}]]
    node = _build(native, generation_protocol=protocol)
    node._pure(messages, thinking=True)
    kind, received, _, kwargs = native.calls[-1]
    assert kind == "chat" and received is messages
    assert kwargs["chat_template_kwargs"] == {"enable_thinking": True}


def test_default_native_path_still_wraps_raw_strings_as_chat(native):
    node = _build(native)
    node._pure(["Question"])
    assert native.calls[-1][0:2] == ("chat", [[{"role": "user", "content": "Question"}]])


@pytest.mark.parametrize("kwargs", [
    {"use_chat_template": False},
    {"generation_protocol": "typo"},
    {"generation_protocol": LEGACY_COMPLETION, "use_chat_template": True},
    {"generation_protocol": LEGACY_COMPLETION, "measured_request_format": CHAT_COMPLETIONS},
])
def test_invalid_protocol_rejected_before_worker_setup_or_engine(native, kwargs):
    with pytest.raises(ValueError):
        _build(native, **kwargs)
    assert native.calls == []


def test_measured_cache_rejects_legacy_before_allocation(native):
    native.cache.get_current = lambda: object()
    with pytest.raises(ValueError, match="restricted to native"):
        _build(native, generation_protocol=LEGACY_COMPLETION)
    assert native.calls == []


@pytest.mark.parametrize("method", ["_pure", "_pure_subprocess"])
def test_native_legacy_cannot_switch_to_subprocess_after_construction(native, method):
    node = _build(native, generation_protocol=LEGACY_COMPLETION, use_chat_template=False)
    node._subprocess = object()
    native.calls.clear()
    with pytest.raises(ValueError, match="restricted to native"):
        getattr(node, method)(["Prompt"])
    assert native.calls == []


def test_query_expansion_forwards_protocol_to_native_constructor(native):
    ns = dict(native.ns, abc=abc, BaseModule=object)
    _definitions(ROOT / "static_rag_evaluator/nodes/util.py",
                 {"make_generator_callable_param"}, ns)
    ns["get_generator_class"] = lambda name: native.cls
    _definitions(ROOT / "static_rag_evaluator/nodes/queryexpansion/base.py",
                 {"BaseQueryExpansion"}, ns)
    expansion = ns["BaseQueryExpansion"](
        "unused", generator_backend="vllm", model="test-model",
        tensor_parallel_size=2, generation_protocol=LEGACY_COMPLETION,
    )
    assert expansion.generator._generation_protocol == LEGACY_COMPLETION
    expansion.generator._pure(["QE prompt"])
    assert native.calls[-1][0] == "generate"


def test_api_rejects_legacy_before_parent_init_or_network():
    class ForbiddenBase:
        def __init__(self, *args, **kwargs):
            pytest.fail("API initialization occurred before protocol validation")

    ns = dict(BaseGenerator=ForbiddenBase, resolve_generation_protocol=resolve_generation_protocol,
              pd=SimpleNamespace(DataFrame=object),
              result_to_dataframe=lambda _: lambda function: function)
    _definitions(ROOT / "static_rag_evaluator/nodes/generator/vllm_api.py", {"VllmAPI"}, ns)
    with pytest.raises(ValueError, match="restricted to native"):
        ns["VllmAPI"]("unused", "test-model", "http://unused", generation_protocol=LEGACY_COMPLETION)


def test_measured_policy_and_prepared_config_cannot_bypass_guard():
    ns = dict(reject_legacy_generation_protocol=reject_legacy_generation_protocol,
              deepcopy=deepcopy)
    _definitions(ROOT / "static_rag_evaluator/measured/evaluator.py",
                 {"apply_measured_generation_defaults", "MeasuredEvaluator"}, ns)
    config = {"node_lines": [{"nodes": [{"modules": [{
        "component": "vllm", "generation_protocol": LEGACY_COMPLETION,
    }]}]}]}
    with pytest.raises(ValueError, match="restricted to native"):
        ns["apply_measured_generation_defaults"](config)
    with pytest.raises(ValueError, match="restricted to native"):
        ns["MeasuredEvaluator"](object()).evaluate(
            config, cache=object(), generation_defaults_applied=True,
        )


def test_subprocess_request_format_rejects_historical_protocol():
    ns = {"MEASURED_REQUEST_FORMAT_KEY": "measured_request_format",
          "REQUEST_FORMAT_CHAT_COMPLETIONS": CHAT_COMPLETIONS}
    _definitions(ROOT / "static_rag_evaluator/measured/vllm_subprocess.py",
                 {"_request_format"}, ns)
    with pytest.raises(ValueError, match="restricted to native"):
        ns["_request_format"]({"generation_protocol": LEGACY_COMPLETION})
