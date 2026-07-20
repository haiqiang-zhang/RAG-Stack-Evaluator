"""CPU mocks for direct measured-evaluator resolved-layout validation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag_stack.runtime_parallelism import RERANKER_MEASURED_DP_COMPONENTS
from rag_stack.static_rag_evaluator.measured.evaluator import MeasuredEvaluator


def _node(stage: str, component: str):
    return SimpleNamespace(
        stage=stage,
        module=SimpleNamespace(component=component, module_param={}),
    )


def _system_config(stage: str, width: int) -> dict:
    performance_stage = (
        "semantic_retrieval_encode"
        if stage == "semantic_retrieval"
        else stage
    )
    engine = (
        "semantic_retrieval"
        if stage == "semantic_retrieval"
        else "passage_reranker"
    )
    devices = [f"cuda:{index}" for index in range(width)]
    return {
        "batch_size_request": 8,
        "retrieval": {
            "faiss_num_threads": 8,
            "faiss_ivf_parallel_mode": 0,
            "num_servers": 1,
        },
        "layout": {
            "performance_stages": {
                performance_stage: {
                    "kind": "gpu",
                    "engine": engine,
                    "devices": devices,
                    "num_chips": width,
                },
            },
            "engines": {
                engine: {
                    "devices": devices,
                    "num_chips": width,
                },
            },
            "resource_groups": [],
        },
    }


def _patch_injection_side_effects(monkeypatch):
    from rag_stack.static_rag_evaluator.embedding import base as embedding_base
    from rag_stack.static_rag_evaluator.measured import cache as measured_cache

    events = []
    monkeypatch.setattr(
        measured_cache,
        "set_current",
        lambda cache: events.append(("cache", cache)),
    )
    monkeypatch.setattr(
        embedding_base,
        "set_embedding_device",
        lambda device: events.append(("embedding", device)),
    )
    return events


def test_direct_measured_evaluator_injects_all_multi_gpu_encode_devices(
    monkeypatch,
):
    events = _patch_injection_side_effects(monkeypatch)
    node = _node("semantic_retrieval", "vectordb")

    MeasuredEvaluator._inject_cache_and_devices(
        {"retrieval": [node]},
        cache="cache-object",
        system_config=_system_config("semantic_retrieval", 4),
    )

    assert events == [
        ("cache", "cache-object"),
        ("embedding", "cuda:0"),
    ]
    assert node.module.module_param["device"] == "cuda:0"
    assert node.module.module_param["devices"] == [
        "cuda:0", "cuda:1", "cuda:2", "cuda:3",
    ]
    assert node.module.module_param["embedding_batch"] == 8


@pytest.mark.parametrize(
    "component",
    ["openvino_reranker", "unknown_reranker"],
)
def test_direct_measured_evaluator_rejects_non_allowlisted_multi_gpu_reranker(
    monkeypatch,
    component,
):
    events = _patch_injection_side_effects(monkeypatch)
    node = _node("passage_reranker", component)

    with pytest.raises(
        NotImplementedError,
        match=rf"measured evaluator resolved system_config.*component '{component}'",
    ):
        MeasuredEvaluator._inject_cache_and_devices(
            {"reranker": [node]},
            cache="cache-object",
            system_config=_system_config("passage_reranker", 4),
        )

    assert events == []
    assert node.module.module_param == {}


def test_direct_measured_evaluator_requires_one_active_reranker_for_dp(
    monkeypatch,
):
    events = _patch_injection_side_effects(monkeypatch)
    nodes = [
        _node("passage_reranker", "colbert_reranker"),
        _node("passage_reranker", "sentence_transformer_reranker"),
    ]

    with pytest.raises(
        NotImplementedError,
        match=r"requires exactly one active component",
    ):
        MeasuredEvaluator._inject_cache_and_devices(
            {"reranker": nodes},
            cache="cache-object",
            system_config=_system_config("passage_reranker", 4),
        )

    assert events == []
    assert all(node.module.module_param == {} for node in nodes)


@pytest.mark.parametrize("component", sorted(RERANKER_MEASURED_DP_COMPONENTS))
def test_direct_measured_evaluator_injects_all_supported_dp_devices(
    monkeypatch,
    component,
):
    events = _patch_injection_side_effects(monkeypatch)
    node = _node("passage_reranker", component)

    MeasuredEvaluator._inject_cache_and_devices(
        {"reranker": [node]},
        cache="cache-object",
        system_config=_system_config("passage_reranker", 4),
    )

    assert events == [
        ("cache", "cache-object"),
        ("embedding", "cuda:0"),
    ]
    assert node.module.module_param["device"] == "cuda:0"
    assert node.module.module_param["devices"] == [
        "cuda:0", "cuda:1", "cuda:2", "cuda:3",
    ]
    assert node.module.module_param["batch"] == 8
