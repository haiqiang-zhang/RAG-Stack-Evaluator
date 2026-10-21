"""Concrete evaluator deployment admission and vLLM request limits."""

import pytest


def test_measured_monot5_forward_batch_is_capped():
    from rag_stack_evaluator.static_rag_evaluator.measured.evaluator import (
        _measured_reranker_forward_batch,
    )

    class Module:
        component = "monot5"

    class Node:
        module = Module()

    assert _measured_reranker_forward_batch({}, Node(), 32) == 8
    assert _measured_reranker_forward_batch({"reranker_max_batch_size": 4}, Node(), 32) == 4


def test_measured_parallelism_gate_reuses_genz_model_metadata():
    from GenZ.Models import ModelConfig
    from rag_stack_evaluator.static_rag_evaluator.measured.vllm_deployment import (
        TrialInvalid,
        _check_nixl_hetero_tp_supported,
        _check_unified_parallelism_supported,
    )

    # No model-name table is involved: a private ModelConfig exercises the
    # same gate as registered Qwen models.  TP=2 reaches its two KV heads, so
    # NIXL cannot fan that producer into a TP=1 decode engine.
    private = ModelConfig(
        model="private/measured-gate",
        num_attention_heads=8,
        num_key_value_heads=2,
        num_decoder_layers=8,
    )
    with pytest.raises(TrialInvalid, match="saturated_producer_fan_in_unsupported"):
        _check_nixl_hetero_tp_supported(
            private, prefill_tp=2, decode_tp=1,
        )

    # The same heterogeneous direction is legal while the producer has not
    # saturated the model's KV-head partition.
    _check_nixl_hetero_tp_supported(
        "Qwen/Qwen2.5-7B-Instruct", prefill_tp=2, decode_tp=1,
    )

    with pytest.raises(TrialInvalid, match="attention_heads_not_divisible"):
        _check_unified_parallelism_supported(
            private, tensor_parallel=3, pipeline_parallel=1,
        )


@pytest.mark.parametrize(
    "prefill_pp,decode_pp",
    [(2, 1), (1, 2)],
)
def test_measured_nixl_gate_rejects_pipeline_parallelism(
    prefill_pp, decode_pp
):
    from rag_stack_evaluator.static_rag_evaluator.measured.vllm_deployment import (
        TrialInvalid,
        _check_nixl_hetero_tp_supported,
    )

    with pytest.raises(TrialInvalid, match="pipeline_parallel_unsupported"):
        _check_nixl_hetero_tp_supported(
            "Qwen/Qwen2.5-14B-Instruct",
            prefill_tp=1,
            prefill_pp=prefill_pp,
            decode_tp=1,
            decode_pp=decode_pp,
        )


def test_measured_query_expansion_pd_uses_the_same_genz_gate():
    from rag_stack_evaluator.static_rag_evaluator.measured.vllm_deployment import (
        TrialInvalid,
        VllmDeploymentManager,
    )

    system_cfg = {
        "batch_size_request": 8,
        "batch_size_decode": 8,
        "vllm": {
            "kv_cache_dtype": "auto",
            "engines": {
                "generator": {
                    "max_num_seqs": 8,
                    "prefill_max_num_seqs": 8,
                    "decode_max_num_seqs": 8,
                },
                "query_expansion": {
                    "max_num_seqs": 8,
                    "prefill_max_num_seqs": 8,
                    "decode_max_num_seqs": 8,
                },
            },
        },
        "layout": {
            "total_gpu_slots": 4,
            "min_total_gpu_slots": 4,
            "max_total_gpu_slots": 4,
            "engines": {
                "generator": {
                    "pd_serving": "collocated_pd",
                    "devices": ["cuda:3"],
                    "num_chips": 1,
                    "tp": 1,
                    "pp": 1,
                },
                "query_expansion": {
                    "pd_serving": "disagg_pd",
                    "devices": ["cuda:0", "cuda:1", "cuda:2"],
                    "num_chips": 3,
                    "query_expansion_prefill": {
                        "devices": ["cuda:0", "cuda:1"],
                        "num_chips": 2,
                        "tp": 2,
                        "pp": 1,
                    },
                    "query_expansion_decode": {
                        "devices": ["cuda:2"],
                        "num_chips": 1,
                        "tp": 1,
                        "pp": 1,
                    },
                },
            },
        },
    }
    nested_cfg = {
        "node_lines": [{
            "nodes": [{
                "stage": "query_expansion",
                "modules": [{
                    "component": "multi_query_expansion",
                    "generator_backend": "vllm",
                    "model": "Qwen/Qwen2.5-1.5B-Instruct",
                }],
            }],
        }, {
            "nodes": [{
                "stage": "generator",
                "modules": [{
                    "component": "vllm",
                    "model": "Qwen/Qwen2.5-7B-Instruct",
                }],
            }],
        }],
    }

    with pytest.raises(
        TrialInvalid, match="saturated_producer_fan_in_unsupported"
    ):
        VllmDeploymentManager(
            ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
        ).prepare_trial(
            cache=object(), nested_cfg=nested_cfg, system_cfg=system_cfg,
        )


def test_vllm_deployment_uses_decode_batch_for_unified_generator():
    from rag_stack_evaluator.static_rag_evaluator.measured.vllm_deployment import (
        VllmDeploymentManager,
    )

    class Cache:
        def __init__(self):
            self.key = None

        def register_main_vllm(self, key):
            self.key = key

        def evict_aux(self):
            return None

    system_cfg = {
        "batch_size_request": 8,
        "batch_size_decode": 32,
        "vllm": {
            "kv_cache_dtype": "auto",
            "engines": {
                "generator": {
                    "max_num_seqs": 32,
                    "prefill_max_num_seqs": 8,
                    "decode_max_num_seqs": 32,
                }
            },
        },
        "layout": {
            "total_gpu_slots": 1,
            "min_total_gpu_slots": 1,
            "max_total_gpu_slots": 1,
            "engines": {
                "generator": {
                    "pd_serving": "collocated_pd",
                    "devices": ["cuda:0"],
                    "num_chips": 1,
                    "tp": 1,
                    "pp": 1,
                }
            },
        },
    }
    nested_cfg = {
        "node_lines": [{
            "nodes": [{
                "stage": "generator",
                "modules": [{
                    "component": "vllm",
                    "model": "Qwen/Qwen2.5-1.5B-Instruct",
                    "dtype": "bfloat16",
                }],
            }]
        }]
    }

    cache = Cache()
    VllmDeploymentManager(["cuda:0"]).prepare_trial(
        cache=cache,
        nested_cfg=nested_cfg,
        system_cfg=system_cfg,
    )

    assert cache.key.max_num_seqs == 32


def test_vllm_deployment_splits_pd_prefill_and_decode_batch():
    from rag_stack_evaluator.static_rag_evaluator.measured.vllm_deployment import (
        VllmDeploymentManager,
    )

    class Cache:
        def __init__(self):
            self.key = None

        def register_main_vllm_pd(self, key):
            self.key = key

        def evict_aux(self):
            return None

    system_cfg = {
        "batch_size_request": 8,
        "batch_size_decode": 64,
        "vllm": {
            "kv_cache_dtype": "auto",
            "engines": {
                "generator": {
                    "max_num_seqs": 64,
                    "prefill_max_num_seqs": 8,
                    "decode_max_num_seqs": 64,
                }
            },
        },
        "layout": {
            "total_gpu_slots": 2,
            "min_total_gpu_slots": 2,
            "max_total_gpu_slots": 2,
            "engines": {
                "generator": {
                    "pd_serving": "disagg_pd",
                    "devices": ["cuda:0", "cuda:1"],
                    "num_chips": 2,
                    "generator_prefill": {
                        "devices": ["cuda:0"],
                        "num_chips": 1,
                        "tp": 1,
                        "pp": 1,
                    },
                    "generator_decode": {
                        "devices": ["cuda:1"],
                        "num_chips": 1,
                        "tp": 1,
                        "pp": 1,
                    },
                }
            },
        },
    }
    nested_cfg = {
        "node_lines": [{
            "nodes": [{
                "stage": "generator",
                "modules": [{
                    "component": "vllm",
                    "model": "Qwen/Qwen2.5-1.5B-Instruct",
                    "dtype": "bfloat16",
                }],
            }]
        }]
    }

    cache = Cache()
    VllmDeploymentManager(["cuda:0", "cuda:1"]).prepare_trial(
        cache=cache,
        nested_cfg=nested_cfg,
        system_cfg=system_cfg,
    )

    assert cache.key.max_num_seqs == 64
    assert cache.key.prefill_max_num_seqs == 8
    assert cache.key.decode_max_num_seqs == 64


def test_vllm_context_guard_clamps_max_tokens(monkeypatch):
    from rag_stack_evaluator.static_rag_evaluator import recording
    from rag_stack_evaluator.static_rag_evaluator.measured.vllm_subprocess import (
        _fit_max_tokens_to_context,
    )

    calls = []

    def fake_count_tokens_batch(texts, model_id=None):
        calls.append((list(texts), model_id))
        return [95]

    monkeypatch.setattr(recording, "count_tokens_batch", fake_count_tokens_batch)

    assert _fit_max_tokens_to_context(
        model="Qwen/Qwen2.5-7B-Instruct",
        prompt="short",
        requested_max_tokens=10,
        served_max_model_len=100,
    ) == 10
    assert calls == []

    assert _fit_max_tokens_to_context(
        model="Qwen/Qwen2.5-7B-Instruct",
        prompt="x" * 200,
        requested_max_tokens=10,
        served_max_model_len=100,
    ) == 5
    assert len(calls) == 1
