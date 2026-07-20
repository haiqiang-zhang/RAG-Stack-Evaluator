"""CPU-only contracts for measured auxiliary-model data parallelism.

All model forwards and constructors are fakes.  These tests must never load a
checkpoint or initialize CUDA; they exercise only replica selection, dispatch,
and result reconstruction.
"""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pandas as pd


def test_reranker_cache_none_builds_one_replica_per_injected_device() -> None:
    from rag_stack.static_rag_evaluator.nodes.passagereranker.base import (
        BasePassageReranker,
    )

    class StubReranker(BasePassageReranker):
        def pure(self, *args, **kwargs):
            raise NotImplementedError

        def _pure(self, *args, **kwargs):
            raise NotImplementedError

    instance = StubReranker.__new__(StubReranker)
    builds: list[tuple[str, str, str]] = []

    def factory(component: str, model_name: str, device: str) -> str:
        builds.append((component, model_name, device))
        return f"replica@{device}"

    replicas = instance._load_replicas(
        None,
        component="fake_reranker",
        model_name="fake/model",
        device="cpu",
        devices=["cuda:1", "cuda:3"],
        factory=factory,
    )

    assert builds == [
        ("fake_reranker", "fake/model", "cuda:1"),
        ("fake_reranker", "fake/model", "cuda:3"),
    ]
    assert replicas == ["replica@cuda:1", "replica@cuda:3"]
    assert instance._replica_devices == ["cuda:1", "cuda:3"]
    assert instance.device == "cuda:1"
    assert instance.model == "replica@cuda:1"
    assert instance._cache_owned is False


def test_llmlingua_cache_none_builds_and_dispatches_every_replica(
    monkeypatch,
) -> None:
    from rag_stack.static_rag_evaluator.measured import cache as measured_cache
    from rag_stack.static_rag_evaluator.nodes.passagecompressor import llmlingua2

    builds: list[tuple[str, str, bool]] = []

    class FakePromptCompressor:
        def __init__(
            self,
            model_name: str,
            *,
            device_map: str,
            use_llmlingua2: bool = False,
        ) -> None:
            self.device = device_map
            self.max_batch_size = None
            builds.append((model_name, device_map, use_llmlingua2))

        def compress_prompt(self, contexts, *, rate, **kwargs):
            del rate, kwargs
            return {
                "compressed_prompt_list": [
                    f"{self.device}:{context}" for context in contexts
                ]
            }

    monkeypatch.setattr(measured_cache, "get_current", lambda: None)
    monkeypatch.setitem(
        sys.modules,
        "llmlingua",
        SimpleNamespace(PromptCompressor=FakePromptCompressor),
    )

    instance = llmlingua2.LLMLingua2(
        ".",
        model_name="fake/llmlingua2",
        devices=["cuda:0", "cuda:2"],
    )

    assert builds == [
        ("fake/llmlingua2", "cuda:0", True),
        ("fake/llmlingua2", "cuda:2", True),
    ]
    assert [replica.device for replica in instance._replicas] == [
        "cuda:0",
        "cuda:2",
    ]
    assert instance._replica_devices == ["cuda:0", "cuda:2"]
    assert instance.llm_lingua is instance._replicas[0]

    queries = [f"q{i}" for i in range(5)]
    contents = [[f"d{i}"] for i in range(5)]
    assert instance._pure(
        queries,
        contents,
        rate=0.5,
        query_batch_size=3,
    ) == [
        "cuda:0:d0",
        "cuda:0:d1",
        "cuda:0:d2",
        "cuda:2:d3",
        "cuda:2:d4",
    ]
    assert [replica.max_batch_size for replica in instance._replicas] == [3, 3]


def test_reranker_dp_preserves_flat_order_and_nested_shape() -> None:
    from rag_stack.static_rag_evaluator.nodes.passagereranker.dp import (
        rerank_flatten_apply_dp,
    )

    calls: list[tuple[str, list[str]]] = []

    def score(items, *, model, batch_size):
        assert batch_size == 7
        calls.append((model, list(items)))
        return [f"{model}:{item}" for item in items]

    result = rerank_flatten_apply_dp(
        score,
        [["a", "b"], ["c"], ["d", "e"]],
        ["r0", "r1"],
        batch_size=7,
    )

    assert sorted(calls) == [
        ("r0", ["a", "b", "c"]),
        ("r1", ["d", "e"]),
    ]
    assert result == [
        ["r0:a", "r0:b"],
        ["r0:c"],
        ["r1:d", "r1:e"],
    ]


def test_tart_dp_keeps_each_instruction_query_paired_with_its_passage(
    monkeypatch,
) -> None:
    from rag_stack.static_rag_evaluator.nodes.passagereranker.tart import tart

    calls: list[tuple[str, str, str, list[tuple[str, str]]]] = []

    def fake_tart_run_model(
        input_texts,
        contents_list,
        model,
        batch_size,
        tokenizer,
        device,
    ):
        assert batch_size == 7
        pairs = [
            (query_texts[0], passage_texts[0])
            for query_texts, passage_texts in zip(input_texts, contents_list)
        ]
        calls.append((model, tokenizer, device, pairs))
        return [float(passage.removeprefix("d")) for _, passage in pairs]

    monkeypatch.setattr(tart, "tart_run_model", fake_tart_run_model)
    instance = tart.Tart.__new__(tart.Tart)
    instance._replicas = [("model0", "tokenizer0"), ("model1", "tokenizer1")]
    instance._replica_devices = ["cuda:0", "cuda:1"]

    contents, ids, scores = instance._pure(
        ["q0", "q1"],
        [["d0", "d1"], ["d2", "d3", "d4"]],
        [["id0", "id1"], ["id2", "id3", "id4"]],
        top_k=5,
        instruction="inst",
        batch=7,
    )

    assert sorted(calls) == [
        (
            "model0",
            "tokenizer0",
            "cuda:0",
            [
                ("inst [SEP] q0", "d0"),
                ("inst [SEP] q0", "d1"),
                ("inst [SEP] q1", "d2"),
            ],
        ),
        (
            "model1",
            "tokenizer1",
            "cuda:1",
            [("inst [SEP] q1", "d3"), ("inst [SEP] q1", "d4")],
        ),
    ]
    assert contents == [["d1", "d0"], ["d4", "d3", "d2"]]
    assert ids == [["id1", "id0"], ["id4", "id3", "id2"]]
    assert scores == [[1.0, 0.0], [4.0, 3.0, 2.0]]


def test_tart_collects_each_forward_scores_with_one_bulk_cpu_transfer(
    monkeypatch,
) -> None:
    """Regression: never coerce one CUDA score at a time.

    The fakes exercise the production control flow without constructing a
    model, loading a checkpoint, or initializing CUDA. ``__iter__`` raises so
    the former ``[float(score[1]) for score in probabilities]`` implementation
    cannot pass this test.
    """
    from rag_stack.static_rag_evaluator.nodes.passagereranker.tart import tart
    import torch.nn.functional as functional

    events: list[tuple[str, object]] = []

    class Encoding(dict):
        def to(self, device):
            events.append(("feature_to", device))
            return self

    class Tokenizer:
        def __call__(
            self,
            texts,
            contents,
            *,
            padding,
            truncation,
            return_tensors,
        ):
            assert padding is True
            assert truncation is True
            assert return_tensors == "pt"
            assert len(texts) == len(contents)
            return Encoding(values=[float(text[1:]) for text in texts])

    class Model:
        def __call__(self, *, values):
            return SimpleNamespace(logits=SimpleNamespace(values=values))

    class BulkProbabilities:
        def __init__(self, values):
            self.values = list(values)

        def __iter__(self):
            raise AssertionError("TART probabilities must not be read row by row")

        def __getitem__(self, key):
            assert key == (slice(None), 1)
            events.append(("positive_class_slice", len(self.values)))
            return self

        def detach(self):
            events.append(("detach", len(self.values)))
            return self

        def cpu(self):
            events.append(("cpu", len(self.values)))
            return self

        def tolist(self):
            events.append(("tolist", len(self.values)))
            return list(self.values)

    def fake_softmax(logits, dim):
        assert dim == 1
        events.append(("softmax", len(logits.values)))
        return BulkProbabilities(logits.values)

    monkeypatch.setattr(functional, "softmax", fake_softmax)

    scores = tart.tart_run_model(
        [["q0"], ["q1"], ["q2"]],
        [["d0"], ["d1"], ["d2"]],
        model=Model(),
        batch_size=2,
        tokenizer=Tokenizer(),
        device="cpu",
    )

    assert scores == [0.0, 1.0, 2.0]
    assert [event for event in events if event[0] == "cpu"] == [
        ("cpu", 2),
        ("cpu", 1),
    ]
    assert [event for event in events if event[0] == "tolist"] == [
        ("tolist", 2),
        ("tolist", 1),
    ]


def test_monot5_dp_routes_each_flattened_pair_to_one_replica(
    monkeypatch,
) -> None:
    from rag_stack.static_rag_evaluator.nodes.passagereranker import monot5

    calls: list[tuple[str, str, str, list[str]]] = []

    def fake_monot5_run_model(
        input_texts,
        model,
        batch_size,
        tokenizer,
        device,
        token_false_id,
        token_true_id,
        execution_report,
    ):
        assert batch_size == 11
        assert (token_false_id, token_true_id) == (10, 20)
        texts = [item[0] for item in input_texts]
        calls.append((model, tokenizer, device, texts))
        # Leaving the report empty is valid: this fake is testing dispatch, not
        # the production forward telemetry hook.
        assert execution_report == {}
        return [float(text.rsplit("d", 1)[1]) for text in texts]

    monkeypatch.setattr(monot5, "monot5_run_model", fake_monot5_run_model)
    instance = monot5.MonoT5.__new__(monot5.MonoT5)
    instance._replicas = [("tokenizer0", "model0"), ("tokenizer1", "model1")]
    instance._replica_devices = ["cuda:0", "cuda:1"]
    instance.token_false_id = 10
    instance.token_true_id = 20
    instance._last_forward_execution_report = None

    contents, ids, scores = instance._pure(
        ["q0", "q1"],
        [["d0", "d1"], ["d2", "d3", "d4"]],
        [["id0", "id1"], ["id2", "id3", "id4"]],
        top_k=5,
        batch=11,
    )

    assert sorted(calls) == [
        (
            "model0",
            "tokenizer0",
            "cuda:0",
            [
                "Query: q0 Document: d0",
                "Query: q0 Document: d1",
                "Query: q1 Document: d2",
            ],
        ),
        (
            "model1",
            "tokenizer1",
            "cuda:1",
            ["Query: q1 Document: d3", "Query: q1 Document: d4"],
        ),
    ]
    assert contents == [["d1", "d0"], ["d4", "d3", "d2"]]
    assert ids == [["id1", "id0"], ["id4", "id3", "id2"]]
    assert scores == [[1.0, 0.0], [4.0, 3.0, 2.0]]
    assert instance.pop_last_forward_execution_report() is None


def test_aux_dp_service_cap_is_local_batch_times_replica_count() -> None:
    from rag_stack.static_rag_evaluator.measured import serving_runtime as sr

    def stage(name: str, component: str) -> dict:
        return {
            "stage": name,
            "node": SimpleNamespace(
                stage=name,
                module=SimpleNamespace(component=component),
            ),
            "params": {},
            "instance": object(),
        }

    stages = [
        stage("passage_reranker", "monot5"),
        stage("passage_compressor", "llmlingua2"),
        # LongLLMLingua uses all devices to shard one model, not as replicas.
        stage("passage_compressor", "longllmlingua"),
    ]
    system_config = {
        "batch_size_request": 3,
        "measured_load_concurrency": 1,
        "measured_warmup_queries": 0,
        "measured_queries": 1,
        "batching": {"dynamic_timeout_s": 0.01},
        "layout": {
            "performance_stages": {
                "passage_reranker": {
                    "kind": "gpu",
                    "engine": "passage_reranker",
                    "devices": ["cuda:0", "cuda:1"],
                },
                "passage_compressor": {
                    "kind": "gpu",
                    "engine": "passage_compressor",
                    "devices": ["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
                },
            },
            "engines": {
                "passage_reranker": {"devices": ["cuda:0", "cuda:1"]},
                "passage_compressor": {
                    "devices": ["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
                },
                "generator": {
                    "pd_serving": "collocated_pd",
                    "devices": ["cuda:0"],
                    "tp": 1,
                    "pp": 1,
                },
            },
            "resource_groups": [],
        },
    }
    runtime = sr.MeasuredServingRuntime(
        owner=SimpleNamespace(),
        stages=stages,
        qa_data=pd.DataFrame({"query": ["q"]}),
        node_lines={},
        system_config=system_config,
        config={"pipeline_runtime": {"rag_dataflow": "sequential"}},
    )

    async def scenario() -> None:
        services = runtime._build_services()
        try:
            assert [service.batch_size for service in services] == [6, 12, 3]
        finally:
            await asyncio.gather(*(service.close() for service in services))

    asyncio.run(scenario())
