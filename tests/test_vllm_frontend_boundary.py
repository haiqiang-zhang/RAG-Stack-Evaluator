from __future__ import annotations

import asyncio
import json

import pytest

from rag_stack.vllm_instrumentation.calibration.runtime_identity import (
    CALIBRATION_RUN_ID_ENV,
)
from rag_stack.vllm_instrumentation.serving_curves.calibration.vllm_frontend_boundary import (
    FRONTEND_BOUNDARY_SCHEMA,
    FrontendBoundaryCoverageError,
    consume_frontend_instrumentation_exclusion_s,
    frontend_boundary_coverage,
    install_frontend_boundary_telemetry,
)
from rag_stack.vllm_instrumentation.serving_curves.calibration.vllm_telemetry import (
    TELEMETRY_ENV,
)


class _FakeAsyncLLM:
    async def add_request(self, request_id, *args, **kwargs):
        del request_id, args, kwargs
        await asyncio.sleep(0)
        return "collector"


class _FakeOutputProcessor:
    def process_outputs(self, *args, **kwargs):
        del args, kwargs
        return "materialized"


class _FakeJSONResponse:
    def render(self, content):
        return json.dumps(content).encode()


class _FakeCompletion:
    async def render_completion_request(self, request):
        del request
        await asyncio.sleep(0)
        return ["tokens"]

    def request_output_to_completion_response(self, value):
        return {"value": value}

    async def create_completion(self, request, raw_request=None):
        del raw_request
        await self.render_completion_request(request)
        await _FakeAsyncLLM().add_request("request")
        _FakeOutputProcessor().process_outputs("output")
        return self.request_output_to_completion_response("done")


class _FakeChat:
    async def render_chat_request(self, request):
        del request
        await asyncio.sleep(0)
        return ["tokens"]

    async def chat_completion_full_generator(self, value):
        await asyncio.sleep(0)
        return {"value": value}

    async def create_chat_completion(self, request, raw_request=None):
        del raw_request
        await self.render_chat_request(request)
        await _FakeAsyncLLM().add_request("real-chat-request-id")
        _FakeOutputProcessor().process_outputs("output")
        return await self.chat_completion_full_generator("done")


def test_production_frontend_hooks_are_coverage_only(monkeypatch, tmp_path):
    path = tmp_path / "stage.audit.jsonl"
    monkeypatch.setenv(TELEMETRY_ENV, str(path))
    monkeypatch.setenv(CALIBRATION_RUN_ID_ENV, "frontend-test")
    assert install_frontend_boundary_telemetry(
        completion_cls=_FakeCompletion,
        async_llm_cls=_FakeAsyncLLM,
        output_processor_cls=_FakeOutputProcessor,
        json_response_cls=_FakeJSONResponse,
    )

    async def exercise():
        response = await _FakeCompletion().create_completion(object())
        return _FakeJSONResponse().render(response)

    assert asyncio.run(exercise()).startswith(b"{")
    records = tuple(
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    )
    assert {record["schema"] for record in records} == {
        FRONTEND_BOUNDARY_SCHEMA
    }
    assert all(record["fit_value"] is False for record in records)
    started = min(float(record["started_monotonic_s"]) for record in records)
    finished = max(float(record["finished_monotonic_s"]) for record in records)

    prefill = frontend_boundary_coverage(
        records,
        run_id="frontend-test",
        phase="prefill",
        window_start_monotonic_s=started - 1.0,
        window_end_monotonic_s=finished + 1.0,
    )
    decode = frontend_boundary_coverage(
        records,
        run_id="frontend-test",
        phase="decode",
        window_start_monotonic_s=started - 1.0,
        window_end_monotonic_s=finished + 1.0,
    )
    assert prefill["preprocess_tokenize"] == 1
    assert prefill["request_dispatch_ipc"] == 1
    assert decode["response_build_postprocess"] == 1
    assert decode["response_json_serialization"] == 1
    assert consume_frontend_instrumentation_exclusion_s() >= 0.0


def test_chat_hooks_bind_dispatch_to_real_async_llm_request_id(
    monkeypatch, tmp_path,
):
    path = tmp_path / "chat-stage.audit.jsonl"
    monkeypatch.setenv(TELEMETRY_ENV, str(path))
    monkeypatch.setenv(CALIBRATION_RUN_ID_ENV, "chat-frontend-test")
    assert install_frontend_boundary_telemetry(
        chat_cls=_FakeChat,
        async_llm_cls=_FakeAsyncLLM,
        output_processor_cls=_FakeOutputProcessor,
        json_response_cls=_FakeJSONResponse,
    )

    async def exercise():
        response = await _FakeChat().create_chat_completion(object())
        return _FakeJSONResponse().render(response)

    assert asyncio.run(exercise()).startswith(b"{")
    records = tuple(
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    )
    dispatch = [
        record for record in records
        if record["kind"] == "request_dispatch_ipc"
    ]
    assert len(dispatch) == 1
    assert dispatch[0]["request_id"] == "real-chat-request-id"
    assert any(record["kind"] == "preprocess_tokenize" for record in records)
    assert any(
        record["kind"] == "response_build_postprocess" for record in records
    )


def test_dispatch_without_real_request_id_fails_closed(monkeypatch, tmp_path):
    path = tmp_path / "bad-dispatch.audit.jsonl"
    monkeypatch.setenv(TELEMETRY_ENV, str(path))
    monkeypatch.setenv(CALIBRATION_RUN_ID_ENV, "bad-dispatch-test")

    class FreshAsyncLLM:
        async def add_request(self, request_id):
            return request_id

    assert install_frontend_boundary_telemetry(
        completion_cls=_FakeCompletion,
        async_llm_cls=FreshAsyncLLM,
        output_processor_cls=_FakeOutputProcessor,
        json_response_cls=_FakeJSONResponse,
    )
    with pytest.raises(RuntimeError, match="real request_id"):
        asyncio.run(FreshAsyncLLM().add_request(None))

    assert asyncio.run(
        FreshAsyncLLM().add_request(request_id="keyword-request-id")
    ) == "keyword-request-id"
    dispatch = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["kind"] == "request_dispatch_ipc"
    ]
    assert dispatch[-1]["request_id"] == "keyword-request-id"


def test_chat_hook_contract_fails_closed_when_vllm_method_is_missing(
    monkeypatch, tmp_path,
):
    path = tmp_path / "missing-chat-method.audit.jsonl"
    monkeypatch.setenv(TELEMETRY_ENV, str(path))
    monkeypatch.setenv(CALIBRATION_RUN_ID_ENV, "missing-chat-method-test")

    class IncompleteChat:
        async def create_chat_completion(self, request, raw_request=None):
            return request, raw_request

        async def render_chat_request(self, request):
            return request

    class FreshAsyncLLM:
        async def add_request(self, request_id):
            return request_id

    class FreshOutputProcessor:
        def process_outputs(self, value):
            return value

    class FreshJSONResponse:
        def render(self, content):
            return json.dumps(content).encode()

    with pytest.raises(
        RuntimeError,
        match="IncompleteChat.chat_completion_full_generator",
    ):
        install_frontend_boundary_telemetry(
            chat_cls=IncompleteChat,
            async_llm_cls=FreshAsyncLLM,
            output_processor_cls=FreshOutputProcessor,
            json_response_cls=FreshJSONResponse,
        )


def test_dispatch_coverage_rejects_missing_real_request_id():
    records = ({
        "schema": FRONTEND_BOUNDARY_SCHEMA,
        "evidence_scope": "active_stage_boundary_coverage_only",
        "run_id": "frontend-test",
        "kind": "request_dispatch_ipc",
        "request_id": None,
        "started_monotonic_s": 1.0,
        "finished_monotonic_s": 2.0,
        "active_span_s": 1.0,
        "fit_value": False,
        "reason": "coverage_only_do_not_add_to_scheduler_interval",
    },)

    with pytest.raises(
        FrontendBoundaryCoverageError,
        match="real AsyncLLM request_id",
    ):
        frontend_boundary_coverage(
            records,
            run_id="frontend-test",
            phase="prefill",
            window_start_monotonic_s=0.5,
            window_end_monotonic_s=2.5,
        )


def test_decode_boundary_fails_closed_without_terminal_serialization():
    records = (
        {
            "schema": FRONTEND_BOUNDARY_SCHEMA,
            "evidence_scope": "active_stage_boundary_coverage_only",
            "run_id": "frontend-test",
            "kind": kind,
            "started_monotonic_s": 1.0,
            "finished_monotonic_s": 2.0,
            "active_span_s": 1.0,
            "fit_value": False,
            "reason": "coverage_only_do_not_add_to_scheduler_interval",
        }
        for kind in ("output_materialization", "response_build_postprocess")
    )

    with pytest.raises(
        FrontendBoundaryCoverageError,
        match="response_json_serialization",
    ):
        frontend_boundary_coverage(
            records,
            run_id="frontend-test",
            phase="decode",
            window_start_monotonic_s=0.5,
            window_end_monotonic_s=2.5,
        )
