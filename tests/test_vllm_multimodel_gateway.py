from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import httpx
from fastapi.testclient import TestClient


SCRIPT = Path(__file__).parents[1] / "scripts" / "vllm_multimodel_gateway.py"
SPEC = importlib.util.spec_from_file_location("vllm_multimodel_gateway", SCRIPT)
assert SPEC and SPEC.loader
gateway = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gateway)


MODELS = {
	"Qwen/Qwen2.5-1.5B-Instruct": "http://backend-1:8101",
	"Qwen/Qwen2.5-14B-Instruct": "http://backend-14:8104",
}


def _handler(request: httpx.Request) -> httpx.Response:
	model = (
		"Qwen/Qwen2.5-1.5B-Instruct"
		if request.url.host == "backend-1"
		else "Qwen/Qwen2.5-14B-Instruct"
	)
	if request.url.path == "/health":
		return httpx.Response(200, text="ok")
	if request.url.path == "/v1/models":
		return httpx.Response(200, json={
			"object": "list",
			"data": [{"id": model, "object": "model", "max_model_len": 32768}],
		})
	payload = json.loads(request.content)
	if request.url.path == "/tokenize":
		return httpx.Response(200, json={"tokens": [1, 2], "routed_model": model})
	if request.url.path == "/detokenize":
		return httpx.Response(200, json={"prompt": "decoded", "routed_model": model})
	if request.url.path == "/v1/chat/completions":
		return httpx.Response(200, json={
			"model": model,
			"choices": [{"message": {"content": payload["model"]}}],
		})
	return httpx.Response(404)


def _client() -> TestClient:
	transport = httpx.MockTransport(_handler)
	return TestClient(gateway.create_app(MODELS, transport=transport))


def test_health_and_models_aggregate_all_backends():
	with _client() as client:
		assert client.get("/health").status_code == 200
		models = client.get("/v1/models").json()["data"]
		assert {entry["id"] for entry in models} == set(MODELS)


def test_models_catalog_is_cached_after_first_success():
	model_calls = []

	def handler(request: httpx.Request) -> httpx.Response:
		if request.url.path == "/v1/models":
			model_calls.append(request.url.host)
			if model_calls.count(request.url.host) > 1:
				return httpx.Response(503, text="backend is saturated")
		return _handler(request)

	transport = httpx.MockTransport(handler)
	with TestClient(gateway.create_app(MODELS, transport=transport)) as client:
		first = client.get("/v1/models")
		second = client.get("/v1/models")

	assert first.status_code == 200
	assert second.status_code == 200
	assert second.json() == first.json()
	assert sorted(model_calls) == sorted(url.split("//", 1)[1].split(":", 1)[0]
		for url in MODELS.values())


def test_models_catalog_failure_is_not_cached():
	failed_once = False
	model_calls = []

	def handler(request: httpx.Request) -> httpx.Response:
		nonlocal failed_once
		if request.url.path == "/v1/models":
			model_calls.append(request.url.host)
		if (
			request.url.path == "/v1/models"
			and request.url.host == "backend-14"
			and not failed_once
		):
			failed_once = True
			return httpx.Response(503, text="backend is saturated")
		return _handler(request)

	transport = httpx.MockTransport(handler)
	with TestClient(gateway.create_app(MODELS, transport=transport)) as client:
		failed = client.get("/v1/models")
		retried = client.get("/v1/models")
		cached = client.get("/v1/models")

	assert failed.status_code == 503
	assert retried.status_code == 200
	assert cached.status_code == 200
	assert cached.json() == retried.json()
	assert {entry["id"] for entry in retried.json()["data"]} == set(MODELS)
	expected_hosts = [url.split("//", 1)[1].split(":", 1)[0]
		for url in MODELS.values()]
	assert sorted(model_calls) == sorted(expected_hosts * 2)


def test_chat_and_tokenizer_routes_by_exact_model():
	model = "Qwen/Qwen2.5-14B-Instruct"
	with _client() as client:
		chat = client.post(
			"/v1/chat/completions",
			json={"model": model, "messages": [{"role": "user", "content": "hi"}]},
		)
		assert chat.status_code == 200
		assert chat.json()["model"] == model
		assert client.post("/tokenize", json={"model": model, "prompt": "hi"}).json()[
			"routed_model"
		] == model
		assert client.post("/detokenize", json={"model": model, "tokens": [1]}).json()[
			"routed_model"
		] == model


def test_unknown_model_fails_without_fallback():
	with _client() as client:
		response = client.post(
			"/v1/chat/completions",
			json={"model": "wrong", "messages": []},
		)
		assert response.status_code == 400
		assert response.json()["detail"]["available"] == sorted(MODELS)
