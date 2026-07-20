#!/usr/bin/env python3
"""Model-aware gateway for several OpenAI-compatible vLLM servers.

vLLM serves one base model per API process.  The optimizer search sends four
different Qwen model IDs to one URI, so this gateway routes each JSON request
by its exact ``model`` field and aggregates ``/v1/models`` and ``/health``.
It also proxies vLLM's root-level ``/tokenize`` and ``/detokenize`` endpoints,
which RAG-Stack uses for prompt truncation and token accounting.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from starlette.background import BackgroundTask


logger = logging.getLogger("uvicorn.error")


HOP_BY_HOP = {
	"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
	"te", "trailers", "transfer-encoding", "upgrade", "content-length",
}


def parse_backend(value: str) -> tuple[str, str]:
	model, separator, url = value.partition("=")
	model = model.strip()
	url = url.strip().rstrip("/")
	if not separator or not model or not url.startswith(("http://", "https://")):
		raise argparse.ArgumentTypeError(
			"backend must be EXACT_MODEL=http://host:port"
		)
	return model, url


def _response_headers(headers: httpx.Headers) -> dict[str, str]:
	return {
		key: value for key, value in headers.items()
		if key.lower() not in HOP_BY_HOP
	}


def create_app(
	backends: dict[str, str],
	*,
	timeout: float = 600.0,
	transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
	if not backends:
		raise ValueError("at least one model backend is required")
	backends = {model: url.rstrip("/") for model, url in backends.items()}

	@asynccontextmanager
	async def lifespan(app: FastAPI):
		app.state.client = httpx.AsyncClient(
			timeout=httpx.Timeout(timeout), transport=transport,
		)
		yield
		await app.state.client.aclose()

	app = FastAPI(title="RAG-Stack multi-model vLLM gateway", lifespan=lifespan)
	app.state.backends = backends
	# Model metadata is immutable for the lifetime of these fixed vLLM
	# processes. Cache the first successful aggregate so generator startup does
	# not fan out to every backend while the judge is saturated. /health remains
	# the live readiness signal and intentionally is not cached.
	app.state.models_catalog = None
	app.state.models_lock = asyncio.Lock()

	async def backend_for_payload(request: Request) -> tuple[str, str, bytes, dict]:
		body = await request.body()
		try:
			payload = json.loads(body)
		except (TypeError, ValueError) as exc:
			raise HTTPException(status_code=400, detail=f"invalid JSON: {exc}") from exc
		model = str(payload.get("model") or "")
		backend = backends.get(model)
		if backend is None:
			raise HTTPException(
				status_code=400,
				detail={"error": "unknown model", "model": model,
						"available": sorted(backends)},
			)
		return model, backend, body, payload

	async def proxy_json(request: Request, path: str) -> Response:
		_, backend, body, payload = await backend_for_payload(request)
		client: httpx.AsyncClient = request.app.state.client
		headers = {
			key: value for key, value in request.headers.items()
			if key.lower() not in HOP_BY_HOP | {"host"}
		}
		upstream_request = client.build_request(
			"POST", f"{backend}{path}", content=body, headers=headers,
		)
		if payload.get("stream"):
			upstream = await client.send(upstream_request, stream=True)
			return StreamingResponse(
				upstream.aiter_raw(),
				status_code=upstream.status_code,
				headers=_response_headers(upstream.headers),
				background=BackgroundTask(upstream.aclose),
			)
		upstream = await client.send(upstream_request)
		return Response(
			content=upstream.content,
			status_code=upstream.status_code,
			headers=_response_headers(upstream.headers),
			media_type=upstream.headers.get("content-type"),
		)

	@app.get("/health")
	async def health(request: Request):
		client: httpx.AsyncClient = request.app.state.client

		async def check(model: str, backend: str):
			try:
				response = await client.get(f"{backend}/health")
				return model, response.status_code, response.text[:200]
			except Exception as exc:  # noqa: BLE001 - surfaced in health response
				return model, 0, str(exc)

		checks = await asyncio.gather(*(
			check(model, backend) for model, backend in backends.items()
		))
		failed = [item for item in checks if item[1] != 200]
		if failed:
			raise HTTPException(
				status_code=503,
				detail={"status": "unhealthy", "backends": checks},
			)
		return {"status": "ok", "models": sorted(backends)}

	@app.get("/v1/models")
	async def models(request: Request):
		cached = request.app.state.models_catalog
		if cached is not None:
			return cached

		client: httpx.AsyncClient = request.app.state.client

		async def get_model(model: str, backend: str):
			response = await client.get(f"{backend}/v1/models")
			response.raise_for_status()
			data = response.json().get("data") or []
			for entry in data:
				if entry.get("id") == model:
					return entry
			raise RuntimeError(
				f"backend {backend} does not advertise configured model {model}"
			)

		# Multiple workers commonly initialize generators together. Single-flight
		# the initial aggregate; failures remain uncached so a later request can
		# retry after a transient backend error.
		async with request.app.state.models_lock:
			cached = request.app.state.models_catalog
			if cached is not None:
				return cached
			try:
				data = await asyncio.gather(*(
					get_model(model, backend) for model, backend in backends.items()
				))
			except Exception as exc:  # noqa: BLE001 - turn backend drift into 503
				logger.warning(
					"model catalog aggregation failed (%s): %s",
					type(exc).__name__, exc,
				)
				raise HTTPException(
					status_code=503,
					detail=f"{type(exc).__name__}: {exc}",
				) from exc
			catalog = {"object": "list", "data": data}
			request.app.state.models_catalog = catalog
			return catalog

	@app.post("/v1/chat/completions")
	async def chat_completions(request: Request):
		return await proxy_json(request, "/v1/chat/completions")

	@app.post("/tokenize")
	async def tokenize(request: Request):
		return await proxy_json(request, "/tokenize")

	@app.post("/detokenize")
	async def detokenize(request: Request):
		return await proxy_json(request, "/detokenize")

	return app


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument(
		"--backend", action="append", required=True, type=parse_backend,
		metavar="MODEL=URL",
	)
	parser.add_argument("--host", default="0.0.0.0")
	parser.add_argument("--port", type=int, default=8000)
	parser.add_argument("--timeout", type=float, default=600.0)
	parser.add_argument("--log-level", default="info")
	args = parser.parse_args()
	backends = dict(args.backend)
	if len(backends) != len(args.backend):
		parser.error("duplicate model names are not allowed")

	import uvicorn

	uvicorn.run(
		create_app(backends, timeout=args.timeout),
		host=args.host,
		port=args.port,
		log_level=args.log_level,
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
