#!/usr/bin/env python
"""Smoke-test a remote vLLM judge server and rag-stack's vLLM client path."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any


def _normalize_base_url(base_url: str) -> str:
	base_url = base_url.strip().rstrip("/")
	if not base_url:
		raise ValueError("base URL is empty")
	if not base_url.endswith("/v1"):
		base_url = f"{base_url}/v1"
	return base_url


def _server_root(base_url: str) -> str:
	return base_url[:-3] if base_url.endswith("/v1") else base_url


def _headers() -> dict[str, str]:
	headers = {"Content-Type": "application/json"}
	api_key = os.environ.get("JUDGE_API_KEY") or os.environ.get("VLLM_JUDGE_API_KEY")
	if api_key:
		headers["Authorization"] = f"Bearer {api_key}"
	return headers


def _read_text(url: str, timeout: float) -> str:
	req = urllib.request.Request(url, headers=_headers())
	with urllib.request.urlopen(req, timeout=timeout) as resp:
		return resp.read().decode("utf-8", errors="replace")


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
	body = json.dumps(payload).encode("utf-8")
	req = urllib.request.Request(url, data=body, headers=_headers(), method="POST")
	with urllib.request.urlopen(req, timeout=timeout) as resp:
		return json.loads(resp.read().decode("utf-8"))


def _fail(message: str) -> None:
	print(f"[check_vllm_judge] ERROR: {message}", file=sys.stderr)
	raise SystemExit(1)


def _format_error(e: BaseException) -> str:
	if isinstance(e, urllib.error.HTTPError):
		body = e.read().decode("utf-8", errors="replace")
		return f"HTTP {e.code}: {body[:1000]}"
	return str(e)


def _dump_pydantic(obj: Any) -> Any:
	if hasattr(obj, "model_dump"):
		return obj.model_dump()
	if hasattr(obj, "dict"):
		return obj.dict()
	return obj


def main() -> int:
	parser = argparse.ArgumentParser()
	parser.add_argument("--base-url", default=os.environ.get("JUDGE_BASE_URL") or os.environ.get("VLLM_JUDGE_BASE_URL"))
	parser.add_argument("--model", default=os.environ.get("JUDGE_MODEL") or os.environ.get("VLLM_JUDGE_MODEL"))
	parser.add_argument("--timeout", type=float, default=float(os.environ.get("JUDGE_TIMEOUT", "180")))
	parser.add_argument("--max-tokens", type=int, default=128)
	parser.add_argument("--structured-output-mode", default=os.environ.get("JUDGE_STRUCTURED_OUTPUT_MODE", "auto"))
	parser.add_argument("--skip-structured", action="store_true")
	args = parser.parse_args()

	if not args.base_url:
		_fail("set JUDGE_BASE_URL=http://<judge-host>:<port>/v1 or pass --base-url")
	if not args.model:
		_fail("set JUDGE_MODEL=<served-model-name> or pass --model")

	base_url = _normalize_base_url(args.base_url)
	root_url = _server_root(base_url)

	print(f"[check_vllm_judge] base_url = {base_url}")
	print(f"[check_vllm_judge] model = {args.model}")
	print(f"[check_vllm_judge] timeout = {args.timeout}s")

	try:
		health = _read_text(f"{root_url}/health", args.timeout)
	except Exception as e:
		_fail(f"/health failed: {_format_error(e)}")
	print(f"[check_vllm_judge] /health OK: {health[:120]!r}")

	try:
		models = _read_text(f"{base_url}/models", args.timeout)
	except Exception as e:
		_fail(f"/v1/models failed: {_format_error(e)}")
	print(f"[check_vllm_judge] /v1/models OK: {models[:300]!r}")

	chat_payload = {
		"model": args.model,
		"messages": [
			{"role": "system", "content": "Reply with exactly: OK"},
			{"role": "user", "content": "Health check."},
		],
		"temperature": 0,
		"max_tokens": 16,
	}
	try:
		chat = _post_json(f"{base_url}/chat/completions", chat_payload, args.timeout)
	except Exception as e:
		_fail(f"/v1/chat/completions failed: {_format_error(e)}")
	content = chat.get("choices", [{}])[0].get("message", {}).get("content")
	if not content:
		_fail(f"chat completion returned empty content: {chat}")
	print(f"[check_vllm_judge] chat OK: {content[:120]!r}")

	if args.skip_structured:
		return 0

	try:
		import anyio
		from pydantic import BaseModel

		from rag_stack.ai_clients import get_ai_client
	except Exception as e:
		_fail(f"could not import rag-stack structured-output path: {e}")

	class JudgeSmoke(BaseModel):
		verdict: str
		score: float

	async def _run_structured() -> JudgeSmoke:
		client = get_ai_client(
			f"vllm/{args.model}",
			base_url=base_url,
			timeout=args.timeout,
			max_concurrency=1,
			structured_output_mode=args.structured_output_mode,
		)
		return await client.structured_output(
			[
				{
					"role": "system",
					"content": "You are a JSON-only evaluator.",
				},
				{
					"role": "user",
					"content": "Return a verdict and score for this claim: 2 + 2 = 4.",
				},
			],
			response_format=JudgeSmoke,
			temperature=0,
			max_tokens=args.max_tokens,
		)

	try:
		structured = anyio.run(_run_structured)
	except Exception as e:
		_fail(f"rag-stack structured_output failed: {e}")
	if not isinstance(structured, JudgeSmoke):
		_fail(f"structured_output returned non-schema object: {structured!r}")
	print(f"[check_vllm_judge] structured_output OK: {_dump_pydantic(structured)}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
