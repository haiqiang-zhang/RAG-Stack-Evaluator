from __future__ import annotations

from unittest.mock import Mock

from rag_stack.static_rag_evaluator.nodes.generator.vllm_api import VllmAPI


def test_max_model_length_matches_requested_model(monkeypatch):
	response = Mock()
	response.raise_for_status.return_value = None
	response.json.return_value = {
		"data": [
			{"id": "small", "max_model_len": 4096},
			{"id": "large", "max_model_len": 32768},
		]
	}
	monkeypatch.setattr(
		"rag_stack.static_rag_evaluator.nodes.generator.vllm_api.requests.get",
		lambda *args, **kwargs: response,
	)
	instance = object.__new__(VllmAPI)
	instance.uri = "http://gateway:8000"
	instance.model = "large"
	instance.request_timeout = 30
	assert instance.get_max_model_length() == 32768


def test_prompt_truncation_reserves_output_budget():
	instance = object.__new__(VllmAPI)
	instance.max_model_len = 10
	instance.max_token_size = 4
	instance.encoding_for_model = lambda prompt: {"tokens": list(range(20))}
	instance.decoding_for_model = lambda tokens: {"prompt": tokens}
	assert instance.truncate_by_token("prompt") == list(range(6))


def test_chat_request_caps_rendered_prompt_to_context_window(monkeypatch):
	instance = object.__new__(VllmAPI)
	instance.uri = "http://gateway:8000"
	instance.model = "small"
	instance.request_timeout = 30
	instance.max_model_len = 32768
	instance.max_token_size = 512

	captured = {}
	response = Mock()
	response.raise_for_status.return_value = None
	response.json.return_value = {"choices": []}

	def fake_post(url, *, json, timeout):
		captured.update(url=url, payload=json, timeout=timeout)
		return response

	monkeypatch.setattr(
		"rag_stack.static_rag_evaluator.nodes.generator.vllm_api.requests.post",
		fake_post,
	)
	instance.call_vllm_api("prompt", max_tokens=512)

	assert captured["payload"]["max_tokens"] == 512
	assert captured["payload"]["truncate_prompt_tokens"] == 32256
	assert captured["payload"]["truncation_side"] == "right"


def test_result_tokenizes_whole_answer_at_most_once():
	instance = object.__new__(VllmAPI)
	instance.call_vllm_api = lambda *args, **kwargs: {
		"choices": [{
			"message": {"content": "one two three"},
			"logprobs": {"content": [
				{"token": "one", "logprob": -0.1},
				{"token": " two", "logprob": -0.2},
				{"token": " three", "logprob": -0.3},
			]},
		}],
	}
	calls = []
	instance.encoding_for_model = lambda text, add_special_tokens=True: (
		calls.append((text, add_special_tokens)) or {"tokens": [11, 12, 13]}
	)
	answer, tokens, logprobs = instance.get_result("prompt")
	assert answer == "one two three"
	assert tokens == [11, 12, 13]
	assert logprobs == [-0.1, -0.2, -0.3]
	assert calls == [("one two three", False)]
