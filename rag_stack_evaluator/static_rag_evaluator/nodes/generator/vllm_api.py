# Portions derived from AutoRAG (https://github.com/Marker-Inc-Korea/AutoRAG), Apache-2.0.
# Modified by the RAG-Stack authors for namespace and runtime integration; see LICENSE.autorag and NOTICE.

import logging
import os
from typing import List, Tuple, Dict, Union
import time

import pandas as pd
import requests
from asyncio import to_thread
from urllib.parse import urlsplit, urlunsplit

from rag_stack_evaluator.static_rag_evaluator.nodes.generator.base import BaseGenerator
from rag_stack_evaluator.static_rag_evaluator.utils.util import get_event_loop, process_batch, result_to_dataframe

logger = logging.getLogger("RAG-Stack")

DEFAULT_MAX_TOKENS = 4096  # Default token limit


def normalize_vllm_server_uri(uri: str) -> str:
	"""Return the server root for OpenAI-compatible and native vLLM routes.

	Configuration often calls the OpenAI endpoint ``base_url`` and therefore
	includes a terminal ``/v1``.  This client also needs vLLM's root-level
	``/tokenize`` and ``/detokenize`` routes, so store one unambiguous server
	root and append each API route exactly once at its call site.
	"""
	expanded = os.path.expandvars(str(uri)).rstrip("/")
	parts = urlsplit(expanded)
	if "$" in expanded or parts.scheme not in ("http", "https") or not parts.netloc:
		raise ValueError(
			"vllm_api uri did not resolve to an HTTP endpoint: "
			f"{uri!r}. Set RAG_LLM_BASE_URL on the optimizer host."
		)
	path = parts.path
	if path.endswith("/v1"):
		path = path[:-3]
	return urlunsplit((parts.scheme, parts.netloc, path.rstrip("/"), "", ""))


class VllmAPI(BaseGenerator):
	def __init__(
		self,
		project_dir,
		model: str,
		uri: str,
		max_tokens: int = None,
		batch: int = 100,
		request_timeout: float = 600.0,
		*args,
		**kwargs,
	):
		"""
		VLLM API Wrapper for OpenAI-compatible chat/completions format.

		:param project_dir: Project directory.
		:param model: Model name (e.g., LLaMA model).
		:param uri: VLLM API server URI.
		:param max_tokens: Maximum token limit.
		    Default is 4096.
		:param batch: Request batch size.
		    Default is 16.
		"""
		super().__init__(project_dir, model, *args, **kwargs)
		assert batch > 0, "Batch size must be greater than 0."
		self.uri = normalize_vllm_server_uri(uri)
		self.batch = batch
		self.request_timeout = float(request_timeout)
		# Use the provided max_tokens if available, otherwise use the default
		self.max_token_size = max_tokens if max_tokens else DEFAULT_MAX_TOKENS
		self.max_model_len = self.get_max_model_length()
		logger.info(f"{model} max model length: {self.max_model_len}")

	@result_to_dataframe(["generated_texts", "generated_tokens", "generated_log_probs"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		kwargs.pop("model", None)  # already captured in self.model
		prompts = self.cast_to_run(previous_result)
		return self._pure(prompts, **kwargs)

	def _pure(
		self,
		prompts: Union[List[str], List[List[Dict]]],
		truncate: bool = True,
		**kwargs,
	) -> Tuple[List[str], List[List[int]], List[List[float]]]:
		"""
		Method to call the VLLM API to generate text.

		:param prompts: List of input prompts.
		:param truncate: Whether to truncate input prompts to fit within the token limit.
		:param kwargs: Additional options (e.g., temperature, top_p).
		:return: Generated text, token lists, and log probability lists.
		"""
		if kwargs.get("logprobs") is not None:
			kwargs.pop("logprobs")
			logger.warning(
				"parameter logprob does not effective. It always set to True."
			)
		if kwargs.get("n") is not None:
			kwargs.pop("n")
			logger.warning("parameter n does not effective. It always set to 1.")

		if truncate:
			prompts = list(map(lambda p: self.truncate_by_token(p), prompts))
		loop = get_event_loop()
		tasks = [to_thread(self.get_result, prompt, **kwargs) for prompt in prompts]
		results = loop.run_until_complete(process_batch(tasks, self.batch))

		answer_result = list(map(lambda x: x[0], results))
		token_result = list(map(lambda x: x[1], results))
		logprob_result = list(map(lambda x: x[2], results))
		return answer_result, token_result, logprob_result

	def truncate_by_token(self, prompt: Union[str, List[Dict]]) -> str:
		"""
		Function to truncate prompts to fit within the maximum token limit.
		"""
		if not isinstance(prompt, str):
			content_list = [msg["content"] for msg in prompt]
			prompt = "\n".join(content_list)
		tokens = self.encoding_for_model(prompt)["tokens"]  # Simple tokenization
		input_budget = max(1, self.max_model_len - self.max_token_size)
		return self.decoding_for_model(tokens[:input_budget])["prompt"]

	def call_vllm_api(self, prompt: Union[str, List[Dict]], **kwargs) -> dict:
		"""
		Calls the VLLM API to get chat/completions responses.

		:param prompt: Input prompt.
		:param kwargs: Additional API options (e.g., temperature, max_tokens).
		:return: API response.
		"""
		max_tokens = min(
			kwargs.get("max_tokens", self.max_token_size), self.max_token_size
		)
		# truncate_by_token() budgets the raw message text, but the chat template
		# can add one or more tokens afterwards.  Ask vLLM to enforce the same
		# budget on the fully rendered chat prompt so an otherwise valid request
		# cannot exceed max_model_len at the API boundary.
		input_budget = max(1, self.max_model_len - max_tokens)
		payload = {
			"model": self.model,
			"messages": parse_prompt(prompt),
			"temperature": kwargs.get("temperature", 0.4),
			"max_tokens": max_tokens,
			"truncate_prompt_tokens": input_budget,
			"truncation_side": "right",
			"logprobs": True,
			"n": 1,
		}
		start_time = time.time()  # Record request start time
		response = requests.post(
			f"{self.uri}/v1/chat/completions",
			json=payload,
			timeout=self.request_timeout,
		)
		end_time = time.time()  # Record request end time

		response.raise_for_status()
		elapsed_time = end_time - start_time  # Calculate elapsed time
		logger.info(
			f"Request chat completions to vllm server completed in {elapsed_time:.2f} seconds"
		)
		return response.json()

	# Additional method: abstract method implementation
	async def astream(self, prompt: Union[str, List[Dict]], **kwargs):
		"""
		Asynchronous streaming method not implemented.
		"""
		raise NotImplementedError("astream method is not implemented for VLLM API yet.")

	def stream(self, prompt: Union[str, List[Dict]], **kwargs):
		"""
		Synchronous streaming method not implemented.
		"""
		raise NotImplementedError("stream method is not implemented for VLLM API yet.")

	def get_result(self, prompt: Union[str, List[Dict]], **kwargs):
		response = self.call_vllm_api(prompt, **kwargs)
		choice = response["choices"][0]
		answer = choice["message"]["content"]

		# Handle cases where logprobs is None
		if choice.get("logprobs") and "content" in choice["logprobs"]:
			content_logprobs = choice["logprobs"]["content"]
			logprobs = [item["logprob"] for item in content_logprobs]
			# Some vLLM versions expose token_id directly. Older versions do not;
			# tokenize the whole answer once in that case. The previous per-token
			# fallback generated tens of thousands of avoidable HTTP requests and
			# returned a nested list instead of token ids.
			direct_ids = [item.get("token_id") for item in content_logprobs]
			if direct_ids and all(isinstance(token_id, int) for token_id in direct_ids):
				tokens = direct_ids
			elif answer:
				tokens = self.encoding_for_model(
					answer, add_special_tokens=False,
				)["tokens"]
			else:
				tokens = []
		else:
			logprobs = []
			tokens = []

		return answer, tokens, logprobs

	def encoding_for_model(self, answer_piece: str, add_special_tokens: bool = True):
		payload = {
			"model": self.model,
			"prompt": answer_piece,
			"add_special_tokens": add_special_tokens,
		}
		response = requests.post(
			f"{self.uri}/tokenize", json=payload, timeout=self.request_timeout,
		)
		response.raise_for_status()
		return response.json()

	def decoding_for_model(self, tokens: list[int]):
		payload = {
			"model": self.model,
			"tokens": tokens,
		}
		response = requests.post(
			f"{self.uri}/detokenize", json=payload, timeout=self.request_timeout,
		)
		response.raise_for_status()
		return response.json()

	def get_max_model_length(self):
		response = requests.get(
			f"{self.uri}/v1/models", timeout=self.request_timeout,
		)
		response.raise_for_status()
		data = response.json().get("data") or []
		for entry in data:
			if entry.get("id") == self.model:
				return int(entry["max_model_len"])
		raise ValueError(
			f"vLLM endpoint {self.uri} does not advertise model {self.model!r}; "
			f"available={[entry.get('id') for entry in data]}"
		)


def parse_prompt(prompt: Union[str, List[Dict]]) -> List[Dict]:
	if isinstance(prompt, str):
		return [{"role": "user", "content": prompt}]
	elif isinstance(prompt, list):
		return prompt
	else:
		raise ValueError("prompt must be a string or a list of dicts.")
