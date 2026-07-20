import logging
from typing import List, Tuple, Union, Dict

import pandas as pd
import tiktoken
from openai import AsyncOpenAI
from tiktoken import Encoding

from rag_stack.static_rag_evaluator.nodes.generator.base import BaseGenerator
from rag_stack.static_rag_evaluator.utils.util import (
	get_event_loop,
	process_batch,
	pop_params,
	result_to_dataframe,
)

logger = logging.getLogger("RAG-Stack")

from rag_stack.model_map import get_context_length


class OpenAILLM(BaseGenerator):
	def __init__(self, project_dir, model: str, batch: int = 100, *args, **kwargs):
		super().__init__(project_dir, model, *args, **kwargs)
		assert batch > 0, "batch size must be greater than 0."
		self.batch = batch

		client_init_params = pop_params(AsyncOpenAI.__init__, kwargs)
		self.client = AsyncOpenAI(**client_init_params)
		# OpenRouter: disable reasoning/thinking by default to prevent
		# token exhaustion. Override with reasoning: true in YAML config.
		self._is_openrouter = bool(
			client_init_params.get("base_url")
			and "openrouter.ai" in str(client_init_params["base_url"])
		)
		self._reasoning = kwargs.pop("reasoning", False)
		try:
			self.tokenizer = tiktoken.encoding_for_model(self.model)
		except KeyError:
			self.tokenizer = tiktoken.get_encoding("o200k_base")

		max_tokens = get_context_length(self.model)
		self.max_token_size = max_tokens - 7  # reserve for chat token overhead

	@result_to_dataframe(["generated_texts", "generated_tokens", "generated_log_probs"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		kwargs.pop("model", None)  # already captured in self.model
		prompts = self.cast_to_run(previous_result)
		return self._pure(prompts, **kwargs)

	def _pure(
		self,
		prompts: Union[List[str], List[List[dict]]],
		truncate: bool = True,
		**kwargs,
	) -> Tuple[List[str], List[List[int]], List[List[float]]]:
		"""
		OpenAI generator module.
		Uses an official openai library for generating answer from the given prompt.
		It returns real token ids and log probs, so you must use this for using token ids and log probs.

		:param prompts: A list of prompts.
		:param llm: A model name for OpenAI.
		    Default is gpt-3.5-turbo.
		:param batch: Batch size for openai api call.
		    If you get API limit errors, you should lower the batch size.
		    Default is 16.
		:param truncate: Whether to truncate the input prompt.
		    Default is True.
		:param api_key: OpenAI API key. You can set this by passing env variable `OPENAI_API_KEY`
		:param kwargs: The optional parameter for openai api call `openai.chat.completion`
		    See https://platform.openai.com/docs/api-reference/chat/create for more details.
		:return: A tuple of three elements.
		    The first element is a list of generated text.
		    The second element is a list of generated text's token ids.
		    The third element is a list of generated text's log probs.
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
			prompts = list(
				map(
					lambda prompt: truncate_by_token(
						prompt, self.tokenizer, self.max_token_size
					),
					prompts,
				)
			)

		openai_chat_params = pop_params(self.client.chat.completions.create, kwargs)
		loop = get_event_loop()
		if (
			self.model.startswith("o1")
			or self.model.startswith("o3")
			or self.model.startswith("o4")
		):
			tasks = [
				self.get_result_reasoning(prompt, **openai_chat_params)
				for prompt in prompts
			]
		elif self.model.startswith("gpt-5"):
			responses_create_params = pop_params(self.client.responses.create, kwargs)
			tasks = [
				self.get_result_gpt_5(prompt, **responses_create_params)
				for prompt in prompts
			]
		else:
			tasks = [
				self.get_result(prompt, **openai_chat_params) for prompt in prompts
			]
		result = loop.run_until_complete(process_batch(tasks, self.batch))
		answer_result = list(map(lambda x: x[0], result))
		token_result = list(map(lambda x: x[1], result))
		logprob_result = list(map(lambda x: x[2], result))
		return answer_result, token_result, logprob_result

	def structured_output(self, prompts: List[str], output_cls, **kwargs):
		if kwargs.get("logprobs") is not None:
			kwargs.pop("logprobs")
			logger.warning(
				"parameter logprob does not effective. It always set to False."
			)
		if kwargs.get("n") is not None:
			kwargs.pop("n")
			logger.warning("parameter n does not effective. It always set to 1.")

		prompts = list(
			map(
				lambda prompt: truncate_by_token(
					prompt, self.tokenizer, self.max_token_size
				),
				prompts,
			)
		)

		openai_chat_params = pop_params(self.client.responses.parse, kwargs)
		loop = get_event_loop()
		tasks = [
			self.get_structured_result(prompt, output_cls, **openai_chat_params)
			for prompt in prompts
		]
		result = loop.run_until_complete(process_batch(tasks, self.batch))
		return result

	async def astream(self, prompt: Union[str, List[Dict]], **kwargs):
		if kwargs.get("logprobs") is not None:
			kwargs.pop("logprobs")
			logger.warning(
				"parameter logprob does not effective. It always set to False."
			)
		if kwargs.get("n") is not None:
			kwargs.pop("n")
			logger.warning("parameter n does not effective. It always set to 1.")

		prompt = truncate_by_token(prompt, self.tokenizer, self.max_token_size)

		openai_chat_params = pop_params(self.client.chat.completions.create, kwargs)

		stream = await self.client.chat.completions.create(
			model=self.model,
			messages=parse_prompt(prompt),
			logprobs=False,
			n=1,
			stream=True,
			**openai_chat_params,
		)
		result = ""
		async for chunk in stream:
			if chunk.choices[0].delta.content is not None:
				result += chunk.choices[0].delta.content
				yield result

	def stream(self, prompt: Union[str, List[Dict]], **kwargs):
		raise NotImplementedError("stream method is not implemented yet.")

	async def get_structured_result(
		self, prompt: Union[str, List[Dict]], output_cls, **kwargs
	):
		if self.model.startswith("gpt-3.5") or self.model in [
			"gpt-4",
			"gpt-4-0613",
			"gpt-4-32k",
			"gpt-4-32k-0613",
			"gpt-4-turbo",
		]:
			raise ValueError("structured output is supported after the gpt-4o model.")

		response = await self.client.responses.parse(
			model=self.model,
			input=parse_prompt(prompt),
			text_format=output_cls,
			**kwargs,
		)
		return response.output_parsed

	async def get_result(self, prompt: Union[str, List[dict]], **kwargs):
		logprobs = True
		messages = parse_prompt(prompt)

		if self._is_openrouter and not self._reasoning and "extra_body" not in kwargs:
			kwargs["extra_body"] = {"reasoning": {"exclude": True}}
		response = await self.client.chat.completions.create(
			model=self.model,
			messages=messages,
			logprobs=logprobs,
			n=1,
			**kwargs,
		)
		choice = response.choices[0]
		answer = choice.message.content
		if answer is None:
			raise ValueError(
				f"Model '{self.model}' returned content=None. "
				f"Full response: {choice.message}"
			)
		if choice.logprobs and choice.logprobs.content:
			logprobs = [x.logprob for x in choice.logprobs.content]
			tokens = [
				self.tokenizer.encode(x.token, allowed_special="all")[0]
				for x in choice.logprobs.content
			]
		else:
			tokens = self.tokenizer.encode(answer, allowed_special="all")
			logprobs = [0.5] * len(tokens)
		if len(tokens) != len(logprobs):
			raise ValueError("tokens and logprobs size is different.")
		return answer, tokens, logprobs

	async def get_result_reasoning(self, prompt: Union[str, List[dict]], **kwargs):
		if not (
			self.model.startswith("o1")
			or self.model.startswith("o3")
			or self.model.startswith("o4")
		):
			raise ValueError("get_result_reasoning is only for o1,o3,o4 models.")
		# The default temperature for the o1 model is 1. 1 is only supported.
		# See https://platform.openai.com/docs/guides/reasoning about beta limitation of o1 models.
		unsupported_params = [
			"temperature",
			"top_p",
			"presence_penalty",
			"frequency_penalty",
			"logprobs",
			"top_logprobs",
			"logit_bias",
		]
		kwargs["max_completion_tokens"] = kwargs.pop("max_tokens", None)
		for unsupported_param in unsupported_params:
			kwargs.pop(unsupported_param, None)
		messages = parse_prompt(prompt)

		if self._is_openrouter and not self._reasoning and "extra_body" not in kwargs:
			kwargs["extra_body"] = {"reasoning": {"exclude": True}}
		response = await self.client.chat.completions.create(
			model=self.model,
			messages=messages,
			n=1,
			**kwargs,
		)
		answer = response.choices[0].message.content
		tokens = self.tokenizer.encode(answer, allowed_special="all")
		pseudo_log_probs = [0.5] * len(tokens)
		return answer, tokens, pseudo_log_probs

	async def get_result_gpt_5(self, prompt: Union[str, List[dict]], **kwargs):
		if not self.model.startswith("gpt-5"):
			raise ValueError("get_result_gpt_5 is only for gpt-5 models.")
		api_key = getattr(self.client, "api_key", None)
		if isinstance(api_key, str) and api_key.startswith("mock_"):
			answer = "Why not"
			tokens = self.tokenizer.encode(answer, allowed_special="all")
			pseudo_log_probs = [0.5] * len(tokens)
			return answer, tokens, pseudo_log_probs
		messages = parse_prompt(prompt)
		instruction = "\n\n".join(
			[msg["content"] for msg in messages if msg["role"] == "system"]
		)
		user_input = "\n\n".join(
			[msg["content"] for msg in messages if msg["role"] == "user"]
		)
		response = await self.client.responses.create(
			model=self.model,
			instructions=instruction,
			input=user_input,
			**kwargs,
		)
		answer: str = response.output_text
		tokens = self.tokenizer.encode(answer, allowed_special="all")
		pseudo_log_probs = [0.5] * len(tokens)
		return answer, tokens, pseudo_log_probs


def truncate_by_token(
	prompt: Union[str, List[Dict]], tokenizer: Encoding, max_token_size: int
):
	if isinstance(prompt, list):
		prompt = tiktoken_messages_to_string(prompt)
	tokens = tokenizer.encode(prompt, allowed_special="all")
	return tokenizer.decode(tokens[:max_token_size])


def tiktoken_messages_to_string(messages: List[Dict[str, str]]) -> str:
	"""Convert chat messages to string format for accurate token counting"""
	formatted_parts = [
		f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>"
		for message in messages
	]
	formatted_parts.append("<|im_start|>assistant")
	full_string = "\n".join(formatted_parts)
	return full_string


def parse_prompt(prompt: Union[str, List[Dict]]) -> List[Dict]:
	if isinstance(prompt, str):
		return [{"role": "user", "content": prompt}]
	elif isinstance(prompt, list):
		return prompt
	else:
		raise ValueError("prompt must be a string or a list of dicts.")
