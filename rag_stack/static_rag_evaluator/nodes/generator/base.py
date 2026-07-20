import abc
import functools
import logging
from pathlib import Path
from typing import Union, Tuple, List

import pandas as pd
from llama_index.core.output_parsers import PydanticOutputParser

from rag_stack.static_rag_evaluator.nodes.generator.registry import get_llama_index_llm_class
from rag_stack.static_rag_evaluator.schema import BaseModule
from rag_stack.static_rag_evaluator.utils import result_to_dataframe
from rag_stack.static_rag_evaluator.utils.util import close_llm_async_client

logger = logging.getLogger("RAG-Stack")


class BaseGenerator(BaseModule, metaclass=abc.ABCMeta):
	def __init__(self, project_dir: str, model: str, *args, **kwargs):
		logger.info(f"Initialize generator node - {self.__class__.__name__}")
		self.model = model

	def __del__(self):
		logger.info(f"Deleting generator module - {self.__class__.__name__}")

	def cast_to_run(self, previous_result: pd.DataFrame, *args, **kwargs):
		logger.info(f"Running generator node - {self.__class__.__name__} module...")
		assert "prompts" in previous_result.columns, (
			"previous_result must contain prompts column."
		)
		prompts = previous_result["prompts"].tolist()
		return prompts

	def structured_output(self, prompts: List[str], output_cls):
		response, _, _ = self._pure(prompts)
		parser = PydanticOutputParser(output_cls)
		result = []
		for res in response:
			try:
				result.append(parser.parse(res))
			except Exception as e:
				logger.warning(
					f"Error parsing response: {e} \nSo returning None instead in this case."
				)
				result.append(None)
		return result

	@abc.abstractmethod
	async def astream(self, prompt: str, **kwargs):
		pass

	@abc.abstractmethod
	def stream(self, prompt: str, **kwargs):
		pass


def generator_node(func):
	@functools.wraps(func)
	@result_to_dataframe(["generated_texts", "generated_tokens", "generated_log_probs"])
	def wrapper(
		project_dir: Union[str, Path], previous_result: pd.DataFrame, model: str, **kwargs
	) -> Tuple[List[str], List[List[int]], List[List[float]]]:
		"""
		This decorator makes a generator module to be a node.
		It automatically extracts prompts from previous_result and runs the generator function.

		:param project_dir: The project directory.
		:param previous_result: The previous result that contains prompts,
		:param model: The model name that you want to use.
		:param kwargs: The extra parameters for initializing the llm instance.
		:return: Pandas dataframe that contains generated texts, generated tokens, and generated log probs.
		    Each column is "generated_texts", "generated_tokens", and "generated_log_probs".
		"""
		logger.info(f"Running generator node - {func.__name__} module...")
		assert "prompts" in previous_result.columns, (
			"previous_result must contain prompts column."
		)
		prompts = previous_result["prompts"].tolist()
		if func.__name__ == "llama_index_llm":
			generator_backend = kwargs.pop("generator_backend", "llama_index_openai")
			llm_class = get_llama_index_llm_class(generator_backend)
			batch = kwargs.pop("batch", 100)
			if llm_class.class_name() in [
				"HuggingFace_LLM",
				"HuggingFaceInferenceAPI",
				"TextGenerationInference",
			]:
				model_name = kwargs.pop("model", None)
				if model_name is not None:
					kwargs["model_name"] = model_name
				else:
					if "model_name" not in kwargs.keys():
						raise ValueError(
							"`model` or `model_name` parameter must be provided for using huggingfacellm."
						)
				kwargs["tokenizer_name"] = kwargs["model_name"]
			kwargs.setdefault("model", model)
			llm_instance = llm_class(**kwargs)
			result = func(prompts=prompts, llm=llm_instance, batch=batch)
			close_llm_async_client(llm_instance)
			del llm_instance
			return result
		else:
			return func(prompts=prompts, llm=llm, **kwargs)

	return wrapper
