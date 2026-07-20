import abc
import logging
import sys
from typing import Dict

import pandas as pd
from llama_index.core.llms import LLM

from rag_stack.static_rag_evaluator.nodes.generator.registry import get_llama_index_llm_class
from rag_stack.static_rag_evaluator.schema import BaseModule
from rag_stack.static_rag_evaluator.utils import result_to_dataframe
from rag_stack.static_rag_evaluator.utils.cast import cast_retrieved_contents
from rag_stack.static_rag_evaluator.utils.util import close_llm_async_client

logger = logging.getLogger("RAG-Stack")


class BasePassageCompressor(BaseModule, metaclass=abc.ABCMeta):
	def __init__(self, project_dir: str, *args, **kwargs):
		logger.info(
			f"Initialize passage compressor node - {self.__class__.__name__} module..."
		)

	def __del__(self):
		if sys.is_finalizing():
			return
		logger.info(
			f"Deleting passage compressor node - {self.__class__.__name__} module..."
		)

	def cast_to_run(self, previous_result: pd.DataFrame, *args, **kwargs):
		# Measured serving calls this once per dynamic batch. Keep the hot path
		# observable at DEBUG without charging synchronous INFO I/O to service.
		logger.debug(
			"Running passage compressor node - %s module...",
			self.__class__.__name__,
		)
		assert "query" in previous_result.columns, (
			"previous_result must contain 'query' column."
		)
		assert len(previous_result) > 0, "previous_result must have at least one row."

		queries = previous_result["query"].tolist()

		return queries, cast_retrieved_contents(previous_result)


class LlamaIndexCompressor(BasePassageCompressor, metaclass=abc.ABCMeta):
	param_list = ["prompt", "chat_prompt", "batch"]

	def __init__(self, project_dir: str, **kwargs):
		"""
		Initialize passage compressor module.

		:param project_dir: The project directory
		:param generator_backend: The LlamaIndex LLM backend name (e.g. llama_index_openrouter).
		:param model: The model name (e.g. qwen/qwen3-14b).
		:param kwargs: Extra parameter for init llm
		"""
		super().__init__(project_dir)
		backend_name = kwargs.pop("generator_backend")
		kwargs_dict = dict(
			filter(lambda x: x[0] not in self.param_list, kwargs.items())
		)
		self.llm: LLM = make_llm(backend_name, kwargs_dict)

	def __del__(self):
		if hasattr(self, "llm"):
			close_llm_async_client(self.llm)
			del self.llm
		super().__del__()

	@result_to_dataframe(["retrieved_contents"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		queries, retrieved_contents = self.cast_to_run(previous_result)
		param_dict = dict(filter(lambda x: x[0] in self.param_list, kwargs.items()))
		result = self._pure(queries, retrieved_contents, **param_dict)
		return list(map(lambda x: [x], result))


def make_llm(backend_name: str, kwargs: Dict) -> LLM:
	llm_class = get_llama_index_llm_class(backend_name)
	return llm_class(**kwargs)
