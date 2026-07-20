import abc
import logging
from pathlib import Path
from typing import List, Union

import pandas as pd

from rag_stack.static_rag_evaluator.nodes.util import make_generator_callable_param
from rag_stack.static_rag_evaluator.schema import BaseModule
from rag_stack.utils.preprocess import validate_qa_dataset

logger = logging.getLogger("RAG-Stack")


class BaseQueryExpansion(BaseModule, metaclass=abc.ABCMeta):
	def __init__(self, project_dir: Union[str, Path], *args, **kwargs):
		logger.info(
			f"Initialize query expansion node - {self.__class__.__name__} module..."
		)
		# set generator module for query expansion
		generator_class, generator_param = make_generator_callable_param(kwargs)
		# Mark the vLLM backend as the AUXILIARY (query-expansion) vLLM so that,
		# on the perf path, it pulls its own cached subprocess (own model+device)
		# instead of the main generator's. No-op for non-vLLM backends and off
		# the cache path (quality eval).
		if generator_class.__name__ == "Vllm":
			generator_param.setdefault("is_aux_vllm", True)
		self.generator = generator_class(project_dir, **generator_param)

	def __del__(self):
		if not hasattr(self, "generator"):
			return
		del self.generator
		logger.info(
			f"Delete query expansion node - {self.__class__.__name__} module..."
		)

	def cast_to_run(self, previous_result: pd.DataFrame, *args, **kwargs):
		logger.info(
			f"Running query expansion node - {self.__class__.__name__} module..."
		)
		validate_qa_dataset(previous_result)

		# find queries columns
		assert "query" in previous_result.columns, (
			"previous_result must have query column."
		)
		queries = previous_result["query"].tolist()
		return queries

	@staticmethod
	def _check_expanded_query(queries: List[str], expanded_queries: List[List[str]]):
		return list(
			map(
				lambda query, expanded_query_list: check_expanded_query(
					query, expanded_query_list
				),
				queries,
				expanded_queries,
			)
		)


def check_expanded_query(query: str, expanded_query_list: List[str]):
	# check if the expanded query is the same as the original query
	expanded_query_list = list(map(lambda x: x.strip(), expanded_query_list))
	return [
		expanded_query if expanded_query else query
		for expanded_query in expanded_query_list
	]
