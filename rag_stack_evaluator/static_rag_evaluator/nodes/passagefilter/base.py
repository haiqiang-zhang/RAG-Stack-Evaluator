# Portions derived from AutoRAG (https://github.com/Marker-Inc-Korea/AutoRAG), Apache-2.0.
# Modified by the RAG-Stack authors for namespace and runtime integration; see LICENSE.autorag and NOTICE.

import abc
import logging
from pathlib import Path
from typing import Union

import pandas as pd

from rag_stack_evaluator.static_rag_evaluator.schema.base import BaseModule
from rag_stack.utils.preprocess import validate_qa_dataset
from rag_stack_evaluator.static_rag_evaluator.utils.cast import cast_retrieve_infos

logger = logging.getLogger("RAG-Stack")


class BasePassageFilter(BaseModule, metaclass=abc.ABCMeta):
	def __init__(self, project_dir: Union[str, Path], *args, **kwargs):
		logger.info(f"Initialize passage filter node - {self.__class__.__name__}")

	def __del__(self):
		logger.info(f"Prompt maker node - {self.__class__.__name__} module is deleted.")

	def cast_to_run(self, previous_result: pd.DataFrame, *args, **kwargs):
		logger.info(
			f"Running passage filter node - {self.__class__.__name__} module..."
		)
		validate_qa_dataset(previous_result)

		# find queries columns
		assert "query" in previous_result.columns, (
			"previous_result must have query column."
		)
		queries = previous_result["query"].tolist()

		retrieve_infos = cast_retrieve_infos(previous_result)
		return (
			queries,
			retrieve_infos["retrieved_contents"],
			retrieve_infos["retrieve_scores"],
			retrieve_infos["retrieved_ids"],
		)
