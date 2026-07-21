# Portions derived from AutoRAG (https://github.com/Marker-Inc-Korea/AutoRAG), Apache-2.0.
# Modified by the RAG-Stack authors for namespace and runtime integration; see LICENSE.autorag and NOTICE.

import logging
from abc import ABCMeta
from pathlib import Path
from typing import Union

import pandas as pd

from rag_stack_evaluator.static_rag_evaluator.schema.base import BaseModule
from rag_stack_evaluator.static_rag_evaluator.utils.cast import cast_retrieved_contents

logger = logging.getLogger("RAG-Stack")


class BasePromptMaker(BaseModule, metaclass=ABCMeta):
	def __init__(self, project_dir: Union[str, Path], *args, **kwargs):
		logger.info(
			f"Initialize prompt maker node - {self.__class__.__name__} module..."
		)

	def __del__(self):
		logger.info(f"Prompt maker node - {self.__class__.__name__} module is deleted.")

	def cast_to_run(self, previous_result: pd.DataFrame, *args, **kwargs):
		logger.debug(
			"Running prompt maker node - %s module...", self.__class__.__name__,
		)
		# get query and retrieved contents from previous_result
		assert "query" in previous_result.columns, (
			"previous_result must have query column."
		)

		query = previous_result["query"].tolist()
		prompt = kwargs.pop("prompt")
		return query, cast_retrieved_contents(previous_result), prompt
