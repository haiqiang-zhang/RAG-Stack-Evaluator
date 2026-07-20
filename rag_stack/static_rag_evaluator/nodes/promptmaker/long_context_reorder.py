import logging
from typing import List

import numpy as np
import pandas as pd

from rag_stack.static_rag_evaluator.nodes.promptmaker.base import BasePromptMaker
from rag_stack.static_rag_evaluator.utils import result_to_dataframe
from rag_stack.static_rag_evaluator.utils.cast import cast_retrieve_scores

logger = logging.getLogger("RAG-Stack")

_summarizer_fallback_logged = False


class LongContextReorder(BasePromptMaker):
	@result_to_dataframe(["prompts"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		query, retrieved_contents, prompt = self.cast_to_run(
			previous_result, *args, **kwargs
		)
		retrieve_scores = cast_retrieve_scores(previous_result)
		return self._pure(prompt, query, retrieved_contents, retrieve_scores)

	def _pure(
		self,
		prompt: str,
		queries: List[str],
		retrieved_contents: List[List[str]],
		retrieve_scores: List[List[float]],
	) -> List[str]:
		"""
		Models struggle to access significant details found
		in the center of extended contexts. A study
		(https://arxiv.org/abs/2307.03172) observed that the best
		performance typically arises when crucial data is positioned
		at the start or conclusion of the input context. Additionally,
		as the input context lengthens, performance drops notably, even
		in models designed for long contexts."

		.. Code:: yaml
		nodes:
		- stage: prompt_maker
		  modules:
		  - component: long_context_reorder
		    prompt: [Answer this question: {query} \n\n {retrieved_contents},
		    Read the passages carefully and answer this question: {query} \n\n Passages: {retrieved_contents}]

		:param prompt: A prompt string.
		:param queries: List of query strings.
		:param retrieved_contents: List of retrieved contents.
		:param retrieve_scores: List of `retrieve scores`.
		:return: Prompts that are made by long context reorder.
		"""

		def long_context_reorder_row(
			_prompt: str,
			_query: str,
			_retrieved_contents: List[str],
			_retrieve_scores: List[float],
		) -> str:
			global _summarizer_fallback_logged
			if isinstance(_retrieved_contents, np.ndarray):
				_retrieved_contents = _retrieved_contents.tolist()
			if not len(_retrieved_contents) == len(_retrieve_scores):
				# Compression changes the content cardinality without carrying a
				# matching score vector. This is an expected fallback, and it can
				# occur for every request in measured serving, so log it only once
				# per process instead of taking the logging lock on the hot path.
				if not _summarizer_fallback_logged:
					_summarizer_fallback_logged = True
					logger.info(
						"Compressed contents do not match retrieval scores; "
						"long_context_reorder passes contents through unchanged "
						"(logged once)."
					)
				return _prompt.format(
					query=_query, retrieved_contents="\n\n".join(_retrieved_contents)
				)
			content_scores = list(zip(_retrieved_contents, _retrieve_scores))
			if not content_scores:
				# Empty retrieval for this query (degenerate config): nothing to
				# reorder — plain prompt instead of crashing on zip(*[]).
				return _prompt.format(query=_query, retrieved_contents="")
			sorted_content_scores = sorted(
				content_scores, key=lambda x: x[1], reverse=True
			)
			content_result, score_result = zip(*sorted_content_scores)
			_retrieved_contents.append(content_result[0])
			contents_str = "\n\n".join(_retrieved_contents)
			return _prompt.format(query=_query, retrieved_contents=contents_str)

		return list(
			map(
				lambda x: long_context_reorder_row(prompt, x[0], x[1], x[2]),
				zip(queries, retrieved_contents, retrieve_scores),
			)
		)
