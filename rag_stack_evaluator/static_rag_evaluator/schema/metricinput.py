# Portions derived from AutoRAG (https://github.com/Marker-Inc-Korea/AutoRAG), Apache-2.0.
# Modified by the RAG-Stack authors for namespace and runtime integration; see LICENSE.autorag and NOTICE.

from dataclasses import dataclass
from typing import Optional, List, Dict, Callable, Any, Union

import numpy as np
import pandas as pd


@dataclass
class MetricInput:
	query: Optional[str] = None
	queries: Optional[List[str]] = None
	retrieval_gt_contents: Optional[List[List[str]]] = None
	retrieved_contents: Optional[List[str]] = None
	# Post-semantic-retrieval contents BEFORE any reranker/compressor mutation.
	# Lets token-overlap metrics isolate "raw retriever quality" over the
	# pre-reranker stages (query_expansion, vectordb, corpus, semantic_retrieval)
	# instead of riding on the full 6-node chain like deepeval_context_*.
	retrieved_contents_semantic: Optional[List[str]] = None
	# NOTE: chunk-UUID retrieval GT (`retrieval_gt`) is deprecated/removed — chunk
	# UUIDs are unstable under the chunk_size search-space dimension. Retrieval GT
	# is TEXT, carried in `retrieval_gt_contents` (references / answer), scored by
	# the token-overlap metrics.
	retrieved_ids: Optional[List[str]] = None
	prompt: Optional[str] = None
	generated_texts: Optional[str] = None
	generation_gt: Optional[List[str]] = None
	generated_log_probs: Optional[List[float]] = None

	def is_fields_notnone(self, fields_to_check: List[str]) -> bool:
		for field in fields_to_check:
			actual_value = getattr(self, field)

			if actual_value is None:
				return False

			try:
				if not type_checks.get(type(actual_value), lambda _: False)(
					actual_value
				):
					return False
			except Exception:
				return False

		return True

	@classmethod
	def from_dataframe(cls, qa_data: pd.DataFrame) -> List["MetricInput"]:
		"""
		Convert a pandas DataFrame into a list of MetricInput instances.
		qa_data: pd.DataFrame: qa_data DataFrame containing metric data.

		:returns: List[MetricInput]: List of MetricInput objects created from DataFrame rows.
		"""
		instances = []

		for _, row in qa_data.iterrows():
			instance = cls()

			for attr_name in cls.__annotations__:
				if attr_name in row:
					value = row[attr_name]

					if isinstance(value, str):
						stripped = value.strip()
						# An emitted-but-empty answer is a real generator outcome,
						# not a structurally missing field.  Answer-dependent quality
						# metrics must score it as zero instead of silently dropping the
						# row from their aggregate.  Keep the old empty->None semantics
						# for every other field so retrieval/context metrics are
						# unchanged.
						setattr(
							instance,
							attr_name,
							stripped
							if attr_name == "generated_texts" or stripped != ""
							else None,
						)
					elif isinstance(value, list):
						# ReAct may legitimately finish without issuing a Search.  The
						# evaluator still emits the retrieval field for that row, with an
						# empty list.  Preserve that presence marker so context-dependent
						# metrics can score the no-retrieval outcome as zero while still
						# rejecting a genuinely absent retrieval column (None).
						setattr(
							instance,
							attr_name,
							value
							if value or attr_name == "retrieved_contents"
							else None,
						)
					else:
						setattr(instance, attr_name, value)

			instances.append(instance)

		return instances

	@staticmethod
	def _check_list(lst_or_arr: Union[List[Any], np.ndarray]) -> bool:
		if isinstance(lst_or_arr, np.ndarray):
			lst_or_arr = lst_or_arr.flatten().tolist()

		if len(lst_or_arr) == 0:
			return False

		for item in lst_or_arr:
			if item is None:
				return False

			item_type = type(item)

			if item_type in type_checks:
				if not type_checks[item_type](item):
					return False
			else:
				return False

		return True


type_checks: Dict[type, Callable[[Any], bool]] = {
	str: lambda x: len(x.strip()) > 0,
	list: MetricInput._check_list,
	np.ndarray: MetricInput._check_list,
	int: lambda _: True,
	float: lambda _: True,
}
