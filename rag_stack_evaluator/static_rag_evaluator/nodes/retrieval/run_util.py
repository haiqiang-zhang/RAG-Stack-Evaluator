# Portions derived from AutoRAG (https://github.com/Marker-Inc-Korea/AutoRAG), Apache-2.0.
# Modified by the RAG-Stack authors for namespace and runtime integration; see LICENSE.autorag and NOTICE.

from typing import List, Union, Dict

import pandas as pd

from rag_stack_evaluator.static_rag_evaluator.evaluation import evaluate_retrieval
from rag_stack_evaluator.static_rag_evaluator.schema.metricinput import MetricInput


def evaluate_retrieval_node(
	result_df: pd.DataFrame,
	metric_inputs: List[MetricInput],
	metrics: Union[List[str], List[Dict]],
) -> pd.DataFrame:
	"""
	Evaluate retrieval node from retrieval node result dataframe.

	:param result_df: The result dataframe from a retrieval node.
	:param metric_inputs: List of metric input schema for AutoRAG.
	:param metrics: Metric list from input strategies.
	:return: Return result_df with metrics columns.
	    The columns will be 'retrieved_contents', 'retrieved_ids', 'retrieve_scores', and metric names.
	"""

	@evaluate_retrieval(
		metric_inputs=metric_inputs,
		metrics=metrics,
	)
	def evaluate_this_module(df: pd.DataFrame):
		return (
			df["retrieved_contents"].tolist(),
			df["retrieved_ids"].tolist(),
			df["retrieve_scores"].tolist(),
		)

	return evaluate_this_module(result_df)
