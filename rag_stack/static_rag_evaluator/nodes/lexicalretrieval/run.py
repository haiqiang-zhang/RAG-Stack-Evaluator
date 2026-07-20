# Portions derived from AutoRAG (https://github.com/Marker-Inc-Korea/AutoRAG), Apache-2.0.
# Modified by the RAG-Stack authors for namespace and runtime integration; see LICENSE.autorag and NOTICE.

import os
from typing import List, Dict, Union

import pandas as pd

from rag_stack.static_rag_evaluator.evaluation import evaluate_retrieval
from rag_stack.static_rag_evaluator.evaluation.retrieval import RETRIEVAL_METRIC_FUNC_DICT
from rag_stack.static_rag_evaluator.schema.metricinput import MetricInput
from rag_stack.static_rag_evaluator.strategy import measure_speed
from rag_stack.static_rag_evaluator.utils.util import apply_recursive, to_list
from rag_stack.security import safe_dataframe_to_csv


def run_lexical_retrieval_node(
	modules: List,
	module_params: List[Dict],
	previous_result: pd.DataFrame,
	node_line_dir: str,
	strategies: Dict,
) -> pd.DataFrame:
	"""
	Run the single lexical retrieval module supplied by the upstream sampler.

	Greedy multi-module selection has been removed. Exactly one module is expected.
	"""
	if len(modules) != 1:
		raise ValueError(
			f"lexical_retrieval expects exactly one module after sampling, "
			f"got {len(modules)}"
		)
	module = modules[0]
	module_param = module_params[0]

	if not os.path.exists(node_line_dir):
		os.makedirs(node_line_dir)
	project_dir = os.environ["PROJECT_DIR"]
	qa_df = pd.read_parquet(
		os.path.join(project_dir, "data", "qa.parquet"), engine="pyarrow"
	)
	retrieval_gt_contents = qa_df["retrieval_gt_contents"].tolist()
	metric_inputs = [
		MetricInput(retrieval_gt_contents=ret_gt_c, query=query, generation_gt=gen_gt)
		for ret_gt_c, query, gen_gt in zip(
			retrieval_gt_contents, qa_df["query"].tolist(), qa_df["generation_gt"].tolist()
		)
	]

	save_dir = os.path.join(node_line_dir, "lexical_retrieval")
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	result, execution_time = measure_speed(
		module.run_evaluator,
		project_dir=project_dir,
		previous_result=previous_result,
		**module_param,
	)
	average_time = execution_time / len(result)

	if strategies.get("metrics"):
		result = evaluate_lexical_retrieval_node(
			result, metric_inputs, strategies.get("metrics")
		)

	filepath = os.path.join(save_dir, "0.parquet")
	result.to_parquet(filepath, index=False)
	filename = os.path.basename(filepath)

	metric_names = strategies.get("metrics") or []
	summary_df = pd.DataFrame(
		{
			"filename": [filename],
			"module_name": [module.__name__],
			"module_params": [module_param],
			"execution_time": [average_time],
			**{metric: [result[metric].mean()] for metric in metric_names},
		}
	)

	previous_result.drop(
		columns=list(RETRIEVAL_METRIC_FUNC_DICT.keys()), inplace=True, errors="ignore"
	)
	result.rename(
		columns={
			"retrieved_contents": "retrieved_contents_lexical",
			"retrieved_ids": "retrieved_ids_lexical",
			"retrieve_scores": "retrieve_scores_lexical",
		},
		inplace=True,
	)
	best_result = pd.concat([previous_result, result], axis=1)
	best_result.to_parquet(
		os.path.join(save_dir, f"best_{os.path.splitext(filename)[0]}.parquet"),
		index=False,
	)
	safe_dataframe_to_csv(summary_df, os.path.join(save_dir, "summary.csv"), index=False)
	return best_result


def evaluate_lexical_retrieval_node(
	result_df: pd.DataFrame,
	metric_inputs: List[MetricInput],
	metrics: Union[List[str], List[Dict]],
) -> pd.DataFrame:
	"""Evaluate retrieval result with the given metrics."""

	@evaluate_retrieval(
		metric_inputs=metric_inputs,
		metrics=metrics,
	)
	def evaluate_this_module(df: pd.DataFrame):
		return (
			df["retrieved_contents_lexical"].tolist(),
			df["retrieved_ids_lexical"].tolist(),
			df["retrieve_scores_lexical"].tolist(),
		)

	return evaluate_this_module(result_df)
