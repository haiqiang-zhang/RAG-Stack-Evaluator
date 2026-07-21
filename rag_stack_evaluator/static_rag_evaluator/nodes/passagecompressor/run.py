# Portions derived from AutoRAG (https://github.com/Marker-Inc-Korea/AutoRAG), Apache-2.0.
# Modified by the RAG-Stack authors for namespace and runtime integration; see LICENSE.autorag and NOTICE.

import os
from typing import List, Dict

import pandas as pd

from rag_stack_evaluator.static_rag_evaluator.evaluation.metric import (
	retrieval_token_recall,
	retrieval_token_precision,
	retrieval_token_f1,
)
from rag_stack_evaluator.static_rag_evaluator.schema.metricinput import MetricInput
from rag_stack_evaluator.static_rag_evaluator.strategy import measure_speed
from rag_stack_evaluator.static_rag_evaluator.utils.util import fetch_contents
from rag_stack.security import safe_dataframe_to_csv


def run_passage_compressor_node(
	modules: List,
	module_params: List[Dict],
	previous_result: pd.DataFrame,
	node_line_dir: str,
	strategies: Dict,
) -> pd.DataFrame:
	"""
	Run the single passage compressor module supplied by the upstream sampler.

	Greedy multi-module selection has been removed. Exactly one module is expected.
	"""
	if len(modules) != 1:
		raise ValueError(
			f"passage_compressor expects exactly one module after sampling, "
			f"got {len(modules)}"
		)
	module = modules[0]
	module_param = module_params[0]

	if not os.path.exists(node_line_dir):
		os.makedirs(node_line_dir)
	project_dir = os.environ["PROJECT_DIR"]
	data_dir = os.path.join(project_dir, "data")
	save_dir = os.path.join(node_line_dir, "passage_compressor")
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	qa_data = pd.read_parquet(os.path.join(data_dir, "qa.parquet"), engine="pyarrow")
	corpus_data = pd.read_parquet(
		os.path.join(data_dir, "corpus.parquet"), engine="pyarrow"
	)

	result, execution_time = measure_speed(
		module.run_evaluator,
		project_dir=project_dir,
		previous_result=previous_result,
		**module_param,
	)
	average_time = execution_time / len(result)

	# Retrieval-GT text comes straight from qa.parquet's retrieval_gt_contents
	# (references / answer, resolved by the dataset loader) — chunker-invariant,
	# no UUID lookup against the per-eval re-chunked corpus.
	retrieval_gt_contents = [
		x.tolist() if hasattr(x, "tolist") else list(x)
		for x in qa_data["retrieval_gt_contents"].tolist()
	]
	metric_inputs = [
		MetricInput(retrieval_gt_contents=ret_cont_gt)
		for ret_cont_gt in retrieval_gt_contents
	]

	if strategies.get("metrics"):
		result = evaluate_passage_compressor_node(
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
			**{
				f"passage_compressor_{metric}": [result[metric].mean()]
				for metric in metric_names
			},
		}
	)

	new_retrieved_contents = result["retrieved_contents"]
	previous_result["retrieved_contents"] = new_retrieved_contents
	result = result.drop(columns=["retrieved_contents"])
	best_result = pd.concat([previous_result, result], axis=1)

	best_result = best_result.rename(
		columns={
			metric_name: f"passage_compressor_{metric_name}"
			for metric_name in metric_names
		}
	)

	best_result.to_parquet(
		os.path.join(save_dir, f"best_{os.path.splitext(filename)[0]}.parquet"),
		index=False,
	)
	safe_dataframe_to_csv(summary_df, os.path.join(save_dir, "summary.csv"), index=False)
	# trace: compress (e.g. llmlingua2 per-chunk encoder) — one call/query. Tokens here are
	# diagnostic (the cost model keys on config n_chunks/chunk_tokens); record context size.
	if "__qid__" in best_result.columns:
		from rag_stack_evaluator.static_rag_evaluator import recording as _rec
		_qids = best_result["__qid__"].tolist()
		_rc_col = next((c for c in ("retrieved_contents", "retrieved_contents_semantic")
						if c in best_result.columns), None)
		_rc = best_result[_rc_col].tolist() if _rc_col else [[]] * len(_qids)
		_m = (module_param.get("model_type") or module_param.get("model")
			  if isinstance(module_param, dict) else None)
		_rec.record_io("passage_compressor", _qids, _rc, model_id=_m)
	return best_result


def evaluate_passage_compressor_node(
	result_df: pd.DataFrame, metric_inputs: List[MetricInput], metrics: List[str]
):
	metric_funcs = {
		retrieval_token_recall.__name__: retrieval_token_recall,
		retrieval_token_precision.__name__: retrieval_token_precision,
		retrieval_token_f1.__name__: retrieval_token_f1,
	}
	for metric_input, generated_text in zip(
		metric_inputs, result_df["retrieved_contents"].tolist()
	):
		metric_input.retrieved_contents = generated_text
	metrics = list(filter(lambda x: x in metric_funcs.keys(), metrics))
	if len(metrics) <= 0:
		raise ValueError(f"metrics must be one of {metric_funcs.keys()}")
	metrics_scores = dict(
		map(
			lambda metric: (
				metric,
				metric_funcs[metric](
					metric_inputs=metric_inputs,
				),
			),
			metrics,
		)
	)
	result_df = pd.concat([result_df, pd.DataFrame(metrics_scores)], axis=1)
	return result_df
