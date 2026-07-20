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


def run_semantic_retrieval_node(
	modules: List,
	module_params: List[Dict],
	previous_result: pd.DataFrame,
	node_line_dir: str,
	strategies: Dict,
) -> pd.DataFrame:
	"""
	Run the single semantic retrieval module supplied by the upstream sampler.

	Greedy multi-module selection has been removed. Exactly one module is expected.
	"""
	if len(modules) != 1:
		raise ValueError(
			f"semantic_retrieval expects exactly one module after sampling, "
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

	save_dir = os.path.join(node_line_dir, "semantic_retrieval")
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	# Reset the encode/search/fetch split accumulator so this stage's summary carries
	# ONLY this run's segments (the FAISS stores add per-call timings; see
	# vectordb/base.py RETRIEVAL_TIMINGS).
	from rag_stack.static_rag_evaluator.vectordb.base import (
		reset_retrieval_timings,
		get_retrieval_timings,
	)
	reset_retrieval_timings()
	result, execution_time = measure_speed(
		module.run_evaluator,
		project_dir=project_dir,
		previous_result=previous_result,
		**module_param,
	)
	average_time = execution_time / len(result)
	_timings = get_retrieval_timings()

	if strategies.get("metrics"):
		result = evaluate_semantic_retrieval_node(
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
			# Retrieval split (per-query means): encode = GPU query-embedding
			# forward, search = FAISS index search, fetch = O(1) document-content
			# lookup after search. Fetch is local content lookup, NOT inter-device
			# communication; GPU↔CPU / GPU↔GPU link movement is modeled separately
			# by cost_model.communication and is not isolated in this measured CSV.
			"encode_time": [_timings["encode_s"] / len(result)],
			"search_time": [_timings["search_s"] / len(result)],
			"fetch_time": [_timings.get("fetch_s", 0.0) / len(result)],
			**{metric: [result[metric].mean()] for metric in metric_names},
		}
	)

	previous_result.drop(
		columns=list(RETRIEVAL_METRIC_FUNC_DICT.keys()), inplace=True, errors="ignore"
	)
	result.rename(
		columns={
			"retrieved_contents": "retrieved_contents_semantic",
			"retrieved_ids": "retrieved_ids_semantic",
			"retrieve_scores": "retrieve_scores_semantic",
		},
		inplace=True,
	)
	best_result = pd.concat([previous_result, result], axis=1)
	best_result.to_parquet(
		os.path.join(save_dir, f"best_{os.path.splitext(filename)[0]}.parquet"),
		index=False,
	)
	safe_dataframe_to_csv(summary_df, os.path.join(save_dir, "summary.csv"), index=False)
	# trace: encode (query embedding, GPU) + retrieve (faiss + O(1) fetch) per query.
	if "__qid__" in best_result.columns:
		from rag_stack.static_rag_evaluator import recording as _rec
		_qids = best_result["__qid__"].tolist()
		_q = (best_result["query"].astype(str).tolist()
			  if "query" in best_result.columns else [""] * len(_qids))
		_rc_col = next((c for c in ("retrieved_contents", "retrieved_contents_semantic")
						if c in best_result.columns), None)
		_rc = best_result[_rc_col].tolist() if _rc_col else [[]] * len(_qids)
		_emb = module_param.get("embedding_model") if isinstance(module_param, dict) else None
		_vdb = module_param.get("vectordb") if isinstance(module_param, dict) else None
		_rec.record_io("semantic_retrieval_encode", _qids, _q, model_id=_emb)  # query → embedding (out=0)
		_rec.record_io("semantic_retrieval_vectorsearch", _qids, _q, out_texts=_rc, model_id=_vdb)  # query → passages
	return best_result


def evaluate_semantic_retrieval_node(
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
			df["retrieved_contents_semantic"].tolist(),
			df["retrieved_ids_semantic"].tolist(),
			df["retrieve_scores_semantic"].tolist(),
		)

	return evaluate_this_module(result_df)
