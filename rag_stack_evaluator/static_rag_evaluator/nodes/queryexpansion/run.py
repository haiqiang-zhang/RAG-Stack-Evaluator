# Portions derived from AutoRAG (https://github.com/Marker-Inc-Korea/AutoRAG), Apache-2.0.
# Modified by the RAG-Stack authors for namespace and runtime integration; see LICENSE.autorag and NOTICE.

import logging
import os
from typing import List, Dict

import pandas as pd

from rag_stack_evaluator.static_rag_evaluator.strategy import measure_speed
from rag_stack.security import safe_dataframe_to_csv

logger = logging.getLogger("RAG-Stack")


def run_query_expansion_node(
	modules: List,
	module_params: List[Dict],
	previous_result: pd.DataFrame,
	node_line_dir: str,
	strategies: Dict,
) -> pd.DataFrame:
	"""
	Run the single query expansion module supplied by the upstream sampler.

	Greedy multi-module selection has been removed. Exactly one module is expected.
	"""
	if len(modules) != 1:
		raise ValueError(
			f"query_expansion expects exactly one module after sampling, "
			f"got {len(modules)}"
		)
	module = modules[0]
	module_param = module_params[0]

	if not os.path.exists(node_line_dir):
		os.makedirs(node_line_dir)
	node_dir = os.path.join(node_line_dir, "query_expansion")
	if not os.path.exists(node_dir):
		os.makedirs(node_dir)
	project_dir = os.environ["PROJECT_DIR"]

	result, execution_time = measure_speed(
		module.run_evaluator,
		project_dir=project_dir,
		previous_result=previous_result,
		**module_param,
	)
	average_time = execution_time / len(result)

	filepath = os.path.join(node_dir, "0.parquet")
	result.to_parquet(filepath, index=False)
	filename = os.path.basename(filepath)

	summary_df = pd.DataFrame(
		{
			"filename": [filename],
			"module_name": [module.__name__],
			"module_params": [module_param],
			"execution_time": [average_time],
		}
	)

	overlap_cols = result.columns.intersection(previous_result.columns)
	if len(overlap_cols) > 0:
		result = result.drop(columns=overlap_cols)
	best_result = pd.concat([previous_result, result], axis=1)

	safe_dataframe_to_csv(summary_df, os.path.join(node_dir, "summary.csv"), index=False)
	best_result.to_parquet(
		os.path.join(node_dir, f"best_{os.path.splitext(filename)[0]}.parquet"),
		index=False,
	)
	# trace: query_expansion is an LLM call → "generate" (input=query, output=expansion).
	if "__qid__" in best_result.columns:
		from rag_stack_evaluator.static_rag_evaluator import recording as _rec
		_qids = best_result["__qid__"].tolist()
		_q = (best_result["query"].astype(str).tolist()
			  if "query" in best_result.columns else [""] * len(_qids))
		_qe = (best_result["queries"].tolist()
			   if "queries" in best_result.columns else _q)
		_m = module_param.get("model") if isinstance(module_param, dict) else None
		_rec.record_io(
			"query_expansion",
			_qids,
			_q,
			out_texts=_qe,
			model_id=_m,
		)
	return best_result
