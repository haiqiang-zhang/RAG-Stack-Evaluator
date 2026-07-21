# Portions derived from AutoRAG (https://github.com/Marker-Inc-Korea/AutoRAG), Apache-2.0.
# Modified by the RAG-Stack authors for namespace and runtime integration; see LICENSE.autorag and NOTICE.

import os
from typing import List, Dict

import pandas as pd
import tokenlog

from rag_stack_evaluator.static_rag_evaluator.strategy import measure_speed
from rag_stack.security import safe_dataframe_to_csv


def run_prompt_maker_node(
	modules: List,
	module_params: List[Dict],
	previous_result: pd.DataFrame,
	node_line_dir: str,
	strategies: Dict,
) -> pd.DataFrame:
	"""
	Run the single prompt maker module supplied by the upstream sampler.

	Greedy multi-module selection has been removed. Exactly one module is expected.
	"""
	if len(modules) != 1:
		raise ValueError(
			f"prompt_maker expects exactly one module after sampling, "
			f"got {len(modules)}"
		)
	module = modules[0]
	module_param = module_params[0]

	if not os.path.exists(node_line_dir):
		os.makedirs(node_line_dir)
	node_dir = os.path.join(node_line_dir, "prompt_maker")
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

	token_logger = tokenlog.getLogger(
		"prompt_maker_0", strategies.get("tokenizer", "openai/gpt-oss-20b")
	)
	token_logger.query_batch(result["prompts"].tolist())
	token_usage = token_logger.get_token_usage() / len(result)

	filepath = os.path.join(node_dir, "0.parquet")
	result.to_parquet(filepath, index=False)
	filename = os.path.basename(filepath)

	summary_df = pd.DataFrame(
		{
			"filename": [filename],
			"module_name": [module.__name__],
			"module_params": [module_param],
			"execution_time": [average_time],
			"average_prompt_token": [token_usage],
		}
	)

	best_result = pd.concat([previous_result, result], axis=1)

	safe_dataframe_to_csv(summary_df, os.path.join(node_dir, "summary.csv"), index=False)
	best_result.to_parquet(
		os.path.join(node_dir, f"best_{os.path.splitext(filename)[0]}.parquet"),
		index=False,
	)
	return best_result
