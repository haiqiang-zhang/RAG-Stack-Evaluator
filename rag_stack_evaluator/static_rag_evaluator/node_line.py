# Portions derived from AutoRAG (https://github.com/Marker-Inc-Korea/AutoRAG), Apache-2.0.
# Modified by the RAG-Stack authors for namespace and runtime integration; see LICENSE.autorag and NOTICE.

import os
import pathlib
from typing import Dict, List, Optional

import pandas as pd

from rag_stack_evaluator.static_rag_evaluator.schema import Node
from rag_stack_evaluator.static_rag_evaluator.utils.util import load_summary_file


def make_node_lines(node_line_dict: Dict) -> List[Node]:
	"""
	This method makes a list of nodes from node line dictionary.
	:param node_line_dict: Node_line_dict loaded from yaml file, or get from user input.
	:return: List of Nodes inside this node line.
	"""
	nodes = node_line_dict.get("nodes")
	if nodes is None:
		raise ValueError("Node line must have 'nodes' key.")
	node_objects = list(map(lambda x: Node.from_dict(x), nodes))
	return node_objects


def run_node_line(
	nodes: List[Node],
	node_line_dir: str,
	previous_result: Optional[pd.DataFrame] = None,
):
	"""
	Run the whole node line by running each node.

	:param nodes: A list of nodes.
	:param node_line_dir: This node line's directory.
	:param previous_result: A result of the previous node line.
	    If None, it loads qa data from data/qa.parquet.
	:return: The final result of the node line.
	"""
	if previous_result is None:
		project_dir = os.environ["PROJECT_DIR"]
		qa_path = os.path.join(project_dir, "data", "qa.parquet")
		if not os.path.exists(qa_path):
			raise ValueError(f"qa.parquet does not exist in {qa_path}.")
		previous_result = pd.read_parquet(qa_path, engine="pyarrow")

	summary_lst = []
	for node in nodes:
		previous_result = node.run(previous_result, node_line_dir)
		node_summary_df = load_summary_file(
			os.path.join(node_line_dir, node.stage, "summary.csv")
		)
		# Single-module nodes write exactly one row to summary.csv after greedy removal.
		row = node_summary_df.iloc[0]
		summary_lst.append(
			{
				"stage": node.stage,
				"best_module_filename": row["filename"],
				"best_module_name": row["module_name"],
				"best_module_params": row["module_params"],
				"best_execution_time": row["execution_time"],
			}
		)

	pd.DataFrame(summary_lst).to_csv(
		os.path.join(node_line_dir, "summary.csv"), index=False
	)
	return previous_result
