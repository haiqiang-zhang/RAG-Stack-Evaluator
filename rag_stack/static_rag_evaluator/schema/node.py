import itertools
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Tuple, Any

import pandas as pd

from rag_stack.static_rag_evaluator.schema.module import Module
from rag_stack.static_rag_evaluator.support import get_support_nodes
from rag_stack.static_rag_evaluator.utils.util import make_combinations, find_key_values

logger = logging.getLogger("RAG-Stack")


# Node-config keys that are never hyperparameters (structural metadata).
# The search-space builder skips these when scanning a node config block
# for sweepable fields.
NODE_SKIP_KEYS = frozenset({"stage", "strategy", "modules", "optional"})


@dataclass
class Node:
	"""
	A single pipeline node with exactly one module.

	Greedy multi-module selection has been removed; each node carries one resolved
	module (chosen by an upstream sampler such as Ax). The YAML config may still
	write either `module: {...}` (preferred) or `modules: [single_item]` (legacy,
	auto-unwrapped).
	"""

	stage: str
	strategy: Dict
	node_params: Dict
	module: Module
	run_node: Callable = field(init=False)

	def __post_init__(self):
		self.run_node = get_support_nodes(self.stage)
		if self.run_node is None:
			raise ValueError(f"Node type {self.stage} is not supported.")

	def get_param_combinations(self) -> Tuple[List[Callable], List[Dict]]:
		"""
		Expand the single module's params into the cartesian product of any
		list-valued parameters. Returns parallel lists `(module_callables,
		module_params)` whose lengths equal the number of combinations.

		Placement keys (``device``, ``devices``) are atomic deployment hints, not
		sweepable params, so they are held OUT of the cartesian product and merged
		back into every combination — otherwise a ``devices`` LIST (injected for
		measured-mode reranker DP / compressor sharding) would be expanded into
		one combination per device, yielding >1 module and breaking the
		single-module node runners.
		"""
		input_dict = {**self.node_params, **self.module.module_param}
		atomic = {k: input_dict.pop(k) for k in ("device", "devices") if k in input_dict}
		combinations = make_combinations(input_dict)
		for combo in combinations:
			combo.update(atomic)
		callables = [self.module.module] * len(combinations)
		return callables, combinations

	@classmethod
	def from_dict(cls, node_dict: Dict) -> "Node":
		_node_dict = deepcopy(node_dict)
		stage = _node_dict.pop("stage")
		strategy = _node_dict.pop("strategy")
		if "module" in _node_dict:
			module = Module.from_dict(_node_dict.pop("module"))
		elif "modules" in _node_dict:
			modules_list = _node_dict.pop("modules")
			if len(modules_list) != 1:
				raise ValueError(
					f"node {stage!r}: greedy multi-module selection has been "
					f"removed. Expected `module:` (scalar) or `modules:` with exactly "
					f"1 item; got {len(modules_list)}."
				)
			module = Module.from_dict(modules_list[0])
		else:
			raise ValueError(
				f"node {stage!r}: missing `module:` (or legacy `modules:[m]`) key"
			)
		node_params = _node_dict
		return cls(stage, strategy, node_params, module)

	def run(self, previous_result: pd.DataFrame, node_line_dir: str) -> pd.DataFrame:
		logger.info(f"Running node {self.stage}...")
		input_callables, input_params = self.get_param_combinations()
		return self.run_node(
			modules=input_callables,
			module_params=input_params,
			previous_result=previous_result,
			node_line_dir=node_line_dir,
			strategies=self.strategy,
		)


def extract_values(node: Node, key: str) -> List[str]:
	"""Extract values for `key` from the single module's `module_param`."""
	module = node.module
	if key not in module.module_param:
		return []
	value = module.module_param[key]
	if isinstance(value, str) or isinstance(value, int):
		return [value]
	elif isinstance(value, list):
		return list(value)
	else:
		raise ValueError(f"{key} must be str, list or int, but got {type(value)}")


def extract_values_from_nodes(nodes: List[Node], key: str) -> List[str]:
	"""Extract `key` values from every node's module, deduplicated."""
	values = list(map(lambda node: extract_values(node, key), nodes))
	return list(set(list(itertools.chain.from_iterable(values))))


def extract_values_from_nodes_strategy(nodes: List[Node], key: str) -> List[Any]:
	"""Extract `key` values from every node's strategy dict, deduplicated."""
	values = []
	for node in nodes:
		value_list = find_key_values(node.strategy, key)
		if value_list:
			values.extend(value_list)
	return values


def module_type_exists(nodes: List[Node], component: str) -> bool:
	"""Return True iff any node uses a module of the given `component`."""
	return any(
		node.module.component.lower() == component.lower() for node in nodes
	)
