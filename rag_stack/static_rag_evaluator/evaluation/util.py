# Portions derived from AutoRAG (https://github.com/Marker-Inc-Korea/AutoRAG), Apache-2.0.
# Modified by the RAG-Stack authors for namespace and runtime integration; see LICENSE.autorag and NOTICE.

from copy import deepcopy
from typing import Union, List, Dict, Tuple, Any

from rag_stack.static_rag_evaluator.embedding.base import EmbeddingModel


# Meta-keys describing how a metric is wired into the optimizer (sub-metric of
# which objective, which RAG knobs feed it). They live alongside the metric
# itself in YAML (single source of truth) but must NOT be forwarded as kwargs
# to the metric function, which would `raise TypeError: unexpected keyword`.
_METRIC_META_KEYS = frozenset({"description"})


def cast_metrics(
	metrics: Union[List[str], List[Dict]],
) -> Tuple[List[str], List[Dict[str, Any]]]:
	"""
	 Turn metrics to list of metric names and parameter list.

	:param metrics: List of string or dictionary.
	:return: The list of metric names and dictionary list of metric parameters.
	"""
	metrics_copy = deepcopy(metrics)
	if not isinstance(metrics_copy, list):
		raise ValueError("metrics must be a list of string or dictionary.")
	if not metrics_copy:
		return [], []
	if isinstance(metrics_copy[0], str):
		return metrics_copy, [{} for _ in metrics_copy]
	elif isinstance(metrics_copy[0], dict):
		# pop 'metric_name' key from dictionary
		metric_names = list(map(lambda x: x.pop("metric_name"), metrics_copy))
		metric_params = [
			dict(
				map(
					lambda x, y: cast_embedding_model(x, y),
					(k for k in metric.keys() if k not in _METRIC_META_KEYS),
					(metric[k] for k in metric.keys() if k not in _METRIC_META_KEYS),
				)
			)
			for metric in metrics_copy
		]
		return metric_names, metric_params
	else:
		raise ValueError("metrics must be a list of string or dictionary.")


def cast_embedding_model(key, value):
	if key == "embedding_model":
		return key, EmbeddingModel.load(value)()
	else:
		return key, value
