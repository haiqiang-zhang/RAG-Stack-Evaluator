# Portions derived from AutoRAG (https://github.com/Marker-Inc-Korea/AutoRAG), Apache-2.0.
# Modified by the RAG-Stack authors for namespace and runtime integration; see LICENSE.autorag and NOTICE.

from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd

from rag_stack_evaluator.static_rag_evaluator.nodes.hybridretrieval.base import HybridRetrieval


class HybridRRF(HybridRetrieval):
	def _pure(self, info: dict, top_k: int, weight: int = 60, rrf_k: int = -1):
		return hybrid_rrf(
			(info["retrieved_ids_semantic"], info["retrieved_ids_lexical"]),
			(info["retrieve_scores_semantic"], info["retrieve_scores_lexical"]),
			top_k,
			weight,
			rrf_k,
		)

	@classmethod
	def run_evaluator(
		cls,
		project_dir: Union[str, Path],
		previous_result: pd.DataFrame,
		*args,
		**kwargs,
	):
		"""
		Run RRF fusion with a single fixed weight from kwargs.
		Greedy weight sweep has been removed; the weight is sampled by the upstream
		optimizer (Ax) and passed in directly.
		"""
		# Strip optional sweep-control kwargs if callers still pass them.
		kwargs.pop("strategy", None)
		kwargs.pop("input_metrics", None)
		kwargs.pop("weight_range", None)
		kwargs.pop("test_weight_size", None)
		instance = cls(project_dir, *args, **kwargs)
		return instance.pure(previous_result, **kwargs)


def hybrid_rrf(
	ids: Tuple,
	scores: Tuple,
	top_k: int,
	weight: int = 60,
	rrf_k: int = -1,
) -> Tuple[List[List[str]], List[List[float]]]:
	"""
	Hybrid RRF function.
	RRF (Rank Reciprocal Fusion) is a method to fuse multiple retrieval results.
	It is common to fuse dense retrieval and sparse retrieval results using RRF.
	To use this function, you must input ids and scores as tuple.
	It is more unique than other retrieval modules because it does not really execute retrieval but just fuses
	the results of other retrieval functions.
	So you have to run more than two retrieval modules before running this function.
	And collect ids and scores result from each retrieval module.
	Make it as a tuple and input it to this function.

	:param ids: The tuple of ids that you want to fuse.
	    The length of this must be the same as the length of scores.
	:param scores: The retrieve scores that you want to fuse.
	    The length of this must be the same as the length of ids.
	:param top_k: The number of passages to be retrieved.
	:param weight: Hyperparameter for RRF.
	    It was originally rrf_k value.
	    Default is 60.
	    For more information, please visit our documentation.
	:param rrf_k: (Deprecated) Hyperparameter for RRF.
	    It was originally rrf_k value. Will remove at a further version.
	:return: The tuple of ids and fused scores that are fused by RRF.
	"""
	assert len(ids) == len(scores), "The length of ids and scores must be the same."
	assert len(ids) > 1, "You must input more than one retrieval results."
	assert top_k > 0, "top_k must be greater than 0."
	assert weight > 0, "rrf_k must be greater than 0."

	if rrf_k != -1:
		weight = int(rrf_k)
	else:
		weight = int(weight)

	id_df = pd.DataFrame({f"id_{i}": id_list for i, id_list in enumerate(ids)})
	score_df = pd.DataFrame(
		{f"score_{i}": score_list for i, score_list in enumerate(scores)}
	)
	df = pd.concat([id_df, score_df], axis=1)

	def rrf_pure_apply(row):
		ids_tuple = tuple(row[[f"id_{i}" for i in range(len(ids))]].values)
		scores_tuple = tuple(row[[f"score_{i}" for i in range(len(scores))]].values)
		return pd.Series(rrf_pure(ids_tuple, scores_tuple, weight, top_k))

	df[["rrf_id", "rrf_score"]] = df.apply(rrf_pure_apply, axis=1)
	return df["rrf_id"].tolist(), df["rrf_score"].tolist()


def rrf_pure(
	ids: Tuple, scores: Tuple, rrf_k: int, top_k: int
) -> Tuple[List[str], List[float]]:
	# SINGLE-LIST FAST PATH (07-09): for one input list RRF degrades to a
	# rank-sort — semantically documented as a no-op fusion — yet the pandas
	# path still cost ~3ms/call and, called once per row per retrieval, was
	# 77% of the measured retrieval stage's service time (react serving:
	# 9.7k calls ≈ 0.72s of every 0.94s per 240-batch). Pure-python replica
	# of the exact same semantics: dict(zip(...)) dedup (last wins),
	# rank-desc with method='min' ties, score = 1/(rank + rrf_k), sort desc,
	# truncate to top_k. Multi-list inputs (real fusion: MQE/HyDE/hybrid)
	# keep the original pandas path untouched.
	if len(ids) == 1:
		m = dict(zip(ids[0], scores[0]))
		ordered = sorted(m.items(), key=lambda kv: kv[1], reverse=True)
		prev_score = None
		prev_rank = 0
		out: List[Tuple[str, float]] = []
		for i, (key, val) in enumerate(ordered, start=1):
			if val != prev_score:
				prev_rank = i
				prev_score = val
			out.append((key, 1.0 / (prev_rank + rrf_k)))
		out.sort(key=lambda kv: kv[1], reverse=True)
		out = out[:top_k]
		return [k for k, _ in out], [v for _, v in out]
	df = pd.concat(
		[pd.Series(dict(zip(_id, score))) for _id, score in zip(ids, scores)], axis=1
	)
	rank_df = df.rank(ascending=False, method="min")
	rank_df = rank_df.fillna(0)
	rank_df["rrf"] = rank_df.apply(lambda row: rrf_calculate(row, rrf_k), axis=1)
	rank_df = rank_df.sort_values(by="rrf", ascending=False)
	return rank_df.index.tolist()[:top_k], rank_df["rrf"].tolist()[:top_k]


def rrf_calculate(row, rrf_k):
	result = 0
	for r in row:
		if r == 0:
			continue
		result += 1 / (r + rrf_k)
	return result
