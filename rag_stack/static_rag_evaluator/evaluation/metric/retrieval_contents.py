"""Content-based retrieval metrics.

These compare retrieved chunk text against GT chunk text via token-level F1
(see ``single_token_f1``). Two design choices are baked into this module:

1. **Pre-reranker source.** ``pred = metric_input.retrieved_contents_semantic``
   — the post-semantic-retrieval text BEFORE the reranker / compressor mutate
   it. This isolates "raw retriever quality" so the metric reflects just the
   pre-reranker stages (query_expansion → vectordb → corpus → semantic_retrieval)
   and is non-redundant with ``deepeval_context_*`` which judge the post-compressor
   pipeline-final context.

2. **Chunker-robust GT.** ``gt = metric_input.retrieval_gt_contents`` — the
   text precomputed once from the original pre-chunked corpus into qa.parquet
   and stored as a column. Independent of any per-eval chunker re-chunking,
   so values stay comparable across the chunker search-space dimension.

If either field is missing (e.g. semantic retrieval skipped), the metric
returns None via the ``@autorag_metric`` decorator's ``fields_to_check``
guard, which the aggregation step filters out.
"""

import itertools
from collections import Counter

import numpy as np

from rag_stack.static_rag_evaluator.evaluation.metric.util import autorag_metric
from rag_stack.static_rag_evaluator.schema.metricinput import MetricInput
from rag_stack.static_rag_evaluator.utils.util import normalize_string


def single_token_f1(ground_truth: str, prediction: str):
	prediction_tokens = normalize_string(prediction).split()
	ground_truth_tokens = normalize_string(ground_truth).split()
	common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
	num_same = sum(common.values())
	if num_same == 0:
		return 0, 0, 0
	precision = 1.0 * num_same / len(prediction_tokens)
	recall = 1.0 * num_same / len(ground_truth_tokens)
	f1 = (2 * precision * recall) / (precision + recall)
	return precision, recall, f1


@autorag_metric(fields_to_check=["retrieved_contents_semantic", "retrieval_gt_contents"])
def retrieval_token_f1(metric_input: MetricInput):
	pred = metric_input.retrieved_contents_semantic
	gt = itertools.chain.from_iterable(metric_input.retrieval_gt_contents)

	calculated_results = list(
		map(lambda x: single_token_f1(x[1], x[0]), list(itertools.product(pred, gt)))
	)
	_, _, result = zip(*calculated_results)
	result_np = np.array(list(result)).reshape(len(pred), -1)
	return result_np.max(axis=1).mean()


@autorag_metric(fields_to_check=["retrieved_contents_semantic", "retrieval_gt_contents"])
def retrieval_token_precision(metric_input: MetricInput):
	pred = metric_input.retrieved_contents_semantic
	gt = itertools.chain.from_iterable(metric_input.retrieval_gt_contents)

	calculated_results = list(
		map(lambda x: single_token_f1(x[1], x[0]), list(itertools.product(pred, gt)))
	)
	result, _, _ = zip(*calculated_results)
	result_np = np.array(list(result)).reshape(len(pred), -1)
	return result_np.max(axis=1).mean()


@autorag_metric(fields_to_check=["retrieved_contents_semantic", "retrieval_gt_contents"])
def retrieval_token_recall(metric_input: MetricInput):
	pred = metric_input.retrieved_contents_semantic
	gt = itertools.chain.from_iterable(metric_input.retrieval_gt_contents)

	calculated_results = list(
		map(lambda x: single_token_f1(x[1], x[0]), list(itertools.product(pred, gt)))
	)
	_, result, _ = zip(*calculated_results)
	result_np = np.array(list(result)).reshape(len(pred), -1)
	return result_np.max(axis=1).mean()
