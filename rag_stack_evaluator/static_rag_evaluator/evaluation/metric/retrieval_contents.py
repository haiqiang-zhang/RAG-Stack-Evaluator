# Portions derived from AutoRAG (https://github.com/Marker-Inc-Korea/AutoRAG), Apache-2.0.
# Modified by the RAG-Stack authors for namespace and runtime integration; see LICENSE.autorag and NOTICE.

"""Token overlap against the effective retrieval output before reranking.

Dense, lexical and hybrid pipelines use the same neutral retrieval snapshot.
Legacy callers that only provide semantic contents retain their old behavior.
Ground truth remains chunker-independent text in ``retrieval_gt_contents``.
"""

import itertools
from collections import Counter

import numpy as np

from rag_stack_evaluator.static_rag_evaluator.evaluation.metric.util import autorag_metric
from rag_stack_evaluator.static_rag_evaluator.schema.metricinput import MetricInput
from rag_stack_evaluator.static_rag_evaluator.utils.util import normalize_string


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


@autorag_metric(fields_to_check=["effective_retrieved_contents", "retrieval_gt_contents"])
def retrieval_token_f1(metric_input: MetricInput):
	pred = metric_input.effective_retrieved_contents
	gt = itertools.chain.from_iterable(metric_input.retrieval_gt_contents)

	calculated_results = list(
		map(lambda x: single_token_f1(x[1], x[0]), list(itertools.product(pred, gt)))
	)
	_, _, result = zip(*calculated_results)
	result_np = np.array(list(result)).reshape(len(pred), -1)
	return result_np.max(axis=1).mean()


@autorag_metric(fields_to_check=["effective_retrieved_contents", "retrieval_gt_contents"])
def retrieval_token_precision(metric_input: MetricInput):
	pred = metric_input.effective_retrieved_contents
	gt = itertools.chain.from_iterable(metric_input.retrieval_gt_contents)

	calculated_results = list(
		map(lambda x: single_token_f1(x[1], x[0]), list(itertools.product(pred, gt)))
	)
	result, _, _ = zip(*calculated_results)
	result_np = np.array(list(result)).reshape(len(pred), -1)
	return result_np.max(axis=1).mean()


@autorag_metric(fields_to_check=["effective_retrieved_contents", "retrieval_gt_contents"])
def retrieval_token_recall(metric_input: MetricInput):
	pred = metric_input.effective_retrieved_contents
	gt = itertools.chain.from_iterable(metric_input.retrieval_gt_contents)

	calculated_results = list(
		map(lambda x: single_token_f1(x[1], x[0]), list(itertools.product(pred, gt)))
	)
	_, result, _ = zip(*calculated_results)
	result_np = np.array(list(result)).reshape(len(pred), -1)
	return result_np.max(axis=1).mean()
