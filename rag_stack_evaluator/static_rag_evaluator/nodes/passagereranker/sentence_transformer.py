# Portions derived from AutoRAG (https://github.com/Marker-Inc-Korea/AutoRAG), Apache-2.0.
# Modified by the RAG-Stack authors for namespace and runtime integration; see LICENSE.autorag and NOTICE.

from typing import List, Tuple

import pandas as pd

from rag_stack_evaluator.static_rag_evaluator.nodes.passagereranker.base import BasePassageReranker
from rag_stack_evaluator.static_rag_evaluator.nodes.passagereranker.dp import rerank_flatten_apply_dp
from rag_stack_evaluator.static_rag_evaluator.utils.util import (
	make_batch,
	select_top_k,
	sort_by_scores,
	pop_params,
	result_to_dataframe,
)


class SentenceTransformerReranker(BasePassageReranker):
	def __init__(
		self,
		project_dir: str,
		model_name: str = "cross-encoder/ms-marco-MiniLM-L-2-v2",
		*args,
		**kwargs,
	):
		"""
		Initialize the Sentence Transformer reranker node.

		When a `cache: ModelCache` kwarg is supplied, the underlying CrossEncoder
		is pulled from the cache (no per-call reload). Otherwise the legacy
		fresh-per-call behavior is preserved.
		"""
		super().__init__(project_dir, *args, **kwargs)
		try:
			import torch
			from sentence_transformers import CrossEncoder
		except ImportError:
			raise ImportError(
				"You have to install AutoRAG[gpu] to use SentenceTransformerReranker"
			)
		from rag_stack_evaluator.static_rag_evaluator.measured.cache import get_current
		cache = get_current()
		kwargs.pop("cache", None)  # strip stale entries if any
		device_override = kwargs.pop("device", None)
		devices = kwargs.pop("devices", None)  # measured DP-replica device list
		default_device = device_override or ("cuda" if torch.cuda.is_available() else "cpu")
		model_params = pop_params(CrossEncoder.__init__, kwargs)
		# Data-parallel replicas (one CrossEncoder per device); single device
		# short-circuits to one model (identical to the previous behaviour).
		self._load_replicas(
			cache,
			component="sentence_transformer_reranker",
			model_name=model_name,
			device=default_device,
			devices=devices,
			factory=lambda mt, mn, dev: CrossEncoder(mn, device=dev, **model_params),
		)

	def __del__(self):
		if getattr(self, "_cache_owned", False):
			# Cache owns the model lifetime; don't free here.
			try:
				super().__del__()
			except Exception:
				pass
			return
		if hasattr(self, "model"):
			del self.model
		super().__del__()

	@result_to_dataframe(["retrieved_contents", "retrieved_ids", "retrieve_scores"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		"""
		Rerank a list of contents based on their relevance to a query using a Sentence Transformer model.

		:param previous_result: The previous result
		:param top_k: The number of passages to be retrieved
		:param batch: The number of queries to be processed in a batch
		:return: pd DataFrame containing the reranked contents, ids, and scores
		"""
		queries, contents_list, scores_list, ids_list = self.cast_to_run(
			previous_result
		)
		top_k = kwargs.get("top_k", 1)
		batch = kwargs.get("batch", 64)
		return self._pure(queries, contents_list, ids_list, top_k, batch)

	def _pure(
		self,
		queries: List[str],
		contents_list: List[List[str]],
		ids_list: List[List[str]],
		top_k: int,
		batch: int = 64,
	) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
		"""
		Rerank a list of contents based on their relevance to a query using a Sentence Transformer model.

		:param queries: The list of queries to use for reranking
		:param contents_list: The list of lists of contents to rerank
		:param ids_list: The list of lists of ids retrieved from the initial ranking
		:param top_k: The number of passages to be retrieved
		:param batch: The number of queries to be processed in a batch

		:return: tuple of lists containing the reranked contents, ids, and scores
		"""
		nested_list = [
			list(map(lambda x: [query, x], content_list))
			for query, content_list in zip(queries, contents_list)
		]
		rerank_scores = rerank_flatten_apply_dp(
			sentence_transformer_run_model,
			nested_list,
			self._replicas,
			batch_size=batch,
		)

		df = pd.DataFrame(
			{
				"contents": contents_list,
				"ids": ids_list,
				"scores": rerank_scores,
			}
		)
		df[["contents", "ids", "scores"]] = df.apply(
			sort_by_scores, axis=1, result_type="expand"
		)
		results = select_top_k(df, ["contents", "ids", "scores"], top_k)

		return (
			results["contents"].tolist(),
			results["ids"].tolist(),
			results["scores"].tolist(),
		)


def sentence_transformer_run_model(input_texts, model, batch_size: int):
	try:
		import torch
	except ImportError:
		raise ImportError(
			"You have to install AutoRAG[gpu] to use SentenceTransformerReranker"
		)
	batch_input_texts = make_batch(input_texts, batch_size)
	results = []
	for batch_texts in batch_input_texts:
		with torch.no_grad():
			pred_scores = model.predict(batch_texts)
		results.extend(pred_scores.tolist())
	return results
