from typing import List, Tuple, Iterable

import pandas as pd

from rag_stack.static_rag_evaluator.nodes.passagereranker.base import BasePassageReranker
from rag_stack.static_rag_evaluator.nodes.passagereranker.dp import rerank_flatten_apply_dp
from rag_stack.static_rag_evaluator.utils.util import (
	make_batch,
	sort_by_scores,
	select_top_k,
	pop_params,
	result_to_dataframe,
)


class FlagEmbeddingReranker(BasePassageReranker):
	def __init__(
		self, project_dir, model_name: str = "BAAI/bge-reranker-large", *args, **kwargs
	):
		"""
		Initialize the FlagEmbeddingReranker module.

		When a `cache: ModelCache` kwarg is supplied, the underlying reranker
		model (FlagReranker or CrossEncoder fallback) is pulled from the
		cache (no per-call reload).
		"""
		super().__init__(project_dir)
		from rag_stack.static_rag_evaluator.measured.cache import get_current
		cache = get_current()
		kwargs.pop("cache", None)
		device_override = kwargs.pop("device", None)
		devices = kwargs.pop("devices", None)  # measured DP-replica device list
		try:
			from FlagEmbedding import FlagReranker
		except Exception:
			try:
				import torch
				from sentence_transformers import CrossEncoder
			except ImportError as exc:
				raise ImportError(
					"FlagEmbeddingReranker requires the 'FlagEmbedding' package or a "
					"compatible sentence-transformers fallback to be installed."
				) from exc
			default_device = device_override or ("cuda" if torch.cuda.is_available() else "cpu")
			model_params = pop_params(CrossEncoder.__init__, kwargs)
			self._load_replicas(
				cache,
				component="flag_embedding_reranker_fallback",
				model_name=model_name,
				device=default_device,
				devices=devices,
				factory=lambda mt, mn, dev: CrossEncoder(mn, device=dev, **model_params),
			)
		else:
			model_params = pop_params(FlagReranker.__init__, kwargs)
			model_params.pop("model_name_or_path", None)

			# Pin each replica to ONE device (FlagReranker would otherwise grab
			# all visible GPUs, conflicting with the derived per-engine
			# placement). ``devices=`` is honoured by recent FlagEmbedding; older
			# versions don't accept it, so fall back to the unpinned constructor.
			def _build_flag(mt, mn, dev):
				try:
					return FlagReranker(model_name_or_path=mn, devices=[dev], **model_params)
				except TypeError:
					return FlagReranker(model_name_or_path=mn, **model_params)

			self._load_replicas(
				cache,
				component="flag_embedding_reranker",
				model_name=model_name,
				device=device_override or "cuda:0",
				devices=devices,
				factory=_build_flag,
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
		queries, contents, _, ids = self.cast_to_run(previous_result)
		top_k = kwargs.pop("top_k")
		batch = kwargs.pop("batch", 64)
		return self._pure(queries, contents, ids, top_k, batch)

	def _pure(
		self,
		queries: List[str],
		contents_list: List[List[str]],
		ids_list: List[List[str]],
		top_k: int,
		batch: int = 64,
	) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
		"""
		Rerank a list of contents based on their relevance to a query using BAAI normal-Reranker model.

		:param queries: The list of queries to use for reranking
		:param contents_list: The list of lists of contents to rerank
		:param ids_list: The list of lists of ids retrieved from the initial ranking
		:param top_k: The number of passages to be retrieved
		:param batch: The number of queries to be processed in a batch
			Default is 64.
		:return: Tuple of lists containing the reranked contents, ids, and scores
		"""
		nested_list = [
			# Coerce non-string contents (NaN from a degenerate retrieval) to str:
			# FlagEmbedding slices pair[1] and crashes on a float.
			list(map(lambda x: [query, x if isinstance(x, str) else str(x)], content_list))
			for query, content_list in zip(queries, contents_list)
		]
		rerank_scores = rerank_flatten_apply_dp(
			flag_embedding_run_model, nested_list, self._replicas, batch_size=batch
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


def flag_embedding_run_model(input_texts, model, batch_size: int):
	try:
		import torch
	except ImportError:
		raise ImportError("FlagEmbeddingReranker requires PyTorch to be installed.")
	batch_input_texts = make_batch(input_texts, batch_size)
	results = []
	for batch_texts in batch_input_texts:
		with torch.no_grad():
			if hasattr(model, "compute_score"):
				pred_scores = model.compute_score(sentence_pairs=batch_texts)
			else:
				pred_scores = model.predict(batch_texts)
		if hasattr(pred_scores, "tolist"):
			pred_scores = pred_scores.tolist()
		if not isinstance(pred_scores, Iterable):
			results.append(pred_scores)
		else:
			results.extend(pred_scores)
	return results
