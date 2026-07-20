from itertools import chain
from typing import List, Tuple

import pandas as pd

from rag_stack.static_rag_evaluator.nodes.passagereranker.base import BasePassageReranker
from rag_stack.static_rag_evaluator.nodes.passagereranker.dp import (
	rerank_flatten_apply_dp,
)
from rag_stack.static_rag_evaluator.nodes.passagereranker.tart.modeling_enc_t5 import (
	EncT5ForSequenceClassification,
)
from rag_stack.static_rag_evaluator.nodes.passagereranker.tart.tokenization_enc_t5 import EncT5Tokenizer
from rag_stack.static_rag_evaluator.utils.util import (
	make_batch,
	sort_by_scores,
	select_top_k,
	result_to_dataframe,
)


class Tart(BasePassageReranker):
	def __init__(self, project_dir: str, *args, **kwargs):
		super().__init__(project_dir)
		try:
			import torch
		except ImportError:
			raise ImportError(
				"torch is not installed. Please install torch first to use TART reranker."
			)
		# Cache-aware load: on the perf path (a ModelCache is current) the
		# (model, tokenizer) bundle loads ONCE + reuses across trials, on the
		# injected device. cache is None (quality path) keeps the legacy load.
		from rag_stack.static_rag_evaluator.measured.cache import get_current
		cache = get_current()
		kwargs.pop("cache", None)
		device_override = kwargs.pop("device", None)
		devices = kwargs.pop("devices", None)
		default_device = device_override or (
			"cuda" if torch.cuda.is_available() else "cpu"
		)
		model_name = "facebook/tart-full-flan-t5-xl"

		def _build(mt, mn, dev):
			return (
				EncT5ForSequenceClassification.from_pretrained(mn).to(dev),
				EncT5Tokenizer.from_pretrained(mn),
			)

		self._load_replicas(
			cache,
			component="tart",
			model_name=model_name,
			device=default_device,
			devices=devices,
			factory=_build,
		)
		self.model, self.tokenizer = self._replicas[0]

	def __del__(self):
		if getattr(self, "_cache_owned", False):
			try:
				super().__del__()
			except Exception:
				pass
			return
		if hasattr(self, "model"):
			del self.model
		if hasattr(self, "tokenizer"):
			del self.tokenizer
		super().__del__()

	@result_to_dataframe(["retrieved_contents", "retrieved_ids", "retrieve_scores"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		queries, contents, _, ids = self.cast_to_run(previous_result)
		top_k = kwargs.pop("top_k")
		instruction = kwargs.pop("instruction", "Find passage to answer given question")
		batch = kwargs.pop("batch", 64)
		return self._pure(queries, contents, ids, top_k, instruction, batch)

	def _pure(
		self,
		queries: List[str],
		contents_list: List[List[str]],
		ids_list: List[List[str]],
		top_k: int,
		instruction: str = "Find passage to answer given question",
		batch: int = 64,
	) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
		"""
		Rerank a list of contents based on their relevance to a query using Tart.
		TART is a reranker based on TART (https://github.com/facebookresearch/tart).
		You can rerank the passages with the instruction using TARTReranker.
		The default model is facebook/tart-full-flan-t5-xl.

		:param queries: The list of queries to use for reranking
		:param contents_list: The list of lists of contents to rerank
		:param ids_list: The list of lists of ids retrieved from the initial ranking
		:param top_k: The number of passages to be retrieved
		:param instruction: The instruction for reranking.
			Note: default instruction is "Find passage to answer given question"
				The default instruction from the TART paper is being used.
				If you want to use a different instruction, you can change the instruction through this parameter
		:param batch: The number of queries to be processed in a batch
		:return: tuple of lists containing the reranked contents, ids, and scores
		"""
		nested_list = [
			[
				(["{} [SEP] {}".format(instruction, query)], [content])
				for content in contents
			]
			for query, contents in zip(queries, contents_list)
		]
		replicas_with_devices = list(zip(self._replicas, self._replica_devices))

		def _run_replica(input_pairs, model, batch_size):
			(replica_model, tokenizer), replica_device = model
			return tart_run_model(
				[pair[0] for pair in input_pairs],
				contents_list=[pair[1] for pair in input_pairs],
				model=replica_model,
				batch_size=batch_size,
				tokenizer=tokenizer,
				device=replica_device,
			)

		rerank_scores = rerank_flatten_apply_dp(
			_run_replica,
			nested_list,
			replicas_with_devices,
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


def tart_run_model(
	input_texts, contents_list, model, batch_size: int, tokenizer, device
):
	try:
		import torch
		import torch.nn.functional as F
	except ImportError:
		raise ImportError(
			"torch is not installed. Please install torch first to use TART reranker."
		)
	flattened_texts = list(chain.from_iterable(input_texts))
	flattened_contents = list(chain.from_iterable(contents_list))
	batch_input_texts = make_batch(flattened_texts, batch_size)
	batch_contents_list = make_batch(flattened_contents, batch_size)
	results = []
	for batch_texts, batch_contents in zip(batch_input_texts, batch_contents_list):
		feature = tokenizer(
			batch_texts,
			batch_contents,
			padding=True,
			truncation=True,
			return_tensors="pt",
		).to(device)
		with torch.no_grad():
			pred_scores = model(**feature).logits
			# Transfer the positive-class probabilities once per forward. Iterating
			# over a CUDA tensor and coercing every row with ``float(...)`` performs
			# one scalar device-to-host transfer (and synchronization) per pair. A
			# reranker worker call can contain thousands of pairs, so that otherwise
			# turns result collection into a material part of measured service time.
			normalized_scores = (
				F.softmax(pred_scores, dim=1)[:, 1]
				.detach()
				.cpu()
				.tolist()
			)
		results.extend(normalized_scores)
	return results
