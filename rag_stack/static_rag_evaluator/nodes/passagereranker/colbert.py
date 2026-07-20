from typing import List, Tuple

import numpy as np
import pandas as pd

from rag_stack.static_rag_evaluator.nodes.passagereranker.base import BasePassageReranker
from rag_stack.static_rag_evaluator.nodes.passagereranker.dp import run_data_parallel
from rag_stack.static_rag_evaluator.utils.util import (
	sort_by_scores,
	select_top_k,
	pop_params,
	result_to_dataframe,
)


class ColbertReranker(BasePassageReranker):
	def __init__(
		self,
		project_dir: str,
		model_name: str = "colbert-ir/colbertv2.0",
		*args,
		**kwargs,
	):
		"""
		Initialize a colbert rerank model for reranking.

		:param project_dir: The project directory
		:param model_name: The model name for Colbert rerank.
			You can choose a colbert model for reranking.
			The default is "colbert-ir/colbertv2.0".
		:param kwargs: Extra parameter for the model.
		"""
		super().__init__(project_dir)
		try:
			import torch
			from transformers import AutoModel, AutoTokenizer
		except ImportError:
			raise ImportError(
				"Pytorch is not installed. Please install pytorch to use Colbert reranker."
			)
		# Cache-aware load: on the perf path (a ModelCache is current) the
		# (model, tokenizer) bundle loads ONCE + reuses across trials, on the
		# injected device. cache is None (quality path) keeps the legacy load.
		from rag_stack.static_rag_evaluator.measured.cache import get_current
		cache = get_current()
		kwargs.pop("cache", None)
		device_override = kwargs.pop("device", None)
		devices = kwargs.pop("devices", None)  # measured DP-replica device list
		default_device = device_override or ("cuda" if torch.cuda.is_available() else "cpu")
		model_params = pop_params(AutoModel.from_pretrained, kwargs)

		def _build(mt, mn, dev):
			return (
				AutoModel.from_pretrained(mn, **model_params).to(dev),
				AutoTokenizer.from_pretrained(mn),
			)

		# Data-parallel replicas: each is a (model, tokenizer) bundle on its own
		# device. ``self.model``/``self.tokenizer`` alias the first (back-compat);
		# ``self._replicas`` + ``self._replica_devices`` drive the DP embedding.
		self._load_replicas(
			cache,
			component="colbert_reranker",
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
		Rerank a list of contents with Colbert rerank models.
		You can get more information about a Colbert model at https://huggingface.co/colbert-ir/colbertv2.0.
		It uses BERT-based model, so recommend using CUDA gpu for faster reranking.

		:param queries: The list of queries to use for reranking
		:param contents_list: The list of lists of contents to rerank
		:param ids_list: The list of lists of ids retrieved from the initial ranking
		:param top_k: The number of passages to be retrieved
		:param batch: The number of queries to be processed in a batch
			Default is 64.

		:return: Tuple of lists containing the reranked contents, ids, and scores
		"""

		# get query and content embeddings — data-parallel across replicas.
		# Each replica is a ((model, tokenizer), device) bundle; the embedding
		# work is split contiguously over them (single replica → one in-line
		# call, identical to the prior behaviour).
		replicas_with_dev = list(zip(self._replicas, self._replica_devices))

		def _embed(replica_dev, strings):
			(model, tokenizer), dev = replica_dev
			return get_colbert_embedding_batch(strings, model, tokenizer, batch, dev)

		query_embedding_list = run_data_parallel(
			replicas_with_dev, queries, _embed
		)
		# Flatten the nested content lists, embed the flat strings in parallel,
		# then reconstruct the nested shape (the DP analog of flatten_apply).
		_cdf = pd.DataFrame({"col1": contents_list}).explode("col1")
		_cdf["result"] = run_data_parallel(
			replicas_with_dev, _cdf["col1"].tolist(), _embed
		)
		content_embedding_list = (
			_cdf.groupby(level=0, sort=False)["result"].apply(list).tolist()
		)
		df = pd.DataFrame(
			{
				"ids": ids_list,
				"query_embedding": query_embedding_list,
				"contents": contents_list,
				"content_embedding": content_embedding_list,
			}
		)
		temp_df = df.explode("content_embedding")
		temp_df["score"] = temp_df.apply(
			lambda x: get_colbert_score(x["query_embedding"], x["content_embedding"]),
			axis=1,
		)
		df["scores"] = (
			temp_df.groupby(level=0, sort=False)["score"].apply(list).tolist()
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


def get_colbert_embedding_batch(
	input_strings: List[str], model, tokenizer, batch_size: int, device: str = "cuda"
) -> List[np.array]:
	try:
		import torch
	except ImportError:
		raise ImportError(
			"Pytorch is not installed. Please install pytorch to use Colbert reranker."
		)
	encoding = tokenizer(
		input_strings,
		return_tensors="pt",
		padding=True,
		truncation=True,
		max_length=model.config.max_position_embeddings,
	)

	input_batches = slice_tokenizer_result(encoding, batch_size, device)
	result_embedding = []
	with torch.no_grad():
		for encoding_batch in input_batches:
			# Move each micro-batch off the accelerator immediately. Keeping
			# every hidden-state tensor on GPU and torch.cat-ing the full
			# passage cohort at the end makes peak VRAM scale with the entire
			# dynamic request batch (12 GiB+ at request batch 128), even though
			# the model forward itself is micro-batched. Preserve the historical
			# per-item [1, seq, hidden] arrays while bounding accelerator memory
			# to one micro-batch.
			host_batch = (
				model(**encoding_batch)
				.last_hidden_state.detach()
				.cpu()
				.numpy()
			)
			result_embedding.extend(
				host_batch[i : i + 1] for i in range(host_batch.shape[0])
			)
	return result_embedding


def slice_tokenizer_result(tokenizer_output, batch_size, device: str = "cuda"):
	input_ids_batches = slice_tensor(tokenizer_output["input_ids"], batch_size, device)
	attention_mask_batches = slice_tensor(
		tokenizer_output["attention_mask"], batch_size, device
	)
	token_type_ids_batches = slice_tensor(
		tokenizer_output.get("token_type_ids", None), batch_size, device
	)
	return [
		{
			"input_ids": input_ids,
			"attention_mask": attention_mask,
			"token_type_ids": token_type_ids,
		}
		for input_ids, attention_mask, token_type_ids in zip(
			input_ids_batches, attention_mask_batches, token_type_ids_batches
		)
	]


def slice_tensor(input_tensor, batch_size, device: str = "cuda"):
	try:
		import torch  # noqa: F401
	except ImportError:
		raise ImportError(
			"Pytorch is not installed. Please install pytorch to use Colbert reranker."
		)
	# Calculate the number of full batches
	num_full_batches = input_tensor.size(0) // batch_size

	# Slice the tensor into batches
	tensor_list = [
		input_tensor[i * batch_size : (i + 1) * batch_size]
		for i in range(num_full_batches)
	]

	# Handle the last batch if it's smaller than batch_size
	remainder = input_tensor.size(0) % batch_size
	if remainder:
		tensor_list.append(input_tensor[-remainder:])

	# Place input tensors on the SAME device as the model (the injected
	# per-component placement), not a hardcoded "cuda" (= cuda:0).
	tensor_list = list(map(lambda x: x.to(device), tensor_list))

	return tensor_list


def get_colbert_score(query_embedding: np.array, content_embedding: np.array) -> float:
	if query_embedding.ndim == 3 and content_embedding.ndim == 3:
		query_embedding = query_embedding.reshape(-1, query_embedding.shape[-1])
		content_embedding = content_embedding.reshape(-1, content_embedding.shape[-1])

	sim_matrix = np.dot(query_embedding, content_embedding.T) / (
		np.linalg.norm(query_embedding, axis=1)[:, np.newaxis]
		* np.linalg.norm(content_embedding, axis=1)
	)
	max_sim_scores = np.max(sim_matrix, axis=1)
	return float(np.mean(max_sim_scores))
