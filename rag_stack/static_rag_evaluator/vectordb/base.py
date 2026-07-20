# Portions derived from AutoRAG (https://github.com/Marker-Inc-Korea/AutoRAG), Apache-2.0.
# Modified by the RAG-Stack authors for namespace and runtime integration; see LICENSE.autorag and NOTICE.

from abc import abstractmethod
from typing import List, Optional, Tuple, Union

from llama_index.embeddings.openai import OpenAIEmbedding

from rag_stack.static_rag_evaluator.utils.util import openai_truncate_by_token
from rag_stack.static_rag_evaluator.embedding.base import EmbeddingModel


# ── Measured retrieval timing split ─────────────────────────────────────────
# The measured path reports retrieval at the same granularity as the cost
# model: encode (GPU query-embedding forward), search (CPU FAISS), and the
# real document-content fetch used after search. The store's query() and the
# semantic-retrieval node's content fetch run deep inside BaseModule.run_evaluator
# (which hides the instance from the node runner), so per-call segments are
# accumulated here at module level: nodes/semanticretrieval/run.py resets before
# the timed run and reads after. The retrieval module runs chunks SEQUENTIALLY (true batching:
# one embed forward + one faiss search of nq=B per chunk, no asyncio
# interleaving), so the accumulated segments are clean walls. One pipeline
# runs at a time, so there is no cross-run interleaving.
RETRIEVAL_TIMINGS = {
	"encode_s": 0.0,
	"search_s": 0.0,
	"fetch_s": 0.0,
	# Complete production-call boundaries for standalone stage calibration.
	# The legacy keys above retain their component-only meaning.
	"encode_active_s": 0.0,
	"vectorsearch_active_s": 0.0,
	"retrieval_wrapper_active_s": 0.0,
}


def reset_retrieval_timings() -> None:
	RETRIEVAL_TIMINGS["encode_s"] = 0.0
	RETRIEVAL_TIMINGS["search_s"] = 0.0
	RETRIEVAL_TIMINGS["fetch_s"] = 0.0
	RETRIEVAL_TIMINGS["encode_active_s"] = 0.0
	RETRIEVAL_TIMINGS["vectorsearch_active_s"] = 0.0
	RETRIEVAL_TIMINGS["retrieval_wrapper_active_s"] = 0.0


def get_retrieval_timings() -> dict:
	return dict(RETRIEVAL_TIMINGS)


def add_retrieval_timing(
	encode_s: float = 0.0,
	search_s: float = 0.0,
	fetch_s: float = 0.0,
	encode_active_s: float = 0.0,
	vectorsearch_active_s: float = 0.0,
	retrieval_wrapper_active_s: float = 0.0,
) -> None:
	RETRIEVAL_TIMINGS["encode_s"] += encode_s
	RETRIEVAL_TIMINGS["search_s"] += search_s
	RETRIEVAL_TIMINGS["fetch_s"] += fetch_s
	RETRIEVAL_TIMINGS["encode_active_s"] += encode_active_s
	RETRIEVAL_TIMINGS["vectorsearch_active_s"] += vectorsearch_active_s
	RETRIEVAL_TIMINGS["retrieval_wrapper_active_s"] += retrieval_wrapper_active_s


class BaseVectorStore:
	support_similarity_metrics = ["l2", "ip", "cosine"]

	def __init__(
		self,
		embedding_model: Union[str, List[dict]],
		similarity_metric: str = "cosine",
		embedding_batch: int = 100,
		embedding_dim: Optional[int] = None,
	):
		import copy

		# Preserve the resolved factory description before legacy dict loaders
		# mutate it.  Measured mode uses this immutable copy to construct one
		# independent query-encoder replica per declared GPU.
		self._embedding_model_config = copy.deepcopy(embedding_model)
		self.embedding = EmbeddingModel.load(embedding_model, embedding_dim=embedding_dim)()
		self.embedding_batch = embedding_batch
		self.embedding.embed_batch_size = embedding_batch
		assert similarity_metric in self.support_similarity_metrics, (
			f"search method {similarity_metric} is not supported"
		)
		self.similarity_metric = similarity_metric
		self.embedding_dim = embedding_dim

	@property
	def embedding_replica_count(self) -> int:
		from rag_stack.static_rag_evaluator.embedding.data_parallel import (
			DataParallelEmbedding,
		)

		if isinstance(self.embedding, DataParallelEmbedding):
			return self.embedding.replica_count
		return 1

	def configure_embedding_data_parallel(self, devices: List[str]) -> int:
		"""Replicate only the query encoder, retaining this one VectorDB server.

		The first embedding is the exact object built by the legacy single-device
		path.  Additional local-HF replicas are created on the remaining devices
		and wrapped by an ordered thread-parallel facade.  Vector-store ``query``
		therefore still performs one search over the combined embedding matrix.
		"""
		from rag_stack.static_rag_evaluator.embedding.base import (
			build_huggingface_embedding_replicas,
		)
		from rag_stack.static_rag_evaluator.embedding.data_parallel import (
			DataParallelEmbedding,
		)

		devs = [str(device) for device in devices if str(device)]
		if len(devs) <= 1:
			return 1
		if isinstance(self.embedding, DataParallelEmbedding):
			if self.embedding.replica_devices != tuple(devs):
				raise RuntimeError(
					"embedding data-parallel devices changed after construction: "
					f"{self.embedding.replica_devices!r} -> {tuple(devs)!r}"
				)
			return self.embedding.replica_count

		replicas = build_huggingface_embedding_replicas(
			self._embedding_model_config,
			devs,
			first_replica=self.embedding,
			embedding_dim=self.embedding_dim,
		)
		self.embedding = DataParallelEmbedding(replicas, devs)
		self.embedding.embed_batch_size = self.embedding_batch
		return self.embedding.replica_count

	@abstractmethod
	async def add(
		self,
		ids: List[str],
		texts: List[str],
	):
		pass

	@abstractmethod
	def add_embedding(self, ids: List[str], embeddings: List[List[float]]):
		"""
		Add the embeddings to the Vector DB.
		"""
		pass

	@abstractmethod
	async def query(
		self, queries: List[str], top_k: int, **kwargs
	) -> Tuple[List[List[str]], List[List[float]]]:
		pass

	@abstractmethod
	async def fetch(self, ids: List[str]) -> List[List[float]]:
		"""
		Fetch the embeddings of the ids.
		"""
		pass

	@abstractmethod
	async def is_exist(self, ids: List[str]) -> List[bool]:
		"""
		Check if the ids exist in the Vector DB.
		"""
		pass

	@abstractmethod
	async def delete(self, ids: List[str]):
		pass

	def truncated_inputs(self, inputs: List[str]) -> List[str]:
		if isinstance(self.embedding, OpenAIEmbedding):
			openai_embedding_limit = 8000
			results = openai_truncate_by_token(
				inputs, openai_embedding_limit, self.embedding.model_name
			)
			return results
		return inputs
