"""Measured query-embedding data parallelism.

The semantic-retrieval service owns one VectorDB/FAISS instance.  Only its
query encoder is replicated: one independent HuggingFace model per declared
GPU.  A dynamic service batch is flattened to sub-queries, split contiguously
across the replicas, encoded concurrently, and reassembled in input order.
The caller then performs one vector-search call over the combined embeddings.

This wrapper deliberately stays duck-typed instead of subclassing llama-index's
Pydantic ``BaseEmbedding``.  Vector stores need only its public embedding
methods/properties, while every other attribute is delegated to the first
replica.  With one replica callers keep the original embedding object and never
construct this wrapper, preserving the legacy single-device path exactly.
"""

from __future__ import annotations

from typing import Any, Sequence

from rag_stack.static_rag_evaluator.utils.data_parallel import run_data_parallel


class DataParallelEmbedding:
	"""Ordered, thread-parallel facade over independent embedding replicas."""

	def __init__(self, replicas: Sequence[Any], devices: Sequence[str]):
		if not replicas:
			raise ValueError("embedding data parallelism requires at least one replica")
		if len(replicas) != len(devices):
			raise ValueError(
				"embedding replica/device cardinality mismatch: "
				f"{len(replicas)} replicas for {len(devices)} devices"
			)
		self._replicas = tuple(replicas)
		self._replica_devices = tuple(str(device) for device in devices)

	@property
	def replicas(self) -> tuple[Any, ...]:
		return self._replicas

	@property
	def replica_devices(self) -> tuple[str, ...]:
		return self._replica_devices

	@property
	def replica_count(self) -> int:
		return len(self._replicas)

	@property
	def embed_batch_size(self) -> Any:
		return getattr(self._replicas[0], "embed_batch_size", None)

	@embed_batch_size.setter
	def embed_batch_size(self, value: Any) -> None:
		# ``batch_size_request`` is a per-replica forward cap.  Each model sees
		# the same cap; a replica shard larger than it is tiled internally by the
		# llama-index embedding implementation.
		for replica in self._replicas:
			setattr(replica, "embed_batch_size", value)

	def get_text_embedding_batch(self, texts, *args, **kwargs):
		return run_data_parallel(
			self._replicas,
			list(texts),
			lambda replica, shard: replica.get_text_embedding_batch(
				shard, *args, **kwargs,
			),
		)

	async def aget_text_embedding_batch(self, texts, *args, **kwargs):
		# Measured FAISS calls the synchronous method from its dedicated aux
		# process.  Keep an async-compatible facade for other vector stores while
		# retaining the exact same ordered DP implementation.
		return self.get_text_embedding_batch(texts, *args, **kwargs)

	def __getattr__(self, name: str) -> Any:
		return getattr(self._replicas[0], name)


__all__ = ["DataParallelEmbedding"]
