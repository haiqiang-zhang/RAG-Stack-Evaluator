# Portions derived from AutoRAG (https://github.com/Marker-Inc-Korea/AutoRAG), Apache-2.0.
# Modified by the RAG-Stack authors for namespace and runtime integration; see LICENSE.autorag and NOTICE.

import itertools
import logging
import os
import time
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

from rag_stack_evaluator.static_rag_evaluator.evaluation.metric.util import (
	calculate_l2_distance,
	calculate_inner_product,
	calculate_cosine_similarity,
)
from rag_stack_evaluator.static_rag_evaluator.nodes.retrieval.base import BaseRetrieval
from rag_stack_evaluator.static_rag_evaluator.nodes.hybridretrieval.hybrid_rrf import rrf_pure
from rag_stack.utils import cast_corpus_dataset, cast_qa_dataset
from rag_stack.utils.preprocess import validate_corpus_dataset, validate_qa_dataset
from rag_stack_evaluator.static_rag_evaluator.utils.util import (
	get_event_loop,
	openai_truncate_by_token,
	flatten_apply,
	result_to_dataframe,
	pop_params,
	convert_inputs_to_list,
	make_batch,
)
from rag_stack_evaluator.static_rag_evaluator.vectordb import load_vectordb_from_yaml
from rag_stack_evaluator.static_rag_evaluator.vectordb.base import (
	BaseVectorStore,
	add_retrieval_timing,
)
from rag_stack_evaluator.static_rag_evaluator import embedding_cache

logger = logging.getLogger("RAG-Stack")


class VectorDB(BaseRetrieval):
	def __init__(self, project_dir: str, vectordb: str = "default", **kwargs):
		"""
		Initialize VectorDB retrieval node.

		:param project_dir: The project directory path.
		:param vectordb: The vectordb name.
			You must configure the vectordb name in the config.yaml file.
			If you don't configure, it uses the default vectordb.
		:param kwargs: The optional arguments.
			Not affected in the init method.
		"""
		raw_devices = kwargs.pop("devices", None)
		if isinstance(raw_devices, str):
			raw_devices = [part.strip() for part in raw_devices.split(",") if part.strip()]
		devices = [str(device) for device in (raw_devices or []) if device]
		device = kwargs.get("device")
		if not devices and device:
			devices = [str(device)]
		super().__init__(project_dir, **kwargs)

		vectordb_config_path = os.path.join(self.resources_dir, "vectordb.yaml")
		self.vector_store = load_vectordb_from_yaml(
			vectordb_config_path, vectordb, project_dir, read_only=True
		)
		self._embedding_replica_count = (
			self.vector_store.configure_embedding_data_parallel(devices)
			if devices else 1
		)
		self.embedding_model = self.vector_store.embedding
		self._contents_by_doc_id = dict(
			zip(self.corpus_df["doc_id"].astype(str), self.corpus_df["contents"])
		)

	def __del__(self):
		if hasattr(self, "vector_store"):
			del self.vector_store
		if hasattr(self, "embedding_model"):
			del self.embedding_model
		super().__del__()

	@result_to_dataframe(
		[
			"retrieved_contents_semantic",
			"retrieved_ids_semantic",
			"retrieve_scores_semantic",
		]
	)
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		queries = self.cast_to_run(previous_result)
		pure_params = pop_params(self._pure, kwargs)
		ids, scores = self._pure(queries, **pure_params)
		contents = self._fetch_contents_o1(ids)
		return contents, ids, scores

	def _fetch_contents_o1(self, ids: List[List[str]]) -> List[List[str]]:
		"""Fetch retrieved texts via the prebuilt doc_id map and time only lookup."""
		start = time.perf_counter()
		try:
			result = []
			for id_list in ids:
				row = []
				for id_ in id_list:
					if not isinstance(id_, str) or id_ == "":
						row.append(None)
						continue
					try:
						row.append(self._contents_by_doc_id[id_])
					except KeyError as e:
						raise ValueError(f"doc_id: {id_} not found in corpus_data.") from e
				result.append(row)
			return result
		finally:
			add_retrieval_timing(fetch_s=time.perf_counter() - start)

	def _pure(
		self,
		queries: List[List[str]],
		top_k: int,
		embedding_batch: int = 128,
		ids: Optional[List[List[str]]] = None,
		nprobe: Optional[int] = None,
		ef_search: Optional[int] = None,
		num_threads: Optional[int] = None,
		parallel_mode: Optional[int] = None,
	) -> Tuple[List[List[str]], List[List[float]]]:
		"""
		VectorDB retrieval function.
		You have to get a chroma collection that is already ingested.
		You have to get an embedding model that is already used in ingesting.

		:param queries: 2-d list of query strings.
		    Each element of the list is a query strings of each row.
		:param top_k: The number of passages to be retrieved.
		:param embedding_batch: The number of queries to be processed in parallel.
		    This is used to prevent API error at the query embedding.
		    Default is 128.
		:param ids: The optional list of ids that you want to retrieve.
		    You don't need to specify this in the general use cases.
		    Default is None.
		:param nprobe: IVF-PQ search-time knob (number of cells to probe).
		    Lives in the retrieval module so it sweeps without rebuilding the
		    trained index. None → use the index's configured default.
		:param ef_search: HNSW search-time knob (query beam width). Lives in the
		    retrieval module so it sweeps without rebuilding the graph. None →
		    use the index's configured default.
		:param num_threads: System-level knob (system.retrieval) — OMP threads
		    for the FAISS search call. None → keep the process default (1).
		:param parallel_mode: System-level knob (system.retrieval) — FAISS IVF
		    threading strategy (0=inter-query, 1=intra-query). Only meaningful
		    with num_threads > 1; HNSW ignores it. None → FAISS default (0).

		:return: The 2-d list contains a list of passage ids that retrieved from vectordb and 2-d list of its scores.
		    It will be a length of queries. And each element has a length of top_k.
		"""
		# if ids are specified, fetch the ids score from Chroma
		if ids is not None:
			return self.__get_ids_scores(queries, ids, embedding_batch)

		# Standalone serving calibration needs the real wrapper glue without
		# charging the timing recorder itself.  Segment the wrapper around each
		# backend query; FAISS owns the encode/vectorsearch core intervals.
		from rag_stack_evaluator.static_rag_evaluator.vectordb.base import add_retrieval_timing
		_wrapper_segment_started = time.perf_counter()

		# Query-time search knobs. These are vectordb-specific (IVF-PQ: nprobe,
		# HNSW: ef_search) and configured on the retrieval MODULE, not the
		# vectordb block — so they're applied per query (via the store's query())
		# without entering the index-build signature, letting the same on-disk
		# index serve every value. Each store honors the knob it understands and
		# ignores the rest; None values are dropped so defaults stand.
		search_params = {
			k: v
			for k, v in (
				("nprobe", nprobe),
				("ef_search", ef_search),
				("num_threads", num_threads),
				("parallel_mode", parallel_mode),
			)
			if v is not None
		}

		# TRUE batching: rows are flattened before encode/search. On the legacy
		# single-replica path they remain grouped into `embedding_batch` chunks,
		# preserving its exact call boundaries. With multiple encoder replicas the
		# service has already aggregated B*replicas logical requests: the complete
		# flattened sub-query set is sent through the ordered DP embedding facade,
		# then searched by this ONE VectorDB/FAISS server in a single call. Each
		# replica still has embed_batch_size=B, so larger MQE shards tile into local
		# forwards without multiplying FAISS searches. Previously
		# embedding_batch was only the asyncio concurrency over per-row nq=1
		# calls, so the GPU and faiss never saw a batch (and the encode/search
		# timing split was polluted by GIL interleaving).
		self.vector_store.embedding.embed_batch_size = embedding_batch
		row_lens = [len(query_list) for query_list in queries]
		flat_queries = [q for query_list in queries for q in query_list]
		query_chunk_size = (
			max(1, len(flat_queries))
			if self._embedding_replica_count > 1
			else embedding_batch
		)

		async def run_batched():
			nonlocal _wrapper_segment_started
			ids_flat: List[List[str]] = []
			scores_flat: List[List[float]] = []
			for start in range(0, len(flat_queries), query_chunk_size):
				chunk = flat_queries[start:start + query_chunk_size]
				_wrapper_segment_done = time.perf_counter()
				add_retrieval_timing(
					retrieval_wrapper_active_s=(
						_wrapper_segment_done - _wrapper_segment_started
					)
				)
				ids_c, scores_c = await self.vector_store.query(
					queries=chunk, top_k=top_k, **search_params
				)
				_wrapper_segment_started = time.perf_counter()
				ids_flat.extend(ids_c)
				scores_flat.extend(scores_c)
			return ids_flat, scores_flat

		loop = get_event_loop()
		ids_flat, scores_flat = loop.run_until_complete(run_batched())

		# Regroup per row. Multi-query rows (MQE / HyDE / decompose) are fused
		# row-locally via Reciprocal Rank Fusion (union + dedup + 1/(rank+k)),
		# NOT even budget-splitting: RRF preserves each query's high-ranked
		# passages, dedups, and lets expansions only add incremental hits. For
		# a single-query row it degrades to a rank-sort (no-op). Same RRF as
		# hybrid retrieval (k=60, AutoRAG default).
		id_result, score_result = [], []
		pos = 0
		for n in row_lens:
			row_ids, row_scores = rrf_pure(
				tuple(ids_flat[pos:pos + n]),
				tuple(scores_flat[pos:pos + n]),
				60,
				top_k,
			)
			pos += n
			pairs = sorted(
				zip(row_scores, row_ids), key=lambda p: p[0], reverse=True
			)
			id_result.append([i for _, i in pairs])
			score_result.append([s for s, _ in pairs])
		_wrapper_segment_done = time.perf_counter()
		add_retrieval_timing(
			retrieval_wrapper_active_s=(
				_wrapper_segment_done - _wrapper_segment_started
			)
		)
		return id_result, score_result

	def __get_ids_scores(self, queries, ids, embedding_batch: int):
		# truncate queries and embedding execution here.
		openai_embedding_limit = 8000
		if isinstance(self.embedding_model, OpenAIEmbedding):
			queries = list(
				map(
					lambda query_list: openai_truncate_by_token(
						query_list,
						openai_embedding_limit,
						self.embedding_model.model_name,
					),
					queries,
				)
			)

		query_embeddings = flatten_apply(
			run_query_embedding_batch,
			queries,
			embedding_model=self.embedding_model,
			batch_size=embedding_batch,
		)

		loop = get_event_loop()

		async def run_fetch(ids):
			final_result = []
			for id_list in ids:
				if len(id_list) == 0:
					final_result.append([])
				else:
					result = await self.vector_store.fetch(id_list)
					final_result.append(result)
			return final_result

		content_embeddings = loop.run_until_complete(run_fetch(ids))

		score_result = list(
			map(
				lambda query_embedding_list, content_embedding_list: get_id_scores(
					query_embedding_list,
					content_embedding_list,
					similarity_metric=self.vector_store.similarity_metric,
				),
				query_embeddings,
				content_embeddings,
			)
		)
		return ids, score_result


async def filter_exist_ids(
	vectordb: BaseVectorStore,
	corpus_data: pd.DataFrame,
) -> pd.DataFrame:
	corpus_data = cast_corpus_dataset(corpus_data)
	validate_corpus_dataset(corpus_data)
	ids = corpus_data["doc_id"].tolist()

	# Query the collection to check if IDs already exist
	existed_bool_list = await vectordb.is_exist(ids=ids)
	# Assuming 'ids' is the key in the response
	new_passage = corpus_data[~pd.Series(existed_bool_list)]
	return new_passage



async def vectordb_ingest_api(
	vectordb: BaseVectorStore,
	corpus_data: pd.DataFrame,
):
	"""
	Ingest given corpus data to the vectordb.
	It truncates corpus content when the embedding model is OpenAIEmbedding to the 8000 tokens.
	Plus, when the corpus content is empty (whitespace), it will be ignored.
	And if there is a document id that already exists in the collection, it will be ignored.

	:param vectordb: A vector stores instance that you want to ingest.
	:param corpus_data: The corpus data that contains doc_id and contents columns.
	"""
	embedding_batch = vectordb.embedding_batch
	if not corpus_data.empty:
		new_contents = corpus_data["contents"].tolist()
		new_ids = corpus_data["doc_id"].tolist()
		content_batches = make_batch(new_contents, embedding_batch)
		id_batches = make_batch(new_ids, embedding_batch)
		for content_batch, id_batch in zip(content_batches, id_batches):
			await vectordb.add(ids=id_batch, texts=content_batch)


def vectordb_ingest_huggingface(
	vectordb: BaseVectorStore,
	corpus_data: pd.DataFrame,
	dataset_name: Optional[str] = None,
	embedding_id: Optional[str] = None,
):
	"""
	Ingest given corpus data to the vectordb using local model.
	When the corpus content is empty (whitespace), it will be ignored.
	And if there is a document id that already exists in the collection, it will be ignored.

	:param vectordb: A vector stores instance that you want to ingest.
	:param corpus_data: The corpus data that contains doc_id and contents columns.
	:param dataset_name: dataset identity, only for the readable cache key/label.
	:param embedding_id: embedding-model id (e.g. ``huggingface_all_mpnet_base_v2``),
		part of the cache key so different models never share vectors.
	"""
	embedding_batch_size = vectordb.embedding_batch
	embedding_model = vectordb.embedding._model
	if corpus_data.empty:
		logger.info("Corpus already ingested in this collection, skipping.")
		return
	new_contents = corpus_data["contents"].tolist()
	new_ids = corpus_data["doc_id"].tolist()
	logger.info("Start embedding corpus data with huggingface model.")
	# Global (rag_stack-wide) embedding cache: encode this (corpus, embedding) ONCE
	# and reuse across every project/case. The FAISS index build (below) stays
	# per-case. Cache failures fall back to a live encode — never blocks eval.
	embeddings = embedding_cache.get_or_encode(
		new_contents,
		lambda: embedding_model.encode(
			new_contents,
			batch_size=embedding_batch_size,
			normalize_embeddings=vectordb.embedding.normalize,
			show_progress_bar=True,
		),
		dataset_name=dataset_name,
		embedding_id=embedding_id,
		normalize=bool(vectordb.embedding.normalize),
	)
	vectordb.add_embedding(new_ids, embeddings)
	logger.info("Finish embedding & ingesting corpus data with huggingface model.")


def run_query_embedding_batch(
	queries: List[str], embedding_model: BaseEmbedding, batch_size: int
) -> List[List[float]]:
	result = []
	for i in range(0, len(queries), batch_size):
		batch = queries[i : i + batch_size]
		embeddings = embedding_model.get_text_embedding_batch(batch)
		result.extend(embeddings)
	return result


@convert_inputs_to_list
def get_id_scores(  # To find the uncalculated score when fuse the scores for the hybrid retrieval
	query_embeddings: List[
		List[float]
	],  # `queries` is input. This is one user input query.
	content_embeddings: List[List[float]],
	similarity_metric: str,
) -> List[
	float
]:  # The most high scores among each query. The length of a result is the same as the contents length.
	"""
	Calculate the highest similarity scores between query embeddings and content embeddings.

	:param query_embeddings: A list of lists containing query embeddings.
	:param content_embeddings: A list of lists containing content embeddings.
	:param similarity_metric: The similarity metric to use ('l2', 'ip', or 'cosine').
	:return: A list of the highest similarity scores for each content embedding.
	"""
	metric_func_dict = {
		"l2": lambda x, y: 1 - calculate_l2_distance(x, y),
		"ip": calculate_inner_product,
		"cosine": calculate_cosine_similarity,
	}
	metric_func = metric_func_dict[similarity_metric]

	result = []
	for content_embedding in content_embeddings:
		scores = []
		for query_embedding in query_embeddings:
			scores.append(
				metric_func(np.array(query_embedding), np.array(content_embedding))
			)
		result.append(max(scores))
	return result
