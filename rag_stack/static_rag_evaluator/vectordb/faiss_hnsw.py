"""Faiss HNSW vector store implementation.

Uses FAISS ``IndexHNSWFlat`` for approximate nearest-neighbor search with a
hierarchical navigable small-world graph. Mirrors the
``rag_stack.cost_model.faiss_hnsw_sim`` adapter contract: the same
``M`` / ``ef_construction`` / ``ef_search`` knobs control graph construction
and search beam width on both the runtime index and the cost model.
"""

import os
import json
import logging
import threading
from typing import Dict, List, Tuple

import faiss
import numpy as np

# Avoid OpenMP runtime conflicts with other libraries (torch, sklearn, etc.)
# that may load a different OpenMP. Without this, index_factory can segfault.
os.environ.setdefault("OMP_NUM_THREADS", "1")
_DEFAULT_FAISS_OMP_THREADS = max(1, int(os.environ.get("OMP_NUM_THREADS", "1")))
faiss.omp_set_num_threads(_DEFAULT_FAISS_OMP_THREADS)

from rag_stack.static_rag_evaluator.vectordb.base import BaseVectorStore
from rag_stack.static_rag_evaluator.vectordb._faiss_threads import faiss_build_threads
from rag_stack.static_rag_evaluator.vectordb._faiss_cache import (
    FAISS_OMP_SEARCH_LOCK,
    atomic_save_faiss_pair,
    faiss_cache_build_lock,
    faiss_cache_metadata_if_ready,
    load_read_only_faiss_pair,
    remove_incomplete_pair,
    stage_faiss_read_file,
)

logger = logging.getLogger(__name__)


class FaissHNSW(BaseVectorStore):
    """Faiss HNSW (IndexHNSWFlat) vector store.

    Stores vectors using FAISS ``IndexHNSWFlat`` and maintains a separate
    id-to-index mapping for string ID support. The graph is built lazily on
    the first ``add_embedding()`` call so we can wrap it in ``IndexIDMap2``
    for stable string ids.
    """

    def __init__(
        self,
        embedding_model,
        similarity_metric: str = "l2",
        embedding_batch: int = 100,
        collection_name: str = "default",
        path: str = "",
        N: int = 1_000_000,
        embedding_dim: int = 768,
        M: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
        read_only: bool = False,
        **kwargs,
    ):
        super().__init__(embedding_model, similarity_metric, embedding_batch, embedding_dim)
        self.collection_name = collection_name
        self.path = path
        self.N = N
        self.d = self.embedding_dim
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self._read_only = bool(read_only)
        # Build-time OMP threads (system.retrieval.faiss_indexing_thread);
        # None → cpu_count-2. META (not an index-content param).
        self._faiss_indexing_thread = kwargs.get("faiss_indexing_thread")

        # id <-> internal index mappings
        self._id_to_idx: Dict[str, int] = {}
        self._idx_to_id: Dict[int, str] = {}
        self._next_idx: int = 0

        self.index = None
        self._search_lock = threading.RLock()
        self._read_cache_entry = None

        if self.path:
            loaded = self._load_if_exists()
            if self._read_only and not loaded:
                raise RuntimeError(
                    f"Required read-only FAISS HNSW cache is unavailable: {self.path}"
                )

    def _get_metric(self) -> int:
        if self.similarity_metric == "l2":
            return faiss.METRIC_L2
        elif self.similarity_metric in ("ip", "cosine"):
            return faiss.METRIC_INNER_PRODUCT
        else:
            raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")

    def _build_index(self):
        """Create the FAISS HNSW index.

        IndexHNSWFlat stores raw vectors plus the navigable graph; no PQ
        compression. ``M`` controls the upper-layer fan-out (layer 0 uses
        ``M0 = 2*M``), ``ef_construction`` the construction beam width, and
        ``ef_search`` the query beam width.
        """
        metric = self._get_metric()
        self.index = faiss.IndexHNSWFlat(self.d, self.M, metric)
        self.index.hnsw.efConstruction = int(self.ef_construction)
        self.index.hnsw.efSearch = int(self.ef_search)

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2-normalize vectors for cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms

    def _prepare_vectors(self, embeddings: List[List[float]]) -> np.ndarray:
        """Convert embeddings to numpy and normalize if cosine metric."""
        vectors = np.array(embeddings, dtype=np.float32)
        if self.similarity_metric == "cosine":
            vectors = self._normalize(vectors)
        return vectors

    def _ensure_index(self):
        if self.index is None:
            self._build_index()

    def _index_path(self) -> str:
        return os.path.join(self.path, f"{self.collection_name}.hnsw.faiss")

    def _meta_path(self) -> str:
        return os.path.join(self.path, f"{self.collection_name}.hnsw.meta.json")

    def _save(self):
        """Persist index and metadata to disk."""
        if not self.path or self.index is None:
            return
        meta = {
            "id_to_idx": self._id_to_idx,
            "idx_to_id": {str(k): v for k, v in self._idx_to_id.items()},
            "next_idx": self._next_idx,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
        }
        atomic_save_faiss_pair(
            self.index, self._index_path(), self._meta_path(), meta,
        )

    def _load_if_exists(self) -> bool:
        """Load index and metadata from disk if available."""
        index_file = self._index_path()
        meta_file = self._meta_path()
        if self._read_only:
            try:
                def _validate(index, meta):
                    if int(meta.get("M", -1)) != int(self.M) or int(
                        meta.get("ef_construction", -1)
                    ) != int(self.ef_construction):
                        raise ValueError(
                            "FAISS HNSW build parameters do not match the cache path"
                        )

                entry = load_read_only_faiss_pair(
                    index_file,
                    meta_file,
                    reader=lambda source: faiss.read_index(
                        stage_faiss_read_file(source), 0,
                    ),
                    validator=_validate,
                )
                if entry is None:
                    return False
                self.index = entry.index
                self._id_to_idx = entry.id_to_idx
                self._idx_to_id = entry.idx_to_id
                self._next_idx = entry.next_idx
                self._search_lock = entry.search_lock
                self._read_cache_entry = entry
                # ``efSearch`` is a runtime knob; query() reapplies its effective
                # value under the shared entry lock immediately before search.
                logger.info(
                    f"Loaded FAISS HNSW index from {index_file} with "
                    f"{self.index.ntotal} vectors (read-only process cache)"
                )
                return True
            except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
                raise RuntimeError(
                    f"Failed to load required read-only FAISS HNSW cache: {self.path}"
                ) from exc

        meta = faiss_cache_metadata_if_ready(index_file, meta_file)
        if meta is None:
            return False
        try:
            # IO_FLAG_MMAP_IFC reproducibly segfaults while reading
            # IndexHNSWFlat in the available Conda FAISS 1.14.1 and 1.14.2
            # builds. The failure cannot be caught in Python, so keep the
            # proven eager reader and stage the immutable file on local NVMe.
            index = faiss.read_index(index_file, 0)
            if int(meta.get("M", -1)) != int(self.M) or int(
                meta.get("ef_construction", -1)
            ) != int(self.ef_construction):
                raise ValueError(
                    "FAISS HNSW build parameters do not match the cache path"
                )
            id_to_idx = meta["id_to_idx"]
            idx_to_id = {int(k): v for k, v in meta["idx_to_id"].items()}
            next_idx = int(meta["next_idx"])
            if not (
                len(id_to_idx) == len(idx_to_id) == next_idx == index.ntotal
            ):
                raise ValueError(
                    "FAISS cache cardinality mismatch: "
                    f"index={index.ntotal}, next_idx={next_idx}, "
                    f"id_to_idx={len(id_to_idx)}, idx_to_id={len(idx_to_id)}"
                )
            self.index = index
            self._id_to_idx = id_to_idx
            self._idx_to_id = idx_to_id
            self._next_idx = next_idx
            # ``efSearch`` is a runtime knob; reapply after load.
            self.index.hnsw.efSearch = int(self.ef_search)
            logger.info(
                f"Loaded FAISS HNSW index from {index_file} with "
                f"{self.index.ntotal} vectors (writable)"
            )
            return True
        except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
            if self._read_only:
                raise RuntimeError(
                    f"Failed to load required read-only FAISS HNSW cache: {self.path}"
                ) from exc
            logger.warning("Ignoring incomplete FAISS HNSW cache %s: %s", self.path, exc)
            return False

    async def add(self, ids: List[str], texts: List[str]):
        texts = self.truncated_inputs(texts)
        # SYNC batched embedding (see query() / faiss_ivf.add): the async path
        # falls back to per-text encode for HuggingFace embeds.
        text_embeddings = self.embedding.get_text_embedding_batch(texts)
        self.add_embedding(ids, text_embeddings)

    def add_embedding(self, ids: List[str], embeddings: List[List[float]]):
        if getattr(self, "_read_only", False):
            raise RuntimeError("Cannot add embeddings to a read-only FAISS HNSW index")
        vectors = self._prepare_vectors(embeddings)
        if self.index is not None and all(id_ in self._id_to_idx for id_ in ids):
            return
        with faiss_cache_build_lock(self.path):
            loaded = self._load_if_exists()
            if loaded and all(id_ in self._id_to_idx for id_ in ids):
                return
            if not loaded:
                remove_incomplete_pair(self._index_path(), self._meta_path())
                self.index = None
                self._id_to_idx = {}
                self._idx_to_id = {}
                self._next_idx = 0

            new_mask = [i for i, id_ in enumerate(ids) if id_ not in self._id_to_idx]
            if not new_mask:
                return
            new_ids = [ids[i] for i in new_mask]
            new_vectors = vectors[new_mask]
            try:
                start_idx = self._next_idx
                for offset, id_ in enumerate(new_ids):
                    idx = start_idx + offset
                    self._id_to_idx[id_] = idx
                    self._idx_to_id[idx] = id_
                self._next_idx = start_idx + len(new_ids)

                self._ensure_index()
                # HNSW graph insertion is serialized per global cache key and uses
                # the configured build thread budget.
                with faiss_build_threads(self._faiss_indexing_thread):
                    self.index.add(new_vectors)
                self._save()
            except Exception:
                self.index = None
                self._id_to_idx = {}
                self._idx_to_id = {}
                self._next_idx = 0
                raise

    async def query(
        self, queries: List[str], top_k: int, **kwargs
    ) -> Tuple[List[List[str]], List[List[float]]]:
        # Timing split: encode (GPU embedding forward) vs FAISS search — the
        # measured analog of the cost model's encode + retrieval stages.
        import time as _time
        from rag_stack.static_rag_evaluator.vectordb.base import add_retrieval_timing
        _encode_active_t0 = _time.perf_counter()
        queries = self.truncated_inputs(queries)
        _t0 = _time.perf_counter()
        # SYNC batched embedding — see faiss_ivf.query: the async path falls back to
        # per-text encode for HuggingFace embeddings (~2× slower), paid every round.
        query_embeddings = self.embedding.get_text_embedding_batch(queries)
        _encode_done = _time.perf_counter()
        add_retrieval_timing(
            encode_s=_encode_done - _t0,
            encode_active_s=_encode_done - _encode_active_t0,
        )
        _vectorsearch_active_t0 = _encode_done
        query_vectors = self._prepare_vectors(query_embeddings)

        if self.index is None or self.index.ntotal == 0:
            add_retrieval_timing(
                vectorsearch_active_s=(
                    _time.perf_counter() - _vectorsearch_active_t0
                )
            )
            return [[] for _ in queries], [[] for _ in queries]

        # Honor any caller-provided ef_search override for this query batch.
        # The underlying read-only index may be shared by later evaluator wrappers,
        # so assignment and search are one critical section.
        ef_search_override = kwargs.get("ef_search")
        effective_ef_search = (
            int(ef_search_override)
            if ef_search_override is not None
            else int(self.ef_search)
        )

        # System-level runtime knob (system.retrieval). Raises the OMP thread
        # count for THIS search only — absent means the configured process
        # default. parallel_mode is IVF-only; HNSW ignores it.
        num_threads = kwargs.get("num_threads")
        effective_num_threads = (
            max(1, int(num_threads))
            if num_threads is not None
            else _DEFAULT_FAISS_OMP_THREADS
        )

        actual_k = min(top_k, self.index.ntotal)
        with FAISS_OMP_SEARCH_LOCK:
            with self._search_lock:
                self.index.hnsw.efSearch = effective_ef_search
                faiss.omp_set_num_threads(effective_num_threads)
                _t1 = _time.perf_counter()
                distances, indices = self.index.search(query_vectors, actual_k)
        add_retrieval_timing(search_s=_time.perf_counter() - _t1)

        all_ids: List[List[str]] = []
        all_scores: List[List[float]] = []
        for i in range(len(queries)):
            result_ids: List[str] = []
            result_scores: List[float] = []
            for j in range(actual_k):
                faiss_idx = int(indices[i][j])
                if faiss_idx == -1:
                    continue
                str_id = self._idx_to_id.get(faiss_idx)
                if str_id is None:
                    continue
                result_ids.append(str_id)

                dist = float(distances[i][j])
                if self.similarity_metric == "l2":
                    score = 1.0 / (1.0 + dist)
                else:
                    score = dist
                result_scores.append(score)
            all_ids.append(result_ids)
            all_scores.append(result_scores)

        add_retrieval_timing(
            vectorsearch_active_s=(
                _time.perf_counter() - _vectorsearch_active_t0
            )
        )
        return all_ids, all_scores

    async def fetch(self, ids: List[str]) -> List[List[float]]:
        results: List[List[float]] = []
        for id_ in ids:
            idx = self._id_to_idx.get(id_)
            if idx is not None and self.index is not None:
                vec = self.index.reconstruct(int(idx))
                results.append(vec.tolist())
            else:
                results.append([])
        return results

    async def is_exist(self, ids: List[str]) -> List[bool]:
        return [id_ in self._id_to_idx for id_ in ids]

    async def delete(self, ids: List[str]):
        raise NotImplementedError(
            "delete is unsupported for immutable shared FAISS caches"
        )
