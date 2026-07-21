"""Faiss IVF vector store (unified IVF-PQ + IVF-Flat).

``index_type`` selects the FAISS index built:
  - ``"pq"``   → IndexIVFPQ  (factory ``IVF{nlist},PQ{M}x{nbits}``):
                 Inverted File Index + Product Quantization (compressed codes).
  - ``"flat"`` → IndexIVFFlat (factory ``IVF{nlist},Flat``):
                 full uncompressed vectors, exact distances at query time.

Both share the same id-mapping, ingest, query, and disk-persistence logic; only
the index factory string and the on-disk shape check differ.
"""

import os
import json
import logging
import threading
from typing import Dict, List, Tuple, Union

import faiss
import numpy as np

# Avoid OpenMP runtime conflicts with other libraries (torch, sklearn, etc.)
# that may load a different OpenMP. Without this, index_factory can segfault.
os.environ.setdefault("OMP_NUM_THREADS", "1")
_DEFAULT_FAISS_OMP_THREADS = max(1, int(os.environ.get("OMP_NUM_THREADS", "1")))
faiss.omp_set_num_threads(_DEFAULT_FAISS_OMP_THREADS)

from rag_stack_evaluator.static_rag_evaluator.vectordb.base import BaseVectorStore
from rag_stack_evaluator.static_rag_evaluator.vectordb._faiss_threads import faiss_build_threads
from rag_stack_evaluator.static_rag_evaluator.vectordb._faiss_cache import (
    FAISS_OMP_SEARCH_LOCK,
    atomic_save_faiss_pair,
    faiss_cache_build_lock,
    faiss_cache_metadata_if_ready,
    load_read_only_faiss_pair,
    remove_incomplete_pair,
    stage_faiss_read_file,
)

logger = logging.getLogger(__name__)


class FaissIVF(BaseVectorStore):
    """Faiss IVF vector store (PQ or Flat, selected by ``index_type``).

    The index is built lazily on the first add_embedding() call and uses the
    configured nlist/M/nbits EXACTLY (no auto-reduction — the optimizer's
    config must be what actually runs). Untrainable configs (e.g. nlist > N)
    fail loudly at faiss train time.
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
        index_type: str = "pq",
        nlist: int = 1024,
        M: int = 96,
        nbits: int = 8,
        nprobe: int = 10,
        read_only: bool = False,
        **kwargs,
    ):
        super().__init__(embedding_model, similarity_metric, embedding_batch, embedding_dim)
        assert index_type in ("pq", "flat"), \
            f"index_type must be 'pq' or 'flat', got {index_type!r}"
        self.collection_name = collection_name
        self.path = path
        self.N = N
        self.d = self.embedding_dim
        self.index_type = index_type
        self.nlist = nlist
        self.M = M
        self.nbits = nbits
        self.nprobe = nprobe
        self._read_only = bool(read_only)
        # Build-time OMP threads (system.retrieval.faiss_indexing_thread);
        # None → cpu_count-2. NOT an index-content param (META), so it never
        # enters the cache path.
        self._faiss_indexing_thread = kwargs.get("faiss_indexing_thread")

        # id <-> internal index mappings
        self._id_to_idx: Dict[str, int] = {}
        self._idx_to_id: Dict[int, str] = {}
        self._next_idx: int = 0

        self.index = None
        self._is_trained = False
        self._search_lock = threading.RLock()
        self._read_cache_entry = None

        # Load from disk if exists
        if self.path:
            loaded = self._load_if_exists()
            if self._read_only and not loaded:
                raise RuntimeError(
                    f"Required read-only FAISS IVF cache is unavailable: {self.path}"
                )

    def _get_metric(self) -> int:
        if self.similarity_metric == "l2":
            return faiss.METRIC_L2
        elif self.similarity_metric in ("ip", "cosine"):
            return faiss.METRIC_INNER_PRODUCT
        else:
            raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")

    def _factory_string(self) -> str:
        if self.index_type == "flat":
            return f"IVF{self.nlist},Flat"
        return f"IVF{self.nlist},PQ{self.M}x{self.nbits}"

    def _build_index(self):
        """Create the FAISS IVF index using index_factory.

        The index is built EXACTLY as configured — no auto-reduction of
        nlist/nbits on short training data. The optimizer's config must be
        what actually runs, otherwise sweep dimensions silently collapse
        (every nlist value maps to the same real index) and the optimizer
        learns a fake landscape. Configs that are untrainable for the corpus
        (e.g. nlist > N) fail loudly at faiss train time and surface as an
        invalid evaluation.
        """
        metric = self._get_metric()
        self.index = faiss.index_factory(self.d, self._factory_string(), metric)
        self.index.nprobe = self.nprobe

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

    def _train_if_needed(self, vectors: np.ndarray):
        """Build and train the index on first call."""
        if not self._is_trained:
            self._build_index()
            self.index.train(vectors)
            # Enable DirectMap so reconstruct() works on IVF indexes
            self.index.make_direct_map()
            self._is_trained = True

    def _index_path(self) -> str:
        return os.path.join(self.path, f"{self.collection_name}.ivf.faiss")

    def _meta_path(self) -> str:
        return os.path.join(self.path, f"{self.collection_name}.ivf.meta.json")

    def _save(self):
        """Persist index and metadata to disk."""
        if not self.path:
            return
        meta = {
            "id_to_idx": self._id_to_idx,
            "idx_to_id": {str(k): v for k, v in self._idx_to_id.items()},
            "next_idx": self._next_idx,
            "is_trained": self._is_trained,
            "index_type": self.index_type,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
        }
        atomic_save_faiss_pair(
            self.index, self._index_path(), self._meta_path(), meta,
        )

    def _on_disk_shape(self, index) -> tuple:
        """Shape tuple used to validate a cached index against the config.

        Flat has no PQ sub-quantizers, so its shape is (d, nlist); PQ adds
        (M, nbits).
        """
        if self.index_type == "flat":
            return (index.d, index.nlist)
        return (index.d, index.nlist, index.pq.M, index.pq.nbits)

    def _expected_shape(self) -> tuple:
        if self.index_type == "flat":
            return (self.d, self.nlist)
        return (self.d, self.nlist, self.M, self.nbits)

    def _load_if_exists(self) -> bool:
        """Load index and metadata from disk if available.

        The loaded index must match the configured shape exactly. A mismatch
        means a stale cache (e.g. built by a different index_type / nlist) —
        discard it and rebuild faithfully on next ingest.
        """
        index_file = self._index_path()
        meta_file = self._meta_path()
        if self._read_only:
            try:
                def _validate(index, _meta):
                    actual = self._on_disk_shape(index)
                    expected = self._expected_shape()
                    if actual != expected:
                        raise ValueError(
                            f"Stale FAISS index {index_file}: on-disk shape "
                            f"{actual} != configured {expected}"
                        )

                entry = load_read_only_faiss_pair(
                    index_file,
                    meta_file,
                    reader=lambda source: faiss.read_index(
                        stage_faiss_read_file(source), faiss.IO_FLAG_MMAP,
                    ),
                    validator=_validate,
                )
                if entry is None:
                    return False
                self.index = entry.index
                self._id_to_idx = entry.id_to_idx
                self._idx_to_id = entry.idx_to_id
                self._next_idx = entry.next_idx
                self._is_trained = bool(entry.metadata["is_trained"])
                self._search_lock = entry.search_lock
                self._read_cache_entry = entry
                # nprobe/parallel_mode are runtime knobs and are applied under
                # the shared entry lock immediately before every search.
                logger.info(
                    f"Loaded FAISS IVF ({self.index_type}) index from {index_file} "
                    f"with {self.index.ntotal} vectors (read-only process cache)"
                )
                return True
            except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
                raise RuntimeError(
                    f"Failed to load required read-only FAISS IVF cache: {self.path}"
                ) from exc

        meta = faiss_cache_metadata_if_ready(index_file, meta_file)
        if meta is None:
            return False
        try:
            index = faiss.read_index(index_file, 0)
            actual = self._on_disk_shape(index)
            expected = self._expected_shape()
            if actual != expected:
                message = (
                    f"Stale FAISS index {index_file}: on-disk shape "
                    f"{actual} != configured {expected}"
                )
                if self._read_only:
                    raise ValueError(message)
                logger.warning("Discarding %s", message)
                return False
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
            self._is_trained = meta["is_trained"]
            self.index.nprobe = self.nprobe
            logger.info(
                f"Loaded FAISS IVF ({self.index_type}) index from {index_file} "
                f"with {self.index.ntotal} vectors "
                f"(writable)"
            )
            return True
        except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
            if self._read_only:
                raise RuntimeError(
                    f"Failed to load required read-only FAISS IVF cache: {self.path}"
                ) from exc
            logger.warning("Ignoring incomplete FAISS IVF cache %s: %s", self.path, exc)
            return False

    async def add(self, ids: List[str], texts: List[str]):
        texts = self.truncated_inputs(texts)
        # SYNC batched embedding — the async aget_text_embedding_batch falls back
        # to per-text encode for HuggingFace embeds (see query()); corpus ingest
        # embeds thousands of chunks, so per-text is especially costly here.
        text_embeddings = self.embedding.get_text_embedding_batch(texts)
        self.add_embedding(ids, text_embeddings)

    def add_embedding(self, ids: List[str], embeddings: List[List[float]]):
        if getattr(self, "_read_only", False):
            raise RuntimeError("Cannot add embeddings to a read-only FAISS IVF index")
        vectors = self._prepare_vectors(embeddings)
        if self.index is not None and all(id_ in self._id_to_idx for id_ in ids):
            return
        with faiss_cache_build_lock(self.path):
            # Another server may have completed this content-addressed build
            # while this process was preparing embeddings or waiting for the
            # NFS lock. Re-check before doing any FAISS work.
            loaded = self._load_if_exists()
            if loaded and all(id_ in self._id_to_idx for id_ in ids):
                return
            if not loaded:
                remove_incomplete_pair(self._index_path(), self._meta_path())
                self.index = None
                self._is_trained = False
                self._id_to_idx = {}
                self._idx_to_id = {}
                self._next_idx = 0

            # Filter out already-existing ids after the lock-time reload.
            new_mask = [i for i, id_ in enumerate(ids) if id_ not in self._id_to_idx]
            if not new_mask:
                return

            new_ids = [ids[i] for i in new_mask]
            new_vectors = vectors[new_mask]

            try:
                # Assign internal indices
                start_idx = self._next_idx
                for offset, id_ in enumerate(new_ids):
                    idx = start_idx + offset
                    self._id_to_idx[id_] = idx
                    self._idx_to_id[idx] = id_

                self._next_idx = start_idx + len(new_ids)
                # Index build (kmeans train + PQ encode) on the configured thread
                # budget, then restored to 1 for calibrated search.
                with faiss_build_threads(self._faiss_indexing_thread):
                    self._train_if_needed(vectors)
                    self.index.add(new_vectors)
                self._save()
            except Exception:
                self.index = None
                self._is_trained = False
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
        from rag_stack_evaluator.static_rag_evaluator.vectordb.base import add_retrieval_timing
        _encode_active_t0 = _time.perf_counter()
        queries = self.truncated_inputs(queries)
        _t0 = _time.perf_counter()
        # SYNC batched embedding (one model.encode over the whole batch). The async
        # aget_text_embedding_batch falls back to PER-TEXT encode for HuggingFace
        # embeddings (no _aget_text_embeddings override) — ~2× slower, and the cost
        # is paid every agentic round. The evaluator is single-threaded here, so
        # blocking is fine.
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

        # Honor a caller-provided nprobe override for this query batch. nprobe
        # is a search-time knob (number of inverted-list cells to probe) set on
        # the retrieval module, so the same trained index is reused across
        # nprobe values. Clamp to nlist.
        nprobe_override = kwargs.get("nprobe")
        effective_nprobe = min(
            int(nprobe_override) if nprobe_override is not None else int(self.nprobe),
            self.nlist,
        )

        # System-level runtime knobs (system.retrieval). num_threads raises the
        # OMP thread count for THIS search only — absent means the configured
        # process default; parallel_mode picks FAISS's inter- (0) vs
        # intra-query (1) threading and only matters with num_threads > 1.
        num_threads = kwargs.get("num_threads")
        effective_num_threads = (
            max(1, int(num_threads))
            if num_threads is not None
            else _DEFAULT_FAISS_OMP_THREADS
        )
        parallel_mode = kwargs.get("parallel_mode")
        effective_parallel_mode = (
            int(parallel_mode) if parallel_mode is not None else 0
        )

        # Clamp top_k to available vectors
        actual_k = min(top_k, self.index.ntotal)

        with FAISS_OMP_SEARCH_LOCK:
            with self._search_lock:
                self.index.nprobe = effective_nprobe
                self.index.parallel_mode = effective_parallel_mode
                faiss.omp_set_num_threads(effective_num_threads)
                _t1 = _time.perf_counter()
                distances, indices = self.index.search(query_vectors, actual_k)
        add_retrieval_timing(search_s=_time.perf_counter() - _t1)

        all_ids = []
        all_scores = []
        for i in range(len(queries)):
            result_ids = []
            result_scores = []
            for j in range(actual_k):
                faiss_idx = int(indices[i][j])
                if faiss_idx == -1:
                    continue
                # Map FAISS internal index back to string id
                str_id = self._idx_to_id.get(faiss_idx)
                if str_id is None:
                    continue
                result_ids.append(str_id)

                dist = float(distances[i][j])
                if self.similarity_metric == "l2":
                    # Convert L2 distance to similarity score
                    score = 1.0 / (1.0 + dist)
                elif self.similarity_metric in ("ip", "cosine"):
                    # Inner product / cosine: higher is better
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
        results = []
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
