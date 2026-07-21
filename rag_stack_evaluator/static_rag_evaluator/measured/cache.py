"""Process-singleton ModelCache.

Replaces the fresh-per-call lifecycle of the original static evaluator that
inflated measured latency by 4-50x. The cache holds:

- vLLM subprocesses (keyed by VllmStartupKey)
- Sentence-transformer embedding models (keyed by (model_name, device))
- Reranker model objects (keyed by (component, model_name, device))
- FAISS / vector indices (keyed by (embedding_model, nlist, M, nbits))
- BM25 corpora (keyed by corpus hash)

Modules pull resources from the cache via `get_*` methods. Lifetimes are
owned by the cache; modules are stateless wrappers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from rag_stack_evaluator.static_rag_evaluator.measured.vllm_subprocess import (
	VllmStartupKey,
	VllmSubprocess,
)
from rag_stack_evaluator.static_rag_evaluator.measured.vllm_pd_pair import (
	VllmPdPair,
	VllmPdPairKey,
)

logger = logging.getLogger("RAG-Stack")


@dataclass(frozen=True)
class FaissKey:
	"""Identifies a built FAISS IVF-PQ index.

	`nprobe` and `parallel_mode` are search-time params and intentionally NOT
	part of the key — the same trained index is reused with different
	search-time settings.
	"""

	embedding_model: str
	nlist: int
	M: int
	nbits: int
	corpus_hash: str


class ModelCache:
	"""Process-singleton cache for vLLM / embedding / reranker / index / bm25."""

	def __init__(self):
		self._vllm: Dict[VllmStartupKey, VllmSubprocess] = {}
		self._inprocess_vllm: Dict[str, Any] = {}
		self._embedding: Dict[Tuple[str, str], Any] = {}
		self._reranker: Dict[Tuple[str, str, str], Any] = {}
		self._faiss: Dict[FaissKey, Any] = {}
		self._bm25: Dict[str, Any] = {}
		# Main-generator vLLM subprocess (separate from judge, which lives on
		# its own startup key + device). The Vllm node consults `get_main_vllm()`
		# to decide whether to use streaming subprocess generation or fall back
		# to the in-process `vllm.LLM` path.
		self._main_vllm_key: Optional[VllmStartupKey] = None
		# PD-disaggregated 1P1D pair (prefill + decode + proxy). Mutually
		# exclusive with `_main_vllm_key` — Vllm node consults `get_main_vllm_pd()`
		# first and falls back to `get_main_vllm()` when no pair is registered.
		self._main_vllm_pd_key: Optional[VllmPdPairKey] = None
		self._vllm_pd: Dict[VllmPdPairKey, VllmPdPair] = {}
		# Auxiliary vLLM subprocess for query_expansion. Independent of the
		# main generator so QE's own model (a separate search dim) + device
		# stay warm simultaneously — the QE node pulls `get_aux_vllm()` instead
		# of the main generator's subprocess (which serves a different model).
		self._aux_vllm_key: Optional[VllmStartupKey] = None
		# PD-disaggregated 1P1D pair for query_expansion (the aux analog of
		# `_main_vllm_pd_key`). When the partition splits QE's prefill/decode
		# into different groups, the QE engine is deployed as its own PD pair —
		# symmetric with the generator. The QE node consults `get_aux_vllm_pd()`
		# before `get_aux_vllm()`. Both aux slots share `self._vllm_pd`.
		self._aux_vllm_pd_key: Optional[VllmPdPairKey] = None
	# --- vLLM ---------------------------------------------------------

	def get_vllm(self, key: VllmStartupKey) -> VllmSubprocess:
		"""Return a vLLM subprocess with the requested startup config.

		If a subprocess for THIS exact key is already running, return it.
		If no subprocess is running, launch one. If a subprocess with a
		DIFFERENT key is running, the caller decides whether to evict it
		(by calling `evict_vllm()` first); this method does NOT auto-evict
		because GPU memory implications vary by caller.
		"""
		if key in self._vllm:
			return self._vllm[key]
		proc = VllmSubprocess(key)
		self._vllm[key] = proc
		return proc

	def evict_vllm(self, key: Optional[VllmStartupKey] = None) -> None:
		"""Shut down a specific vLLM subprocess (or all if key is None).

		``key is None`` tears down EVERYTHING — single subprocesses AND the
		PD-disaggregated pairs (main + aux) — so a full eviction can't leave a
		dangling PD pair behind ``get_main/aux_vllm_pd()``."""
		if key is None:
			for proc in list(self._vllm.values()):
				proc.shutdown()
			self._vllm.clear()
			self._main_vllm_key = None
			self._aux_vllm_key = None
			for pair in list(self._vllm_pd.values()):
				pair.shutdown()
			self._vllm_pd.clear()
			self._main_vllm_pd_key = None
			self._aux_vllm_pd_key = None
		elif key in self._vllm:
			self._vllm[key].shutdown()
			del self._vllm[key]
			if self._main_vllm_key == key:
				self._main_vllm_key = None
			if self._aux_vllm_key == key:
				self._aux_vllm_key = None

	def register_main_vllm(self, key: VllmStartupKey) -> VllmSubprocess:
		"""Register a vLLM subprocess as the main generator. If a different
		main was previously registered, tear it down (frees GPU memory) and
		launch a new one. The judge subprocess (registered separately on its
		own key and device) is untouched.
		"""
		if self._main_vllm_key is not None and self._main_vllm_key != key:
			logger.info(
				f"Main vLLM startup key changed; evicting "
				f"{self._main_vllm_key.device} → relaunching on {key.device}"
			)
			self.evict_vllm(self._main_vllm_key)
		# Unified and 1P1D are mutually exclusive: a previously-registered main
		# PD pair must be torn down or `get_main_vllm_pd()` (checked first) would
		# shadow this unified engine.
		self._evict_pd("main")
		if key not in self._vllm:
			self._vllm[key] = VllmSubprocess(key)
		self._main_vllm_key = key
		return self._vllm[key]

	def get_main_vllm(self) -> Optional[VllmSubprocess]:
		"""Return the main generator's vLLM subprocess, or None if unset."""
		if self._main_vllm_key is None:
			return None
		return self._vllm.get(self._main_vllm_key)

	def register_aux_vllm(self, key: VllmStartupKey) -> VllmSubprocess:
		"""Register a vLLM subprocess as the AUXILIARY (query-expansion)
		generator. Independent of the main generator (different model/device),
		so both can stay warm simultaneously. Relaunches when the key changes
		(e.g. the QE model search dim moves between trials). The main generator
		and judge subprocesses are untouched.
		"""
		if self._aux_vllm_key is not None and self._aux_vllm_key != key:
			logger.info(
				f"Aux (query-expansion) vLLM startup key changed; evicting "
				f"{self._aux_vllm_key.device} → relaunching on {key.device}"
			)
			self.evict_vllm(self._aux_vllm_key)
		# Mirror the main slot: a stale aux PD pair must go (get_aux_vllm_pd is
		# checked before get_aux_vllm).
		self._evict_pd("aux")
		if key not in self._vllm:
			self._vllm[key] = VllmSubprocess(key)
		self._aux_vllm_key = key
		return self._vllm[key]

	def get_aux_vllm(self) -> Optional[VllmSubprocess]:
		"""Return the auxiliary (query-expansion) vLLM subprocess, or None."""
		if self._aux_vllm_key is None:
			return None
		return self._vllm.get(self._aux_vllm_key)

	def register_aux_vllm_pd(self, key: VllmPdPairKey) -> VllmPdPair:
		"""Register a PD-disaggregated 1P1D pair as the AUXILIARY (query-
		expansion) engine — the aux analog of :meth:`register_main_vllm_pd`.
		Evicts a stale aux PD pair and the unified aux subprocess (mutually
		exclusive). The main generator + judge are untouched."""
		if self._aux_vllm_pd_key is not None and self._aux_vllm_pd_key != key:
			logger.info(
				f"Aux PD pair changed; evicting "
				f"{self._aux_vllm_pd_key.prefill_device}/"
				f"{self._aux_vllm_pd_key.decode_device} → launching new"
			)
			old = self._vllm_pd.pop(self._aux_vllm_pd_key, None)
			if old is not None:
				old.shutdown()
			self._aux_vllm_pd_key = None
		if self._aux_vllm_key is not None:
			self.evict_vllm(self._aux_vllm_key)
		if key not in self._vllm_pd:
			self._vllm_pd[key] = VllmPdPair(key)
		self._aux_vllm_pd_key = key
		return self._vllm_pd[key]

	def get_aux_vllm_pd(self) -> Optional[VllmPdPair]:
		"""Return the PD pair backing query-expansion, or None if not set."""
		if self._aux_vllm_pd_key is None:
			return None
		return self._vllm_pd.get(self._aux_vllm_pd_key)

	def evict_aux(self) -> None:
		"""Tear down the auxiliary (query-expansion) engine — unified subprocess
		AND PD pair. Called by the deployment manager when a trial does NOT use
		query expansion, so a prior trial's aux vLLM doesn't leak its GPU memory
		into later trials (eviction-on-register only fires when aux IS used)."""
		if self._aux_vllm_key is not None:
			self.evict_vllm(self._aux_vllm_key)
		self._evict_pd("aux")

	def _evict_pd(self, role: str) -> None:
		"""Tear down the main/aux PD pair (if registered) and clear its slot."""
		attr = "_main_vllm_pd_key" if role == "main" else "_aux_vllm_pd_key"
		pd_key = getattr(self, attr)
		if pd_key is not None:
			pair = self._vllm_pd.pop(pd_key, None)
			if pair is not None:
				pair.shutdown()
			setattr(self, attr, None)

	def register_main_vllm_pd(self, key: VllmPdPairKey) -> VllmPdPair:
		"""Register a PD-disaggregated 1P1D pair as the main generator. If a
		different PD pair was registered, tear it down. Also evicts any
		collocated `_main_vllm_key` because the two modes are mutually
		exclusive — `get_main_vllm_pd()` is checked before `get_main_vllm()`.
		"""
		if self._main_vllm_pd_key is not None and self._main_vllm_pd_key != key:
			logger.info(
				f"Main PD pair changed; evicting "
				f"{self._main_vllm_pd_key.prefill_device}/"
				f"{self._main_vllm_pd_key.decode_device} → launching new"
			)
			old = self._vllm_pd.pop(self._main_vllm_pd_key, None)
			if old is not None:
				old.shutdown()
			self._main_vllm_pd_key = None
		if self._main_vllm_key is not None:
			# Free GPU memory used by single-vllm path before launching a PD pair
			# (they would clash on cuda devices).
			self.evict_vllm(self._main_vllm_key)
		if key not in self._vllm_pd:
			self._vllm_pd[key] = VllmPdPair(key)
		self._main_vllm_pd_key = key
		return self._vllm_pd[key]

	def get_main_vllm_pd(self) -> Optional[VllmPdPair]:
		"""Return the PD pair backing the main generator, or None if not set."""
		if self._main_vllm_pd_key is None:
			return None
		return self._vllm_pd.get(self._main_vllm_pd_key)

	# --- In-process vLLM (cached LLM instance, no subprocess) ---------

	def get_inprocess_vllm(
		self,
		model: str,
		factory: Callable[[str], Any],
	) -> Any:
		"""Return a cached in-process vLLM `LLM` instance keyed by model name.

		Use this for SMOKE/Dev where a single fixed model is sufficient and
		the subprocess HTTP layer is overkill. The factory is called only
		on first request per model; subsequent trials reuse the loaded LLM.
		"""
		if model not in self._inprocess_vllm:
			logger.info(f"Loading in-process vLLM model {model!r}")
			self._inprocess_vllm[model] = factory(model)
		return self._inprocess_vllm[model]

	def evict_inprocess_vllm(self, model: Optional[str] = None) -> None:
		"""Drop a specific in-process vLLM (or all)."""
		if model is None:
			self._inprocess_vllm.clear()
		elif model in self._inprocess_vllm:
			del self._inprocess_vllm[model]

	# --- Embedding ----------------------------------------------------

	def get_embedding(
		self,
		model_name: str,
		device: str,
		factory: Callable[[str, str], Any],
	) -> Any:
		"""Return a cached embedding model; build via `factory(model, device)`."""
		k = (model_name, device)
		if k not in self._embedding:
			logger.info(f"Loading embedding model {model_name!r} on {device}")
			self._embedding[k] = factory(model_name, device)
		return self._embedding[k]

	# --- Reranker -----------------------------------------------------

	def get_reranker(
		self,
		component: str,
		model_name: str,
		device: str,
		factory: Callable[[str, str, str], Any],
	) -> Any:
		"""Return a cached reranker; build via factory if missing."""
		k = (component, model_name, device)
		if k not in self._reranker:
			logger.info(
				f"Loading reranker {component}/{model_name!r} on {device}"
			)
			self._reranker[k] = factory(component, model_name, device)
		return self._reranker[k]

	def get_reranker_replicas(
		self,
		component: str,
		model_name: str,
		devices: List[str],
		factory: Callable[[str, str, str], Any],
	) -> List[Any]:
		"""Return one cached reranker replica per device (data-parallel).

		The reranker is a small cross-encoder with no HF tensor-parallel path,
		so multi-GPU = N independent model copies in measured runtime.  This
		runtime capability is deliberately separate from whether RAG-CM has
		operation-scoped evidence for replica composition. Each
		replica is cached by ``(component, model_name, device)`` exactly like
		:meth:`get_reranker`, so a single-device call short-circuits to one
		already-warm model. ``devices`` empty → one replica on ``cuda:0``."""
		devs = list(devices) or ["cuda:0"]
		return [self.get_reranker(component, model_name, d, factory) for d in devs]

	# --- FAISS --------------------------------------------------------

	def get_faiss_index(
		self,
		key: FaissKey,
		factory: Callable[[FaissKey], Any],
	) -> Any:
		"""Return a cached FAISS index; train+encode via factory if missing."""
		if key not in self._faiss:
			logger.info(
				f"Building FAISS index for "
				f"{key.embedding_model}/nlist={key.nlist}/M={key.M}/nbits={key.nbits}"
			)
			self._faiss[key] = factory(key)
		return self._faiss[key]

	# --- BM25 ---------------------------------------------------------

	def get_bm25(
		self,
		corpus_hash: str,
		factory: Callable[[str], Any],
	) -> Any:
		"""Return a cached BM25 index keyed by corpus hash."""
		if corpus_hash not in self._bm25:
			logger.info(f"Building BM25 index for corpus_hash={corpus_hash}")
			self._bm25[corpus_hash] = factory(corpus_hash)
		return self._bm25[corpus_hash]

	# --- HF-model eviction (small-VRAM hygiene) -------------------------
	#
	# Cached embeddings / rerankers / the llmlingua compressor are in-process
	# CUDA-resident objects that persist across trials BY DESIGN (reuse). On
	# small cards (e.g. 24 GiB 3090) that residue can starve a later trial's
	# vLLM launch (its startup free-memory check) or poison retries after an
	# OOM — making genuinely-feasible arms fail and bias the optimizer. These
	# helpers drop the refs and return the VRAM.

	@staticmethod
	def _norm_dev(device: str) -> str:
		d = str(device)
		return "cuda:0" if d == "cuda" else d

	def _release_cuda(self) -> None:
		import gc
		gc.collect()
		try:
			import torch
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
		except Exception:  # noqa: BLE001 — best-effort VRAM release
			pass

	def evict_hf_models_on_devices(self, devices) -> None:
		"""Evict cached HF models (embeddings + rerankers/compressor) living on
		any of ``devices``, freeing their VRAM. Called by the deployment
		manager before launching a vLLM on those devices so stale models from
		earlier trials can't starve the launch. FAISS/BM25 (CPU) untouched."""
		targets = {self._norm_dev(d) for d in devices}
		emb_victims = [k for k in self._embedding if self._norm_dev(k[1]) in targets]
		rer_victims = [k for k in self._reranker if self._norm_dev(k[2]) in targets]
		for k in emb_victims:
			logger.info(f"Evicting embedding {k[0]!r} from {k[1]} (vLLM needs the device)")
			del self._embedding[k]
		for k in rer_victims:
			logger.info(f"Evicting reranker {k[0]}/{k[1]!r} from {k[2]} (vLLM needs the device)")
			del self._reranker[k]
		# ALWAYS gc + empty_cache — even with no dict victims. Refs dropped by
		# an earlier evict_all may have been pinned by the then-live exception
		# traceback; by now those frames are dead and this releases their VRAM
		# back to the driver (observed: 2.96 GiB allocator-hoarded on cuda:0
		# starving a later vLLM launch). Costs ~tens of ms per trial launch.
		self._release_cuda()

	def evict_all_hf_models(self) -> None:
		"""Evict ALL cached HF models (embeddings + rerankers/compressor).
		Called after a failed measured eval so partially-loaded models from
		the failed attempt can't poison retries or subsequent trials. Runs
		gc + empty_cache even when the dicts are already empty — the deferred
		(dirty-flag) cleanup relies on exactly that to release VRAM that the
		failure-time eviction couldn't (exception traceback pinned the refs)."""
		if self._embedding or self._reranker:
			logger.info(
				f"Evicting all cached HF models after failed eval "
				f"({len(self._embedding)} embeddings, {len(self._reranker)} rerankers)"
			)
			self._embedding.clear()
			self._reranker.clear()
		self._release_cuda()

	# --- Lifecycle ----------------------------------------------------

	def shutdown(self) -> None:
		"""Tear down all subprocess resources and clear caches."""
		self.evict_vllm()
		for pair in list(self._vllm_pd.values()):
			pair.shutdown()
		self._vllm_pd.clear()
		self._main_vllm_pd_key = None
		self._aux_vllm_pd_key = None
		# Drop in-process refs so vLLM's __del__ runs (frees GPU memory).
		self._inprocess_vllm.clear()
		# Embedding / reranker / FAISS / BM25 are in-process objects that
		# free on dereference; explicit clear assists GC.
		self._embedding.clear()
		self._reranker.clear()
		self._faiss.clear()
		self._bm25.clear()

	def __enter__(self) -> "ModelCache":
		set_current(self)
		return self

	def __exit__(self, exc_type, exc, tb) -> None:
		set_current(None)
		self.shutdown()


# Module-level "current cache" — modules pull from this instead of from
# their kwargs, so the cache reference never enters `module_param` (which
# gets stringified by some downstream serialization paths).
_CURRENT_CACHE: Optional[ModelCache] = None


def set_current(cache: Optional[ModelCache]) -> None:
	"""Set the process-wide current ModelCache that modules pull from."""
	global _CURRENT_CACHE
	_CURRENT_CACHE = cache


def get_current() -> Optional[ModelCache]:
	"""Return the process-wide current ModelCache (or None if not set)."""
	return _CURRENT_CACHE
