from typing import List, Optional

import logging

import pandas as pd

from rag_stack.static_rag_evaluator.nodes.passagecompressor.base import BasePassageCompressor
from rag_stack.static_rag_evaluator.utils.data_parallel import run_data_parallel
from rag_stack.static_rag_evaluator.utils.util import pop_params, result_to_dataframe

logger = logging.getLogger("RAG-Stack")

# Official LLMLingua-2 token-classifier checkpoint (multilingual BERT-base
# distilled from GPT-4 compression on MeetingBank). Unlike LongLLMLingua's ~7B
# causal LM, this is a ~0.1-0.5B encoder: a single bidirectional forward over a
# batch of chunks. It fits on one GPU, so — unlike LongLLMLingua — there is no
# model sharding; the measured multi-GPU mode is DATA PARALLELISM (one replica
# per device).
_DEFAULT_MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
_DEFAULT_RATE = 0.5  # fraction of tokens to KEEP (reduce_rate = 1 - rate)
# Fixed per-forward CHUNK cap (the library's own default). NOT a search dim and
# NOT batch-derived: it bounds one BERT forward, so a request batch's chunks
# tile into ceil(pooled / cap) forwards. Therefore it NEVER OOMs from a large
# batch_size_request (more forwards, not bigger ones); BERT-base chunks are tiny
# (~10 MB at seq_len 512), so this is safe even when collocated. Override per
# machine with a SCALAR system.system_design_space.compressor_max_batch_size.
_DEFAULT_MAX_BATCH_SIZE = 50


class LLMLingua2(BasePassageCompressor):
	"""LLMLingua-2 prompt/passage compressor (token classification).

	The fast, batchable sibling of :class:`LongLLMLingua`. Both run through
	llmlingua's ``PromptCompressor``, but this one sets ``use_llmlingua2=True``
	so ``compress_prompt`` dispatches to the token-classification path — a
	single BERT forward that keeps/drops tokens, no perplexity, no causal LM.
	Token-pruning = prefill-only, so it shares LongLLMLingua's
	``passage_compressor_token_pruning_sim`` cost model.

	**Cross-query batching.** The global request-batch knob ``batch_size_request``
	(a search-space dimension; = number of QUERIES per batch) is injected as
	``query_batch_size``. Because LLMLingua-2's token scoring is
	query-independent (task-agnostic — no question is fed in), ``query_batch_size``
	queries are compressed in fused calls: the service batch is first split
	contiguously across data-parallel replicas, then each replica fuses up to
	``query_batch_size`` queries. Their passages are pooled and the library
	batches the pooled chunks through the BERT (``max_batch_size``), then results
	are regrouped 1:1 back to each query. This fills every available replica as
	production batched serving does. The fused path is rate-based
	(``use_context_level_filter=False`` → each passage compressed independently),
	which is what makes the 1:1 regroup exact; an absolute per-query
	``target_token`` budget is incompatible with sharing a forward and is not
	supported here.

	**Multi-GPU.** One replica per device in the engine's device list; the
	query-batches are split contiguously across replicas and run concurrently
	(see :func:`run_data_parallel`). Measured analog of the cost model's
	compressor single-chip + RAGO ``scale_up`` → qps×N.
	"""

	def __init__(
		self, project_dir: str, model_name: str = _DEFAULT_MODEL, **kwargs
	):
		try:
			from llmlingua import PromptCompressor
		except ImportError:
			raise ImportError(
				"LLMLingua-2 is not installed. Please install it by running `pip install llmlingua`."
			)

		super().__init__(project_dir)
		# Cache-aware load: on the perf path (a ModelCache is current) the BERT
		# replicas are loaded ONCE + reused across trials; on the quality path
		# (cache is None) the fresh per-call load is preserved.
		from rag_stack.static_rag_evaluator.measured.cache import get_current
		cache = get_current()
		kwargs.pop("cache", None)
		device_override = kwargs.pop("device", None)
		devices = kwargs.pop("devices", None)  # measured DP-replica device list
		# llmlingua2-specific construction knob (max_batch_size / max_force_token)
		llmlingua2_config = dict(kwargs.pop("llmlingua2_config", {}) or {})
		base_init_params = pop_params(PromptCompressor.__init__, kwargs)
		base_init_params["use_llmlingua2"] = True
		if llmlingua2_config:
			base_init_params["llmlingua2_config"] = llmlingua2_config

		# One replica per device (data parallelism). The factory places EACH
		# replica on the device it is handed (never a fixed closure device), so
		# the cache can key replicas by (component, model_name, device).
		devs = [str(d) for d in (devices or ([device_override] if device_override else [])) if d]
		if not devs:
			devs = ["cuda:0"]

		def _build(mt, mn, dev):
			params = dict(base_init_params)
			params.setdefault("device_map", dev)
			return PromptCompressor(model_name=mn, **params)

		if cache is not None:
			self._cache_owned = True
			self._replicas = cache.get_reranker_replicas(
				component="llmlingua2",
				model_name=model_name,
				devices=devs,
				factory=_build,
			)
		else:
			self._cache_owned = False
			# A measured AuxProcessStage is a spawned child and cannot see the
			# parent ModelCache.  Honor its injected devices instead of silently
			# collapsing a measured DP layout to the first GPU.  Ordinary quality
			# callers do not inject ``devices`` and still build exactly one model.
			self._replicas = [
				_build("llmlingua2", model_name, replica_device)
				for replica_device in devs
			]
		self._replica_devices = devs
		# Back-compat handle (first replica) for any single-model callers.
		self.llm_lingua = self._replicas[0]

	def __del__(self):
		if getattr(self, "_cache_owned", False):
			try:
				super().__del__()
			except Exception:
				pass
			return
		if hasattr(self, "_replicas"):
			del self._replicas
		if hasattr(self, "llm_lingua"):
			del self.llm_lingua
		super().__del__()

	@result_to_dataframe(["retrieved_contents"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		queries, retrieved_contents = self.cast_to_run(previous_result)
		results = self._pure(queries, retrieved_contents, **kwargs)
		return list(map(lambda x: [x], results))

	def _pure(
		self,
		queries: List[str],
		contents: List[List[str]],
		rate: float = _DEFAULT_RATE,
		query_batch_size: Optional[int] = None,
		max_batch_size: Optional[int] = None,
		**kwargs,
	) -> List[str]:
		"""
		Compress retrieved passages with LLMLingua-2, batched across queries.

		Queries are split across GPU replicas first, then each replica's shard is
		grouped into batches of at most ``query_batch_size`` (the injected
		``batch_size_request``). Replica shards run concurrently; a single replica
		short-circuits to the same in-line fused calls as before.

		:param queries: The queries for retrieved passages.
		:param contents: The contents of retrieved passages.
		:param rate: Fraction of tokens to KEEP (default 0.5). Applied per
		    passage, so it composes correctly under cross-query batching.
		:param query_batch_size: Queries per fused batch (= ``batch_size_request``).
		    Defaults to all queries in one batch when unset (quality-only path,
		    where grouping does not change the rate-based output).
		:param max_batch_size: CHUNKS (passages) per BERT forward — a per-forward
		    cap (NOT a search dim). Default tracks the request batch
		    (= ``query_batch_size`` = ``batch_size_request``); falls back to
		    ``_DEFAULT_MAX_BATCH_SIZE`` only off the measured path. The request
		    batch's chunks tile into ``ceil(pooled / cap)`` forwards, so a large
		    ``query_batch_size`` only adds forwards, never enlarges one → no OOM.
		    Set per-trial on the cached replicas (weights independent — only the
		    internal DataLoader batch). Explicit override via a SCALAR
		    ``system.system_design_space.compressor_max_batch_size`` per machine.
		:param kwargs: Extra args forwarded to ``compress_prompt`` (e.g.
		    ``force_tokens``). ``target_token`` is dropped — incompatible with
		    cross-query batching (use ``rate``).
		:return: One compressed string per query, in input order.
		"""
		items = list(zip(queries, contents))
		if not items:
			return []
		# Per-forward chunk cap. Read at compress time by the library (DataLoader
		# batch in __compress); weights are independent of it, so set it on every
		# (possibly cached) replica without a rebuild. Work beyond the cap tiles
		# into more forwards → never OOMs from a large request batch. Default
		# tracks the request batch (= batch_size_request) rather than a magic
		# constant; the fixed fallback applies only off the measured path (no
		# batch info). Explicit max_batch_size always wins.
		if max_batch_size is not None:
			cap = int(max_batch_size)
		elif query_batch_size:
			cap = int(query_batch_size)
		else:
			cap = _DEFAULT_MAX_BATCH_SIZE
		for replica in self._replicas:
			try:
				replica.max_batch_size = cap
			except Exception:  # noqa: BLE001 — never fail the eval on this
				logger.warning(f"LLMLingua2: could not set max_batch_size={cap}")
		# target_token cannot be honoured per-query once queries share a forward.
		if kwargs.pop("target_token", None) is not None:
			logger.warning(
				"LLMLingua2 ignores `target_token` under cross-query batching; "
				"use `rate` (fraction of tokens to keep) instead."
			)
		bsz = int(query_batch_size) if query_batch_size else len(items)
		bsz = max(bsz, 1)
		def _run(replica, sub_items):
			out: List[str] = []
			for i in range(0, len(sub_items), bsz):
				batch = sub_items[i:i + bsz]
				out.extend(_compress_query_batch(replica, batch, rate, **kwargs))
			return out

		# Split individual queries before forming fused calls. Splitting the
		# already-formed batch list leaves a second replica idle whenever the
		# service supplies exactly one request-sized batch (the common b32/DP2
		# measured path).
		return run_data_parallel(self._replicas, items, _run)


def _compress_query_batch(
	llm_lingua, batch_items: List, rate: float = _DEFAULT_RATE, **kwargs
) -> List[str]:
	"""Compress one batch of ``(query, contents)`` items in a single fused call.

	All passages across the batch's queries are pooled into one context list;
	the library batches their chunks through the BERT and returns a per-element
	``compressed_prompt_list`` that we regroup 1:1 back to each query. Falls back
	to per-query calls if an element was internally re-chunked (rare: a passage
	longer than the model's max_seq_len), which would break 1:1 alignment.
	"""
	from llmlingua import PromptCompressor

	counts: List[int] = []
	flat: List[str] = []
	for _query, contents in batch_items:
		elems = [c for ctx in contents for c in ctx.split("\n\n")]
		counts.append(len(elems))
		flat.extend(elems)
	if not flat:
		return ["" for _ in batch_items]

	params = pop_params(PromptCompressor.compress_prompt, dict(kwargs))
	# Token-level filter only → each passage compressed independently → the
	# returned list aligns 1:1 with `flat`, so we can regroup by query.
	params["use_context_level_filter"] = False
	res = llm_lingua.compress_prompt(flat, rate=rate, **params)
	comp_list = res.get("compressed_prompt_list") or []

	if len(comp_list) != len(flat):
		return [
			_compress_single(llm_lingua, contents, rate, **kwargs)
			for _query, contents in batch_items
		]

	out: List[str] = []
	idx = 0
	for c in counts:
		seg = comp_list[idx:idx + c]
		idx += c
		out.append("\n\n".join(x for x in seg if x))
	return out


def _compress_single(
	llm_lingua, contents: List[str], rate: float = _DEFAULT_RATE, **kwargs
) -> str:
	"""Per-query compression fallback (one query, its passages)."""
	from llmlingua import PromptCompressor

	elems = [c for ctx in contents for c in ctx.split("\n\n")]
	if not elems:
		return ""
	params = pop_params(PromptCompressor.compress_prompt, dict(kwargs))
	res = llm_lingua.compress_prompt(elems, rate=rate, **params)
	return res["compressed_prompt"]
