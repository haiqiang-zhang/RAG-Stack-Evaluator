import asyncio
import gc
import logging
import os
from copy import deepcopy
from typing import List, Optional, Tuple, Union

import pandas as pd

from rag_stack.static_rag_evaluator.nodes.generator.base import BaseGenerator
from rag_stack.static_rag_evaluator.utils import result_to_dataframe
from rag_stack.static_rag_evaluator.utils.util import pop_params, to_list, is_chat_prompt
from rag_stack.static_rag_evaluator.measured.vllm_env import (
	ensure_python_env_lib_in_ld_library_path,
)
from rag_stack.static_rag_evaluator.measured.vllm_subprocess import (
	MEASURED_REQUEST_FORMAT_KEY,
	REQUEST_FORMAT_CHAT_COMPLETIONS,
)

logger = logging.getLogger("RAG-Stack")


# ``torch.cuda.mem_get_info`` is only a placement snapshot.  vLLM starts a
# worker process after that snapshot and profiles CUDA graphs/activations before
# allocating its KV cache; on a shared card, a launch whose free memory merely
# equals vLLM's requested reservation can cross the boundary meanwhile.  Keep a
# small absolute margin outside the reservation.  One GiB is material on the
# 24GB quality host without needlessly excluding otherwise usable 80GB cards.
_PLACEMENT_FREE_HEADROOM_BYTES = 1 * 1024**3


def _prepare_vllm_prompts(
	prompts: Union[List[str], List[List[dict]]],
	*,
	use_chat_template: bool = True,
) -> Union[List[str], List[List[dict]]]:
	"""Wrap raw strings as user messages for the global chat contract.

	Raw local completions are intentionally unsupported: calibration, quality,
	and measured execution must all apply the model's chat template and the
	assistant-generation boundary. Already structured chat prompts are kept
	byte-for-byte unchanged.
	"""
	if use_chat_template is not True:
		raise ValueError(
			"local vLLM raw-completion mode is disabled; "
			"use_chat_template must be true"
		)
	if not prompts or not isinstance(prompts[0], str):
		return prompts
	return [[{"role": "user", "content": prompt}] for prompt in prompts]


def _quality_slot_keep_pids():
	"""Engine-core pids of the quality slot's LIVE cached engine (07-04 fix:
	an aux/QE node's __del__ used to reap ALL EngineCores — murdering the
	slot's cached main engine mid-run; subsequent slot HITs then handed out a
	dead engine → EngineDeadError ×3 → false INVALID)."""
	try:
		from rag_stack.static_rag_evaluator.engine_slot import quality_engine_slot
		return quality_engine_slot().keep_pids()
	except Exception:  # noqa: BLE001
		return None


def _quality_slot_key(model: str, input_kwargs: dict):
	"""Canonical startup key shared by placement and slot lookup."""
	return (model, tuple(sorted((k, str(v)) for k, v in input_kwargs.items())))


def _quality_slot_has_live_key(key) -> bool:
	"""Return whether the quality slot can satisfy ``key`` without a build."""
	try:
		from rag_stack.static_rag_evaluator.engine_slot import quality_engine_slot

		return quality_engine_slot().has_live_key(key)
	except Exception:  # noqa: BLE001 — advisory fast path only
		return False


def _clear_quality_slot_for_engine_fit(logger) -> bool:
	"""Release this process's idle quality engine before a larger build.

	The quality slot deliberately survives evaluation boundaries, but it must
	not reserve a GPU when the next (usually auxiliary) model cannot fit on any
	card.  Clearing the slot is quality-semantics-neutral: it only gives up an
	engine-reuse optimization, and the next main-generator use rebuilds it.
	"""
	try:
		from rag_stack.static_rag_evaluator.engine_slot import quality_engine_slot

		slot = quality_engine_slot()
		if not slot.occupied:
			return False
		logger.info(
			"vLLM adapt: no GPU has a safe free-memory margin for the "
			"requested engine while the "
			"quality engine slot is occupied; clearing the cached engine and "
			"resampling free memory"
		)
		slot.clear()
		return True
	except Exception as exc:  # noqa: BLE001 — placement may still use retry policy
		logger.warning(f"vLLM adapt: quality-slot clear failed ({exc})")
		return False


def _force_kill_engine_core_orphans(timeout: float = 5.0, keep_pids=None) -> None:
	# vllm V1 engine (0.16+, tested through 0.20) spawns each worker as a
	# separate subprocess (named "VLLM::EngineCore" / "EngineCore_DP*"). The
	# parent's `del llm_engine` drops Python refs but does NOT always cause
	# the subprocess to exit — the ProcessGroupNCCL "destroy_process_group()
	# was not called" warning is the smoke signal. Result: ghost EngineCores
	# accumulate across trials and eat GPU memory until subsequent vllm.LLM()
	# inits OOM. Walk our direct + indirect children and reap anything that
	# looks like an EngineCore: SIGTERM with a short timeout, then SIGKILL.
	try:
		import psutil
		import signal
	except ImportError:
		return
	try:
		me = psutil.Process()
		victims = []
		for c in me.children(recursive=True):
			try:
				name = c.name() or ""
				cmd = " ".join(c.cmdline() or [])
			except (psutil.NoSuchProcess, psutil.AccessDenied):
				continue
			if keep_pids and c.pid in keep_pids:
				continue  # a live cached engine (quality slot) — never an orphan
			if "EngineCore" in name or "EngineCore" in cmd or "VLLM::" in name:
				victims.append(c)
		for p in victims:
			try:
				p.send_signal(signal.SIGTERM)
			except (psutil.NoSuchProcess, ProcessLookupError):
				continue
		if victims:
			_, alive = psutil.wait_procs(victims, timeout=timeout)
			for p in alive:
				try:
					p.kill()
				except (psutil.NoSuchProcess, ProcessLookupError):
					continue
			if victims:
				logger.info(
					f"vLLM: reaped {len(victims)} EngineCore subprocess(es) "
					f"({len(alive)} required SIGKILL)"
				)
	except Exception as e:
		logger.warning(f"vLLM: EngineCore reaper failed: {e}")


def _chosen_to_physical(chosen, n_gpus: int) -> List[str]:
	"""Translate torch LOGICAL device ordinals into PHYSICAL GPU ids.

	``_adapt_engine_resources`` picks devices through torch, whose ordinals
	index this process's visible set; ``_maybe_pin_cvd`` keeps ids that
	numerically match visible physical ids as absolute. Under a non-0-based
	CUDA_VISIBLE_DEVICES (e.g. "1,2,3") the two disagree — logical 2 is
	physical GPU 3, but the numeric-subset check would pin physical GPU 2
	(07-08: on a shared box every engine whose emptiest-GPU choice omitted
	logical 0 landed on the one full card and died at request_memory). So the
	pin is emitted in PHYSICAL ids here. With no (or non-numeric, or stale —
	shorter than torch's view) prior restriction, logical == physical.
	"""
	import os as _os
	_cvd = _os.environ.get("CUDA_VISIBLE_DEVICES")
	_vis = [v.strip() for v in _cvd.split(",") if v.strip()] if _cvd else []
	if _vis and len(_vis) >= n_gpus and all(v.isdigit() for v in _vis):
		return [_vis[i] for i in sorted(chosen)]
	return [str(i) for i in sorted(chosen)]


def _model_weights_gb(model: str) -> Optional[float]:
	"""Estimate the bf16/fp16 weight footprint from the parameter count in the
	model name (``...-7B-...`` → 7e9 × 2 bytes). Returns None when the name
	carries no parseable size — the adapter then leaves the config untouched."""
	import re
	m = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?:$|[-_./])", model)
	if not m:
		return None
	return float(m.group(1)) * 2.0  # 1e9 params × 2 bytes = GB


def _adapt_engine_resources(
	model: str,
	input_kwargs: dict,
	device_spec,
	logger,
	*,
	allow_quality_slot_reuse: bool = False,
) -> object:
	"""Fit the requested engine onto the LOCAL GPUs (quality answers are
	placement-independent; performance comes from the cost model).

	The YAML's ``gpu_memory_utilization`` is written for the modeled system
	(80GB-class cards) where a flat 0.4 fits every model in the sweep. On a
	small-VRAM box (24GB 3090s) that same fraction is physically infeasible
	for 7B+ (weights alone exceed util×total) and every engine defaults onto
	cuda:0 while the other GPUs idle. This adapter, applied ONLY to the
	in-process ``vllm.LLM`` path:

	  * estimates the weight footprint from the model name;
	  * picks the smallest tensor_parallel_size ∈ {1,2,4} whose per-GPU need
	    (weights/tp + activations/graphs + KV headroom) fits in 0.92 of a
	    card, and RAISES gpu_memory_utilization to that need (never lowers
	    the configured value — big-VRAM behavior is unchanged);
	  * pins the engine to the emptiest GPU(s) via the existing ``device``
	    mechanism (unless the YAML pinned one explicitly), spreading
	    co-resident engines instead of stacking them on cuda:0;
	  * briefly waits for the previous node's engine teardown when the chosen
	    GPUs are still draining (per-eval serial phase lag).

	Returns the (possibly synthesized) device spec; mutates ``input_kwargs``
	(gpu_memory_utilization / tensor_parallel_size) in place. Any failure
	falls back to the untouched config."""
	_OVERHEAD_GB = 3.0   # activations + cudagraphs + engine fixed cost
	_KV_MIN_GB = 2.0     # minimum useful KV-cache room
	_UTIL_CAP = 0.92
	try:
		import time as _time

		import torch
		if not torch.cuda.is_available():
			return device_spec
		n_gpus = torch.cuda.device_count()
		if n_gpus == 0:
			return device_spec
		weights_gb = _model_weights_gb(model)
		if weights_gb is None:
			return device_spec
		total_gb = min(
			torch.cuda.get_device_properties(i).total_memory
			for i in range(n_gpus)
		) / 1024**3
		cfg_util = float(input_kwargs.get("gpu_memory_utilization", 0.9))
		cfg_tp = int(input_kwargs.get("tensor_parallel_size", 1) or 1)

		def _need_gb(tp: int) -> float:
			return weights_gb / tp + _OVERHEAD_GB + _KV_MIN_GB

		tp = cfg_tp
		while _need_gb(tp) > _UTIL_CAP * total_gb and tp * 2 <= min(4, n_gpus):
			tp *= 2
		# Throughput widening DISABLED (07-05): on the 4×3090 box multi-worker
		# engines die with rising frequency over process lifetime (NCCL/shm
		# leakage across engine rebuild cycles — 7B@TP2 and 7B/14B@TP4 all hit
		# VllmWorker deaths). Minimal-fit TP only: TP>1 ONLY when weights
		# physically demand it (14B→TP2); single-GPU engines are NCCL-free.
		need_util = min(_UTIL_CAP, _need_gb(tp) / total_gb)
		util = max(cfg_util, need_util)
		if _need_gb(tp) > _UTIL_CAP * total_gb:
			logger.warning(
				f"vLLM adapt: {model} (~{weights_gb:.1f}GB weights) exceeds "
				f"{_UTIL_CAP:.2f}×{total_gb:.1f}GB even at tp={tp} — launching "
				f"anyway with util={util:.2f}; expect OOM."
			)
		if (tp, util) != (cfg_tp, cfg_util):
			input_kwargs["tensor_parallel_size"] = tp
			input_kwargs["gpu_memory_utilization"] = round(util, 3)
			logger.info(
				f"vLLM adapt: {model} weights≈{weights_gb:.1f}GB on "
				f"{total_gb:.1f}GB cards → tensor_parallel_size={tp}, "
				f"gpu_memory_utilization={cfg_util:.2f}→{util:.3f}"
			)

		# Multi-GPU TP stability on the 4×3090 box (07-06). ROOT CAUSE found:
		# TP>1 engines died INTERMITTENTLY during generation NOT because of CUDA
		# graphs per se, but because ``NCCL_P2P_DISABLE=1`` forced the all-reduce
		# onto the SHM (host shared-memory) transport, whose CUDA-graph replay
		# faults on these no-NVLink cards ("Worker proc died / Executor failed").
		# Validated: with P2P ENABLED, 14B@TP2 + CUDA graphs runs 60 generations
		# in ~2s, zero deaths; the historical P2P hang was the disagg
		# p2pnccl CONNECTOR, not plain TP all-reduce. So P2P is left ENABLED for
		# these agent evals (launch no longer exports NCCL_P2P_DISABLE) and
		# CUDA graphs are kept (fast). enforce_eager is forced ONLY as a fallback
		# when something external still disables P2P (→ SHM path), keeping the
		# SHM route stable at a ~15% speed cost.
		# 07-06 UPDATE: P2P alone did NOT eliminate the death — a 14B@TP2 eval
		# still died under P2P+graphs in the live run (offline 60-gen tests pass
		# because they stress it far less than a 100-query eval, and the failure
		# rises with process lifetime — NCCL/shm state accumulates across the
		# many engine build/teardown cycles). Belt-and-suspenders: keep P2P (the
		# better transport) AND force enforce_eager for EVERY TP>1 engine to
		# remove CUDA-graph replay, the common factor across both transports'
		# deaths. ~15% slower decode; a dead 14B eval wastes ~28 min. Residual
		# intermittent deaths are handled gracefully (retries + GP-exclusion).
		if tp > 1 and str(input_kwargs.get("enforce_eager", "")).lower() not in ("true", "1", "yes"):
			input_kwargs["enforce_eager"] = True
			logger.info(
				f"vLLM adapt: {model} tp={tp}>1 → enforce_eager=True "
				"(CUDA-graph replay is flaky for multi-GPU TP on this box under "
				"BOTH P2P and SHM; ~15% slower but far more stable)"
			)

		# Placement asks whether a NEW engine can fit. A live same-key quality
		# engine needs no build at all, so its own reservation must not be treated
		# as pressure and evicted. This check must precede memory sampling/waiting;
		# otherwise a single-card 7B cache hit tears itself down on every eval.
		if (
			allow_quality_slot_reuse
			and tp <= 1
			and device_spec is None
			and _quality_slot_has_live_key(_quality_slot_key(model, input_kwargs))
		):
			logger.info(
				"vLLM adapt: live same-key quality engine slot hit; "
				"skipping placement and preserving the cached engine"
			)
			return None

		if device_spec is not None:  # YAML pin wins
			return device_spec
		# Pick the tp emptiest GPUs; wait briefly for the previous engine's
		# teardown if none has both the reservation and safe launch headroom
		# (serial per-eval phase lag).
		need_bytes = int(util * total_gb * 1024**3)
		required_free_bytes = need_bytes + _PLACEMENT_FREE_HEADROOM_BYTES
		deadline = _time.time() + 90.0
		chosen: List[int] = []
		# ACTIVE reclaim (07-06): the previous engine's teardown may lag, and a
		# TP>1 build failing to find room is the dominant multi-worker death on
		# the 4×3090 box (14B@TP2 needs BOTH GPUs ~21GB-free at once; a lingering
		# orphan EngineCore holding memory starves it → VllmWorker dies). Rather
		# than passively wait and then launch into insufficient memory, reap
		# orphan EngineCores up front so their GPU memory is released, then poll.
		# The standalone 14B@TP2 launch works cleanly — the ONLY difference in a
		# long run is accumulated engine state, so a hard reclaim closes the gap.
		_reaped = False
		_slot_checked = False
		while True:
			free = []
			for i in range(n_gpus):
				try:
					f, _ = torch.cuda.mem_get_info(i)
				except Exception:
					f = 0
				free.append((f, i))
			free.sort(reverse=True)
			chosen = [i for f, i in free[:tp]]
			if all(f >= required_free_bytes for f, _ in free[:tp]):
				break
			# The cross-eval quality slot is an optimization, not a GPU
			# reservation.  If it is the reason no card can fit the next
			# engine, release it *before* orphan reaping and resample.  The
			# reaper intentionally spares slot pids, so omitting this step can
			# otherwise wait for the full deadline and launch into a guaranteed
			# OOM (for example cached 3B main -> 7B query decomposer).
			if not _slot_checked:
				_slot_checked = True
				if _clear_quality_slot_for_engine_fit(logger):
					_time.sleep(2.0)
					continue
			# Not enough room yet — reap orphans ONCE, then keep polling.
			if not _reaped:
				_reaped = True
				try:
					_force_kill_engine_core_orphans(keep_pids=_quality_slot_keep_pids())
				except Exception as _e:  # noqa: BLE001
					logger.warning(f"vLLM adapt: orphan reap failed ({_e})")
				_time.sleep(2.0)
				continue
			if _time.time() >= deadline:
				least_selected_free = min(f for f, _ in free[:tp])
				if least_selected_free < required_free_bytes:
					logger.warning(
						f"vLLM adapt: proceeding with GPUs {chosen} after "
						f"teardown wait + orphan reap — least selected free "
						f"{least_selected_free/1024**3:.1f}GB < safe launch "
						f"{required_free_bytes/1024**3:.1f}GB "
						f"({need_bytes/1024**3:.1f}GB reservation + "
						f"{_PLACEMENT_FREE_HEADROOM_BYTES/1024**3:.1f}GB headroom)"
					)
				break
			_time.sleep(3.0)
		phys = _chosen_to_physical(chosen, n_gpus)
		logger.info(
			f"vLLM adapt: pinning to emptiest GPU(s) {sorted(chosen)} "
			f"(physical {','.join(phys)}; selected free "
			f"{min(f for f, _ in free[:tp])/1024**3:.1f}GB, safe launch "
			f"{required_free_bytes/1024**3:.1f}GB)"
		)
		return ",".join(phys)
	except Exception as e:  # never break the launch path over the adapter
		logger.warning(f"vLLM adapt: skipped ({e})")
		return device_spec


def _build_inprocess_engine(owner, factory):
	"""Build an in-process engine and synchronously undo a failed CVD pin.

	A partially constructed :class:`Vllm` can stay reachable until a later GC
	cycle.  Deferring ``CUDA_VISIBLE_DEVICES`` restoration to ``__del__`` then
	poisons the next retry's logical-to-physical device mapping.  Construction
	failure is the authoritative boundary, so restore there immediately.
	"""
	try:
		return factory()
	except BaseException:
		owner._restore_cvd()
		raise


class Vllm(BaseGenerator):
	def __init__(self, project_dir: str, model: str, **kwargs):
		super().__init__(project_dir, model, **kwargs)
		ensure_python_env_lib_in_ld_library_path(logger=logger)
		try:
			from vllm import SamplingParams, LLM
		except ImportError:
			raise ImportError(
				"Please install vllm library. You can install it by running `pip install vllm`."
			)

		model_from_kwargs = kwargs.pop("model", None)
		model = model if model_from_kwargs is None else model_from_kwargs

		input_kwargs = deepcopy(kwargs)
		sampling_params_init_params = pop_params(
			SamplingParams.from_optional, input_kwargs
		)
		input_kwargs.pop("thinking", None)
		use_chat_template = input_kwargs.pop("use_chat_template", True)
		if use_chat_template is not True:
			raise ValueError(
				"local vLLM raw-completion mode is disabled; "
				"use_chat_template must be true"
			)
		self._use_chat_template = use_chat_template
		# Request-only replay metadata: measured validates the global
		# chat-completions contract. It is not a vLLM EngineArg.
		input_kwargs.pop("measured_request_format", None)
		input_kwargs.pop("_measured_source_max_tokens", None)
		# Cache-managed lifetime: pull the current cache from the module-level
		# registry (never via kwargs — would break dict serialization).
		from rag_stack.static_rag_evaluator.measured.cache import get_current
		cache = get_current()
		self._cache = cache
		# Strip kwargs that aren't valid vLLM init args.
		input_kwargs.pop("cache", None)
		# `is_aux_vllm`: set by BaseQueryExpansion so this generator pulls the
		# AUXILIARY (query-expansion) cached vLLM — its own model/device — rather
		# than the main generator's subprocess (which serves a different model).
		is_aux_vllm = bool(input_kwargs.pop("is_aux_vllm", False))
		# `device`: YAML hint for pinning the in-process vLLM to a specific GPU
		# subset. Accepted forms:
		#   "cuda:3"            → CUDA_VISIBLE_DEVICES=3
		#   "cuda:2,cuda:3"     → CUDA_VISIBLE_DEVICES=2,3 (use with tensor_parallel_size=2)
		#   "2,3"               → CUDA_VISIBLE_DEVICES=2,3
		# Mechanism: we export CUDA_VISIBLE_DEVICES BEFORE constructing
		# ``vllm.LLM(...)``; vllm spawns its workers in subprocesses which
		# inherit the new env and thus see only the pinned GPUs. The MAIN
		# Python process's ``torch.cuda`` view is unchanged (already cached
		# at optimizer construction), so the optimizer's BoTorch tensors on
		# e.g. ``tkwargs.device=cuda:N`` keep working as long as N was
		# visible at process start.
		# `_cvd_prev` is the sentinel for __del__'s restore step. `None` means
		# "we did not mutate CVD" → skip restore. `(value,)` means "the env
		# var was unset before we touched it" → re-unset it on teardown.
		# Any other string means "restore the env var to this previous value".
		self._cvd_prev: Optional[object] = None
		self._cvd_set: Optional[str] = None
		self._quality_engine_slot = None
		device_spec = input_kwargs.pop("device", None)
		# NOTE: CUDA_VISIBLE_DEVICES is pinned LAZILY (see _maybe_pin_cvd) — only
		# when we actually fall back to the in-process ``vllm.LLM`` path below. A
		# cached subprocess / PD pair owns its GPUs in a SEPARATE process, so
		# mutating THIS process's CVD for it is a dead side-effect whose restore
		# would hinge on GC ordering; we skip it entirely in that case.
		# Prefer the PD pair when registered (real TTFT/TPOT under prefill/decode
		# disaggregation), then fall back to single subprocess, then to the
		# legacy in-process LLM. The Vllm node treats the PD pair as a drop-in
		# for the single subprocess — both expose
		# `generate_batch_streaming(prompts, sampling_params)`.
		self._subprocess = None
		if cache is not None:
			if is_aux_vllm:
				# Query-expansion: use its OWN aux engine (own model+device).
				# Prefer the aux PD pair (1P1D) when registered, then the unified
				# aux subprocess — symmetric with the main generator below. Falls
				# through to the model-keyed in-process cache when neither is
				# registered — never the main generator's subprocess.
				self._subprocess = cache.get_aux_vllm_pd() or cache.get_aux_vllm()
			else:
				self._subprocess = cache.get_main_vllm_pd() or cache.get_main_vllm()
		if self._subprocess is not None:
			self._cache_owned = True
			self.vllm_model = None
			logger.info(
				f"Vllm generator using subprocess {self._subprocess.base_url} "
				f"(real TTFT/TPOT via streaming, type={type(self._subprocess).__name__})"
			)
		else:
			# In-process LLM fallback: fit (util, tp, device) to the LOCAL
			# GPUs — the YAML values target the modeled big-VRAM system — then
			# pin CVD so vllm's spawned workers inherit it (restored by
			# _restore_cvd in __del__).
			device_spec = _adapt_engine_resources(
				model,
				input_kwargs,
				device_spec,
				logger,
				allow_quality_slot_reuse=(cache is None and not is_aux_vllm),
			)
			self._maybe_pin_cvd(device_spec)
			if cache is not None:
				self._cache_owned = True
				self.vllm_model = _build_inprocess_engine(
					self,
					lambda: cache.get_inprocess_vllm(
						model, factory=lambda m: LLM(m, **input_kwargs)
					),
				)
			elif not is_aux_vllm and int(input_kwargs.get("tensor_parallel_size", 1) or 1) <= 1:
				# QUALITY path, MAIN generator, SINGLE-GPU only: adjacent-eval
				# engine reuse via the path-neutral slot (07-04). Keyed by the
				# FULL startup signature — any engine-affecting kwarg change tears
				# down and rebuilds, so co-residency can't happen. Sampling params
				# are per-generate-call, so reuse is quality-semantics-neutral.
				# Aux (QE) engines stay per-eval fresh: they vary more and the
				# slot holds exactly one engine.
				from rag_stack.static_rag_evaluator.engine_slot import (
					quality_engine_slot,
				)
				_slot_key = _quality_slot_key(model, input_kwargs)
				self._cache_owned = True  # slot owns lifetime, not this node
				self._quality_engine_slot = quality_engine_slot()
				self.vllm_model = _build_inprocess_engine(
					self,
					lambda: self._quality_engine_slot.get(
						_slot_key, lambda: LLM(model, **input_kwargs)
					),
				)
			else:
				# NEVER CACHE MULTI-GPU (TP>1) ENGINES (07-06, user hypothesis
				# confirmed): the quality slot keeps ONE engine alive ACROSS evals
				# and exempts its EngineCore workers from the orphan reaper
				# (keep_pids). For a TP>1 (only 14B→TP2) engine — the sole NCCL
				# user — that cross-eval persistence means a reused (HIT) or
				# spared engine carries NCCL/CUDA collective state that degrades
				# over the process lifetime, and eventually a graph replay /
				# all-reduce faults at the CUDA level ("Worker proc died /
				# Executor failed", no clean traceback). TP1 engines are
				# NCCL-free so their reuse is safe and stays cached above. A TP>1
				# engine is instead built FRESH and, with _cache_owned=False,
				# fully torn down by __del__ after this eval — isolating its NCCL
				# lifetime to a single eval, the in-process analogue of the
				# measured path's per-trial subprocess.
				self._cache_owned = False
				self.vllm_model = _build_inprocess_engine(
					self, lambda: LLM(model, **input_kwargs)
				)

		# delete not sampling param keys in the kwargs (including 'cache', 'device')
		kwargs_keys = list(kwargs.keys())
		for key in kwargs_keys:
			if key not in sampling_params_init_params:
				kwargs.pop(key)

	def _maybe_pin_cvd(self, device_spec) -> None:
		"""Pin CUDA_VISIBLE_DEVICES from the YAML ``device`` hint for the
		in-process ``vllm.LLM`` path (its spawned workers inherit the env).
		Snapshots the prior value into ``self._cvd_prev`` so ``_restore_cvd``
		(in __del__) restores it — otherwise the restriction leaks into the next
		multiprocessing-spawn node and crashes its extra workers on invalid
		ordinals. Called ONLY on the in-process fallback, never for a cached
		subprocess / PD pair (which owns its GPUs in its own process)."""
		if device_spec is None:
			return
		import os as _os
		raw = [d.strip() for d in str(device_spec).split(",") if d.strip()]
		cuda_ids = [d[len("cuda:"):] if d.startswith("cuda:") else d for d in raw]
		# YAML device ids are LOGICAL indices into THIS process's visible set
		# (standard CUDA semantics). When a launcher already restricted
		# CUDA_VISIBLE_DEVICES (e.g. a benchmark pair "2,3"), writing the raw
		# id would ESCAPE the pair — "cuda:1" under CVD=2,3 must pin physical
		# GPU 3, not 1 (07-04: a pair-2,3 run's aux engine landed on phys 1
		# and collided with an unrelated job there). No prior CVD → logical
		# == physical, absolute semantics unchanged.
		_prior_cvd = _os.environ.get("CUDA_VISIBLE_DEVICES")
		if _prior_cvd:
			_visible = [v.strip() for v in _prior_cvd.split(",") if v.strip()]
			if set(cuda_ids).issubset(set(_visible)):
				logger.info(
					f"vLLM: device spec {cuda_ids} already refers to visible physical GPUs "
					f"{_visible}; keeping absolute ids under prior CUDA_VISIBLE_DEVICES"
				)
			else:
				_mapped = []
				for _cid in cuda_ids:
					try:
						_mapped.append(_visible[int(_cid)])
					except IndexError:
						# Fewer visible GPUs than the YAML's logical id (e.g. a
						# single-GPU box running a two-GPU-era config): clamp to
						# the LAST visible device — the aux engine co-resides with
						# the main one (util budgets already split per engine)
						# instead of pinning a nonexistent ordinal and crashing.
						logger.warning(
							f"vLLM: logical device {_cid} exceeds visible set "
							f"{_visible}; clamping to {_visible[-1]} (co-resident)"
						)
						_mapped.append(_visible[-1])
					except ValueError:
						_mapped.append(_cid)
				cuda_ids = _mapped
		self._cvd_prev = (
			_os.environ["CUDA_VISIBLE_DEVICES"]
			if "CUDA_VISIBLE_DEVICES" in _os.environ
			else (None,)  # sentinel for "was unset"
		)
		_os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(cuda_ids)
		self._cvd_set = _os.environ["CUDA_VISIBLE_DEVICES"]
		logger.info(
			f"vLLM: pinning CUDA_VISIBLE_DEVICES={_os.environ['CUDA_VISIBLE_DEVICES']} "
			f"(from YAML device={device_spec!r}; prior={self._cvd_prev!r})"
		)

	def _restore_cvd(self) -> None:
		# Undo any CUDA_VISIBLE_DEVICES mutation we did in __init__. Idempotent
		# — sets _cvd_prev back to None so a repeat call is a no-op. See the
		# init-time snapshot for the leak this prevents (multiprocessing-spawn
		# children inheriting our pinning and crashing on invalid ordinals).
		prev = getattr(self, "_cvd_prev", None)
		if prev is None:
			return
		import os as _os
		# LIFO guard (07-08): a failed engine's __del__ can run AFTER the next
		# retry attempt already pinned its own CVD — restoring then would
		# clobber the newer pin (observed: attempt 2 snapshotting attempt 1's
		# leaked value as its "prior"). Only restore while the env still holds
		# OUR pin; otherwise a newer node owns it and we stand down.
		_set = getattr(self, "_cvd_set", None)
		if _set is not None and _os.environ.get("CUDA_VISIBLE_DEVICES") != _set:
			logger.info(
				f"vLLM: skipping CUDA_VISIBLE_DEVICES restore — env holds "
				f"{_os.environ.get('CUDA_VISIBLE_DEVICES')!r}, not our pin "
				f"{_set!r} (a newer engine owns it)"
			)
			self._cvd_prev = None
			return
		if isinstance(prev, tuple):  # sentinel: was unset
			_os.environ.pop("CUDA_VISIBLE_DEVICES", None)
		else:
			_os.environ["CUDA_VISIBLE_DEVICES"] = prev
		logger.info(
			f"vLLM: restored CUDA_VISIBLE_DEVICES → "
			f"{_os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
		)
		self._cvd_prev = None

	def _invalidate_quality_engine_slot(self, exc: BaseException) -> None:
		slot = getattr(self, "_quality_engine_slot", None)
		if slot is None:
			return
		reason = f"{type(exc).__name__}: {exc}"
		try:
			slot.invalidate(reason)
		except Exception as clear_exc:  # noqa: BLE001
			logger.warning(f"vLLM: failed to invalidate quality engine slot: {clear_exc}")

	@staticmethod
	def _is_engine_dead_exception(exc: BaseException) -> bool:
		text = f"{type(exc).__name__}: {exc}"
		return (
			"EngineDeadError" in text
			or "EngineCore encountered" in text
			or "Executor failed" in text
		)

	def __del__(self):
		if not hasattr(self, "vllm_model"):
			# We may have raised before vllm_model was assigned but AFTER
			# mutating CVD — still need to restore.
			self._restore_cvd()
			return
		# When the cache owns this vLLM instance (in-process LLM or remote
		# subprocess), do NOT tear it down here — the cache owns lifetimes.
		# Tearing down would invalidate cached references held by subsequent
		# calls.
		if getattr(self, "_cache_owned", False):
			try:
				super().__del__()
			except Exception:
				pass
			self._restore_cvd()
			return
		# Subprocess path: vllm_model is None and we own nothing — let parent
		# clean up and return.
		if self.vllm_model is None:
			try:
				super().__del__()
			except Exception:
				pass
			self._restore_cvd()
			return
		try:
			import torch
			import contextlib

			if torch.cuda.is_available():
				from vllm.distributed.parallel_state import (
					destroy_model_parallel,
					destroy_distributed_environment,
				)

				destroy_model_parallel()
				destroy_distributed_environment()
				if hasattr(self.vllm_model.llm_engine, "model_executor"):
					del self.vllm_model.llm_engine.model_executor
				del self.vllm_model
				with contextlib.suppress(AssertionError):
					torch.distributed.destroy_process_group()
				gc.collect()
				_force_kill_engine_core_orphans(keep_pids=_quality_slot_keep_pids())
				torch.cuda.empty_cache()
				torch.cuda.synchronize()
		except ImportError:
			del self.vllm_model
			_force_kill_engine_core_orphans(keep_pids=_quality_slot_keep_pids())

		# Restore env AFTER subprocess reaping. The subprocess was already
		# fork()'d with the restricted view at __init__ time, so restoring
		# now doesn't affect that subprocess — it just unblocks subsequent
		# multiprocessing-spawn nodes in the pipeline.
		self._restore_cvd()
		super().__del__()

	@result_to_dataframe(["generated_texts", "generated_tokens", "generated_log_probs"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		kwargs.pop("model", None)  # already captured in self.model
		prompts = self.cast_to_run(previous_result)
		thinking = kwargs.pop("thinking", False)
		return self._pure(prompts, thinking=thinking, **kwargs)

	def _pure(
		self,
		prompts: Union[List[str], List[List[dict]]],
		thinking: bool = False,
		**kwargs,
	) -> Tuple[List[str], List[List[int]], List[List[float]]]:
		# Subprocess streaming path: real per-query TTFT/TPOT via OpenAI stream.
		# Logprobs are NOT collected here (the streaming chat-completions API
		# doesn't surface them in a uniformly-shaped way across vLLM versions);
		# callers that need logprobs must use the in-process path.
		if self._subprocess is not None:
			return self._pure_subprocess(prompts, **kwargs)

		prompts = _prepare_vllm_prompts(
			prompts,
			use_chat_template=self._use_chat_template,
		)

		"""
		Vllm module.
		It gets the VLLM instance and returns generated texts by the input prompt.
		You can set logprobs to get the log probs of the generated text.
		Default logprobs is 1.

		:param prompts: A list of prompts or a list of chat prompts.
		:param thinking: A boolean that indicates whether to think when generating text.
			Default is False.
			Effective when set True and using chat prompts.
			You can learn how to use chat prompt at `chat_fstring` module documentation.
		:param kwargs: The extra parameters for generating the text.
		:return: A tuple of three elements.
		    The first element is a list of generated text.
		    The second element is a list of generated text's token ids.
		    The third element is a list of generated text's log probs.
		"""
		try:
			from vllm.outputs import RequestOutput
			from vllm import SamplingParams, LLM
			from vllm.logprobs import SampleLogprobs
		except ImportError:
			raise ImportError(
				"Please install vllm library. You can install it by running `pip install vllm`."
			)

		if "logprobs" not in kwargs:
			kwargs["logprobs"] = 1

		sampling_params = pop_params(SamplingParams.from_optional, kwargs)
		generate_params = SamplingParams(**sampling_params)
		try:
			if not is_chat_prompt(prompts):
				raise ValueError(
					"local vLLM generation requires chat-formatted prompts"
				)
			chat_template_kwargs = kwargs.pop("chat_template_kwargs", {})
			chat_template_kwargs["enable_thinking"] = thinking
			chat_kwargs = pop_params(LLM.chat, kwargs)
			results: List[RequestOutput] = self.vllm_model.chat(
				prompts,
				generate_params,
				chat_template_kwargs=chat_template_kwargs,
				**chat_kwargs,
			)
		except Exception as exc:
			if self._is_engine_dead_exception(exc):
				self._invalidate_quality_engine_slot(exc)
			self._restore_cvd()
			raise
		generated_texts = list(map(lambda x: x.outputs[0].text, results))
		generated_token_ids = list(map(lambda x: x.outputs[0].token_ids, results))
		log_probs: List[SampleLogprobs] = list(
			map(lambda x: x.outputs[0].logprobs, results)
		)
		generated_log_probs = list(
			map(
				lambda x: list(map(lambda y: y[0][y[1]].logprob, zip(x[0], x[1]))),
				zip(log_probs, generated_token_ids),
			)
		)
		return (
			to_list(generated_texts),
			to_list(generated_token_ids),
			to_list(generated_log_probs),
		)

	def _pure_subprocess(
		self,
		prompts: Union[List[str], List[List[dict]]],
		**kwargs,
	) -> Tuple[List[str], List[List[int]], List[List[float]]]:
		"""Generate all prompts concurrently against the cached vLLM subprocess.

		This path is retained for direct module use. The measured service runtime
		calls the subprocess per request so it can record end-to-end request
		timing across all stages.
		"""
		if is_chat_prompt(prompts):
			# Subprocess streaming uses the chat-completions endpoint by
			# default. If the caller passed pre-rendered chat prompts we'd
			# need a different code path; fall back to in-process for safety.
			raise NotImplementedError(
				"Chat-formatted prompts are not yet supported through the "
				"streaming subprocess path. Use the in-process vLLM path "
				"by not registering a main-vLLM subprocess on the cache."
			)
		flat_prompts: List[str] = list(prompts)
		sampling_params = {
			"temperature": kwargs.get("temperature", 1.0),
			"max_tokens": kwargs.get("max_tokens", 512),
		}
		if self._use_chat_template:
			sampling_params[MEASURED_REQUEST_FORMAT_KEY] = (
				REQUEST_FORMAT_CHAT_COMPLETIONS
			)
		# Forward optional controls the agentic loops rely on. ReAct passes
		# `stop=["Observation", ...]` so each round halts at the right boundary;
		# without it the deployed engine runs to max_tokens and the loop parser
		# (Thought/Action/Observation) breaks.
		if kwargs.get("stop"):
			sampling_params["stop"] = kwargs["stop"]
		if kwargs.get("top_p") is not None:
			sampling_params["top_p"] = kwargs["top_p"]
		# Fire all queries concurrently so vLLM continuous batching activates.
		# We assume we're in a synchronous calling context (pipeline runner
		# is sync in SMOKE); if a parent event loop ever exists, run in a
		# fresh thread via asyncio.run inside a worker.
		#
		# ALWAYS use the NON-streaming batch path (both VllmSubprocess and the
		# PD pair provide `generate_batch`). Token-by-token SSE streaming makes
		# one asyncio loop process B×output_tokens events, which at e2e batch
		# sizes makes the Python loop (not the GPU) the bottleneck and depresses
		# measured qps ~3.6–4.5× vs the true server throughput (the GenZ-roofline
		# CM). Non-streaming issues one completion per request, so the generator
		# stage wall-clock reflects real continuous-batching throughput. TTFT is
		# sacrificed (no per-token signal); `QueryPerf` tolerates first_token_ts=None.
		_gen_batch = self._subprocess.generate_batch
		try:
			loop = asyncio.get_running_loop()
		except RuntimeError:
			loop = None
		if loop is None:
			texts, perf_dicts = asyncio.run(
				_gen_batch(flat_prompts, sampling_params)
			)
		else:
			# Parent already in a loop — schedule on a temporary thread.
			import concurrent.futures
			with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
				future = pool.submit(
					lambda: asyncio.run(
						_gen_batch(flat_prompts, sampling_params)
					)
				)
				texts, perf_dicts = future.result()
		# We have no token-id / logprob info from the streaming chat endpoint,
		# but `run.py::run_generator_node` uses `generated_tokens.apply(len).mean()`
		# to populate `average_output_token` in summary.csv. Synthesize a list
		# of correct length (with placeholder ids) so the downstream stat
		# remains accurate.
		token_ids: List[List[int]] = [[0] * d["n_output_tokens"] for d in perf_dicts]
		logprobs: List[List[float]] = [[0.0] * d["n_output_tokens"] for d in perf_dicts]
		return to_list(texts), token_ids, logprobs

	async def astream(self, prompt: str, **kwargs):
		raise NotImplementedError

	def stream(self, prompt: str, **kwargs):
		raise NotImplementedError
