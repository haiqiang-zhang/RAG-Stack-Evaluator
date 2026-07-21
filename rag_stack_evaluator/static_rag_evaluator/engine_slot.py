"""Path-neutral single-slot holder for one in-process vLLM engine.

Extracted engineering (07-04, user directive): the QUALITY path wants
adjacent-eval engine reuse (the ~45-60s per-eval engine load is pure tax when
consecutive arms share a generator), but must NOT reach into the measured
path's ``ModelCache`` — measured owns real-deployment lifetimes with its own
main/aux/PD semantics. This module is the shared, dependency-free primitive
both sides can build on: ONE live engine keyed by its full startup signature,
evict-on-key-change with a complete teardown, and the slot's EngineCore child
pids exposed so per-eval GPU sweepers can spare the cached engine while still
reaping true orphans.

Reuse is QUALITY-SEMANTICS-NEUTRAL: sampling params are per-``generate`` call,
never engine state, so a reused engine produces the same outputs as a fresh
one — only wall-clock changes.
"""
from __future__ import annotations

import gc
import logging
from typing import Any, Callable, Hashable, Optional, Set

logger = logging.getLogger("RAG-Stack")


def _live_engine_core_pids() -> Set[int]:
	"""Pids of this process's (recursive) children that look like EngineCores."""
	try:
		import psutil
	except ImportError:
		return set()
	out: Set[int] = set()
	try:
		me = psutil.Process()
		for child in me.children(recursive=True):
			try:
				name = " ".join([child.name(), " ".join(child.cmdline())])
			except (psutil.NoSuchProcess, psutil.AccessDenied):
				continue
			if "EngineCore" in name:
				out.add(child.pid)
	except Exception:  # noqa: BLE001 — advisory bookkeeping only
		pass
	return out


def teardown_inprocess_engine(engine: Any) -> None:
	"""Full teardown of an in-process ``vllm.LLM`` (mirrors Vllm.__del__)."""
	try:
		import contextlib

		import torch

		if torch.cuda.is_available():
			from vllm.distributed.parallel_state import (
				destroy_distributed_environment,
				destroy_model_parallel,
			)

			destroy_model_parallel()
			destroy_distributed_environment()
			if hasattr(engine, "llm_engine") and hasattr(engine.llm_engine, "model_executor"):
				del engine.llm_engine.model_executor
			del engine
			with contextlib.suppress(AssertionError):
				torch.distributed.destroy_process_group()
			gc.collect()
			torch.cuda.empty_cache()
			torch.cuda.synchronize()
		else:
			del engine
			gc.collect()
	except Exception as exc:  # noqa: BLE001 — teardown must never propagate
		logger.warning(f"engine_slot: teardown error swallowed: {exc}")


class InProcessEngineSlot:
	"""One live in-process engine, keyed by its full startup signature."""

	def __init__(self, name: str = "engine-slot"):
		self._name = name
		self._key: Optional[Hashable] = None
		self._engine: Any = None
		self._child_pids: Set[int] = set()

	def get(self, key: Hashable, factory: Callable[[], Any]) -> Any:
		"""Return the cached engine for ``key``; rebuild on key change.

		Key change performs a COMPLETE synchronous teardown of the old engine
		(including reaping its EngineCore children) before the new build, so
		two engines never co-reside.
		"""
		if self._engine is not None and self._key == key:
			if self._engine_alive():
				logger.info(f"[{self._name}] HIT {self._fmt_key(key)} — reusing engine")
				return self._engine
			logger.warning(
				f"[{self._name}] cached engine for {self._fmt_key(key)} is DEAD "
				f"(engine cores gone) — rebuilding"
			)
			self.clear()
		if self._engine is not None:
			logger.info(
				f"[{self._name}] key change {self._fmt_key(self._key)} -> "
				f"{self._fmt_key(key)} — tearing down old engine"
			)
			self.clear()
		before = _live_engine_core_pids()
		engine = factory()
		self._child_pids = _live_engine_core_pids() - before
		self._key = key
		self._engine = engine
		logger.info(
			f"[{self._name}] BUILT {self._fmt_key(key)} "
			f"(engine cores: {sorted(self._child_pids) or 'in-process'})"
		)
		return engine

	def clear(self) -> None:
		"""Tear down the held engine (if any) and reap its engine cores."""
		engine, pids = self._engine, set(self._child_pids)
		self._engine, self._key, self._child_pids = None, None, set()
		if engine is not None:
			teardown_inprocess_engine(engine)
		if pids:
			try:
				import os
				import signal

				for pid in pids:
					try:
						os.kill(pid, signal.SIGKILL)
					except ProcessLookupError:
						pass
			except Exception:  # noqa: BLE001
				pass

	def invalidate(self, reason: str = "") -> None:
		"""Drop the cached engine after a runtime failure.

		Liveness probes can miss vLLM engines whose EngineCore process still
		exists but has already transitioned into an unusable internal state.
		Runtime exceptions are the authoritative signal in that case.
		"""
		if self._engine is None:
			return
		suffix = f": {reason}" if reason else ""
		logger.warning(f"[{self._name}] invalidating cached engine{suffix}")
		self.clear()

	def _engine_alive(self) -> bool:
		"""Liveness probe: an in-process v1 engine is alive iff its EngineCore
		children are. No cores recorded (pure in-process) → trust the object."""
		if not self._child_pids:
			return True
		try:
			import psutil
			return all(psutil.pid_exists(pid) for pid in self._child_pids)
		except ImportError:
			return True

	def keep_pids(self) -> Set[int]:
		"""Child pids a GPU sweeper must SPARE (the live cached engine)."""
		return set(self._child_pids)

	def has_live_key(self, key: Hashable) -> bool:
		"""Whether ``key`` can reuse the currently held engine immediately."""
		return (
			self._engine is not None
			and self._key == key
			and self._engine_alive()
		)

	@property
	def occupied(self) -> bool:
		return self._engine is not None

	@staticmethod
	def _fmt_key(key: Hashable) -> str:
		s = str(key)
		return s if len(s) <= 90 else s[:87] + "..."


# The QUALITY path's slot (module singleton). The measured path keeps its own
# ModelCache and never touches this.
_QUALITY_SLOT: Optional[InProcessEngineSlot] = None


def quality_engine_slot() -> InProcessEngineSlot:
	global _QUALITY_SLOT
	if _QUALITY_SLOT is None:
		_QUALITY_SLOT = InProcessEngineSlot(name="quality-engine-slot")
	return _QUALITY_SLOT
