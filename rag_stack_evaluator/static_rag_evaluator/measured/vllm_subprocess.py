"""vLLM subprocess wrapper.

Replaces the in-process `vllm.LLM(model)` pattern that re-loaded the model
on every generator call. A single `VllmSubprocess` per startup-config stays
up across queries and trials, giving realistic continuous-batching behavior
and meaningful QPS / TTFT / TPOT measurements.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import requests

from rag_stack.logging_utils import PROJECT_LOG_FILE_ENV, PROJECT_LOG_STDIO_REDIRECTED_ENV
from rag_stack_evaluator.static_rag_evaluator.measured.vllm_env import (
	configure_vllm_worker_env,
)

logger = logging.getLogger("RAG-Stack")

# Local vLLM requests can legitimately sit behind saturated closed-loop queues;
# phase wall caps, not HTTP read timeouts, bound measured runtime.
_DEFAULT_STREAM_TIMEOUT_S = 900.0
_DEFAULT_HEALTH_TIMEOUT_S = 600.0
_STREAM_TIMEOUT_ENV = "RAG_STACK_VLLM_STREAM_TIMEOUT_S"
_HEALTH_TIMEOUT_ENV = "RAG_STACK_VLLM_HEALTH_TIMEOUT_S"
_MAX_CONCURRENCY_ENV = "RAG_STACK_VLLM_MAX_CONCURRENCY"
_API_SERVER_COUNT_ENV = "RAG_STACK_VLLM_API_SERVER_COUNT"
_DISABLE_CUSTOM_ALL_REDUCE_ENV = "RAG_STACK_VLLM_DISABLE_CUSTOM_ALL_REDUCE"
_TP2_DISABLE_DIRECT_P2P_ENV = "RAG_STACK_VLLM_TP2_DISABLE_DIRECT_P2P"
_CALIBRATION_RUN_ID_ENV = "RAG_STACK_VLLM_CALIBRATION_RUN_ID"
_COMPONENT_TELEMETRY_PATH_ENV = "RAG_STACK_LLM_COMPONENT_TELEMETRY_PATH"
_COMPONENT_TELEMETRY_RUN_ID_ENV = "RAG_STACK_LLM_COMPONENT_TELEMETRY_RUN_ID"
_COMPONENT_TELEMETRY_WORKER = (
	"rag_stack_evaluator.vllm_instrumentation.llm_sim.calibration.component_telemetry."
	"VllmComponentTelemetryWorker"
)
_CONTEXT_CLAMP_LOGGED: set[tuple[str, int, int]] = set()

# Calibration, quality, and measured execution share one request boundary:
# OpenAI-compatible ``/v1/chat/completions`` with each raw prompt wrapped as a
# user message. Raw ``/v1/completions`` is retained only as a named legacy
# constant so stale configs fail with an actionable error.
MEASURED_REQUEST_FORMAT_KEY = "measured_request_format"
REQUEST_FORMAT_COMPLETIONS = "completions"
REQUEST_FORMAT_CHAT_COMPLETIONS = "chat_completions"


class RetryableVllmStartupError(RuntimeError):
	"""Transient vLLM process-start failure which merits a clean relaunch.

	This is deliberately distinct from deployment/configuration infeasibility:
	callers must not turn it into ``TrialInvalid``.  The controller's existing
	generic-INVALID policy then retries after the measured provider tears down the
	failed process group.
	"""


_STARTUP_ADDRESS_IN_USE_MARKERS = (
	"eaddrinuse",
	"address already in use",
)
_STARTUP_OUTPUT_TAIL_LINES = 2_048


def _request_format(sampling_params: Dict[str, Any]) -> str:
	value = str(
		sampling_params.get(
			MEASURED_REQUEST_FORMAT_KEY,
			REQUEST_FORMAT_CHAT_COMPLETIONS,
		)
	).strip().lower()
	if value != REQUEST_FORMAT_CHAT_COMPLETIONS:
		raise ValueError(
			f"unsupported {MEASURED_REQUEST_FORMAT_KEY}={value!r}; expected "
			f"{REQUEST_FORMAT_CHAT_COMPLETIONS!r}; raw completions are disabled"
		)
	return value


def _chat_request_budget(
	requested_max_tokens: int,
	served_max_model_len: Optional[int],
) -> Tuple[int, Optional[int]]:
	"""Mirror ``VllmAPI``'s chat request budget.

	The legacy API backend kept the configured output cap fixed and asked the
	server to truncate the rendered chat prompt to ``context - max_tokens``.
	Raw measured completions retain their existing fit/clamp policy.
	"""
	max_tokens = max(1, int(requested_max_tokens))
	try:
		context = int(served_max_model_len) if served_max_model_len else 0
	except (TypeError, ValueError):
		context = 0
	truncate_prompt_tokens = max(1, context - max_tokens) if context > 0 else None
	return max_tokens, truncate_prompt_tokens


def _stream_timeout_s() -> float:
	raw = os.environ.get(_STREAM_TIMEOUT_ENV, "").strip()
	if not raw:
		return _DEFAULT_STREAM_TIMEOUT_S
	try:
		return max(5.0, float(raw))
	except ValueError:
		logger.warning(
			f"{_STREAM_TIMEOUT_ENV}={raw!r} is invalid; using "
			f"{_DEFAULT_STREAM_TIMEOUT_S:.0f}s"
		)
		return _DEFAULT_STREAM_TIMEOUT_S


def _health_timeout_s(default_s: float = _DEFAULT_HEALTH_TIMEOUT_S) -> float:
	raw = os.environ.get(_HEALTH_TIMEOUT_ENV, "").strip()
	if not raw:
		return default_s
	try:
		return max(30.0, float(raw))
	except ValueError:
		logger.warning(
			f"{_HEALTH_TIMEOUT_ENV}={raw!r} is invalid; using "
			f"{default_s:.0f}s"
		)
		return default_s


def _openai_stream_timeout():
	import httpx

	timeout_s = _stream_timeout_s()
	return httpx.Timeout(
		timeout_s,
		connect=min(10.0, timeout_s),
		read=timeout_s,
		write=min(30.0, timeout_s),
		pool=min(10.0, timeout_s),
	)


def _positive_int_env(name: str) -> Optional[int]:
	raw = os.environ.get(name, "").strip()
	if not raw:
		return None
	try:
		value = int(raw)
	except ValueError:
		logger.warning(f"{name}={raw!r} is invalid; ignoring")
		return None
	if value <= 0:
		logger.warning(f"{name}={raw!r} must be positive; ignoring")
		return None
	return value


def resolve_vllm_api_server_count() -> int:
	"""Resolve the measured vLLM frontend process count.

	The default remains the historical single API server.  Scale-out is an
	explicit hardware/runtime choice because it consumes additional host CPU and
	memory even though the underlying GPU engine topology is unchanged.
	"""
	raw = os.environ.get(_API_SERVER_COUNT_ENV, "").strip()
	if not raw:
		return 1
	try:
		count = int(raw)
	except ValueError as exc:
		raise ValueError(
			f"{_API_SERVER_COUNT_ENV} must be a positive integer, got {raw!r}"
		) from exc
	if count <= 0:
		raise ValueError(
			f"{_API_SERVER_COUNT_ENV} must be a positive integer, got {raw!r}"
		)
	return count


def _stream_concurrency_limit(
	n_prompts: int,
	default_limit: int,
	env_name: str = _MAX_CONCURRENCY_ENV,
) -> int:
	env_limit = _positive_int_env(env_name)
	limit = env_limit if env_limit is not None else default_limit
	try:
		limit = int(limit)
	except (TypeError, ValueError):
		limit = 1
	return max(1, min(max(n_prompts, 1), limit))


def _query_served_max_model_len(base_url: str) -> Optional[int]:
	try:
		r = requests.get(f"{base_url}/models", timeout=5.0)
		r.raise_for_status()
		payload = r.json()
		data = payload.get("data") if isinstance(payload, dict) else None
		if not data:
			return None
		model_info = data[0] if isinstance(data[0], dict) else {}
		for key in ("max_model_len", "max_model_length", "context_length"):
			value = model_info.get(key)
			if value is None:
				continue
			value = int(value)
			if value > 0:
				return value
	except Exception as exc:  # noqa: BLE001
		logger.debug("could not query vLLM served max_model_len: %s", exc)
	return None


@lru_cache(maxsize=32)
def _hf_model_context_len(model: str) -> Optional[int]:
	try:
		from transformers import AutoConfig

		from rag_stack.model_map import resolve_tokenizer_name

		cfg = AutoConfig.from_pretrained(resolve_tokenizer_name(model))
		candidates = [
			getattr(cfg, "max_position_embeddings", None),
			getattr(cfg, "max_sequence_length", None),
			getattr(cfg, "seq_length", None),
			getattr(cfg, "n_positions", None),
		]
		valid = [int(v) for v in candidates if v is not None and 0 < int(v) < 10_000_000]
		if valid:
			return max(valid)
	except Exception as exc:  # noqa: BLE001
		logger.debug("could not resolve HF context length for %s: %s", model, exc)
	return None


def _resolve_served_max_model_len(
	base_url: str,
	model: str,
	key_max_model_len: int,
) -> Optional[int]:
	try:
		key_len = int(key_max_model_len)
	except (TypeError, ValueError):
		key_len = 0
	if key_len > 0:
		return key_len
	return _query_served_max_model_len(base_url) or _hf_model_context_len(model)


def _fit_request_to_context(
	*,
	model: str,
	prompt: str,
	requested_max_tokens: int,
	served_max_model_len: Optional[int],
) -> Tuple[int, Optional[int]]:
	requested = max(1, int(requested_max_tokens))
	if not served_max_model_len or served_max_model_len <= 0:
		return requested, None
	limit = int(served_max_model_len)
	# Fast safe path: BPE token count cannot exceed plain character count for
	# normal benchmark prompts, so only long prompts pay tokenizer overhead.
	if len(str(prompt)) + requested <= limit:
		return requested, None
	try:
		from rag_stack_evaluator.static_rag_evaluator.recording import count_tokens_batch

		input_tokens = int(count_tokens_batch([prompt], model)[0])
	except Exception as exc:  # noqa: BLE001
		logger.debug("could not count prompt tokens for context guard: %s", exc)
		return requested, None
	allowed = limit - input_tokens
	if allowed >= requested:
		return requested, None
	fitted = max(1, min(requested, allowed))
	truncate_prompt_tokens = None
	if input_tokens + fitted > limit:
		truncate_prompt_tokens = max(1, limit - fitted)
	log_key = (model, requested, limit)
	if log_key not in _CONTEXT_CLAMP_LOGGED:
		_CONTEXT_CLAMP_LOGGED.add(log_key)
		logger.info(
			"vLLM request max_tokens clamped for context: model=%s "
			"input_tokens=%d requested=%d fitted=%d truncate_prompt_tokens=%s "
			"max_model_len=%d",
			model,
			input_tokens,
			requested,
			fitted,
			truncate_prompt_tokens,
			limit,
		)
	return fitted, truncate_prompt_tokens



async def _afit_request_to_context(
	*,
	model: str,
	prompt: str,
	requested_max_tokens: int,
	served_max_model_len,
):
	"""Async wrapper: the cheap char-count fast path stays inline; the
	HF-tokenizer path (long prompts only) runs in a worker thread so it
	never blocks the event loop between completion reads."""
	requested = max(1, int(requested_max_tokens))
	try:
		limit = int(served_max_model_len) if served_max_model_len else 0
	except (TypeError, ValueError):
		limit = 0
	if limit <= 0 or len(str(prompt)) + requested <= limit:
		return requested, None
	return await asyncio.to_thread(
		_fit_request_to_context,
		model=model,
		prompt=prompt,
		requested_max_tokens=requested_max_tokens,
		served_max_model_len=served_max_model_len,
	)

def _fit_max_tokens_to_context(
	*,
	model: str,
	prompt: str,
	requested_max_tokens: int,
	served_max_model_len: Optional[int],
) -> int:
	max_tokens, _ = _fit_request_to_context(
		model=model,
		prompt=prompt,
		requested_max_tokens=requested_max_tokens,
		served_max_model_len=served_max_model_len,
	)
	return max_tokens


@dataclass(frozen=True)
class VllmStartupKey:
	"""Identifying tuple for a vLLM subprocess.

	Two requests with the same key share a process; differing keys require
	teardown + relaunch.

	Multi-GPU support: `device` may be a comma-separated list (e.g.,
	``"cuda:2,cuda:3"``); the number of GPUs must equal
	``tensor_parallel_size * pipeline_parallel_size``. Stored as a string so
	the dataclass stays hashable.
	"""

	model: str
	device: str = "cuda:0"
	max_num_seqs: int = 64
	# None ⇒ do NOT pass --max-num-batched-tokens; let vLLM pick its own (much
	# larger) default. A small hardcoded cap (the old 2048) throttles prefill for
	# long-context agentic runs (react contexts grow to ~2k+ tokens), inflating
	# latency. Only set this to pin a specific value for a study.
	max_num_batched_tokens: "int | None" = None
	# ``None`` preserves vLLM's deployment default.  Standalone component/stage
	# calibration pins this to ``False`` because automatic prefix caching would
	# turn a repeated full-prompt prefill anchor into a cached/chunked shape.
	enable_prefix_caching: "bool | None" = None
	# -1 ⇒ ask vLLM to auto-fit the served context length to the launch-time GPU
	# memory budget. This avoids failing measured runs on models whose HF config
	# advertises a long context (e.g. 32k) that does not fit in the trial layout.
	max_model_len: int = -1
	kv_cache_dtype: str = "auto"
	gpu_memory_utilization: float = 0.85
	tensor_parallel_size: int = 1
	pipeline_parallel_size: int = 1
	dtype: str = "bfloat16"
	# Number of OpenAI-compatible frontend processes attached to the same GPU
	# EngineCore.  This is part of the cache identity because changing it changes
	# the deployed server process graph and host-side capacity.
	api_server_count: int = 1


def _vllm_server_command_prefix(key: VllmStartupKey) -> List[str]:
	"""Return the launch prefix without touching CUDA or host resources."""
	count = int(key.api_server_count)
	if count <= 0:
		raise ValueError(f"api_server_count must be positive, got {count}")
	telemetry_path = os.environ.get("RAG_STACK_STAGE_TELEMETRY_PATH")
	timing_dispatch = os.environ.get(
		"RAG_STACK_V1_TIMING_DISPATCH_COUNTER", ""
	).strip().lower() in {"1", "true", "yes", "on"}
	if telemetry_path is not None or timing_dispatch:
		if telemetry_path is not None and timing_dispatch:
			raise ValueError(
				"stage proof telemetry and production timing dispatch counter "
				"must use separate servers"
			)
		if timing_dispatch and count != 1:
			raise ValueError(
				"production timing dispatch counter requires exactly one "
				f"API frontend, got api_server_count={count}"
			)
	if telemetry_path is not None:
		if not telemetry_path.strip():
			raise ValueError(
				"RAG_STACK_STAGE_TELEMETRY_PATH cannot be empty"
			)
		if count != 1:
			raise ValueError(
				"stage-cycle calibration telemetry requires exactly one "
				f"API frontend, got api_server_count={count}"
			)
		run_id = (
			os.environ.get(_CALIBRATION_RUN_ID_ENV)
			or os.environ.get(_COMPONENT_TELEMETRY_RUN_ID_ENV)
		)
		if run_id is None or not run_id.strip():
			raise ValueError(
				f"{_CALIBRATION_RUN_ID_ENV} is required with stage telemetry"
			)
		return [
			sys.executable,
			"-m",
			"rag_stack_evaluator.vllm_instrumentation.serving_curves.calibration.vllm_server",
			"--model",
			key.model,
		]
	if timing_dispatch:
		return [
			sys.executable,
			"-m",
			"rag_stack_evaluator.vllm_instrumentation.serving_curves.calibration.vllm_server",
			"--model",
			key.model,
		]
	if count == 1:
		# Preserve the historical launcher byte-for-byte for the default path.
		return [
			sys.executable,
			"-m",
			"vllm.entrypoints.openai.api_server",
			"--model",
			key.model,
		]
	# ``vllm.entrypoints.openai.api_server.__main__`` always calls run_server()
	# directly and therefore ignores --api-server-count.  Only the top-level
	# ``serve`` subcommand dispatches to vLLM's run_multi_api_server().
	return [
		sys.executable,
		"-m",
		"vllm.entrypoints.cli.main",
		"serve",
		key.model,
		"--api-server-count",
		str(count),
	]


def _component_telemetry_worker_args(
	env: Dict[str, str], key: VllmStartupKey
) -> List[str]:
	"""Return an opt-in worker class or fail before starting vLLM.

	The stage stat logger may be enabled in the same diagnostic run.  Both then
	share a run/cycle identity, while publication policy still decides which
	independently scoped observations are admissible for each calibration fit.
	"""
	path = env.get(_COMPONENT_TELEMETRY_PATH_ENV)
	if path is None:
		return []
	if not path.strip():
		raise ValueError(f"{_COMPONENT_TELEMETRY_PATH_ENV} cannot be empty")
	run_id = (
		env.get(_CALIBRATION_RUN_ID_ENV)
		or env.get(_COMPONENT_TELEMETRY_RUN_ID_ENV)
	)
	if run_id is None or not run_id.strip():
		raise ValueError(
			f"{_CALIBRATION_RUN_ID_ENV} is required with component telemetry"
		)
	if int(key.api_server_count) != 1:
		raise ValueError(
			"component calibration telemetry requires exactly one API frontend"
		)
	if int(key.pipeline_parallel_size) != 1:
		raise ValueError(
			"component telemetry cannot price PP>1 until vLLM exposes timed PP "
			"receive/send boundaries"
		)
	return ["--worker-cls", _COMPONENT_TELEMETRY_WORKER]


def _prefix_caching_args(key: VllmStartupKey) -> List[str]:
	"""Render an explicit prefix-cache policy without changing measured defaults."""
	value = key.enable_prefix_caching
	if value is None:
		return []
	if not isinstance(value, bool):
		raise ValueError("enable_prefix_caching must be bool or None")
	return ["--enable-prefix-caching" if value else "--no-enable-prefix-caching"]


def _find_free_port() -> int:
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
		s.bind(("127.0.0.1", 0))
		return s.getsockname()[1]


def _retryable_startup_port_failure(proc: subprocess.Popen) -> Optional[str]:
	"""Return the exact captured EADDRINUSE line, if this child emitted one."""
	persisted = getattr(proc, "_rag_stack_startup_port_failure", None)
	if isinstance(persisted, str) and persisted:
		return persisted
	tail = getattr(proc, "_rag_stack_output_tail", ())
	for line in reversed(tuple(tail)):
		lowered = line.lower()
		if any(marker in lowered for marker in _STARTUP_ADDRESS_IN_USE_MARKERS):
			return line.strip()
	return None


def _wait_for_health(
	base_url: str,
	timeout_s: Optional[float] = None,
	proc: Optional[subprocess.Popen] = None,
) -> None:
	if timeout_s is None:
		timeout_s = _health_timeout_s()
	deadline = time.monotonic() + timeout_s
	last_err: Optional[Exception] = None
	while time.monotonic() < deadline:
		# A distributed worker can report a failed TCPStore bind while the API
		# parent is still alive waiting for its EngineCore.  Detect that exact
		# transient immediately instead of burning the full health timeout.
		if proc is not None:
			port_failure = _retryable_startup_port_failure(proc)
			if port_failure is not None:
				raise RetryableVllmStartupError(
					"vLLM distributed rendezvous failed during startup: "
					f"{port_failure}"
				)
		# Fail FAST if the subprocess already died (e.g. EngineCore OOM / "No
		# available memory for the cache blocks") — otherwise we'd poll /health
		# for the full timeout (default 600s) on a process that will never serve.
		if proc is not None and proc.poll() is not None:
			# The reader thread can trail process exit by a few scheduler ticks.
			# Give it a bounded drain and re-check the exact transient marker before
			# emitting the generic (and deployment-infeasible) startup exception.
			_join_output_tee(proc, timeout_s=0.5)
			port_failure = _retryable_startup_port_failure(proc)
			if port_failure is not None:
				raise RetryableVllmStartupError(
					"vLLM distributed rendezvous failed during startup: "
					f"{port_failure}"
				)
			raise RuntimeError(
				f"vLLM subprocess exited (returncode={proc.returncode}) during "
				f"startup before /health came up — see the subprocess log above "
				f"for the root cause (commonly an init OOM)."
			)
		try:
			r = requests.get(f"{base_url}/health", timeout=2.0)
			if r.status_code == 200:
				return
		except Exception as exc:
			last_err = exc
		time.sleep(1.0)
	raise RuntimeError(
		f"vLLM health endpoint at {base_url}/health did not come up "
		f"within {timeout_s}s. Last error: {last_err}"
	)


def _start_output_tee(
	proc: subprocess.Popen,
	label: str,
	log_path: Optional[str],
	output_tail: "deque[str]",
) -> Optional[threading.Thread]:
	"""Mirror a child process' merged stdout/stderr to terminal and project log."""
	stream = proc.stdout
	if stream is None:
		return None
	stdio_redirected = os.environ.get(PROJECT_LOG_STDIO_REDIRECTED_ENV) == "1"

	def _pump() -> None:
		fh = None
		try:
			if log_path and not stdio_redirected:
				try:
					fh = open(log_path, "a", encoding="utf-8", buffering=1)
				except Exception as exc:  # noqa: BLE001
					logger.warning(
						"subprocess log tee could not open %s for %s: %s",
						log_path,
						label,
						exc,
					)
			for line in stream:
				output_tail.append(line)
				# A failed distributed launch can print thousands of traceback
				# lines after the first EADDRINUSE diagnostic.  Retain the first
				# exact marker separately so bounded-tail eviction cannot turn an
				# infrastructure port race into a deployment-invalid result.
				lowered = line.lower()
				if (
					getattr(proc, "_rag_stack_startup_port_failure", None) is None
					and any(
						marker in lowered
						for marker in _STARTUP_ADDRESS_IN_USE_MARKERS
					)
				):
					setattr(
						proc,
						"_rag_stack_startup_port_failure",
						line.strip(),
					)
				try:
					sys.stderr.write(line)
					sys.stderr.flush()
				except Exception:  # noqa: BLE001
					pass
				if fh is not None:
					try:
						fh.write(line)
						fh.flush()
					except Exception:  # noqa: BLE001
						pass
		except Exception as exc:  # noqa: BLE001
			logger.warning("subprocess log tee failed for %s: %s", label, exc)
		finally:
			if fh is not None:
				with contextlib.suppress(Exception):
					fh.close()
			with contextlib.suppress(Exception):
				stream.close()

	thread = threading.Thread(
		target=_pump,
		name=f"rag-stack-log-tee-{label}",
		daemon=True,
	)
	thread.start()
	return thread


def _join_output_tee(proc: subprocess.Popen, timeout_s: float = 2.0) -> None:
	thread = getattr(proc, "_rag_stack_log_tee_thread", None)
	if thread is not None and thread.is_alive():
		thread.join(timeout=timeout_s)


# Every vLLM we launch carries this env marker. The startup reclaim uses it to
# recognise our own orphaned servers (left behind by a previous run that died
# via SIGKILL/OOM-kill, where teardown never ran) without touching unrelated
# vLLM processes that may belong to other tools on a shared box.
_OWNED_ENV_MARKER = "RAG_STACK_VLLM_OWNED"


def _kill_proc_group(
	proc: Optional[subprocess.Popen],
	*,
	term_timeout: float = 15.0,
	kill_timeout: float = 5.0,
) -> None:
	"""Tear down a vLLM subprocess AND its entire session / process group.

	vLLM's ``api_server`` spawns ``VLLM::EngineCore`` worker subprocesses that
	hold the GPU memory. Signalling only the api_server parent (``terminate()``)
	can orphan those workers — especially when the parent is wedged or has
	already died (then ``proc.children()`` is empty because they re-parented to
	init). Because we launch every vLLM with ``start_new_session=True`` (its own
	process group, ``pgid == api_server pid``, inherited by the workers), we can
	reap the whole tree atomically via ``os.killpg`` regardless of parent
	liveness: SIGTERM the group, wait, then SIGKILL-sweep any stragglers."""
	if proc is None:
		return
	import os
	import signal

	pgid: Optional[int] = None
	try:
		pgid = os.getpgid(proc.pid)
	except (ProcessLookupError, OSError):
		pgid = None

	def _signal_group(sig: int) -> None:
		if pgid is not None:
			try:
				os.killpg(pgid, sig)
				return
			except (ProcessLookupError, OSError):
				pass
		# Fallback when the group is gone / unavailable: signal the parent only.
		try:
			proc.send_signal(sig)
		except (ProcessLookupError, OSError):
			pass

	if proc.poll() is None:
		_signal_group(signal.SIGTERM)
		try:
			proc.wait(timeout=term_timeout)
		except subprocess.TimeoutExpired:
			pass
	# Always SIGKILL the group afterwards: even on a clean parent exit, an
	# EngineCore worker can linger. ESRCH (group already empty) is expected.
	_signal_group(signal.SIGKILL)
	try:
		proc.wait(timeout=kill_timeout)
	except subprocess.TimeoutExpired:
		pass
	_join_output_tee(proc)


def reclaim_orphaned_vllm(logger_=logger) -> int:
	"""Kill vLLM servers orphaned by a previously-crashed run and free their VRAM.

	When an optimize run dies via SIGKILL / the OOM-killer, no teardown runs and
	its vLLM ``api_server`` + ``EngineCore`` children re-parent to init (pid 1)
	while still pinning GPU memory — they then starve the next (or resumed) run's
	launches. Called at measured-mode startup to clear them.

	Safety: targets only processes WE spawned (``_OWNED_ENV_MARKER`` in their
	environ) that have re-parented to init (``ppid == 1``). A live concurrent
	run's servers still have a live parent (``ppid != 1``) and are never touched.
	Returns the number of process groups reaped."""
	try:
		import os
		import signal

		import psutil
	except ImportError:
		return 0
	pgids: set = set()
	for p in psutil.process_iter(["pid", "ppid"]):
		try:
			if p.info.get("ppid") != 1:
				continue
			if p.environ().get(_OWNED_ENV_MARKER) != "1":
				continue
			pgids.add(os.getpgid(p.info["pid"]))
		except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError, OSError):
			continue
	for pgid in pgids:
		try:
			os.killpg(pgid, signal.SIGKILL)
		except (ProcessLookupError, OSError):
			continue
	if pgids:
		logger_.warning(
			f"reclaimed {len(pgids)} orphaned vLLM process group(s) from a prior "
			f"run (SIGKILL/OOM-killed run that skipped teardown)"
		)
	return len(pgids)


def _popen_with_output_tee(
	cmd: List[str],
	*,
	env: Dict[str, str],
	label: str,
	capture_tail: bool = False,
) -> subprocess.Popen:
	"""Launch a subprocess.

	If a project log is active, tee the child stdout/stderr to the terminal and
	that log file. When an app parent already redirects stdio into the same log,
	only the stdio path writes the file to avoid duplicate lines.
	"""
	# start_new_session=True puts the vLLM api_server (and its EngineCore worker
	# children, which inherit the new pgid) in their own process group so
	# `_kill_proc_group` can reap the whole tree atomically. The owner marker
	# lets `reclaim_orphaned_vllm` recognise our servers after a crashed run.
	env = dict(env)
	env[_OWNED_ENV_MARKER] = "1"
	log_path = os.environ.get(PROJECT_LOG_FILE_ENV)
	if not log_path and not capture_tail:
		return subprocess.Popen(cmd, env=env, start_new_session=True)
	env.setdefault("PYTHONUNBUFFERED", "1")
	proc = subprocess.Popen(
		cmd,
		env=env,
		stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT,
		text=True,
		bufsize=1,
		errors="replace",
		start_new_session=True,
	)
	# Retain bounded startup output for health classification.  The exact
	# EADDRINUSE marker is also persisted separately by the tee thread, so a
	# long distributed traceback cannot evict it.
	output_tail: deque[str] = deque(maxlen=_STARTUP_OUTPUT_TAIL_LINES)
	setattr(proc, "_rag_stack_output_tail", output_tail)
	thread = _start_output_tee(proc, label, log_path, output_tail)
	if thread is not None:
		setattr(proc, "_rag_stack_log_tee_thread", thread)
	logger.info(
		"subprocess stdout/stderr tee active for %s%s",
		label,
		f" -> {log_path}" if log_path else " (bounded startup capture)",
	)
	return proc


class VllmSubprocess:
	"""Spawns and supervises one vLLM OpenAI-compatible API server."""

	def __init__(self, key: VllmStartupKey):
		self.key = key
		self.port = _find_free_port()
		self.base_url = f"http://127.0.0.1:{self.port}/v1"
		self._async_client: Any = None
		self._http_by_loop: Dict[int, Any] = {}
		self._served_max_model_len: Optional[int] = None
		# Multi-GPU device handling: device may be "cuda:2" or "cuda:2,cuda:3"
		# (TP*PP > 1). Build CUDA_VISIBLE_DEVICES from the cuda indices.
		raw_devices = [d.strip() for d in key.device.split(",") if d.strip()]
		cuda_ids: list[str] = []
		for d in raw_devices:
			cuda_ids.append(d[len("cuda:") :] if d.startswith("cuda:") else d)
		if not cuda_ids:
			cuda_ids = ["0"]
		expected_chips = key.tensor_parallel_size * key.pipeline_parallel_size
		if len(cuda_ids) != expected_chips:
			raise ValueError(
				f"VllmStartupKey: device={key.device!r} resolves to "
				f"{len(cuda_ids)} GPU(s) but TP*PP={expected_chips}. "
				f"Caller must pass exactly tensor_parallel_size * "
				f"pipeline_parallel_size GPUs in `device`."
			)
		env = os.environ.copy()
		configure_vllm_worker_env(env=env, logger=logger)
		env["CUDA_VISIBLE_DEVICES"] = ",".join(cuda_ids)
		stage_telemetry_enabled = (
			env.get("RAG_STACK_STAGE_TELEMETRY_PATH") is not None
		)
		if stage_telemetry_enabled:
			# vLLM 0.18.1 exposes the exact per-cycle prefill/decode work
			# shape only through its opt-in MFU debug context breakdown.  The
			# custom stat logger consumes that shape; it never infers an active
			# batch from the number of HTTP clients.
			env["VLLM_DEBUG_MFU_METRICS"] = "1"
		tp2_direct_p2p_disabled = (
			key.tensor_parallel_size == 2
			and env.get(_TP2_DISABLE_DIRECT_P2P_ENV, "").strip().lower()
			in {"1", "true", "yes", "on"}
		)
		if tp2_direct_p2p_disabled:
			# Scope the transport workaround to this TP2 subprocess. A case can
			# also contain a healthy TP4 engine whose communication path must stay
			# unchanged.
			env["NCCL_P2P_DISABLE"] = "1"
		# Account for CUDA-graph memory INSIDE gpu_memory_utilization. By default
		# vLLM allocates cudagraph capture memory (~1-2.5 GiB) OUTSIDE the util
		# budget, so its real footprint exceeds util*total — collocated engines
		# then OOM even though each believes it's within budget ("phantom" OOM
		# with space free). This env makes the footprint ≤ util*total, so the
		# derived per-engine split actually holds. (Default in vLLM v0.19+.)
		env.setdefault("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", "1")
		# Size util to memory ACTUALLY free now (waits out eviction lag; lowers
		# util if a co-tenant persists) so a transient/legitimate occupant can't
		# cause a phantom "free < desired utilization" launch refusal.
		from rag_stack_evaluator.static_rag_evaluator.measured.gpu_mem import effective_util
		eff_util = effective_util(cuda_ids, float(key.gpu_memory_utilization), model=key.model)
		cmd = [
			# sys.executable, NOT "python": the subprocess must use the SAME
			# interpreter/env as the parent — bare "python" resolves via PATH
			# and can hit a different env without vllm installed.
			*_vllm_server_command_prefix(key),
			"--host", "127.0.0.1",
			"--port", str(self.port),
			# Access logs emit one synchronous line per measured request. At high
			# QPS that logging path becomes benchmark work rather than diagnostics.
			"--disable-uvicorn-access-log",
			"--max-model-len", str(key.max_model_len),
			"--max-num-seqs", str(key.max_num_seqs),
			"--kv-cache-dtype", key.kv_cache_dtype,
			"--gpu-memory-utilization", str(eff_util),
			"--tensor-parallel-size", str(key.tensor_parallel_size),
			"--pipeline-parallel-size", str(key.pipeline_parallel_size),
			"--dtype", key.dtype,
			*_prefix_caching_args(key),
			*_component_telemetry_worker_args(env, key),
			*(["--enable-mfu-metrics"] if stage_telemetry_enabled else []),
			# Engine stats are ON by default here (r21): the occupancy probe
			# reads vllm:num_requests_running/waiting plus the prefix-cache
			# hit counters and they are the saturation-admissibility proof
			# every measured record must carry. The PrometheusStatLogger
			# negative-increment abort bug that once forced stats off bites
			# only PD/KV-source token accounting — those engines launch via
			# vllm_pd_pair, which keeps its own hardcoded --disable-log-stats.
			# Set RAG_STACK_VLLM_ENGINE_STATS=0 to force stats off here.
			*(
				["--disable-log-stats"]
				if os.environ.get(
					"RAG_STACK_VLLM_ENGINE_STATS", "1"
				).strip().lower() in {"0", "false", "no", "off"}
				else []
			),
		]
		if tp2_direct_p2p_disabled or env.get(
			_DISABLE_CUSTOM_ALL_REDUCE_ENV, ""
		).strip().lower() in {"1", "true", "yes", "on"}:
			# Some PCIe-only systems cannot safely run vLLM's direct-P2P custom
			# all-reduce for mixed TP/PP layouts. Keep this opt-in so normal
			# launches and their measured communication backend remain unchanged.
			cmd.append("--disable-custom-all-reduce")
		# Only cap the per-iteration token budget if explicitly pinned; otherwise
		# let vLLM choose (its default is far larger than the old hardcoded 2048,
		# which throttled long-context prefill).
		if key.max_num_batched_tokens is not None:
			cmd += ["--max-num-batched-tokens", str(key.max_num_batched_tokens)]
		if tp2_direct_p2p_disabled:
			logger.info(
				"vLLM TP2 direct-P2P workaround on %s: "
				"NCCL_P2P_DISABLE=1, custom all-reduce disabled",
				key.device,
			)
		logger.info(f"Launching vLLM subprocess on {key.device}: {' '.join(cmd)}")
		self.proc = _popen_with_output_tee(
			cmd,
			env=env,
			label=f"vllm:{key.device}:{self.port}",
			capture_tail=True,
		)
		try:
			_wait_for_health(f"http://127.0.0.1:{self.port}", proc=self.proc)
		except Exception:
			self.shutdown()
			raise
		self._served_max_model_len = _resolve_served_max_model_len(
			self.base_url,
			self.key.model,
			self.key.max_model_len,
		)
		logger.info(f"vLLM subprocess ready at {self.base_url}")

	def shutdown(self) -> None:
		# Kill the whole process group (api_server + EngineCore workers), not
		# just the parent handle — workers are what pin the GPU memory.
		_kill_proc_group(self.proc)
		logger.info(f"vLLM subprocess on {self.key.device} shut down")

	# --- Streaming generation (real TTFT/TPOT measurement) -----------

	def _client(self):
		if self._async_client is None:
			from openai import AsyncOpenAI
			self._async_client = AsyncOpenAI(
				base_url=self.base_url,
				api_key="EMPTY",
				timeout=_openai_stream_timeout(),
			)
		return self._async_client

	def _http_session(self):
		"""Loop-keyed shared aiohttp session for the RAW completions path.

		Why raw aiohttp and not the openai client: at high call rates the
		openai-python client's request building + pydantic response parsing
		costs ~3.2x throughput on the driver's single event loop (measured
		07-08 on a 1.5B TP2 engine at population 1024: raw aiohttp 928
		calls/s vs cached AsyncOpenAI 287 calls/s, same engine, same loop).
		That client-side cost suppressed every high-call-rate measured
		number (react x large batch) to a ~110 calls/s harness ceiling the
		real engine does not have.

		Sessions bind to the event loop they are created in and measured
		runs one asyncio.run() per trial, so key by loop identity; stale
		sessions of dead loops are dropped (their sockets are already gone
		with the per-trial vLLM teardown). connector limit=0: admission is
		gated upstream by the stage service's max_inflight semaphore.
		"""
		import aiohttp
		loop = asyncio.get_running_loop()
		sess = self._http_by_loop.get(id(loop))
		if sess is None or sess.closed:
			sess = aiohttp.ClientSession(
				# keepalive_timeout BELOW the server's (uvicorn defaults to
				# 5s): the client discards idle pooled sockets before the
				# server can close them under us. Low-call-rate trials (big
				# models at request batch 1-2, seconds between generates)
				# otherwise grab a socket the server just closed →
				# ServerDisconnectedError — resilience the old openai client
				# provided via its automatic connection-error retry.
				connector=aiohttp.TCPConnector(limit=0, keepalive_timeout=4.0),
				timeout=aiohttp.ClientTimeout(total=_stream_timeout_s()),
			)
			self._http_by_loop[id(loop)] = sess
		return sess

	async def aclose_http(self):
		"""Close this loop's raw session (called by the serving runtime
		before its event loop ends; best-effort)."""
		loop = asyncio.get_running_loop()
		sess = self._http_by_loop.pop(id(loop), None)
		if sess is not None and not sess.closed:
			await sess.close()

	async def _raw_completion(
		self,
		payload: Dict[str, Any],
		*,
		request_format: str = REQUEST_FORMAT_CHAT_COMPLETIONS,
	) -> Dict[str, Any]:
		"""POST a chat-completions JSON body; return the parsed body.

		Error semantics match the PD path: HTTP>=400 raises with the
		response BODY included (vLLM puts the actionable cause there).

		One immediate retry on a stale-keep-alive connection error
		(ServerDisconnected / ECONNRESET before any response): the request
		never reached the server, and the retry runs on a fresh socket —
		the same resilience the openai client's built-in connection-error
		retry used to provide. Anything on the second attempt propagates.
		"""
		import aiohttp
		request_format = _request_format({
			MEASURED_REQUEST_FORMAT_KEY: request_format,
		})
		endpoint = (
			"chat/completions"
			if request_format == REQUEST_FORMAT_CHAT_COMPLETIONS
			else "completions"
		)
		sess = self._http_session()
		for attempt in (0, 1):
			try:
				async with sess.post(
					f"{self.base_url}/{endpoint}", json=payload,
				) as r:
					if r.status >= 400:
						body = (await r.text())[:2000]
						raise RuntimeError(
							f"vLLM {endpoint} request failed (HTTP {r.status}): {body}"
						)
					return await r.json()
			except (aiohttp.ServerDisconnectedError, aiohttp.ClientOSError) as e:
				if attempt == 1:
					raise
				logger.warning(
					f"raw {endpoint}: stale connection ({type(e).__name__}: {e}); "
					f"retrying once on a fresh socket"
				)
		raise RuntimeError("unreachable")  # loop always returns or raises

	async def generate_one_stream(
		self,
		prompt: str,
		sampling_params: Dict[str, Any],
	) -> Tuple[str, Dict[str, Any]]:
		"""Stream-generate one prompt; return (text, perf-dict).

		`perf-dict` has the same fields as :class:`QueryPerf` (kept dict-shaped
		here to avoid this module depending on `performance.py`):
		``{request_send_ts, first_token_ts, last_token_ts, n_output_tokens}``.
		"""
		client = self._client()
		request_format = _request_format(sampling_params)
		requested_max_tokens = int(sampling_params.get("max_tokens", 512))
		if request_format == REQUEST_FORMAT_CHAT_COMPLETIONS:
			max_tokens, truncate_prompt_tokens = _chat_request_budget(
				requested_max_tokens,
				self._served_max_model_len,
			)
		else:
			max_tokens, truncate_prompt_tokens = await _afit_request_to_context(
				model=self.key.model,
				prompt=prompt,
				requested_max_tokens=requested_max_tokens,
				served_max_model_len=self._served_max_model_len,
			)
		send_ts = time.perf_counter()
		first_token_ts: Optional[float] = None
		text_parts: List[str] = []
		n_output_tokens = 0
		create_kwargs = dict(
			model=self.key.model,
			temperature=float(sampling_params.get("temperature", 1.0)),
			max_tokens=max_tokens,
			stream=True,
			stream_options={"include_usage": True},
		)
		if request_format == REQUEST_FORMAT_CHAT_COMPLETIONS:
			create_kwargs["messages"] = [{"role": "user", "content": prompt}]
			create_kwargs["logprobs"] = True
			create_kwargs["n"] = 1
		else:
			create_kwargs["prompt"] = prompt
		if truncate_prompt_tokens is not None:
			create_kwargs["extra_body"] = {
				"truncate_prompt_tokens": truncate_prompt_tokens,
			}
			if request_format == REQUEST_FORMAT_CHAT_COMPLETIONS:
				create_kwargs["extra_body"]["truncation_side"] = "right"
		# Forward optional sampling controls when present. ``stop`` is REQUIRED
		# for agentic loops (ReAct halts each round at "Observation"); without it
		# the model runs to max_tokens and the Thought/Action/Observation parser
		# breaks. ``top_p`` is forwarded when the caller pins it.
		if sampling_params.get("stop"):
			create_kwargs["stop"] = sampling_params["stop"]
		if sampling_params.get("top_p") is not None:
			create_kwargs["top_p"] = float(sampling_params["top_p"])
		# The global chat contract applies the same model template and
		# assistant-generation boundary as quality and calibration.
		if request_format == REQUEST_FORMAT_CHAT_COMPLETIONS:
			stream = await client.chat.completions.create(**create_kwargs)
		else:
			stream = await client.completions.create(**create_kwargs)
		async for chunk in stream:
			if getattr(chunk, "usage", None) is not None and chunk.usage is not None:
				n_output_tokens = int(getattr(chunk.usage, "completion_tokens", 0) or 0)
			if not chunk.choices:
				continue
			if request_format == REQUEST_FORMAT_CHAT_COMPLETIONS:
				piece = getattr(getattr(chunk.choices[0], "delta", None), "content", None)
			else:
				piece = getattr(chunk.choices[0], "text", None)
			if piece:
				if first_token_ts is None:
					first_token_ts = time.perf_counter()
				text_parts.append(piece)
		last_token_ts = time.perf_counter()
		text = "".join(text_parts)
		# Fallback if usage chunk was missing: estimate tokens ≈ chars/4
		if n_output_tokens <= 0:
			n_output_tokens = max(len(text) // 4, 1)
		return text, {
			"request_send_ts": send_ts,
			"first_token_ts": first_token_ts,
			"last_token_ts": last_token_ts,
			"n_output_tokens": n_output_tokens,
		}

	async def generate_one(
		self,
		prompt: str,
		sampling_params: Dict[str, Any],
	) -> Tuple[str, Dict[str, Any]]:
		"""NON-streaming generate; returns (text, perf-dict) with the same shape as
		:meth:`generate_one_stream` but ``first_token_ts=None``.

		The measured ReAct loop can keep many closed-loop requests active, while
		each vLLM engine admits up to its configured ``max_num_seqs``. Token-by-
		token STREAMING makes the client process ~B×output_tokens SSE events; the
		single loop can't deliver them promptly, so the per-request token timing
		(and thus ``last_token_ts`` → e2e/tpot) is inflated ~3.6× — a pure harness
		artifact (the aggregate qps / tokens-per-s stay correct). Non-streaming
		yields ONE completion event per request, so the loop keeps up and
		``last_token_ts - request_send_ts`` is the ACCURATE per-request generate
		latency. Continuous batching is unaffected (it is server-side, independent
		of streaming). TTFT is sacrificed (no per-token signal) — acceptable when
		e2e is the target."""
		request_format = _request_format(sampling_params)
		requested_max_tokens = int(sampling_params.get("max_tokens", 512))
		if request_format == REQUEST_FORMAT_CHAT_COMPLETIONS:
			max_tokens, truncate_prompt_tokens = _chat_request_budget(
				requested_max_tokens,
				self._served_max_model_len,
			)
		else:
			max_tokens, truncate_prompt_tokens = await _afit_request_to_context(
				model=self.key.model,
				prompt=prompt,
				requested_max_tokens=requested_max_tokens,
				served_max_model_len=self._served_max_model_len,
			)
		send_ts = time.perf_counter()
		# RAW aiohttp JSON call (NOT the openai client): identical request
		# fields, ~3.2x cheaper on the driver loop at high call rates — see
		# _http_session. extra_body keys are top-level body keys in vLLM's
		# OpenAI-compatible server, so truncate_prompt_tokens moves inline.
		payload: Dict[str, Any] = {
			"model": self.key.model,
			"temperature": float(sampling_params.get("temperature", 1.0)),
			"max_tokens": max_tokens,
			"stream": False,
		}
		if request_format == REQUEST_FORMAT_CHAT_COMPLETIONS:
			payload["messages"] = [{"role": "user", "content": prompt}]
			payload["logprobs"] = True
			payload["n"] = 1
		else:
			payload["prompt"] = prompt
		if truncate_prompt_tokens is not None:
			payload["truncate_prompt_tokens"] = truncate_prompt_tokens
			if request_format == REQUEST_FORMAT_CHAT_COMPLETIONS:
				payload["truncation_side"] = "right"
		if sampling_params.get("stop"):
			payload["stop"] = sampling_params["stop"]
		if sampling_params.get("top_p") is not None:
			payload["top_p"] = float(sampling_params["top_p"])
		if sampling_params.get("ignore_eos") is not None:
			payload["ignore_eos"] = bool(sampling_params["ignore_eos"])
		data = await self._raw_completion(payload, request_format=request_format)
		done_ts = time.perf_counter()
		choices = data.get("choices") or []
		if request_format == REQUEST_FORMAT_CHAT_COMPLETIONS:
			message = (choices[0].get("message") or {}) if choices else {}
			text = message.get("content") or ""
		else:
			text = (choices[0].get("text") or "") if choices else ""
		usage = data.get("usage") or {}
		n_input_tokens = int(usage.get("prompt_tokens") or 0)
		n_output_tokens = int(usage.get("completion_tokens") or 0)
		if n_output_tokens <= 0:
			n_output_tokens = max(len(text) // 4, 1)
		return text, {
			"request_send_ts": send_ts,
			"first_token_ts": None,
			"last_token_ts": done_ts,
			"n_input_tokens": n_input_tokens,
			"n_output_tokens": n_output_tokens,
		}

	async def generate_batch_streaming(
		self,
		prompts: List[str],
		sampling_params: Dict[str, Any],
	) -> Tuple[List[str], List[Dict[str, Any]]]:
		"""Stream prompts with bounded client-side concurrency.

		vLLM still performs continuous batching inside each window; this prevents
		the client from opening more simultaneous streams than the configured
		batching capacity.
		"""
		limit = _stream_concurrency_limit(
			len(prompts),
			default_limit=self.key.max_num_seqs,
		)
		if limit < len(prompts):
			logger.info(
				f"vLLM streaming concurrency limited: prompts={len(prompts)} "
				f"concurrency={limit} (max_num_seqs={self.key.max_num_seqs})"
			)
		sem = asyncio.Semaphore(limit)
		results: List[Optional[Tuple[str, Dict[str, Any]]]] = [None] * len(prompts)

		async def run_one(idx: int, prompt: str) -> None:
			async with sem:
				results[idx] = await self.generate_one_stream(prompt, sampling_params)

		await asyncio.gather(*(run_one(i, p) for i, p in enumerate(prompts)))
		assert all(r is not None for r in results)
		results = [r for r in results if r is not None]
		return [r[0] for r in results], [r[1] for r in results]

	async def generate_batch(
		self,
		prompts: List[str],
		sampling_params: Dict[str, Any],
	) -> Tuple[List[str], List[Dict[str, Any]]]:
		"""NON-streaming batch generate — the qps-accurate counterpart of
		:meth:`generate_batch_streaming`.

		Token-by-token SSE streaming makes ONE asyncio loop process
		``B × output_tokens`` events; at e2e batch sizes (B=32, ~370 tok →
		~12k events) the Python event loop — not the GPU — becomes the
		bottleneck, inflating the generator stage wall-clock ~3.6–4.5× and
		thus DEPRESSING measured qps by the same factor (the CM, GenZ-roofline,
		predicts the true GPU throughput). Issuing one NON-streaming completion
		per request (``generate_one``) means the loop handles only ``B`` events,
		so the stage wall-clock reflects real server-side continuous-batching
		throughput. Continuous batching is server-side and identical either way.
		TTFT is sacrificed (no per-token signal → ``first_token_ts=None``);
		acceptable when e2e qps/latency is the target (see ``generate_one``)."""
		limit = _stream_concurrency_limit(
			len(prompts),
			default_limit=self.key.max_num_seqs,
		)
		if limit < len(prompts):
			logger.info(
				f"vLLM batch concurrency limited: prompts={len(prompts)} "
				f"concurrency={limit} (max_num_seqs={self.key.max_num_seqs})"
			)
		sem = asyncio.Semaphore(limit)
		results: List[Optional[Tuple[str, Dict[str, Any]]]] = [None] * len(prompts)

		async def run_one(idx: int, prompt: str) -> None:
			async with sem:
				results[idx] = await self.generate_one(prompt, sampling_params)

		await asyncio.gather(*(run_one(i, p) for i, p in enumerate(prompts)))
		assert all(r is not None for r in results)
		results = [r for r in results if r is not None]
		return [r[0] for r in results], [r[1] for r in results]
