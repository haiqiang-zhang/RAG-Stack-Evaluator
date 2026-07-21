"""1P1D (one-prefill-one-decode) vLLM disaggregation via **NixlConnector**.

Spawns two vLLM subprocesses:

  - a prefill vLLM  (``--kv-transfer-config kv_role=kv_producer``)
  - a decode  vLLM  (``--kv-transfer-config kv_role=kv_consumer``)

KV is transferred prefill→decode by vLLM's NixlConnector (UCX backend). There is
**no proxy process**: a request is run in two steps over plain HTTP, exactly like
vLLM's ``nixl_integration`` test —

  1. POST prefill ``/v1/chat/completions`` with
     ``kv_transfer_params={"do_remote_decode": True}`` and ``max_tokens=1``; the
     prefill engine computes+registers the KV and returns a filled
     ``kv_transfer_params`` (remote_engine_id / remote_block_ids / remote_host /
     remote_port / remote_request_id).
  2. POST decode with that ``kv_transfer_params`` (now ``do_remote_prefill=True``);
     the decode engine pulls the KV via nixl and generates.

Why NixlConnector and not P2pNcclConnector: on consumer GPUs (RTX 3090: no NVLink,
PCIe P2P disabled) the raw cross-process NCCL send/recv used by P2pNcclConnector
hangs (comm inits, but the data collective never completes). nixl/UCX falls back to
a working transport (CUDA-IPC / host staging) and completes the transfer. See
memory ``disagg-nixl-vs-p2pnccl-3090``.

Environment (driver-560 / cu12 box): needs the **cu12** nixl backend
(``nixl`` meta + ``nixl-cu12``, NOT ``nixl-cu13``) and vllm ≤ 0.18.1; do NOT put the
cu13 runtime libs on ``LD_LIBRARY_PATH``.

Usage:

    pair = VllmPdPair(VllmPdPairKey(
        model="Qwen/Qwen2.5-3B-Instruct",
        prefill_device="cuda:2", decode_device="cuda:3",
        gpu_memory_utilization=0.7, dtype="bfloat16",
    ))
    texts, perfs = await pair.generate_batch(prompts, sampling_params)
    pair.shutdown()
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from rag_stack_evaluator.static_rag_evaluator.measured.vllm_subprocess import (
    MEASURED_REQUEST_FORMAT_KEY,
    REQUEST_FORMAT_CHAT_COMPLETIONS,
    _DISABLE_CUSTOM_ALL_REDUCE_ENV,
    _TP2_DISABLE_DIRECT_P2P_ENV,
    _chat_request_budget,
    _fit_request_to_context,
    _health_timeout_s,
    _kill_proc_group,
    _openai_stream_timeout,
    _popen_with_output_tee,
    _prefix_caching_args,
    _resolve_served_max_model_len,
    _stream_concurrency_limit,
    _stream_timeout_s,
    _wait_for_health,
    _request_format,
)
from rag_stack_evaluator.static_rag_evaluator.measured.vllm_env import (
    configure_vllm_worker_env,
)

logger = logging.getLogger("RAG-Stack")

_GIB = 1024 ** 3
_PD_MAX_CONCURRENCY_ENV = "RAG_STACK_VLLM_PD_MAX_CONCURRENCY"
# Keep exactly one request queued immediately in front of the prefill engine.
# This is feed-ahead, not extra engine capacity: it hides frontend/HTTP bubbles
# while bounding producer handles to at most engine max_num_seqs + one.
_PD_PREFILL_FEEDER_SLOTS = 1
# Health/startup budget per engine (model load + KV profiling + graph capture).
_HEALTH_TIMEOUT_S = 600.0
# NIXL binds its side channel only after model load/compile. A port returned by
# bind(("127.0.0.1", 0)) is in Linux's ephemeral range and can be reused by an
# outbound connection during that long gap. Keep side channels outside the
# kernel's configured ephemeral range instead.
_DEFAULT_EPHEMERAL_PORT_RANGE = (32_768, 60_999)
# vLLM 0.18.1's local multiprocess executor selects its torch-distributed
# rendezvous with ``get_open_port()`` and closes the probe socket before its TP
# rank 0 creates TCPStore.  P and D start concurrently and open many loopback
# connections during that gap, so a kernel-ephemeral rendezvous port can be
# reused before the delayed bind.  Use the same non-ephemeral policy as NIXL's
# delayed side-channel bind, with parent-process claims shared by both uses.
_VLLM_RENDEZVOUS_PORT_BLOCK_SIZE = 8
_claimed_delayed_bind_ports: set[int] = set()
_delayed_bind_port_lock = threading.Lock()


def _nixl_side_port_candidates() -> List[int]:
    ephemeral_start, ephemeral_end = _DEFAULT_EPHEMERAL_PORT_RANGE
    try:
        with open(
            "/proc/sys/net/ipv4/ip_local_port_range", encoding="ascii"
        ) as range_file:
            ephemeral_start, ephemeral_end = map(int, range_file.read().split())
    except (OSError, ValueError):
        pass

    ranges: List[tuple[int, int]] = []
    if ephemeral_end < 65_535:
        ranges.append((ephemeral_end + 1, 65_535))
    # Avoid the conventional low service-port area even when it is outside the
    # ephemeral interval.
    if ephemeral_start > 10_000:
        ranges.append((10_000, ephemeral_start - 1))
    return [port for start, end in ranges for port in range(start, end + 1)]


def _claim_free_delayed_bind_port_block(count: int = 1) -> tuple[int, ...]:
    """Claim one contiguous non-ephemeral block for a delayed child bind.

    vLLM 0.18.1's ``MultiprocExecutor._init_executor`` has exactly one
    ``get_open_port()`` call for this local executor's rendezvous, and the
    NIXL path has no other local-executor call consuming ``VLLM_PORT``.  Eight
    ports are nevertheless claimed so a future/internal upward scan from that
    base still cannot enter the sibling P/D engine's range.
    """
    if count < 1:
        raise ValueError("delayed-bind port block size must be positive")
    candidates = _nixl_side_port_candidates()
    if not candidates:
        raise RuntimeError("No unprivileged TCP ports exist outside the ephemeral range")
    candidate_set = set(candidates)
    first_offset = os.getpid() % len(candidates)
    with _delayed_bind_port_lock:
        for step in range(len(candidates)):
            first_port = candidates[(first_offset + step) % len(candidates)]
            block = tuple(range(first_port, first_port + count))
            if any(port not in candidate_set for port in block):
                continue
            if any(port in _claimed_delayed_bind_ports for port in block):
                continue
            available = True
            for port in block:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    try:
                        # Match vLLM's own `_get_open_port` wildcard probe.
                        # A listener bound on another local interface must also
                        # make this candidate unavailable to TCPStore.
                        sock.bind(("", port))
                    except OSError:
                        available = False
                        break
            if not available:
                continue
            # Prevent another pair in this parent process from selecting the
            # same range before either vLLM child reaches its delayed bind.
            _claimed_delayed_bind_ports.update(block)
            return block
    raise RuntimeError(
        f"No free delayed-bind TCP port block of size {count} exists outside "
        "the ephemeral range"
    )


def _find_free_nixl_side_port() -> int:
    """Return a distinct claimed listener port outside the ephemeral range."""
    return _claim_free_delayed_bind_port_block()[0]


@dataclass(frozen=True)
class VllmPdPairKey:
    """Identifying tuple for a 1P1D PD pair.

    Two requests with the same key share a pair; differing keys require teardown
    + relaunch of both vllm subprocesses.
    """

    model: str
    prefill_device: str = "cuda:0"
    decode_device: str = "cuda:1"
    max_num_seqs: int = 64
    prefill_max_num_seqs: int | None = None
    decode_max_num_seqs: int | None = None
    max_num_batched_tokens: "int | None" = None   # None ⇒ let vLLM default (see VllmSubprocessKey)
    enable_prefix_caching: "bool | None" = None
    max_model_len: int = -1
    kv_cache_dtype: str = "auto"
    gpu_memory_utilization: float = 0.85
    prefill_tensor_parallel_size: int = 1
    prefill_pipeline_parallel_size: int = 1
    decode_tensor_parallel_size: int = 1
    decode_pipeline_parallel_size: int = 1
    dtype: str = "bfloat16"


@dataclass(frozen=True)
class VllmPdRoleTelemetry:
    """One calibration-owned role logger identity.

    This is launch provenance, not a measured deployment or search-space knob.
    Normal measured pairs never construct it.
    """

    path: str
    run_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path.strip():
            raise ValueError("PD role telemetry path must be non-empty")
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("PD role telemetry run_id must be non-empty")


@dataclass(frozen=True)
class VllmPdCalibrationTelemetry:
    """Role-separated direct-disaggregation telemetry launch contract."""

    prefill: VllmPdRoleTelemetry
    decode: VllmPdRoleTelemetry

    def __post_init__(self) -> None:
        if self.prefill.path == self.decode.path:
            raise ValueError("P/D telemetry paths must be distinct")
        if self.prefill.run_id == self.decode.run_id:
            raise ValueError("P/D telemetry run_ids must be distinct")

    def for_role(self, role: str) -> VllmPdRoleTelemetry:
        if role == "prefill":
            return self.prefill
        if role == "decode":
            return self.decode
        raise ValueError(f"unknown PD telemetry role {role!r}")


def _resolve_devices(device_str: str) -> List[str]:
    """Split 'cuda:2' or 'cuda:2,cuda:3' into ['2','3'] cuda ids."""
    out: List[str] = []
    for d in device_str.split(","):
        d = d.strip()
        if not d:
            continue
        out.append(d[len("cuda:") :] if d.startswith("cuda:") else d)
    return out or ["0"]


def _role_max_num_seqs(key: VllmPdPairKey, role: str) -> int:
    role_specific = getattr(key, f"{role}_max_num_seqs", None)
    if role_specific is not None:
        return max(1, int(role_specific))
    return max(1, int(getattr(key, "max_num_seqs", 64)))


def _pd_role_engine_limits(key: VllmPdPairKey) -> Dict[str, int]:
    """Role-local engine residency configured at vLLM launch."""
    return {
        "prefill": _role_max_num_seqs(key, "prefill"),
        "decode": _role_max_num_seqs(key, "decode"),
    }


def pd_prefill_feeder_slots() -> int:
    """Runtime-policy feed-ahead depth shared with replay fingerprints."""
    return _PD_PREFILL_FEEDER_SLOTS


def _pd_role_admission_limits(key: VllmPdPairKey) -> Dict[str, int]:
    """Role-local client admission for one P/D pair.

    Unlike a unified frontend, a NIXL producer keeps its KV blocks pinned after
    the prefill response until the decode side fetches them.  A single prefill
    feed-ahead slot keeps the next HTTP request ready while the engine's resident
    sequences finish; the P permit remains held through D admission, so even a
    blocked decode can create at most one handle beyond the P engine cap.

    Decode remains strict. Reusing the generic 20% HTTP slack there could admit
    dozens of additional handles before the consumer fetches them.
    """
    engine_limits = _pd_role_engine_limits(key)
    return {
        "prefill": engine_limits["prefill"] + pd_prefill_feeder_slots(),
        "decode": engine_limits["decode"],
    }


def _pd_stage_admission_limit(key: VllmPdPairKey) -> int:
    """Maximum client P/D work in progress across both role-local gates."""
    return sum(_pd_role_admission_limits(key).values())


@dataclass
class _PdRoleAdmissionState:
    role: str
    engine_max_num_seqs: int
    admission_limit: int
    semaphore: asyncio.Semaphore
    submitted: int = 0
    acquired: int = 0
    released: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    inflight: int = 0
    waiting: int = 0
    max_inflight_observed: int = 0
    max_waiting_observed: int = 0
    total_queue_wait_s: float = 0.0
    total_residency_s: float = 0.0
    total_handoff_wait_s: float = 0.0
    window_start: Optional[Dict[str, float]] = None
    window_end: Optional[Dict[str, float]] = None
    window_max_inflight_observed: int = 0
    window_max_waiting_observed: int = 0

    def counters(self) -> Dict[str, float]:
        return {
            "submitted": float(self.submitted),
            "acquired": float(self.acquired),
            "released": float(self.released),
            "completed": float(self.completed),
            "failed": float(self.failed),
            "cancelled": float(self.cancelled),
            "total_queue_wait_s": float(self.total_queue_wait_s),
            "total_residency_s": float(self.total_residency_s),
            "total_handoff_wait_s": float(self.total_handoff_wait_s),
        }

    def mark_window_start(self) -> None:
        self.window_start = self.counters()
        self.window_end = None
        self.window_max_inflight_observed = self.inflight
        self.window_max_waiting_observed = self.waiting

    def mark_window_end(self) -> None:
        if self.window_start is not None and self.window_end is None:
            self.window_end = self.counters()

    def windowed_counters(self) -> Dict[str, float]:
        if self.window_start is None:
            return self.counters()
        end = self.window_end or self.counters()
        return {
            key: value - self.window_start.get(key, 0.0)
            for key, value in end.items()
        }


@dataclass
class _PdRoleAdmissionLease:
    state: _PdRoleAdmissionState
    acquired_ts: float
    released: bool = False


def _empty_remote_params() -> dict:
    """kv_transfer_params the prefill request carries: 'I will be decoded remotely'."""
    return {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }


class VllmPdPair:
    """Manages a 1P1D vLLM pair (NixlConnector). No proxy process.

    The Vllm generator node treats this like a ``VllmSubprocess`` (duck-typed):
    it uses ``.base_url`` (cosmetic, points at the decode engine) and
    ``generate_batch_streaming`` / ``generate_one_stream``. Lifetimes are owned
    here — ``shutdown()`` tears down both subprocesses.
    """

    def __init__(
        self,
        key: VllmPdPairKey,
        *,
        calibration_telemetry: VllmPdCalibrationTelemetry | None = None,
    ):
        self.key = key
        self.calibration_telemetry = calibration_telemetry
        # Both API ports are selected before either child is launched, while
        # each bind(0)-then-close probe immediately releases its result.  Two
        # independent probes can therefore return the same ephemeral port;
        # vLLM enables SO_REUSEPORT, so P and D then both start successfully on
        # that one port and the kernel load-balances requests across the wrong
        # roles.  Claim the API pair atomically from the same
        # process-lifetime, non-ephemeral allocator used for the other delayed
        # child binds.  This makes role identity a construction invariant, not
        # a probabilistic startup outcome.
        self.api_ports = _claim_free_delayed_bind_port_block(2)
        self.prefill_port, self.decode_port = self.api_ports
        if self.prefill_port == self.decode_port:
            raise RuntimeError("P/D API listener ports must be distinct")
        # ``VLLM_PORT`` is the supported vLLM base for internal port scans.
        # Keep bounded disjoint blocks for P and D; the first port is the
        # current local multiprocess executor's torch rendezvous.
        self.prefill_rendezvous_ports = _claim_free_delayed_bind_port_block(
            _VLLM_RENDEZVOUS_PORT_BLOCK_SIZE
        )
        self.decode_rendezvous_ports = _claim_free_delayed_bind_port_block(
            _VLLM_RENDEZVOUS_PORT_BLOCK_SIZE
        )
        self.prefill_rendezvous_port = self.prefill_rendezvous_ports[0]
        self.decode_rendezvous_port = self.decode_rendezvous_ports[0]
        # Per-engine nixl side-channel ports (KV-handle metadata exchange). Must
        # differ since both bind on 127.0.0.1. Unlike the API ports, these bind
        # only after model startup, so keep them outside the ephemeral range.
        self.prefill_side_port = _find_free_nixl_side_port()
        self.decode_side_port = _find_free_nixl_side_port()
        # Cosmetic: the node logs this. Actual generation goes through our
        # two-step methods, not a single OpenAI endpoint.
        self.base_url = f"http://127.0.0.1:{self.decode_port}/v1"

        self.prefill_proc: Optional[subprocess.Popen] = None
        self.decode_proc: Optional[subprocess.Popen] = None
        self._http_by_loop = {}
        # A cached pair can be used by more than one ``asyncio.run`` over its
        # lifetime.  Semaphores bind to the loop once contended, so keep one
        # independent role-gate set per concrete loop object (not merely a
        # process-global pair of semaphores).
        self._role_admission_by_loop: Dict[
            int, tuple[asyncio.AbstractEventLoop, Dict[str, _PdRoleAdmissionState]]
        ] = {}
        self._last_role_admission_stats: Optional[Dict[str, Dict[str, Any]]] = None
        self._served_max_model_len: Optional[int] = None

        try:
            self._launch_engine("prefill")
            self._launch_engine("decode")
            health_timeout_s = _health_timeout_s(_HEALTH_TIMEOUT_S)
            _wait_for_health(f"http://127.0.0.1:{self.prefill_port}", timeout_s=health_timeout_s, proc=self.prefill_proc)
            _wait_for_health(f"http://127.0.0.1:{self.decode_port}", timeout_s=health_timeout_s, proc=self.decode_proc)
            self._served_max_model_len = _resolve_served_max_model_len(
                self.base_url,
                self.key.model,
                self.key.max_model_len,
            )
        except Exception:
            self.shutdown()
            raise

        logger.info(
            f"VllmPdPair (Nixl) ready: prefill={self.prefill_port}@{key.prefill_device} "
            f"(side {self.prefill_side_port}), decode={self.decode_port}@{key.decode_device} "
            f"(side {self.decode_side_port}); engine caps "
            f"P={_role_max_num_seqs(key, 'prefill')}, "
            f"D={_role_max_num_seqs(key, 'decode')}; client admission "
            f"P={_pd_role_admission_limits(key)['prefill']}, "
            f"D={_pd_role_admission_limits(key)['decode']}, "
            f"outer={_pd_stage_admission_limit(key)}"
        )

    # --- subprocess launcher -----------------------------------------

    def _launch_engine(self, role: str) -> None:
        is_prefill = role == "prefill"
        device = self.key.prefill_device if is_prefill else self.key.decode_device
        port = self.prefill_port if is_prefill else self.decode_port
        side_port = self.prefill_side_port if is_prefill else self.decode_side_port
        tp = (
            self.key.prefill_tensor_parallel_size
            if is_prefill
            else self.key.decode_tensor_parallel_size
        )
        pp = (
            self.key.prefill_pipeline_parallel_size
            if is_prefill
            else self.key.decode_pipeline_parallel_size
        )
        kv_role = "kv_producer" if is_prefill else "kv_consumer"
        max_num_seqs = _role_max_num_seqs(self.key, role)
        cuda_ids = _resolve_devices(device)
        expected_chips = tp * pp
        if len(cuda_ids) != expected_chips:
            raise ValueError(
                f"VllmPdPair {role}: device={device!r} resolves to {len(cuda_ids)} GPU(s) "
                f"but TP*PP={expected_chips}. Pass exactly that many GPUs."
            )

        env = os.environ.copy()
        configure_vllm_worker_env(env=env, logger=logger)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(cuda_ids)
        rendezvous_port = (
            self.prefill_rendezvous_port
            if is_prefill
            else self.decode_rendezvous_port
        )
        # Avoid vLLM's probe-close-delayed-bind race on a random ephemeral port.
        # Each P/D child gets a distinct non-ephemeral claimed scan range.
        env["VLLM_PORT"] = str(rendezvous_port)
        role_telemetry = (
            None
            if self.calibration_telemetry is None
            else self.calibration_telemetry.for_role(role)
        )
        if role_telemetry is not None:
            # The same custom wrapper used by standalone stage calibration
            # installs one exact scheduler-cycle logger in this role's API
            # process.  P and D use distinct paths/run IDs so their evidence
            # can never be joined by filename or launch order guesses.
            env["RAG_STACK_STAGE_TELEMETRY_PATH"] = role_telemetry.path
            env["RAG_STACK_VLLM_CALIBRATION_RUN_ID"] = role_telemetry.run_id
            env["VLLM_SERVER_DEV_MODE"] = "1"
            env["VLLM_DEBUG_MFU_METRICS"] = "1"
            # The wrapper installs a Prometheus subclass which retains only
            # scheduler/NIXL metrics.  This prevents vLLM's known negative PD
            # finished-request token statistic from aborting B256 requests.
            env["RAG_STACK_PD_CALIBRATION_SAFE_PROMETHEUS"] = "1"
        tp2_direct_p2p_disabled = (
            tp == 2
            and env.get(_TP2_DISABLE_DIRECT_P2P_ENV, "").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        if tp2_direct_p2p_disabled:
            # A collocated case may contain non-TP2 engines. Apply the NCCL
            # workaround only to this affected PD role's child environment.
            env["NCCL_P2P_DISABLE"] = "1"
        # NixlConnector side channel (KV-handle metadata). Loopback on one node.
        env["VLLM_NIXL_SIDE_CHANNEL_HOST"] = "127.0.0.1"
        env["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(side_port)
        # UCX transport selection + NHD/HND layout required by the connector.
        env.setdefault("UCX_NET_DEVICES", "all")
        env["VLLM_KV_CACHE_LAYOUT"] = "HND"
        # Hybrid SSM models such as Qwen3.6 require DS conv state layout for
        # NixlConnector's 3-read Mamba state transfer.
        env.setdefault("VLLM_SSM_CONV_STATE_LAYOUT", "DS")

        # Size util to memory ACTUALLY free now (eviction-lag wait + co-tenant
        # downscale) — avoids phantom "free < desired utilization" refusals.
        from rag_stack_evaluator.static_rag_evaluator.measured.gpu_mem import effective_util
        eff_util = effective_util(
            cuda_ids, float(self.key.gpu_memory_utilization), model=self.key.model
        )

        kv_cfg = json.dumps({"kv_connector": "NixlConnector", "kv_role": kv_role})
        server_module = (
            "rag_stack_evaluator.vllm_instrumentation.serving_curves.calibration.vllm_server"
            if role_telemetry is not None
            else "vllm.entrypoints.openai.api_server"
        )
        cmd = [
            sys.executable, "-m", server_module,
            "--model", self.key.model,
            "--host", "127.0.0.1",
            "--port", str(port),
            # Keep per-request uvicorn I/O out of the measured service path.
            "--disable-uvicorn-access-log",
            # CUDA graphs ON (vLLM default). Historically this launched with
            # --enforce-eager, copied from vLLM's upstream nixl_integration
            # test recipe — but the NixlConnector has no eager requirement,
            # and the A/B on this box (2026-06-10, Qwen2.5-1.5B, 16 streams)
            # showed graphs are stable through the PD pair with IDENTICAL
            # greedy outputs and 2.2x faster decode (TPOT 12.7 -> 5.8 ms,
            # matching the unified engine). Only cost: ~9s extra launch time
            # for graph capture, outside the timed window.
            "--max-model-len", str(self.key.max_model_len),
            "--max-num-seqs", str(max_num_seqs),
            "--kv-cache-dtype", self.key.kv_cache_dtype,
            "--gpu-memory-utilization", str(eff_util),
            "--tensor-parallel-size", str(tp),
            "--pipeline-parallel-size", str(pp),
            "--dtype", self.key.dtype,
            "--kv-transfer-config", kv_cfg,
            *_prefix_caching_args(self.key),
            # MFU debug context is the exact scheduler-shape source consumed
            # by direct calibration.  Default loggers remain disabled for all
            # PD engines.  In calibration the wrapper's custom telemetry and
            # scheduler-only Prometheus logger still run, preserving NIXL
            # evidence without the unsafe request/iteration histograms.
            *(["--enable-mfu-metrics"] if role_telemetry is not None else []),
            "--disable-log-stats",
        ]
        if tp2_direct_p2p_disabled or env.get(
            _DISABLE_CUSTOM_ALL_REDUCE_ENV, ""
        ).strip().lower() in {"1", "true", "yes", "on"}:
            # Each PD role can still be tensor-parallel. Keep this opt-in so
            # systems with working direct P2P retain vLLM's normal backend.
            cmd.append("--disable-custom-all-reduce")
        if self.key.max_num_batched_tokens is not None:
            cmd += ["--max-num-batched-tokens", str(self.key.max_num_batched_tokens)]
        if tp2_direct_p2p_disabled:
            logger.info(
                "vLLM PD %s TP2 direct-P2P workaround on %s: "
                "NCCL_P2P_DISABLE=1, custom all-reduce disabled",
                role,
                device,
            )
        logger.info(
            f"Launching {role} vLLM (Nixl) on {device} (port={port}, "
            f"kv_role={kv_role}, side_port={side_port}, "
            f"rendezvous_base_port={rendezvous_port}, "
            f"max_num_seqs={max_num_seqs}, util={eff_util:.3f})"
        )
        proc = _popen_with_output_tee(
            cmd,
            env=env,
            label=f"pd-{role}:{device}:{port}",
            capture_tail=True,
        )
        if is_prefill:
            self.prefill_proc = proc
        else:
            self.decode_proc = proc

    # --- generation (two-step prefill→decode handoff) ----------------


    def _http_session(self):
        """Loop-keyed shared aiohttp session for RAW prefill/decode calls.

        Replaces the per-call httpx.AsyncClient + per-call AsyncOpenAI decode
        client: at high call rates the openai client's pydantic layer costs
        ~3.2x driver-loop throughput (measured 07-08: raw 928 vs openai 287
        calls/s on the same engine), and per-call client churn is pure waste.
        Sessions bind to their creation loop (one asyncio.run per trial) —
        key by loop identity; connector limit=0 (admission gated upstream).
        """
        import aiohttp
        import asyncio as _asyncio
        loop = _asyncio.get_running_loop()
        sess = self._http_by_loop.get(id(loop))
        if sess is None or sess.closed:
            sess = aiohttp.ClientSession(
                # keepalive_timeout below the server's 5s default — see
                # vllm_subprocess._http_session: idle pooled sockets must be
                # discarded client-side before uvicorn closes them under us.
                connector=aiohttp.TCPConnector(limit=0, keepalive_timeout=4.0),
                timeout=aiohttp.ClientTimeout(total=_stream_timeout_s()),
            )
            self._http_by_loop[id(loop)] = sess
        return sess

    def role_engine_limits(self) -> Dict[str, int]:
        return _pd_role_engine_limits(self.key)

    def role_base_url(self, role: str) -> str:
        """Canonical role URL used by calibration pause/metrics RPCs."""

        if role == "prefill":
            port = self.prefill_port
        elif role == "decode":
            port = self.decode_port
        else:
            raise ValueError(f"unknown PD role {role!r}")
        return f"http://127.0.0.1:{port}/v1"

    def role_admission_limits(self) -> Dict[str, int]:
        return _pd_role_admission_limits(self.key)

    def stage_admission_limit(self) -> int:
        return _pd_stage_admission_limit(self.key)

    def _role_admission_states(self) -> Dict[str, _PdRoleAdmissionState]:
        loop = asyncio.get_running_loop()
        by_loop = getattr(self, "_role_admission_by_loop", None)
        if by_loop is None:
            # Defensive for lightweight ``object.__new__`` tests and old cached
            # objects restored inside a long-lived Python process.
            by_loop = {}
            self._role_admission_by_loop = by_loop
        loop_id = id(loop)
        entry = by_loop.get(loop_id)
        if entry is not None and entry[0] is loop:
            return entry[1]

        engine_limits = self.role_engine_limits()
        admission_limits = self.role_admission_limits()
        states = {
            role: _PdRoleAdmissionState(
                role=role,
                engine_max_num_seqs=engine_limits[role],
                admission_limit=admission_limit,
                semaphore=asyncio.Semaphore(admission_limit),
            )
            for role, admission_limit in admission_limits.items()
        }
        by_loop[loop_id] = (loop, states)
        return states

    async def _acquire_role(self, role: str) -> _PdRoleAdmissionLease:
        states = self._role_admission_states()
        if role not in states:
            raise ValueError(f"unknown PD role {role!r}")
        state = states[role]
        state.submitted += 1
        state.waiting += 1
        state.max_waiting_observed = max(
            state.max_waiting_observed, state.waiting
        )
        if state.window_start is not None and state.window_end is None:
            state.window_max_waiting_observed = max(
                state.window_max_waiting_observed, state.waiting
            )
        queued_ts = time.perf_counter()
        try:
            await state.semaphore.acquire()
        except asyncio.CancelledError:
            state.cancelled += 1
            raise
        except BaseException:
            state.failed += 1
            raise
        finally:
            state.waiting -= 1

        acquired_ts = time.perf_counter()
        state.acquired += 1
        state.total_queue_wait_s += acquired_ts - queued_ts
        state.inflight += 1
        state.max_inflight_observed = max(
            state.max_inflight_observed, state.inflight
        )
        if state.window_start is not None and state.window_end is None:
            state.window_max_inflight_observed = max(
                state.window_max_inflight_observed, state.inflight
            )
        return _PdRoleAdmissionLease(state=state, acquired_ts=acquired_ts)

    @staticmethod
    def _release_role(
        lease: Optional[_PdRoleAdmissionLease],
        *,
        failed: bool = False,
        cancelled: bool = False,
        handoff_wait_s: float = 0.0,
    ) -> None:
        if lease is None or lease.released:
            return
        lease.released = True
        state = lease.state
        state.inflight -= 1
        state.released += 1
        if cancelled:
            state.cancelled += 1
        elif failed:
            state.failed += 1
        else:
            state.completed += 1
        state.total_residency_s += time.perf_counter() - lease.acquired_ts
        state.total_handoff_wait_s += max(0.0, float(handoff_wait_s))
        state.semaphore.release()

    def mark_role_admission_window_start(self) -> None:
        for state in self._role_admission_states().values():
            state.mark_window_start()

    def mark_role_admission_window_end(self) -> None:
        for state in self._role_admission_states().values():
            state.mark_window_end()

    @staticmethod
    def _role_admission_stats_from_states(
        states: Dict[str, _PdRoleAdmissionState],
    ) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for role, state in states.items():
            counters = state.windowed_counters()
            acquired = counters["acquired"]
            released = counters["released"]
            completed = counters["completed"]
            out[role] = {
                "engine_max_num_seqs": state.engine_max_num_seqs,
                "admission_limit": state.admission_limit,
                "submitted": counters["submitted"],
                "acquired": acquired,
                "released": released,
                "completed": completed,
                "failed": counters["failed"],
                "cancelled": counters["cancelled"],
                "current_inflight": state.inflight,
                "current_waiting": state.waiting,
                "max_inflight_observed": (
                    state.window_max_inflight_observed
                    if state.window_start is not None
                    else state.max_inflight_observed
                ),
                "max_waiting_observed": (
                    state.window_max_waiting_observed
                    if state.window_start is not None
                    else state.max_waiting_observed
                ),
                "avg_queue_wait_s": (
                    counters["total_queue_wait_s"] / acquired
                    if acquired else 0.0
                ),
                "avg_residency_s": (
                    counters["total_residency_s"] / released
                    if released else 0.0
                ),
                "avg_handoff_wait_s": (
                    counters["total_handoff_wait_s"] / completed
                    if completed else 0.0
                ),
            }
        return out

    def role_admission_stats(self) -> Dict[str, Dict[str, Any]]:
        """Return current-loop stats, or the snapshot frozen during close.

        ``MeasuredServingRuntime`` closes HTTP sessions before annotating its
        summary.  Preserve the just-closed loop's role counters while still
        dropping loop-bound semaphores so a cached pair can be reused safely by
        a later ``asyncio.run``.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        by_loop = getattr(self, "_role_admission_by_loop", None) or {}
        if loop is not None:
            entry = by_loop.get(id(loop))
            if entry is not None and entry[0] is loop:
                return self._role_admission_stats_from_states(entry[1])
        return dict(getattr(self, "_last_role_admission_stats", None) or {})

    async def aclose_http(self):
        import asyncio as _asyncio
        loop = _asyncio.get_running_loop()
        sess = self._http_by_loop.pop(id(loop), None)
        if sess is not None and not sess.closed:
            await sess.close()
        by_loop = getattr(self, "_role_admission_by_loop", None)
        if by_loop is not None:
            entry = by_loop.get(id(loop))
            if entry is not None and entry[0] is loop:
                self._last_role_admission_stats = (
                    self._role_admission_stats_from_states(entry[1])
                )
                by_loop.pop(id(loop), None)

    async def _raw_post(
        self,
        port: int,
        payload: dict,
        request_id: str,
        *,
        request_format: str = REQUEST_FORMAT_CHAT_COMPLETIONS,
    ) -> dict:
        """POST chat-completions with body-preserving errors.

        The request format must match on both sides of a PD handoff: changing
        chat rendering between prefill and decode would make the KV token
        sequence invalid. One immediate retry handles a stale keep-alive
        connection, matching ``VllmSubprocess._raw_completion``.
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
                    f"http://127.0.0.1:{port}/v1/{endpoint}",
                    json=payload,
                    headers={"X-Request-Id": request_id},
                ) as r:
                    if r.status >= 400:
                        body = (await r.text())[:2000]
                        raise RuntimeError(
                            f"1P1D request failed (HTTP {r.status}, "
                            f"request_id={request_id}): {body}"
                        )
                    return await r.json()
            except (aiohttp.ServerDisconnectedError, aiohttp.ClientOSError) as e:
                if attempt == 1:
                    raise
                logger.warning(
                    f"1P1D {endpoint}: stale connection ({type(e).__name__}: {e}); "
                    f"retrying once on a fresh socket"
                )
        raise RuntimeError("unreachable")  # loop always returns or raises

    def _decode_client(self):
        """Fresh AsyncOpenAI bound to the CURRENT event loop (the node runs each
        batch under its own ``asyncio.run``, so we must NOT cache across loops)."""
        from openai import AsyncOpenAI
        return AsyncOpenAI(
            base_url=f"http://127.0.0.1:{self.decode_port}/v1",
            api_key="EMPTY",
            timeout=_openai_stream_timeout(),
        )

    async def _gen_one(
        self,
        prompt,
        sampling_params,
        http: httpx.AsyncClient,
        oai,
        stream: bool = False,
    ):
        temperature = float(sampling_params.get("temperature", 1.0))
        request_format = _request_format(sampling_params)
        requested_max_tokens = int(sampling_params.get("max_tokens", 512))
        if request_format == REQUEST_FORMAT_CHAT_COMPLETIONS:
            max_tokens, truncate_prompt_tokens = _chat_request_budget(
                requested_max_tokens,
                self._served_max_model_len,
            )
        else:
            max_tokens, truncate_prompt_tokens = _fit_request_to_context(
                model=self.key.model,
                prompt=prompt,
                requested_max_tokens=requested_max_tokens,
                served_max_model_len=self._served_max_model_len,
            )
        request_id = f"pd-{uuid.uuid4().hex}"
        send_ts = time.perf_counter()

        # Step 1: prefill computes/registers KV and returns the remote handle.
        # Chat mode must use the same rendered messages here and on decode.
        prefill_body: Dict[str, Any] = {
            "model": self.key.model,
            "max_tokens": 1,
            "temperature": temperature,
            "stream": False,
            "kv_transfer_params": _empty_remote_params(),
        }
        if request_format == REQUEST_FORMAT_CHAT_COMPLETIONS:
            prefill_body["messages"] = [{"role": "user", "content": prompt}]
            prefill_body["n"] = 1
        else:
            prefill_body["prompt"] = prompt
        if truncate_prompt_tokens is not None:
            prefill_body["truncate_prompt_tokens"] = truncate_prompt_tokens
            if request_format == REQUEST_FORMAT_CHAT_COMPLETIONS:
                prefill_body["truncation_side"] = "right"
        prefill_lease: Optional[_PdRoleAdmissionLease] = None
        decode_lease: Optional[_PdRoleAdmissionLease] = None
        prefill_done_ts: Optional[float] = None
        try:
            prefill_lease = await self._acquire_role("prefill")
            rp_data = await self._raw_post(
                self.prefill_port,
                prefill_body,
                request_id,
                request_format=request_format,
            )
            ktp = rp_data.get("kv_transfer_params")
            if not ktp:
                raise RuntimeError(
                    "prefill returned no kv_transfer_params "
                    f"(request_id={request_id})"
                )
            prefill_done_ts = time.perf_counter()

            # NIXL keeps the producer blocks pinned after this HTTP response
            # until a consumer fetches the handle.  Keep the P permit while
            # waiting for D admission, so ready handles can never exceed the P
            # engine's own max_num_seqs.  Once D is reserved it is safe to
            # release P immediately; holding P through decode would serialize
            # the whole deployment at the tiny prefill cap.
            decode_lease = await self._acquire_role("decode")
        except asyncio.CancelledError:
            self._release_role(prefill_lease, cancelled=True)
            raise
        except BaseException:
            self._release_role(prefill_lease, failed=True)
            # Defensive only: there is no await between assigning the D lease
            # and leaving this try block, but never leak it if future code adds
            # synchronous validation there.
            self._release_role(decode_lease, failed=True)
            raise
        else:
            assert prefill_done_ts is not None
            self._release_role(
                prefill_lease,
                handoff_wait_s=time.perf_counter() - prefill_done_ts,
            )

        decode_failed = False
        decode_cancelled = False
        try:
            # Step 2: decode pulls KV via NIXL and generates the visible answer.
            decode_extra: Dict[str, Any] = {}
            if sampling_params.get("stop"):
                decode_extra["stop"] = sampling_params["stop"]
            if sampling_params.get("top_p") is not None:
                decode_extra["top_p"] = float(sampling_params["top_p"])
            decode_extra_body: Dict[str, Any] = {"kv_transfer_params": ktp}
            if truncate_prompt_tokens is not None:
                decode_extra_body["truncate_prompt_tokens"] = truncate_prompt_tokens
            first_token_ts: Optional[float] = None
            n_output_tokens = 0
            if not stream:
                decode_body: Dict[str, Any] = {
                    "model": self.key.model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False,
                    **decode_extra_body,
                    **decode_extra,
                }
                if request_format == REQUEST_FORMAT_CHAT_COMPLETIONS:
                    decode_body["messages"] = [
                        {"role": "user", "content": prompt}
                    ]
                    decode_body["logprobs"] = True
                    decode_body["n"] = 1
                    decode_body["truncation_side"] = "right"
                else:
                    decode_body["prompt"] = prompt
                data = await self._raw_post(
                    self.decode_port,
                    decode_body,
                    request_id,
                    request_format=request_format,
                )
                last_token_ts = time.perf_counter()
                choices = data.get("choices") or []
                if request_format == REQUEST_FORMAT_CHAT_COMPLETIONS:
                    message = (choices[0].get("message") or {}) if choices else {}
                    text = message.get("content") or ""
                else:
                    text = (choices[0].get("text") or "") if choices else ""
                usage = data.get("usage") or {}
                n_output_tokens = int(usage.get("completion_tokens") or 0)
                if n_output_tokens <= 0:
                    n_output_tokens = max(len(text) // 4, 1)
                return text, {
                    "request_send_ts": send_ts,
                    "first_token_ts": None,
                    "last_token_ts": last_token_ts,
                    "n_output_tokens": n_output_tokens,
                }

            text_parts: List[str] = []
            stream_kwargs: Dict[str, Any] = {
                "model": self.key.model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
                "extra_body": decode_extra_body,
                "extra_headers": {"X-Request-Id": request_id},
                **decode_extra,
            }
            if request_format == REQUEST_FORMAT_CHAT_COMPLETIONS:
                stream_kwargs["messages"] = [
                    {"role": "user", "content": prompt}
                ]
                stream_kwargs["logprobs"] = True
                stream_kwargs["n"] = 1
                stream_kwargs["extra_body"]["truncation_side"] = "right"
                sse = await oai.chat.completions.create(**stream_kwargs)
            else:
                stream_kwargs["prompt"] = prompt
                sse = await oai.completions.create(**stream_kwargs)
            async for chunk in sse:
                if (
                    getattr(chunk, "usage", None) is not None
                    and chunk.usage is not None
                ):
                    n_output_tokens = int(
                        getattr(chunk.usage, "completion_tokens", 0) or 0
                    )
                if not chunk.choices:
                    continue
                if request_format == REQUEST_FORMAT_CHAT_COMPLETIONS:
                    piece = getattr(
                        getattr(chunk.choices[0], "delta", None),
                        "content",
                        None,
                    )
                else:
                    piece = getattr(chunk.choices[0], "text", None)
                if piece:
                    if first_token_ts is None:
                        first_token_ts = time.perf_counter()
                    text_parts.append(piece)
            last_token_ts = time.perf_counter()
            text = "".join(text_parts)
            if n_output_tokens <= 0:
                n_output_tokens = max(len(text) // 4, 1)
            return text, {
                "request_send_ts": send_ts,
                "first_token_ts": first_token_ts,
                "last_token_ts": last_token_ts,
                "n_output_tokens": n_output_tokens,
            }
        except asyncio.CancelledError:
            decode_cancelled = True
            raise
        except BaseException:
            decode_failed = True
            raise
        finally:
            self._release_role(
                decode_lease,
                failed=decode_failed,
                cancelled=decode_cancelled,
            )

    async def generate_one(self, prompt: str, sampling_params: dict):
        """NON-streaming generate one prompt through the PD pair (prefill→decode).

        Matches ``VllmSubprocess.generate_one`` so the Vllm node uses the same
        non-streaming path in single-subprocess and PD modes (first_token_ts=None).
        """
        # Non-streaming path uses the shared raw session; http/oai params of
        # _gen_one are unused there (kept for the streaming path).
        return await self._gen_one(prompt, sampling_params, None, None, stream=False)

    async def generate_batch(self, prompts, sampling_params):
        """NON-streaming batch — qps-accurate counterpart of
        ``generate_batch_streaming`` (see ``VllmSubprocess.generate_batch``)."""
        limit = _stream_concurrency_limit(
            len(prompts),
            default_limit=_pd_stage_admission_limit(self.key),
            env_name=_PD_MAX_CONCURRENCY_ENV,
        )
        sem = asyncio.Semaphore(limit)
        results: List[Any] = [None] * len(prompts)

        # Non-streaming batch rides the shared raw session (see _http_session);
        # no per-batch httpx/openai client churn.
        async def run_one(idx: int, prompt: str) -> None:
            async with sem:
                results[idx] = await self._gen_one(
                    prompt, sampling_params, None, None, stream=False)

        await asyncio.gather(*(run_one(i, p) for i, p in enumerate(prompts)))
        assert all(r is not None for r in results)
        return [r[0] for r in results], [r[1] for r in results]

    async def generate_one_stream(self, prompt: str, sampling_params: dict):
        """Stream-generate one prompt through the PD pair (prefill→decode).

        Signature/return matches ``VllmSubprocess.generate_one_stream`` so the
        Vllm node can branch transparently between single-subprocess and PD modes.
        """
        async with httpx.AsyncClient(timeout=_stream_timeout_s()) as http:
            oai = self._decode_client()
            try:
                return await self._gen_one(prompt, sampling_params, http, oai, stream=True)
            finally:
                await oai.close()

    async def generate_batch_streaming(self, prompts, sampling_params):
        # Every prompt prefills, pushes KV over nixl, then decodes. Keep the
        # pipeline batch intact but bound simultaneous in-flight requests.
        limit = _stream_concurrency_limit(
            len(prompts),
            default_limit=_pd_stage_admission_limit(self.key),
            env_name=_PD_MAX_CONCURRENCY_ENV,
        )
        if limit < len(prompts):
            logger.info(
                "PD vLLM (Nixl) streaming concurrency limited: prompts=%d "
                "concurrency=%d (stage_admission_limit=%d, env=%s)",
                len(prompts), limit, _pd_stage_admission_limit(self.key),
                _PD_MAX_CONCURRENCY_ENV,
            )
        sem = asyncio.Semaphore(limit)
        results: List[Any] = [None] * len(prompts)

        async with httpx.AsyncClient(timeout=_stream_timeout_s()) as http:
            oai = self._decode_client()

            async def run_one(idx: int, prompt: str) -> None:
                async with sem:
                    results[idx] = await self._gen_one(
                        prompt, sampling_params, http, oai, stream=True
                    )

            try:
                await asyncio.gather(*(run_one(i, p) for i, p in enumerate(prompts)))
            finally:
                await oai.close()

        assert all(r is not None for r in results)
        return [r[0] for r in results], [r[1] for r in results]

    # --- cleanup ------------------------------------------------------

    def shutdown(self) -> None:
        # Delayed-bind ports stay CLAIMED for the parent-process lifetime —
        # never released on shutdown. Releasing them here (0752d869) made the
        # pid-offset scan hand the SAME ports to consecutive PD pairs while a
        # killed engine could still linger in GPU teardown. The next delayed
        # NIXL/TCPStore bind then raced that corpse. One pair claims 2 API
        # ports, 2 side ports, and two 8-port rendezvous ranges; ~27k candidates
        # support over 1,300 pair launches in one optimizer process, well beyond
        # a campaign.
        for name, proc in (
            ("prefill", self.prefill_proc),
            ("decode", self.decode_proc),
        ):
            if proc is None:
                continue
            # Group-kill each engine (api_server + EngineCore workers) — see
            # `_kill_proc_group`. A half-dead engine's workers pin VRAM too.
            _kill_proc_group(proc)
            logger.info(f"VllmPdPair: {name} subprocess shut down")
