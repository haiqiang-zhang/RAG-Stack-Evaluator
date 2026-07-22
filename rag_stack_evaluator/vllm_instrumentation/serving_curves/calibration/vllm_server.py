"""CM-family-neutral vLLM server entrypoint with opt-in telemetry.

The command line intentionally mirrors
``python -m vllm.entrypoints.openai.api_server`` for vLLM 0.18.1.  When
``RAG_STACK_STAGE_TELEMETRY_PATH`` is present, the wrapper patches the exact
function symbol imported by ``vllm.v1.engine.async_llm`` before constructing
the server's ``AsyncLLM``.
"""

from __future__ import annotations

import os
from types import ModuleType

from .vllm_frontend_boundary import install_frontend_boundary_telemetry
from .vllm_telemetry import TELEMETRY_ENV, VllmStageTelemetryStatLogger
from .vllm_timing_dispatch import install_timing_dispatch_counter


PD_SAFE_PROMETHEUS_ENV = "RAG_STACK_PD_CALIBRATION_SAFE_PROMETHEUS"


def _prometheus_stat_logger_base() -> type:
    """Load vLLM's aggregate Prometheus logger only inside the server child."""

    from vllm.v1.metrics.loggers import PrometheusStatLogger

    return PrometheusStatLogger


def _pd_safe_prometheus_stat_logger(base: type) -> type:
    """Keep scheduler/NIXL metrics while dropping broken PD request stats.

    vLLM 0.18.1 can report a negative finished-request prefill-token value for
    KV producer/consumer requests.  Its normal Prometheus logger observes that
    value as a histogram sample and aborts the live request.  Direct P/D
    calibration needs scheduler and NIXL counters, but none of the
    request/iteration histograms, so this aggregate logger deliberately passes
    only scheduler and multi-modal-cache stats to the upstream implementation.

    Subclassing the upstream aggregate Prometheus logger is significant: its
    ``StatLoggerManager`` then recognizes this as the sole Prometheus provider
    and does not append the unsafe default logger.
    """

    class VllmPdSafePrometheusStatLogger(base):
        def record(
            self,
            scheduler_stats,
            iteration_stats,
            mm_cache_stats=None,
            engine_idx=0,
        ):
            del iteration_stats
            return super().record(
                scheduler_stats,
                None,
                mm_cache_stats=mm_cache_stats,
                engine_idx=engine_idx,
            )

    VllmPdSafePrometheusStatLogger.__name__ = (
        "VllmPdSafePrometheusStatLogger"
    )
    VllmPdSafePrometheusStatLogger.__qualname__ = (
        "VllmPdSafePrometheusStatLogger"
    )
    return VllmPdSafePrometheusStatLogger


def install_telemetry_stat_logger(async_llm_module: ModuleType) -> bool:
    """Install the sole calibration stat logger when telemetry is enabled."""

    output_path = os.environ.get(TELEMETRY_ENV)
    if output_path is None:
        return False
    if not output_path.strip():
        raise RuntimeError(f"{TELEMETRY_ENV} cannot be empty")

    factories: list[type] = [VllmStageTelemetryStatLogger]
    if os.environ.get(PD_SAFE_PROMETHEUS_ENV) == "1":
        factories.append(
            _pd_safe_prometheus_stat_logger(_prometheus_stat_logger_base())
        )

    # AsyncLLM imported the loader into its own module namespace.  Patching the
    # metrics module would therefore be too late; replace this exact symbol.
    def _load_calibration_stat_logger_factories() -> list[type]:
        return list(factories)

    async_llm_module.load_stat_logger_plugin_factories = (
        _load_calibration_stat_logger_factories
    )
    return True


def main() -> None:
    # These imports are intentionally local: importing the telemetry schema or
    # its unit tests must not import torch, vLLM, uvloop, or initialize CUDA.
    from rag_stack_evaluator.vllm_env import configure_vllm_worker_env

    configure_vllm_worker_env()

    from vllm.entrypoints.utils import cli_env_setup

    cli_env_setup()

    import uvloop
    import vllm.v1.engine.async_llm as async_llm

    telemetry_enabled = install_telemetry_stat_logger(async_llm)
    timing_dispatch_enabled = install_timing_dispatch_counter(
        async_llm.AsyncLLM
    )
    if telemetry_enabled and timing_dispatch_enabled:
        raise RuntimeError(
            "stage proof telemetry and production timing dispatch counter "
            "must run in separate servers"
        )
    if telemetry_enabled:
        # Coverage spans are not added to the scheduler-cycle wall interval.
        # They attest that this saturated run exercised every production
        # frontend operation required by the active-stage contract.
        install_frontend_boundary_telemetry(async_llm_cls=async_llm.AsyncLLM)

    from vllm.entrypoints.openai.api_server import run_server
    from vllm.entrypoints.openai.cli_args import (
        make_arg_parser,
        validate_parsed_serve_args,
    )
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    # This wrapper deliberately follows api_server.__main__, whose run_server
    # path is one frontend.  Refuse a future/parser-specific multi-frontend
    # value rather than producing interleaved per-process sequences.
    api_server_count = getattr(args, "api_server_count", None)
    if telemetry_enabled and api_server_count not in (None, 1):
        raise RuntimeError(
            "stage-cycle telemetry requires exactly one API frontend; "
            f"got api_server_count={api_server_count!r}"
        )

    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()


__all__ = [
    "PD_SAFE_PROMETHEUS_ENV",
    "install_frontend_boundary_telemetry",
    "install_telemetry_stat_logger",
    "install_timing_dispatch_counter",
    "main",
]
