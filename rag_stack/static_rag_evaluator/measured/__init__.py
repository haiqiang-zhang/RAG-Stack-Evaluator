"""Measured-performance subsystem of the static RAG evaluator.

Everything that turns a pipeline run into REAL on-hardware performance lives here:
the run-spanning resource cache, the per-trial vLLM deployment + subprocess
launch/teardown machinery, the perf-record summarizer, and the two orchestration
classes (:class:`MeasuredEvaluator`, which composes the pure-quality
``StaticRAGEvaluatorQualityOnly`` via its shared ``_run_pipeline`` core, and
:class:`MeasuredProvider`, the controller-facing run-spanning context manager).

The pure-quality (cost-model) path in the parent package never imports this
subpackage; the only coupling is the shared nodes' ``cache.get_current()`` seam,
which returns ``None`` in quality mode.
"""

# Leaf-first re-exports (keeps the in-package import order acyclic).
from rag_stack.static_rag_evaluator.measured.performance import QueryPerf, summarize
from rag_stack.static_rag_evaluator.measured.cache import ModelCache, FaissKey
from rag_stack.static_rag_evaluator.measured.vllm_subprocess import (
    VllmStartupKey,
    reclaim_orphaned_vllm,
)
from rag_stack.static_rag_evaluator.measured.vllm_deployment import (
    VllmDeploymentManager,
    TrialInvalid,
)
from rag_stack.static_rag_evaluator.measured.evaluator import MeasuredEvaluator
from rag_stack.static_rag_evaluator.measured.provider import (
    MeasuredProvider,
    ProviderResult,
)

__all__ = [
    "QueryPerf",
    "summarize",
    "ModelCache",
    "FaissKey",
    "VllmStartupKey",
    "reclaim_orphaned_vllm",
    "VllmDeploymentManager",
    "TrialInvalid",
    "MeasuredEvaluator",
    "MeasuredProvider",
    "ProviderResult",
]
