"""Unified generator backend registry.

Single source of truth for all LLM backends used across the pipeline:
- Generator node backends (openai_llm, vllm, vllm_api)
- LlamaIndex LLM backends (llama_index_openai, llama_index_openrouter, etc.)

Usage:
    from rag_stack.static_rag_evaluator.nodes.generator.registry import (
        get_generator_class,
        get_llama_index_llm_class,
    )
"""

import importlib
import logging
from typing import Any

from llama_index.core.base.llms.types import CompletionResponse
from llama_index.core.llms.mock import MockLLM
from llama_index.llms.bedrock import Bedrock
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.openrouter import OpenRouter

logger = logging.getLogger("RAG-Stack")


class AutoRAGBedrock(Bedrock):
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        return self.complete(prompt, formatted=formatted, **kwargs)


# ---------------------------------------------------------------------------
# Unified backend registry
# ---------------------------------------------------------------------------
# Values are either:
#   - A class (for LlamaIndex LLM backends, available at import time)
#   - A (module_path, class_name) tuple (for generator modules, lazy-imported)

GENERATOR_BACKENDS: dict[str, Any] = {
    # Project generator modules (full pipeline generators with tokenization, log probs)
    "openai_llm": ("rag_stack.static_rag_evaluator.nodes.generator.openai_llm", "OpenAILLM"),
    "vllm": ("rag_stack.static_rag_evaluator.nodes.generator.vllm", "Vllm"),
    "vllm_api": ("rag_stack.static_rag_evaluator.nodes.generator.vllm_api", "VllmAPI"),
    # LlamaIndex LLM backends (used by compressor, reranker, llama_index_llm generator)
    "llama_index_openai": OpenAI,
    "llama_index_openailike": OpenAILike,
    "llama_index_openrouter": OpenRouter,
    "llama_index_mock": MockLLM,
    "llama_index_bedrock": AutoRAGBedrock,
    # `llama_index_vllm` is registered below in the optional-backend block.
}

# Optional LlamaIndex backends (require extra dependencies)
try:
    from llama_index.llms.huggingface import HuggingFaceLLM
    GENERATOR_BACKENDS["llama_index_huggingface"] = HuggingFaceLLM
except ImportError:
    pass

try:
    from llama_index.llms.ollama import Ollama
    GENERATOR_BACKENDS["llama_index_ollama"] = Ollama
except ImportError:
    pass

# In-process vllm — drives `tree_summarize` / `refine` / `rankgpt` /
# LlamaIndexLLM generator from local vllm.LLM without spinning up a separate
# `vllm serve` HTTP process. We use a thin subclass that strips `best_of` from
# SamplingParams kwargs (vllm 0.13+ removed the field; the upstream adapter
# at llama-index-llms-vllm==0.7.0 still emits it). Verified through vllm 0.20.2.
# Each construction loads a fresh vllm.LLM, so only one such node should be
# active at a time. The EngineCore reaper in nodes/generator/vllm.py handles
# between-node cleanup.
try:
    from rag_stack.static_rag_evaluator.nodes.generator.llama_index_vllm_adapter import (
        LlamaIndexVllm,
    )
    GENERATOR_BACKENDS["llama_index_vllm"] = LlamaIndexVllm
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def _resolve(entry):
    """Resolve a registry entry to a class (lazy import if needed)."""
    if isinstance(entry, tuple):
        module_path, class_name = entry
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    return entry


def get_generator_class(name: str):
    """Get any generator backend class by name.

    Works for both project generator modules (openai_llm, vllm, ...) and
    LlamaIndex LLM backends (llama_index_openai, llama_index_openrouter, ...).
    """
    if name not in GENERATOR_BACKENDS:
        raise KeyError(
            f"Unknown generator backend '{name}'. "
            f"Available: {sorted(GENERATOR_BACKENDS.keys())}"
        )
    return _resolve(GENERATOR_BACKENDS[name])


def get_llama_index_llm_class(name: str):
    """Get a LlamaIndex LLM class by name.

    Accepts names with or without the 'llama_index_' prefix for convenience.
    Used by passage compressor, reranker, and LlamaIndexLLM generator.
    """
    # Try exact name first
    if name in GENERATOR_BACKENDS:
        cls = _resolve(GENERATOR_BACKENDS[name])
        return cls
    # Try with prefix
    prefixed = f"llama_index_{name}"
    if prefixed in GENERATOR_BACKENDS:
        cls = _resolve(GENERATOR_BACKENDS[prefixed])
        return cls
    raise KeyError(
        f"Unknown LlamaIndex LLM backend '{name}'. "
        f"Available: {[k for k in sorted(GENERATOR_BACKENDS.keys()) if k.startswith('llama_index_')]}"
    )


def is_llama_index_backend(name: str) -> bool:
    """Check if a backend name refers to a LlamaIndex LLM backend."""
    return name.startswith("llama_index_")
