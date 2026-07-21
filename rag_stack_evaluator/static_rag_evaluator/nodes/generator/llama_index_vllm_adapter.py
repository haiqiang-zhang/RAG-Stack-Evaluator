"""Thin subclass of `llama_index.llms.vllm.Vllm` that patches version-skew
between `llama-index-llms-vllm==0.7.0` and `vllm>=0.13` (verified through 0.20.2).

Upstream `Vllm._model_kwargs` always emits ``best_of`` (even when ``None``),
which `SamplingParams(**params)` then rejects with
``TypeError: Unexpected keyword argument 'best_of'`` because vllm removed that
field. We strip the key so the official adapter stays usable on current vllm.

If LlamaIndex releases a fix upstream, this file can be deleted and the
registry can point at the official class directly.
"""

from __future__ import annotations

from typing import Any, Dict

from llama_index.llms.vllm import Vllm as _OfficialVllm


class LlamaIndexVllm(_OfficialVllm):
    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        kwargs = super()._model_kwargs
        # vllm.SamplingParams no longer accepts `best_of` (deprecated since
        # vllm 0.6, removed in 0.13+). Drop it unconditionally; upstream
        # `Vllm.best_of` defaults to None anyway.
        kwargs.pop("best_of", None)
        return kwargs
