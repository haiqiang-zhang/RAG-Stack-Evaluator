# Portions derived from AutoRAG (https://github.com/Marker-Inc-Korea/AutoRAG), Apache-2.0.
# Modified by the RAG-Stack authors for namespace and runtime integration; see LICENSE.autorag and NOTICE.

import logging
import sys


class LazyInit:
	def __init__(self, factory, *args, **kwargs):
		self._factory = factory
		self._args = args
		self._kwargs = kwargs
		self._instance = None

	def __call__(self):
		if self._instance is None:
			self._instance = self._factory(*self._args, **self._kwargs)
		return self._instance

	def __getattr__(self, name):
		if self._instance is None:
			self._instance = self._factory(*self._args, **self._kwargs)
		return getattr(self._instance, name)


logger = logging.getLogger("RAG-Stack")


def handle_exception(exc_type, exc_value, exc_traceback):
	logger = logging.getLogger("RAG-Stack")
	logger.error("Unexpected exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception

try:
	import transformers

	transformers.logging.set_verbosity_error()
except ImportError:
	logger.info(
		"You are using API version of RAG-Stack."
		"To use local version, run pip install '.[gpu]'"
	)


# --- llama_index dispatcher monkey-patch -----------------------------------
# `Dispatcher.span()`'s async_wrapper bare-calls `active_span_id.reset(token)`
# in its `finally` block. When the wrapped coroutine is awaited via
# `asyncio.gather(...)`, the token was set in a different
# `contextvars.Context` than the one running the finally block, and reset
# raises `ValueError: <Token ...> was created in a different Context`. The
# sibling sync wrapper already wraps reset in `context.run(...)`. We wrap
# `active_span_id` itself with a tolerant proxy whose returned tokens
# swallow that specific ValueError on `.reset()`.
try:
	import llama_index_instrumentation.span as _li_span

	class _TolerantToken:
		"""Token proxy whose `reset` swallows the cross-Context ValueError."""

		__slots__ = ("_inner", "_var")

		def __init__(self, inner, var):
			self._inner = inner
			self._var = var

		@property
		def old_value(self):
			return self._inner.old_value

		@property
		def var(self):
			return self._var

	class _TolerantContextVar:
		"""ContextVar proxy that returns _TolerantToken from .set()."""

		def __init__(self, inner):
			self._inner = inner

		def set(self, value):
			return _TolerantToken(self._inner.set(value), self._inner)

		def get(self, *args, **kwargs):
			return self._inner.get(*args, **kwargs)

		def reset(self, tok):
			try:
				inner_tok = tok._inner if isinstance(tok, _TolerantToken) else tok
				self._inner.reset(inner_tok)
			except ValueError:
				# Token created in a different Context; harmless to skip.
				pass

	_li_span.active_span_id = _TolerantContextVar(_li_span.active_span_id)
	# Also patch the re-export in the dispatcher module so any code
	# that imported the name before our patch sees the proxy too.
	import llama_index_instrumentation.dispatcher as _li_dispatcher
	_li_dispatcher.active_span_id = _li_span.active_span_id
	logger.debug("Patched llama_index_instrumentation.span.active_span_id (tolerant reset)")
except Exception as _exc:  # noqa: BLE001
	logger.debug(f"Could not patch llama_index dispatcher: {_exc}")


# Public API exposed for baselines and cost-model-coupled callers.
from rag_stack.static_rag_evaluator.static_rag_evaluator import (
	StaticRAGEvaluatorQualityOnly,
)
# Measured subsystem public API (re-exported from the measured/ subpackage).
from rag_stack.static_rag_evaluator.measured import (
	ModelCache,
	FaissKey,
	VllmStartupKey,
	QueryPerf,
	summarize,
	MeasuredEvaluator,
	MeasuredProvider,
)

__all__ = [
	"StaticRAGEvaluatorQualityOnly",
	"ModelCache",
	"FaissKey",
	"VllmStartupKey",
	"QueryPerf",
	"summarize",
	"MeasuredEvaluator",
	"MeasuredProvider",
]
