"""Compatibility alias for RAG-Stack's shared recording implementation.

All recorder paths resolve to the same module object, preserving ContextVars,
tokenizer caches, and historical monkeypatch targets. New integrations should
use ``rag_stack.evaluation.tracing``.
"""

import sys

from rag_stack.evaluation import _recording

sys.modules[__name__] = _recording
