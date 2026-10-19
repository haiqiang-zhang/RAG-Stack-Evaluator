"""Compatibility import for the shared, lightweight recording implementation.

Both module paths intentionally resolve to the same module object. Existing
wrappers, ContextVars, tokenizer caches, and monkeypatches therefore keep their
original behavior. External integrations should use ``rag_stack_evaluator.tracing``.
"""

import sys

from rag_stack_evaluator import _recording

sys.modules[__name__] = _recording
