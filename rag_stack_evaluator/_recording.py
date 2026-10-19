"""Compatibility alias for RAG-Stack's shared recorder.

The module object is shared so ContextVars, tokenizer caches, and monkeypatches
remain identical across historical evaluator and current host import paths.
"""

import sys

from rag_stack.evaluation import _recording

sys.modules[__name__] = _recording
