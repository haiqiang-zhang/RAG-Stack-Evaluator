"""Compatibility alias for the hooks owned by RAG-Stack.

New integrations should import ``rag_stack.evaluation.tracing`` directly.
"""

import sys

from rag_stack.evaluation import tracing

sys.modules[__name__] = tracing
