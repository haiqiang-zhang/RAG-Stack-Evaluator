"""Compatibility alias for the RAG-Stack-owned evaluator API.

New backends import ``rag_stack.evaluation.api`` directly.
"""
import sys
from rag_stack.evaluation import api as _host_api

sys.modules[__name__] = _host_api
