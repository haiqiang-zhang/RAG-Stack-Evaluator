"""CM-family-neutral vLLM runtime instrumentation.

The package contains opt-in sensors and server wrappers shared by measured
runtime and calibration implementations.  Importing the package itself never
imports vLLM, torch, or initializes CUDA.
"""
