# Evaluator tests

The static evaluator owns concrete quality execution, retrieval nodes, measured
execution and runtime instrumentation. The host repository tests the public
backend API, global tracing, optimizer, cost model and their integration with an
evaluator. Keep one authoritative copy of each test at its implementation owner.

| Directory | Scope |
| --- | --- |
| `contracts/` | Static backend adapter, import boundary and supported compatibility imports |
| `dataset/` | Dataset binding, corpus activation and shared cache routing |
| `quality/` | Quality metrics, judge cardinality, generation cache and pipeline semantics |
| `retrieval/` | Dense/BM25/fusion execution, embeddings and FAISS stores |
| `measured/` | Admission, stage lifecycle, warmup, measured traces and scoring isolation |
| `runtime/` | Worker IPC, model lifecycle, vLLM requests and instrumentation |

Run from this repository root with the host package installed:

```bash
python -m pytest
python -m pytest tests/retrieval tests/quality
```

From the host checkout, select this project's configuration explicitly:

```bash
python -m pytest -c RAG-Stack-Evaluator/pyproject.toml RAG-Stack-Evaluator/tests
```

Run the host and evaluator suites separately; both are independent Python test
packages. The default suite uses fake models/services and temporary data. It
does not start GPU model servers or call judge APIs. Native FAISS tests use
small deterministic CPU indexes. A few IPC/import tests start local Python
children. Install the evaluator's `test` and `faiss` extras when needed.

Use `tmp_path` and `monkeypatch` for files, cache roots and environment changes.
For rate and deadline assertions, use a local virtual event-loop clock and
distinct completion timestamps, as in `measured/test_serving_phase_contract.py`,
instead of depending on submillisecond scheduling on the test machine.
Assertions should cover behavior and interface contracts, not old benchmark
paths, source snapshots or machine-specific data. Put real model/endpoint
experiments in an explicitly invoked script rather than `test_*.py`.
