# Self-contained examples

These examples generate their own QA and pre-chunked corpus Parquet files,
construct the evaluator with Python objects, and pass the resolved pipeline as
an in-memory `dict`. They do not read a YAML config, call an external API, or
download a model, and they do not require a GPU. The fixture uses an
eight-dimensional mock embedding, a one-document FAISS HNSW index, and the
generator's built-in local mock-response path. It demonstrates API wiring, not
retrieval-model or answer quality.

Run them from a compatible RAG-Stack host checkout after installing the host,
the evaluator, and the FAISS extra:

```bash
uv pip install -e 'RAG-Stack-Evaluator[faiss]' -e .

python RAG-Stack-Evaluator/example/quality_retrieval.py
python RAG-Stack-Evaluator/example/metrics_override.py
python RAG-Stack-Evaluator/example/trace_callback.py
```

Each script creates an isolated project under `/tmp` and prints its location.
Pass `--work-dir /absolute/path` to keep artifacts in a chosen directory.

## What each example shows

- `quality_retrieval.py` exercises the complete path-based quality API and
  prints retrieval recall, precision, and F1.
- `metrics_override.py` passes `metrics_override` to score only one metric
  without mutating the resolved pipeline dictionary.
- `trace_callback.py` receives the canonical trace through `on_trace_ready`,
  verifies that it is the same object returned as `__execution_dag__`, and
  writes it to `trace.json`.
