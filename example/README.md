# Evaluator examples

The [input and output contract](../docs/input_contract.md) is authoritative.
These scripts deserialize caller-supplied JSON before calling the core API.
They pass QA/corpus as separate arguments, never through either config.

[algo_config.json](algo_config.json) and [system_config.json](system_config.json)
are a complete resolved generator-only example and its one-GPU deployment.
The algorithm uses Qwen2.5-1.5B-Instruct and the local `f1` metric; executing it
requires the corresponding local model runtime. Replace the pair with a
resolved host export for retrieval, hybrid retrieval or another deployment.
No model is loaded by `--help`.

All four direct examples require pre-chunked corpus data and
`corpus_runtime.chunker: {}`. For a raw corpus or nonempty chunker, use the
host's `python -m rag_stack replay ALGO_JSON SYSTEM_JSON --qa QA --corpus CORPUS`
command, which owns chunking and current corpus statistics. Do not pass a
resolved algorithm dict as the dataset manager's host `config` argument.

## Quality

```bash
python RAG-Stack-Evaluator/example/quality.py \
  --algo-config RAG-Stack-Evaluator/example/algo_config.json \
  --qa data/qa.parquet --corpus data/chunks.parquet \
  --project-dir /tmp/evaluator-quality
```

`quality.py`, `quality_metrics_override.py` and `quality_trace.py` use the host
registry and return `EvaluationResult`. Their JSON output has `metrics`,
`trace`, `workflow`, `runtime_stats` and `metadata` fields. `--pipeline-config`
remains an alias for `--algo-config` for existing invocations.

Override metrics with:

```bash
python RAG-Stack-Evaluator/example/quality_metrics_override.py \
  --algo-config RAG-Stack-Evaluator/example/algo_config.json \
  --qa data/qa.parquet --corpus data/chunks.parquet \
  --project-dir /tmp/evaluator-metrics --metric f1 --metric em
```

Receive the canonical serving trace before scoring and save it separately:

```bash
python RAG-Stack-Evaluator/example/quality_trace.py \
  --algo-config RAG-Stack-Evaluator/example/algo_config.json \
  --qa data/qa.parquet --corpus data/chunks.parquet \
  --project-dir /tmp/evaluator-trace
```

The callback receives the same object returned in `result.trace`; judge calls
are not serving operations in that trace.

## Measured performance

```bash
python RAG-Stack-Evaluator/example/measured.py \
  --algo-config RAG-Stack-Evaluator/example/algo_config.json \
  --system-config RAG-Stack-Evaluator/example/system_config.json \
  --qa data/qa.parquet --corpus data/chunks.parquet \
  --project-dir /tmp/evaluator-measured --gpu cuda:0
```

This example uses the same `static_gt` registry factory. Its adapter forwards
the concrete engine's measured execution methods. An arbitrary evaluator that
only implements public `evaluate` is not enough for `MeasuredProvider`.
The provider returns `performance_score`, `quality`, `raw_performance` and
`performance_execution_trace`, not `EvaluationResult`, and runs as a context
manager on explicitly assigned GPUs.

Choose a fresh project directory when changing data. No example treats the
saved config file name as a dataset selector or duplicates QA/corpus inside it.
