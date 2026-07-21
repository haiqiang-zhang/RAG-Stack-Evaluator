# Examples

Each file is a complete evaluator call: it loads caller-owned inputs into
Python objects, constructs the evaluator, invokes it, and writes the returned
result. The examples do not share a helper module, so each call flow can be
copied independently.

The core evaluator APIs receive Python dictionaries. These scripts accept JSON
files only as a command-line convenience and deserialize them before calling
the evaluator; they never pass a config path or string to the core API.

All examples expect a compatible RAG-Stack host environment and a pre-chunked
QA/corpus pair. `pipeline.json` must contain one fully resolved pipeline dict.
`system.json` must contain one fully resolved deployment dict produced by the
compatible host.

## Quality result

```bash
python RAG-Stack-Evaluator/example/quality.py \
  --pipeline-config pipeline.json \
  --qa data/qa.parquet \
  --corpus data/corpus.parquet \
  --project-dir outputs/quality
```

## Quality result with selected metrics

```bash
python RAG-Stack-Evaluator/example/quality_metrics_override.py \
  --pipeline-config pipeline.json \
  --qa data/qa.parquet \
  --corpus data/corpus.parquet \
  --project-dir outputs/quality-override \
  --metric retrieval_token_recall \
  --metric retrieval_token_f1
```

## Quality result and canonical trace

```bash
python RAG-Stack-Evaluator/example/quality_trace.py \
  --pipeline-config pipeline.json \
  --qa data/qa.parquet \
  --corpus data/corpus.parquet \
  --project-dir outputs/quality-trace
```

## Measured quality and performance result

Run this only on hardware exclusively assigned to the measured provider:

```bash
python RAG-Stack-Evaluator/example/measured.py \
  --pipeline-config pipeline.json \
  --system-config system.json \
  --qa data/qa.parquet \
  --corpus data/corpus.parquet \
  --project-dir outputs/measured \
  --gpu cuda:0
```
