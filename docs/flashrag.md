# FlashRAG quality backend

The optional FlashRAG backend is implemented in
[`rag_stack_evaluator.flashrag_quality_evaluator`](../rag_stack_evaluator/flashrag_quality_evaluator/).
The evaluator project owns the adapter, config translation, pipeline factory,
custom pipelines, dataset conversion, index orchestration, scoring, and monitor
integration. Its nested [`FlashRAG`](../FlashRAG/) submodule owns the framework
and the patched call sites that produce monitor events. RAG-Stack owns the
optimizer, search-space resolution, cost model, and shared IR/trace contracts.

`static_gt` remains the default backend. FlashRAG supports iterative and
agentic quality evaluation and emits canonical traces for cost-model replay.
It does not provide the measured-performance backend; the host requires
`static_gt` when `system.performance_source: measured` is selected.

## Installation

From the RAG-Stack host root, initialize recursive submodules and install the
host and evaluator using the host's documented CPU or CUDA environment first.
Then install the pinned optional framework:

```bash
git submodule update --init --recursive RAG-Stack-Evaluator
uv pip install -e RAG-Stack-Evaluator/FlashRAG
```

When developing from the evaluator repository root in that same environment:

```bash
git submodule update --init --recursive FlashRAG
uv pip install -e FlashRAG
```

The nested gitlink pins the required fork, including monitor hooks and runtime
patches. A same-version FlashRAG package from a package index is not a substitute
for that checkout. Do not maintain a second host-level `FlashRAG/` checkout.
There is no mandatory FlashRAG dependency for static evaluation: the evaluator
loads framework modules only when the FlashRAG backend is selected.

## Public API

The shared evaluator interface lives in
[`rag_stack_evaluator.base.BaseEvaluator`](../rag_stack_evaluator/base.py).
Construct a backend through the lazy factory:

```python
from rag_stack_evaluator.factory import create_quality_evaluator

# dataset_manager is the caller's owner-managed dataset for this project.
evaluator = create_quality_evaluator(
    dataset_manager=dataset_manager,
    project_dir=dataset_manager.project_dir,
    backend="flashrag",
    flashrag_options={"framework": "vllm", "monitor": "full"},
)
quality = evaluator.evaluate(
    resolved_pipeline_config,
    run_dir=run_dir,
    metrics_override=None,
    on_trace_ready=None,
)
```

The factory also accepts `dataset` and `project_dir` for callers with a dataset
object. Its default backend is `static_gt`. `metrics_override`, when provided,
replaces the configured scoring metrics for this call without mutating the
input config. `on_trace_ready`, when provided and monitoring produces a trace,
receives the same envelope returned under `quality["__execution_dag__"]`.
FlashRAG scores inside its pipeline run, so this callback runs after scoring;
callback failures are logged and do not invalidate the quality result.

Direct imports remain available:

```python
from rag_stack_evaluator.flashrag_quality_evaluator import FlashRAGQualityEvaluator
from rag_stack_evaluator.static_rag_evaluator import StaticRAGEvaluatorQualityOnly
```

Host configuration and search-space code can query supported metadata without
importing FlashRAG's heavyweight runtime:

```python
from rag_stack_evaluator.factory import supported_metrics, supported_pipeline_modes

metric_names = supported_metrics("flashrag")
pipeline_modes = supported_pipeline_modes("flashrag")
```

Add backend modes and metrics in the evaluator project. The host consumes this
metadata rather than maintaining another copy of backend capabilities.

## Resolved input contract

`evaluate` accepts an in-memory, fully resolved pipeline mapping. The host
converts `algo_search_space.pipeline.methods` into this mapping before calling
the evaluator. Do not pass an optimizer search space, lists of candidate
parameters, or a `methods` block to `evaluate`.

The relevant fields are:

| Field | Meaning |
| --- | --- |
| `node_lines` | Concrete retrieval and generation nodes; each node has one resolved module in `modules`. |
| `vectordb` | Concrete retrieval configuration, including `embedding_model` and index settings. The FlashRAG adapter uses the first store. |
| `corpus_runtime.chunker` | One concrete chunker mapping, or `{}` for pre-chunked data. |
| `pipeline_runtime.mode` | One selected method, such as `sequential`, `ircot`, `react`, or `a_rag`. |
| Other `pipeline_runtime` keys | Concrete method constructor arguments, such as `max_iter`. |
| `eval_backend_setting.metrics` | FlashRAG-native metric names, or mappings with `metric_name`. |
| `eval_backend_setting.flashrag` | Framework, model paths, generation/retrieval overrides, and monitor settings. |

The dataset manager supplies QA and corpus data and resolves the requested
chunking for each evaluation. QA rows use `qid`, `query`, and accepted answers
in `generation_gt`; corpus rows use `doc_id` and `contents`. The adapter writes
FlashRAG JSONL with `id`, `question`, and `golden_answers` for QA, and `id` and
`contents` for the corpus. Retrieval metrics use FlashRAG's own semantics;
legacy passage-ID `retrieval_gt` is not forwarded.

`pipeline_runtime.mode` takes precedence over
`eval_backend_setting.flashrag.pipeline_type`, which defaults to `sequential`.
The factory supports `sequential`, `adaptive_rag`, `iter_retgen`, `flare`,
`ircot`, `self_ask`, `searchr1`, `corag`, `search_o1`, `react`, and `a_rag`.
Use `supported_pipeline_modes("flashrag")` for the authoritative list.

Per-call `eval_backend_setting.flashrag` values override constructor-level
`flashrag_options` for translation, index configuration, and monitoring. The
materialized QA dataset name is selected separately from the constructor's
`flashrag_options.dataset_name`; a per-call `dataset_name` does not override it.
When the constructor omits that option, the QA content hash supplies the name.
For a host optimization config, select the backend with:

```yaml
global:
  eval_backend: flashrag
  rag_ir_mode: dynamic
system:
  performance_source: cost_model
eval_backend_setting:
  metrics:
    - metric_name: em
    - metric_name: f1
    - metric_name: rouge-l
  flashrag:
    framework: vllm
    monitor: full
    gpu_memory_utilization: 0.85
    model2path:
      e5: intfloat/e5-base-v2
```

This is a settings fragment, not a complete optimization config. Dataset,
hardware, optimizer, and search-space sections still belong to the host config.
Metrics use the FlashRAG vocabulary, including `em`, `sub_em`, `f1`, `acc`,
`recall`, `precision`, `bleu`, `rouge_score`, `rouge-1`, `rouge-2`, `rouge-l`,
`llm_judge`, `input_tokens`, `retrieval_recall`, and `retrieval_precision`.
The adapter normalizes `rouge_l` to `rouge-l`; static evaluator metric names are
not automatically translated.

## Outputs and trace ownership

Evaluation returns a flat mapping of metric names to numeric scores. With
monitoring enabled, it also returns `__execution_dag__`: the canonical quality
trace envelope with per-query component calls. Keys beginning with `__` are
metadata and must not become optimization objectives.

The monitor is enabled by default (`monitor: full`). It records framework
retrieval, generation, and other instrumented calls; the adapter converts
these records into the host's shared `rag_stack.rag_ir` trace contract. The
host consumes that contract for dynamic cost-model replay and does not
implement FlashRAG monitoring or inspect the framework's internal call graph.
Disabling the monitor removes this trace output and is unsuitable for the
host's trace-driven FlashRAG cost-model path. The pinned fork is required for
monitor hook support.

Framework monitor state is cleared after a pipeline run, including when the
run fails. The evaluator owns its execution and cleanup lifecycle; the host
owns optimization scheduling and the choice of project/run directories.

## Cache and artifact layout

The caller's `project_dir` and selected `run_dir` must be writable. FlashRAG
artifacts are scoped to the project:

```text
<project_dir>/
  _flashrag/
    datasets/<dataset-name>/test.jsonl
    corpus/<corpus-hash>/corpus.jsonl
    indexes/<corpus-hash>/...
  _flashrag_run/                     # Default when run_dir is omitted.
```

By default, the dataset name incorporates the QA content hash. Corpus
materialization and index paths incorporate the corpus content hash; index
reuse also depends on retrieval method and FAISS index type. A-RAG additionally
builds a BM25 index over the same corpus. Supplying a fixed `dataset_name`
requires the caller to keep it associated with the same QA content because
existing JSONL files are reused.

These project-local artifacts are separate from static evaluator shared
embedding/index caches configured by `RAG_STACK_CACHE_DIR`. Moving the adapter
into the evaluator package does not change existing project paths or require
regenerating cached FlashRAG artifacts.

## Development boundary

Keep adapter behavior and backend unit tests in this repository. RAG-Stack
integration tests should verify backend selection, resolved inputs, returned
scores/traces, and optional dependency behavior. Changes to framework hooks or
framework pipeline implementations belong in the nested `FlashRAG` repository;
update its gitlink when accepting such changes. Do not add host-side evaluator
implementations or compatibility copies.

Run the evaluator's FlashRAG tests from the evaluator root in the compatible
host environment. Real model execution additionally needs the corresponding
model files, framework dependencies, and caller-assigned hardware. CPU unit
tests of translation, data conversion, and monitor contracts do not establish
real-model quality or performance results.
