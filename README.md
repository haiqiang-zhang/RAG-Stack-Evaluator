# RAG-Stack Evaluator

RAG-Stack Evaluator contains the static quality evaluator, the real-hardware
measured evaluator, and the vLLM instrumentation used by measured runs. It
preserves the existing Python imports:

```python
from rag_stack.static_rag_evaluator import (
    StaticRAGEvaluatorQualityOnly,
    MeasuredProvider,
)
```

This distribution contributes `rag_stack.static_rag_evaluator` and
`rag_stack.vllm_instrumentation` to the `rag_stack` namespace. The repository
deliberately does **not** contain `rag_stack/__init__.py`.

This is currently a RAG-Stack host subproject, not a standalone replacement
for the host package. The evaluator continues to consume RAG-Stack's shared
dataset, IR, layout, security, and cost-model contracts through `rag_stack.*`.
Install and run it through a compatible RAG-Stack checkout; the direct install
commands below are for developing the submodule in that host environment.

## Installation

Use Python 3.10 or newer and install exactly one CUDA stack when local GPU
models or measured evaluation are required:

```bash
# API/CPU components; add `faiss` unless faiss-cpu 1.14.1 is already present.
uv pip install -e '.[faiss]'

# NVIDIA driver 525-579.
uv pip install -e '.[cu12,faiss]'

# NVIDIA driver >=580.
uv pip install -e '.[cu13,faiss]'
```

Do not install the `cu12` and `cu13` extras together. Native benchmark
environments may provide `faiss-cpu=1.14.1` through conda and omit the `faiss`
extra. The supported integration checks this repository out as the
`RAG-Stack-Evaluator` submodule and installs it through the parent workspace,
so the host's shared `rag_stack` contracts and this namespace package are
available together.

## Input Contract v1

This section is the public, versioned boundary of the evaluator. Changes that
break it require a new major contract version. Inputs outside this contract are
caller errors even if a particular implementation happens to accept them.

### 1. Resolved pipeline config only

Both quality and measured evaluation accept one **fully resolved pipeline
configuration**. Every active vector database, node, component, model,
parameter, and deployment value must already be concrete.

The evaluator does **not** accept an optimizer search space. Do not pass:

- `algo_search_space`, optimizer configuration, objectives, or candidate arms;
- `{range: [...]}` values, categorical choice dictionaries, or lists that mean
  "choose one";
- multiple candidate modules for a node; or
- unresolved optional nodes or conditional parameters.

Structural lists defined by this contract, such as `node_lines`, `nodes`,
`metrics`, and device lists, remain lists. A node must use the singular
`module` field and contain exactly one resolved `component`. Upstream search or
optimization code is responsible for selecting an arm and resolving it before
calling the evaluator.

A resolved quality configuration has this shape:

```yaml
dataset:
  dataset_name: example

corpus_runtime:
  chunker: {}

pipeline_runtime:
  mode: sequential

vectordb:
  - name: example_hnsw
    db_type: faiss_hnsw
    embedding_model: mock
    embedding_dim: 768
    collection_name: example
    path: /absolute/path/to/project/resources/faiss
    similarity_metric: cosine
    M: 32
    ef_construction: 200
    ef_search: 64

node_lines:
  - node_line_name: retrieval
    nodes:
      - stage: semantic_retrieval
        strategy:
          metrics: []
          strategy: mean
        top_k: 4
        module:
          component: vectordb
          vectordb: example_hnsw
          ef_search: 64

eval_backend_setting:
  metrics:
    - metric_name: retrieval_token_recall
    - metric_name: retrieval_token_precision
    - metric_name: retrieval_token_f1
```

The required top-level evaluator fields are:

- `node_lines`: ordered pipeline lines, each with a stable `node_line_name` and
  an ordered `nodes` list;
- `vectordb`: the concrete stores referenced by active retrieval nodes, or an
  empty list when the pipeline does not use a vector database;
- `eval_backend_setting.metrics`: a list of metric names or metric mappings
  containing `metric_name`; and
- `corpus_runtime.chunker`: one concrete chunker mapping or `{}`.

`pipeline_runtime` is optional and defaults to sequential execution. ReAct
requires concrete `rag_dataflow: react` and `max_iter` values. Measured calls
may also include resolved runtime policy under the pipeline config, but the
physical deployment belongs in the separate `system_config` described below.

### 2. QA Parquet

The QA input is a Parquet file. Each row must contain:

| Column | Type | Meaning |
| --- | --- | --- |
| `qid` | `str` | Stable question identifier. |
| `query` | `str` | Query presented to the pipeline. |
| `generation_gt` | `str` or `list[str]` | One or more accepted answers. |

The optional `references` column contains retrieval ground-truth text as a
string or list of strings. The legacy spelling `reference` is accepted and
normalized to `references`. If neither is present, `generation_gt` is used as
the retrieval ground-truth text.

`retrieval_gt` chunk IDs are deprecated. They are ignored and removed because
chunk IDs are not stable when chunking changes. Retrieval metrics compare
reference text with retrieved text.

### 3. Corpus Parquet

The normal, pre-chunked corpus contains:

| Column | Type | Meaning |
| --- | --- | --- |
| `doc_id` | string-compatible | Stable ID of the chunk. |
| `contents` | `str` | Chunk text. |
| `metadata` | `dict` | Chunk metadata. Missing navigation/timestamp keys are normalized. |

A raw corpus supplied for evaluator-owned chunking contains `doc_id` and
`contents`; the legacy text-column name `texts` is accepted. A raw corpus must
not contain `start_end_idx`. A corpus carrying `start_end_idx` is treated as
already chunked and must not be passed through another chunker.

The caller must keep `qid` and `doc_id` stable within one project. All input
paths must identify Parquet files.

### 4. Quality API

The stable path-based quality API is:

```python
from rag_stack.static_rag_evaluator.dataset import DatasetManager
from rag_stack.static_rag_evaluator import StaticRAGEvaluatorQualityOnly

dataset = DatasetManager(
    project_dir=project_dir,
    config=resolved_pipeline_config,
    qa_data_path=qa_parquet,
    corpus_data_path=corpus_parquet,
)
evaluator = StaticRAGEvaluatorQualityOnly(
    dataset_manager=dataset,
    project_dir=project_dir,
)

quality = evaluator.evaluate(
    resolved_pipeline_config,
    run_dir=run_dir,
    metrics_override=None,
    on_trace_ready=None,
)
```

`resolved_pipeline_config` is the contract from section 1. `metrics_override`,
when supplied, is a concrete list of metrics and changes scoring only.
`on_trace_ready`, when supplied, is called with the same canonical quality
trace envelope later returned under `quality["__execution_dag__"]`; hook
failures are advisory and do not invalidate the quality run.

### 5. MeasuredProvider and system_config

Measured evaluation uses the same evaluator and resolved pipeline config. The
physical deployment is a second, fully resolved mapping named `system_config`:

```python
from rag_stack.static_rag_evaluator import MeasuredProvider

with MeasuredProvider(
    evaluator,
    available_gpus=["cuda:0"],
    n_queries=None,
    selection="max_throughput",
) as provider:
    result = provider.evaluate(
        resolved_pipeline_config,
        system_config,
        run_dir=run_dir,
        force_disagg=False,
        require_admissible=True,
    )
```

`available_gpus` is an explicit list of `"cuda:N"` device IDs the provider owns
for the duration of the context. `selection` is `max_throughput` or
`min_latency`. `n_queries=None` evaluates the full QA input; an integer selects
the first N rows.

`system_config` must be a single concrete deployment, not a design space. Its
contract is:

```yaml
batch_size_request: 4
measured_load_concurrency: 4
measured_warmup_queries: 4
measured_queries: 16

batching:
  dynamic_timeout_s: 0.0

retrieval:
  faiss_num_threads: 1
  faiss_ivf_parallel_mode: 0
  num_servers: 1

vllm:
  kv_cache_dtype: auto
  engines:
    generator:
      max_num_seqs: 4

layout:
  engines:
    generator:
      pd_serving: collocated_pd
      devices: [cuda:0]
      num_chips: 1
      tp: 1
      pp: 1
```

Every active GPU stage must be assigned to concrete devices. A disaggregated
generator additionally supplies concrete `generator_prefill` and
`generator_decode` role mappings under its engine, each with `devices`,
`num_chips`, `tp`, and `pp`. Counts, batching settings, cache dtype, placement,
and parallelism must not contain optimizer choices.

`MeasuredProvider` must be used as a context manager. Calling `evaluate`
outside its context is an error.

### 6. Outputs

Quality evaluation returns a mapping of metric name to score. It may also
contain reserved metadata keys:

- `__execution_dag__`: canonical quality trace envelope; and
- `__fp_hit__`: generation-fingerprint donor marker when deduplication applies.

Callers must not treat keys beginning with `__` as optimization metrics.

Measured evaluation returns `ProviderResult` with these stable fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `performance_score` | `float` | Higher-is-better scalar selected by `selection`. |
| `quality` | `dict` | Quality metrics and reserved quality metadata. |
| `raw_performance` | `dict` | Measured latency, throughput, stationarity, and deployment evidence. |
| `performance_execution_trace` | `dict` | Canonical completion-window workload trace. |

An infeasible deployment raises `TrialInvalid`. With
`require_admissible=True`, a non-publishable measurement window raises
`MeasuredGTInadmissibleError` after preserving diagnostic artifacts when a
`run_dir` was supplied.

### 7. Filesystem and process lifecycle

`project_dir` and `run_dir` must be writable. The evaluator owns these paths
under `project_dir`:

- `data/` for the resume-bound QA/corpus and canonical raw corpus;
- `resources/` for vector indexes and reusable resources;
- the caller-selected `run_dir`, or `_static_run/` when none is supplied; and
- evaluator-managed cache/artifact files below those locations.

Once `project_dir/data/` is populated, subsequent runs reuse that bound dataset
for resume safety; changing the input path does not silently replace it. Use a
new project directory for different data.

Measured mode may set process-level runtime defaults, load in-process GPU
models, launch vLLM process groups, open local service ports, and write
diagnostics. Entering `MeasuredProvider` reclaims evaluator-owned orphaned vLLM
processes. Its run-spanning cache may reuse a matching vLLM deployment across
calls; replacing a deployment and leaving the context release evaluator-owned
processes and cached models. The caller must not concurrently assign the same
GPU devices or project directory to another measured provider.

## License and attribution

This project includes code derived from AutoRAG. See `LICENSE.autorag` and
`NOTICE` for the Apache License 2.0 terms and required attribution.
