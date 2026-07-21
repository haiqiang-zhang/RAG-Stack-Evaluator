# RAG-Stack Evaluator

RAG-Stack Evaluator contains the static quality evaluator, the real-hardware
measured evaluator, and the vLLM instrumentation used by measured runs. It
exposes them through its own top-level Python package:

```python
from rag_stack_evaluator.static_rag_evaluator import (
    StaticRAGEvaluatorQualityOnly,
    MeasuredProvider,
)
```

This distribution owns the `rag_stack_evaluator` package, including
`rag_stack_evaluator.static_rag_evaluator` and
`rag_stack_evaluator.vllm_instrumentation`. It does not contribute modules to
the host's `rag_stack` namespace.

This is currently a RAG-Stack host subproject, not a standalone replacement
for the host package. The evaluator continues to consume RAG-Stack's shared
dataset, IR, layout, security, and cost-model contracts through `rag_stack.*`.
Install and run it through a compatible RAG-Stack checkout; the direct install
commands below are for developing the submodule in that host environment.

## Installation

Compatibility for this release is explicit:

| Evaluator | Required host |
| --- | --- |
| `rag-stack-evaluator==0.0.1` | [`RAG-Stack==0.0.1`](https://github.com/haiqiang-zhang/rag-stack) source checkout containing the `RAG-Stack-Evaluator` submodule integration |

The evaluator wheel deliberately does not declare the host as a dependency,
because the host in turn pins this evaluator as a submodule/workspace member.
Installing this repository alone can therefore build successfully and import
the package root, but evaluator submodules that consume host-owned contracts
still require RAG-Stack. From a compatible host revision, use the host-root
installation flow:

```bash
git clone --recurse-submodules https://github.com/haiqiang-zhang/rag-stack.git
cd rag-stack
# If needed, check out a host revision that pins
# `rag-stack-evaluator==0.0.1`; it may not yet be the default branch.
# Create and activate the conda environment as documented by the host first.
uv pip install -e 'RAG-Stack-Evaluator[cu12]' -e '.[cu12]'
# On NVIDIA driver >=580, use cu13 in both editable requirements instead.
```

The commands below are only for developing this submodule after the compatible
host is already installed. Use Python 3.10 or newer and install exactly one
CUDA stack when local GPU models or measured evaluation are required:

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
`RAG-Stack-Evaluator` submodule and installs both local projects as editable
distributions (or resolves them through the parent workspace), so the host's
shared `rag_stack` contracts and the evaluator package are available together.

## Network safety

The retained vLLM launch tools default to `0.0.0.0` for compatibility with
existing benchmark hosts. The multi-model gateway has no authentication layer.
Do not expose either service to an untrusted network. For local use, bind
explicitly to loopback:

```bash
JUDGE_HOST=127.0.0.1 \
  bash scripts/start_vllm_judge_server.sh

python scripts/vllm_multimodel_gateway.py \
  --host 127.0.0.1 --port 8000 \
  --backend MODEL=http://127.0.0.1:8101
```

For remote use, restrict access with the host firewall or an authenticated
reverse proxy. A vLLM API key can be passed as an extra server argument (for
example `-- --api-key "$JUDGE_API_KEY"`), but this does not add authentication
to the gateway itself.

## Tests

Run evaluator tests from an initialized compatible host checkout so the shared
`rag_stack.*` contracts are present. A lightweight, non-GPU check is:

```bash
uv pip install -e 'RAG-Stack-Evaluator[test,faiss]' -e .
python -m pytest \
  RAG-Stack-Evaluator/tests/test_vllm_multimodel_gateway.py
```

The complete suite includes FAISS, model-backend, vLLM, and measured-runtime
cases. Use `RAG-Stack-Evaluator[test,cu12,faiss]` or
`RAG-Stack-Evaluator[test,cu13,faiss]` on a dedicated compatible GPU host; do
not run hardware tests on a shared benchmark server.

## Input Contract

This section defines the evaluator's single public input contract. Inputs
outside this contract are caller errors even if a particular implementation
happens to accept them.

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
`modules`, `metrics`, and device lists, remain lists. Every node must use
`modules` with exactly one mapping containing one resolved `component`.
Although the quality-only parser also accepts singular `module`, the shared
quality-and-measured contract does not: measured deployment consumes the
single-item `modules` list. Upstream search or optimization code is responsible
for selecting an arm and resolving it before calling the evaluator.

A caller passes the resolved configuration as an in-memory Python `dict`. The
core API does not accept a YAML path, YAML file, or YAML string:

```python
resolved_pipeline_config = {
    "dataset": {"dataset_name": "example"},
    "corpus_runtime": {"chunker": {}},
    "pipeline_runtime": {"mode": "sequential"},
    "vectordb": [
        {
            "name": "example_hnsw",
            "db_type": "faiss_hnsw",
            "embedding_model": "mock",
            "embedding_dim": 768,
            "collection_name": "example",
            "path": "/absolute/path/to/project/resources/faiss",
            "similarity_metric": "cosine",
            "M": 32,
            "ef_construction": 200,
            "ef_search": 64,
        },
    ],
    "node_lines": [
        {
            "node_line_name": "retrieval",
            "nodes": [
                {
                    "stage": "semantic_retrieval",
                    "strategy": {"metrics": [], "strategy": "mean"},
                    "top_k": 4,
                    "modules": [
                        {
                            "component": "vectordb",
                            "vectordb": "example_hnsw",
                            "ef_search": 64,
                        },
                    ],
                },
            ],
        },
        {
            "node_line_name": "generation",
            "nodes": [
                {
                    "stage": "prompt_maker",
                    "modules": [
                        {
                            "component": "fstring",
                            "prompt": (
                                "Question: {query}\n"
                                "Context: {retrieved_contents}\n"
                                "Answer:"
                            ),
                        },
                    ],
                },
                {
                    "stage": "generator",
                    "modules": [
                        {
                            "component": "vllm",
                            "model": "Qwen/Qwen2.5-1.5B-Instruct",
                            "max_tokens": 128,
                            "temperature": 0.0,
                        },
                    ],
                },
            ],
        },
    ],
    "eval_backend_setting": {
        "metrics": [
            {"metric_name": "retrieval_token_recall"},
            {"metric_name": "retrieval_token_precision"},
            {"metric_name": "retrieval_token_f1"},
        ],
    },
}

quality = evaluator.evaluate(resolved_pipeline_config)
```

The required top-level evaluator fields are:

- `node_lines`: ordered pipeline lines, each with a stable `node_line_name`, an
  ordered `nodes` list, and exactly one concrete item in every node's `modules`;
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
| `start_end_idx` | pair of integers | Source-text offsets; also marks the corpus as already chunked. |

A raw corpus supplied through the RAG-Stack host's owner-managed dataset path
contains `doc_id` and `contents`; the legacy text-column name `texts` is
accepted. A raw corpus must not contain `start_end_idx`. A corpus carrying
`start_end_idx` is treated as already chunked and must not be passed through
another chunker.

The direct path-based API in section 4 is intentionally narrower: it accepts a
pre-chunked corpus and `corpus_runtime.chunker: {}`. A raw corpus or a non-empty
runtime chunker needs the host-owned
`rag_stack.dataset_manager.DatasetManager(config=owner_config, ...)`
path because a chunk-cache miss must derive fresh token statistics. Here
`owner_config` is the host's full configuration document (including its corpus
and pipeline definition), not the resolved runtime mapping passed to
`evaluate`.

The caller must keep `qid` and `doc_id` stable within one project. All input
paths must identify Parquet files.

### 4. Quality API

The stable path-based quality API is:

```python
from rag_stack_evaluator.static_rag_evaluator import DatasetEvalManager
from rag_stack_evaluator.static_rag_evaluator import StaticRAGEvaluatorQualityOnly

dataset = DatasetEvalManager(
    project_dir=project_dir,
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

This direct, config-less `DatasetEvalManager` form is for a pre-chunked corpus and
requires `resolved_pipeline_config["corpus_runtime"]["chunker"] == {}`. The
fully resolved runtime mapping belongs on `evaluate`. Passing that runtime
mapping as `DatasetEvalManager.config` is incorrect because a non-empty manager
config is the host's full optimizer/configuration document used to derive
cost-model token statistics, not the section 1 runtime shape. Use the
owner-managed form described in section 3 for raw or dynamically chunked input.

`resolved_pipeline_config` is the contract from section 1. `metrics_override`,
when supplied, is a concrete list of metrics and changes scoring only.
`on_trace_ready`, when supplied and a recorded canonical trace is produced, is
called with the same envelope later returned under
`quality["__execution_dag__"]`; hook failures are advisory and do not invalidate
the quality run.

### 5. MeasuredProvider and system_config

Measured evaluation uses the same evaluator and resolved pipeline config. The
physical deployment is a second, fully resolved mapping named `system_config`:

```python
from rag_stack_evaluator.static_rag_evaluator import MeasuredProvider

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

`system_config` must be a single concrete deployment, not a design space. The
canonical producer is the compatible RAG-Stack host's
`PerformanceContext.resolve_system_config(...)`; callers should pass through
that complete result or a persisted JSON copy of it. It is not merely the
`layout.engines` block. The following in-memory Python `dict` fragment
illustrates the required derived records for a one-GPU generator. A real result
contains equivalent records for every active performance stage:

```python
system_config = {
    "batch_size_request": 4,
    "measured_load_concurrency": 4,
    "measured_warmup_queries": 4,
    "measured_queries": 16,
    "batching": {"dynamic_timeout_s": 0.0},
    "retrieval": {
        "faiss_num_threads": 1,
        "faiss_ivf_parallel_mode": 0,
        "num_servers": 1,
    },
    "vllm": {
        "kv_cache_dtype": "auto",
        "engines": {"generator": {"max_num_seqs": 4}},
    },
    "layout": {
        "total_gpu_slots": 1,
        "min_total_gpu_slots": 1,
        "max_total_gpu_slots": 1,
        "performance_stage_order": [
            "generator_prefill",
            "generator_decode",
        ],
        "resource_groups": [
            {
                "id": "gpu:0",
                "kind": "gpu",
                "devices": ["cuda:0"],
                "performance_stages": [
                    "generator_prefill",
                    "generator_decode",
                ],
            },
        ],
        "performance_stages": {
            "generator_prefill": {
                "kind": "gpu",
                "resource": "gpu:0",
                "devices": ["cuda:0"],
                "num_chips": 1,
                "engine": "generator",
                "role": "prefill",
                "tp": 1,
                "pp": 1,
            },
            "generator_decode": {
                "kind": "gpu",
                "resource": "gpu:0",
                "devices": ["cuda:0"],
                "num_chips": 1,
                "engine": "generator",
                "role": "decode",
                "tp": 1,
                "pp": 1,
            },
        },
        "engines": {
            "generator": {
                "pd_serving": "collocated_pd",
                "devices": ["cuda:0"],
                "num_chips": 1,
                "performance_stages": [
                    "generator_prefill",
                    "generator_decode",
                ],
                "tp": 1,
                "pp": 1,
            },
        },
        "gpu_occupants": {
            "cuda:0": ["generator_prefill", "generator_decode"],
        },
    },
}
```

The complete `layout` must include `total_gpu_slots`, ordered performance-stage
records, resource groups, engine projections, and GPU occupants that agree with
one another. Every active GPU stage must be assigned to concrete devices. A
disaggregated generator additionally supplies concrete `generator_prefill` and
`generator_decode` role mappings under `layout.engines.generator`, each with
`devices`, `num_chips`, `tp`, and `pp`. Counts, batching settings, cache dtype,
placement, and parallelism must not contain optimizer choices. Missing derived
layout records are invalid input rather than defaults.

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
- run-specific evaluator artifacts below those locations.

Content-addressed embedding and FAISS caches are separate from `project_dir`.
In the supported host integration, `rag_stack` sets `RAG_STACK_CACHE_DIR` to
`<host-checkout>/.cache/rag_stack`. Without that bootstrap, the fallback is
derived from the installed evaluator source location, which may be unwritable
for a non-editable wheel. Set `RAG_STACK_CACHE_DIR` explicitly to a writable
shared cache root in that case. The legacy `RAG_STACK_EMBED_CACHE` override, when
set, still takes precedence for embedding vectors only.

Once `project_dir/data/` is populated, subsequent runs reuse that bound dataset
for resume safety; changing the input path does not silently replace it. Use a
new project directory for different data.

Measured mode may set process-level runtime defaults, load in-process GPU
models, launch vLLM process groups, open local service ports, and write
diagnostics. Entering `MeasuredProvider` reclaims evaluator-owned orphaned vLLM
processes. Every `evaluate` call tears down all vLLM processes it launched,
whether the call succeeds or fails; leaving the context also releases cached
in-process models. The caller must not concurrently assign the same GPU devices
or project directory to another measured provider.

## License and attribution

Most redistributed derived code is under Apache License 2.0; see
`LICENSE.autorag` and `NOTICE` for AutoRAG attribution. Two bundled TART model
and tokenizer implementation files are under CC BY-NC 4.0; see `LICENSE.tart`
and `NOTICE`. That TART-derived portion is restricted to non-commercial use, so
the combined distribution is not Apache-only and is not suitable for commercial
use without replacing those files or obtaining separate permission.
