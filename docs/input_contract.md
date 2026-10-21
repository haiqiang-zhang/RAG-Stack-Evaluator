# Evaluator input and output contract

This document is the authoritative input/output contract for static quality
and measured evaluation. The README and runnable examples link here.
Inputs outside this contract are caller errors even if a particular
implementation happens to accept them. `static_gt` is the default backend;
other providers document their own settings and native metrics in the host.

## 1. Resolved pipeline config only

Both quality and measured evaluation accept one **fully resolved pipeline
configuration**. Every active vector database, node, component, model,
parameter, and deployment value must already be concrete.

`algo_config.json` stores this resolved algorithm mapping. `system_config.json`
stores one concrete deployment mapping. File names are CLI conventions; the
core API receives deserialized dictionaries. QA and corpus are separate caller
arguments; neither config contains `dataset`, dataset names, training/test
selection, or QA/corpus paths. Reusing a config pair with different QA/corpus is
an explicit caller choice. Project/run directories and credentials are also
invocation inputs, not algorithm fields.
The host CM replay command accepts corpus profile identity separately as
`--dataset-name`; this label selects existing FAISS workload profiles and is
runtime metadata, not a field in either persisted config.

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
core API does not accept config paths or YAML/JSON strings.

The runnable [algorithm example](../example/algo_config.json) is a complete
resolved generation pipeline with no retrieval; it pairs with the
[one-GPU deployment example](../example/system_config.json). A retrieval pipeline
additionally supplies its resolved `vectordb` entries and retrieval stages.
These examples require local model runtime resources when actually executed.

The required top-level evaluator fields are:

- `node_lines`: ordered pipeline lines, each with a stable `node_line_name`, an
  ordered `nodes` list, and exactly one concrete item in every node's `modules`;
- `vectordb`: the concrete stores referenced by active retrieval nodes, or an
  empty list when the pipeline does not use a vector database;
- `eval_backend_setting.metrics`: concrete metric names or mappings containing
  `metric_name` when quality scoring is enabled; measured performance-only calls
  and serving-only trace previews use `eval_backend_setting.performance_only: true`
  instead; and
- `corpus_runtime.chunker`: one concrete chunker mapping or `{}`.

`pipeline_runtime` is optional and defaults to sequential execution. ReAct
requires concrete `rag_dataflow: react` and `max_iter` values. Measured calls
may also include resolved runtime policy under the pipeline config, but the
physical deployment belongs in the separate `system_config` described below.
Persisted algorithms do not contain `system`, `deployment_presets`,
`system_design_space`, or other optimizer choices. Hardware, interconnect,
batching and concrete retrieval execution settings belong to `system_config`.
Do not persist data-derived `vectordb.N`: the dataset owner derives corpus size
from the explicit input. Legacy internal execution mappings may bind the
selected system under `pipeline_config.system`; this is runtime adaptation,
not an additional field in the algorithm file.

For the current serial hybrid workflow, dense `semantic_retrieval` (`vectordb`),
`lexical_retrieval` (`bm25`) and `hybrid_retrieval` (`hybrid_rrf`) are three
ordered nodes, followed by passage processing or generation. Each node still
has exactly one resolved module. This is sequential execution; parallel hybrid
and ReAct hybrid are not part of the supported input contract. A valid
quality/measured pipeline does not imply analytical CM coverage: unsupported
BM25/fusion cost bindings must fail explicitly.

## 2. QA Parquet

The QA input is an explicit `qa_data_path` constructor argument (or `--qa` in
the examples and host replay CLI). It is not read from either config. The
Parquet file contains these columns:

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

## 3. Corpus Parquet

Corpus selection is an explicit `corpus_data_path` constructor argument
(or `--corpus`). The normal, pre-chunked corpus contains:

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
`owner_config` is produced by the host's
`rag_stack.evaluator_contract.dataset_owner_config_from_resolved(algo_config)`.
It contains only concrete corpus/pipeline choices in the owner's internal shape.
QA/corpus remain separate `DatasetManager` constructor arguments.
It is not a third persisted config and must not be sent to `evaluate`. The host
owns chunking, current corpus-size/token-stat derivation, and this projection;
do not pass the resolved algorithm dict directly as `DatasetEvalManager.config`.
For raw corpora, use the host `python -m rag_stack replay ALGO_JSON SYSTEM_JSON
--qa QA_PARQUET --corpus CORPUS_PARQUET ...` entry point.

The caller must keep `qid` and `doc_id` stable within one project. All input
paths must identify Parquet files.

## 4. Quality API

The public quality API returns the host's typed `EvaluationResult`:

```python
from rag_stack.evaluation.registry import create_quality_evaluator
from rag_stack_evaluator.static_rag_evaluator import DatasetEvalManager

dataset = DatasetEvalManager(
    project_dir=project_dir,
    qa_data_path=qa_parquet,
    corpus_data_path=corpus_parquet,
)
evaluator = create_quality_evaluator(
    dataset_manager=dataset,
    project_dir=project_dir,
    backend="static_gt",
)
result = evaluator.evaluate(
    algo_config,  # deserialized, fully resolved dict
    run_dir=run_dir,
    metrics_override=None,
    on_trace_ready=None,
)
print(result.metrics)
trace = result.trace
```

This direct, config-less dataset manager requires a pre-chunked corpus and
`algo_config["corpus_runtime"]["chunker"] == {}`. Raw/dynamically chunked corpora
use the host-owned path in section 3. `metrics_override` changes scoring only.
If a canonical trace is produced, `on_trace_ready` receives the same object
returned as `result.trace`. Callback failures are advisory.

Recorded traces carry `provenance.trace_origin: observed` and
`provenance.producer: rag-stack-evaluator`. The legacy post-hoc fallback carries
`trace_origin: reconstructed` and never invokes the observed-trace callback.
Canonical shape alone does not establish an observed origin. Consumers such as
CM Preview require an observed marker before reusing a saved trace; older
unmarked traces require a fresh run for that workflow.

For a serving-only preview, set `eval_backend_setting.performance_only: true`
on a runtime copy and pass `metrics_override=[]`. The pipeline still executes
its retrieval, fusion, and generation stages and records their real calls.
Both final scoring and explicitly configured node-level metrics are disabled;
empty metrics do not represent a quality score. This flag does not select the
measured backend or change the deployment and generation parameters.

The concrete `StaticRAGEvaluatorQualityOnly` class is also supported. Its
`evaluate` method returns the legacy flat metric dict, with a canonical trace
under `__execution_dag__` when present. The registry adapts this concrete engine
to `EvaluationResult`. Do not call `.get("__execution_dag__")` on a typed result
or assume that a direct engine returns `EvaluationResult`.

## 5. MeasuredProvider and system_config

Measured evaluation uses the same resolved algorithm config and a separate
concrete deployment dict. `MeasuredProvider` requires static-engine execution
capabilities such as `_run_pipeline`; implementing only the generic public
`Evaluator.evaluate` protocol is insufficient. Both a concrete
`StaticRAGEvaluatorQualityOnly` and the registry's `static_gt` adapter provide
those capabilities. The adapter forwards the private execution methods; the
measured path does not call its typed quality `evaluate` method.

```python
from rag_stack_evaluator.static_rag_evaluator import MeasuredProvider

# evaluator is the registry-created static_gt instance from section 4.
with MeasuredProvider(
    evaluator,
    available_gpus=["cuda:0"],
    n_queries=None,
    selection="max_throughput",
) as provider:
    result = provider.evaluate(
        algo_config,
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
`layout.engines` block. The [system example](../example/system_config.json) contains the complete
one-GPU generator deployment paired with the algorithm example. A retrieval or
multi-stage pipeline needs corresponding stage/engine/resource records from the
host's resolver. Do not copy only `layout.engines` or infer missing records.

The complete `layout` must include `total_gpu_slots`, ordered performance-stage
records, resource groups, engine projections, and GPU occupants that agree with
one another. Every active GPU stage must be assigned to concrete devices. A
disaggregated generator additionally supplies concrete `generator_prefill` and
`generator_decode` role mappings under `layout.engines.generator`, each with
`devices`, `num_chips`, `tp`, and `pp`. Counts, batching settings, cache dtype,
placement, and parallelism must not contain optimizer choices. Missing derived
layout records are invalid input rather than defaults.

Measured runtime currently applies its 512-token generation cap by default.
The existing runtime-only `_measured_source_max_tokens` marker preserves an
explicit source cap during replay. This marker belongs to execution adaptation,
not the persisted algorithm schema. The shipped example uses an explicit 512
cap so its quality and measured requests agree. Transport/default adaptation
must preserve source sampling choices when replaying other config pairs.

`MeasuredProvider` must be used as a context manager. Calling `evaluate`
outside its context is an error.

## 6. Outputs

Public registry quality evaluation returns `EvaluationResult`:

| Field | Meaning |
| --- | --- |
| `metrics` | Finite metric scores. |
| `trace` | Canonical serving RAG-IR, or `None` when unavailable. |
| `workflow` | Workflow schema supplied/derived by the backend. |
| `runtime_stats` | Optional aggregates for explicit static CM mode; not a substitute for a dynamic trace. |
| `metadata` | Reserved host metadata. |

The concrete legacy engine instead returns a flat metric mapping. Only that
legacy mapping may include these reserved keys:

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

## 7. Filesystem and process lifecycle

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

Published benchmark results need the config pair, final numerical results and a
run note. Scratch traces, resource caches, attempts and diagnostics can be used
during evaluation; they are not additional replay configuration inputs.

Measured mode may set process-level runtime defaults, load in-process GPU
models, launch vLLM process groups, open local service ports, and write
diagnostics. Entering `MeasuredProvider` reclaims evaluator-owned orphaned vLLM
processes. Every `evaluate` call tears down all vLLM processes it launched,
whether the call succeeds or fails; leaving the context also releases cached
in-process models. The caller must not concurrently assign the same GPU devices
or project directory to another measured provider.
