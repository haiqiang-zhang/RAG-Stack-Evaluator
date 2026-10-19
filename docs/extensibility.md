# Evaluator API Reference

An evaluator supplies two observations for each resolved candidate: scalar
quality scores and RAG-IR, the pair `(workflow, trace)`. RAG-Stack owns search,
objective selection, caching, and performance modeling. An external backend owns
pipeline execution, tracing its serving components, and quality scoring.

## 1. Select an extension boundary

| Need | Extension point | What remains reusable |
| --- | --- | --- |
| Score the existing static pipeline with another judge | Generation/retrieval metric adapter | Static execution, instrumentation, aggregation |
| Run and score a different pipeline/evaluation framework | `Evaluator` protocol and `BackendSpec` | Optimizer, quality store, RAG-CM for supported components |
| Capture an external pipeline's component I/O | `TraceSession` public hooks | Canonical RAG-IR validation and replay |

The backend is replaceable; replacing the entire `rag-stack-evaluator`
distribution is unnecessary. That package remains the integration layer for
backend discovery, shared dataset adapters, and instrumentation. External
plugins can live in separate distributions, without editing its factory or the
host controller. Existing `BaseEvaluator` subclasses and flat dictionaries remain
supported for built-in backends.

## 2. Public Python API

```python
from rag_stack_evaluator.api import EvaluationResult, Evaluator, publish_trace
from rag_stack_evaluator.factory import BackendSpec, EvaluatorContext, register_backend
from rag_stack_evaluator.tracing import TraceSession
```

These imports do not load model runtimes. Final trace/workflow validation imports
the compatible host's lightweight `rag_stack.rag_ir` contract. The evaluator
package still requires a compatible RAG-Stack installation for actual execution.

### Construction context

| `EvaluatorContext` field | Meaning |
| --- | --- |
| `dataset_manager` | Host-owned active QA/corpus data and chunk cache; do not construct a second owner |
| `project_dir` | Project directory, if supplied |
| `options` | Selected backend's `eval_backend_setting.backend_options` dictionary |
| `dataset` | Legacy direct-construction argument; optional |

A backend factory accepts one context and returns any object implementing:

```python
def evaluate(self, config, run_dir=None, metrics_override=None,
             on_trace_ready=None) -> EvaluationResult:
    ...
```

`config` is a resolved candidate, with one selected component and concrete knob
values. Translate those choices to the external runner; silently ignoring them
would invalidate an optimization study. Honor `metrics_override` per call without
changing the caller's config or default metric list. `run_dir` is the destination
for per-evaluation artifacts. Let execution/scoring errors propagate so the host
can record failure; do not turn them into successful zero scores.

### Result contract

```python
EvaluationResult(
    metrics={"answer_quality": 0.83},
    trace=trace,
    workflow=workflow,  # WorkflowSchema or its JSON dictionary; optional
)
```

* `metrics`: named, finite numeric scalar scores. Booleans, NaN and infinity are
  rejected. The interface imposes no universal `[0, 1]` range. Configure objective
  direction and aggregation for each metric's meaning.
* `trace`: one complete invocation for every active QA question, joined by its
  stable question ID. External backends must return it; missing/extra/duplicate
  requests fail validation. Query order may differ from dataset order.
* `workflow`: `rag_stack.rag_ir.WorkflowSchema` or its dictionary form. Omit it
  only when the resolved host configuration already describes the runner's
  structure. Supply it explicitly when the adapter owns orchestration.

`normalize_evaluation_result` is the compatibility boundary. It translates the
result to existing `__execution_dag__` / `__workflow_schema__` fields, preserves
the trace object's identity, validates metrics and trace, and serializes the
workflow into `trace.provenance.workflow_schema`. The host removes metadata from
optimizer objectives and binds the workflow when pricing live or cached traces.
Only built-in legacy flat results retain the existing aggregate/static fallback.

### Early trace publication

Finish serving before calling judges, then optionally publish:

```python
trace = recorder.finish(expected_question_ids=question_ids)
publish_trace(trace, workflow=workflow, on_trace_ready=on_trace_ready)
scores = score_predictions(predictions)
return EvaluationResult(metrics=scores, trace=trace, workflow=workflow)
```

This lets the host overlap cost modeling with slow judging. `publish_trace`
validates and attaches the workflow **before** the callback. It logs callback
failures and continues scoring; the host can retry cost modeling afterward.
Return the same trace object and treat its contents as read-only after publication.
A callback is optional; an adapter can simply return the completed result.

## 3. Instrument component boundaries

Use context managers around existing component calls. No inheritance, global
monkeypatch, or framework-specific callback base class is required.

```python
recorder = TraceSession()
with recorder.query(question_id):
    # Existing code renders/tokenizes the actual model prompt.
    prompt_ids = tokenizer.encode(rendered_prompt)
    with recorder.stage("generator", model_id=model_id) as call:
        call.input(token_ids=prompt_ids, text=rendered_prompt)
        answer = generator.generate(prompt_ids)
        call.output(token_ids=answer.token_ids, text=answer.text)
trace = recorder.finish(expected_question_ids=[question_id])
```

The snippet assumes the runner already supplies its tokenizer, generator, and
output token IDs; it does not construct them. Record the actual chat template,
retrieval context, and tokens used by the model. Generated token IDs exclude the
prompt. For streaming, aggregate the completed response before recording output.

### Hook payloads

| Input/output argument | Single request | Batch |
| --- | --- | --- |
| Actual token IDs | `token_ids=[1, 2]` | `token_ids=[[1, 2], [3]]` |
| Exact runtime token count | `token_count=2` | `token_counts=[2, 1]` |
| Text with the component's explicit tokenizer callable | `text=prompt, tokenizer=encode` | `texts=prompts, tokenizer=encode` |
| Optional exact bytes | `byte_count=12` | `byte_counts=[12, 8]` |

Choose exactly one token representation. Optional `text`/`texts` alongside
explicit tokens provides UTF-8 byte sizes; raw text and token IDs are not retained
in the trace. Text alone fails: the hooks never download a tokenizer or estimate
counts from whitespace. Missing byte sizes remain absent so CM fallback semantics
are preserved. For a one-request batch, `token_ids` is a flat list.

Both hooks are required, including `output(token_count=0)` for a non-generative
component. Counts must come from the relevant component/tokenizer. For retrieval,
output token count is zero; retrieved text belongs in the downstream generator
prompt's input count and in the quality scorer's prediction record.

```python
with recorder.batch(question_ids):
    with recorder.stage("generator", model_id=model_id) as call:
        call.input(token_counts=prompt_counts, texts=prompts)
        answers = generate_batch(prompts)
        call.output(token_counts=completion_counts, texts=answers)
```

This records one call per question, without embedding runtime batch shape in
RAG-IR. Payload lengths must match the batch exactly.

### Stage semantics

| Trace stage | Boundary to wrap |
| --- | --- |
| `query_expansion` | Expansion model call; exact expansion prompt/completion |
| `semantic_retrieval_encode` | Query embedding forward pass |
| `semantic_retrieval_vectorsearch` | Vector index search; separate from encoding |
| `lexical_retrieval` | Lexical retrieval operation |
| `passage_reranker` | Reranking operation, using its component's accounting convention |
| `passage_compressor` | Compression operation |
| `generator` | Each generator call, including intermediate agent iterations |

Semantic retrieval is always split into encoding and vector search. Judge calls
are excluded: they measure quality, not candidate serving cost. Instrument each
boundary once. Public hooks target uninstrumented external components; wrapping
an already instrumented static component can double-count calls, and deferred
legacy text records are rejected by public finalization.

### Ordering, concurrency, and failures

Each query has ordered calls with monotonically increasing `step_idx` and a
terminal generator call. Different queries can run in concurrent asyncio tasks
or threads; enter a query/batch scope in each worker using the shared session.
ContextVars restore the enclosing session and IDs on scope exit. Overlapping
stages for the same query are rejected because the current IR cannot represent
parallel branches within one request.

Exceptions, missing hooks, repeated hooks, unfinished scopes, and wrong question
coverage make the session unpublishable. `finish` freezes it against further
writes. Repeating `finish` with the same IDs returns the same trace object.

## 4. Describe the workflow for performance modeling

A canonical trace is JSON:

```json
{
  "queries": [{
    "question_id": "turn-1",
    "calls": [{
      "stage": "generator", "step_idx": 0,
      "input_tokens": 128, "output_tokens": 24, "model_id": "model-name"
    }]
  }]
}
```

The companion `WorkflowSchema` describes `dataflow_type` (`sequential` or
`agentic`), performance-stage names, component-presence flags, and optional
corpus/runtime statistics. Performance stages differ from trace stages: for
example one `generator` call is priced using `generator_prefill` and
`generator_decode`.

```python
from rag_stack.rag_ir import build_workflow_schema
workflow = build_workflow_schema(resolved_config)
```

For custom orchestration, construct an explicit schema describing the actual
runner. Its performance stages and presence flags must match the host's resolved
cost-model components. Explicit corpus statistics are retained; omitted corpus
statistics are supplied from the host's resolved dataset metadata. Every observed trace stage must have a corresponding resolved cost-model
component; configured conditional stages may remain unobserved. Complete
instrumentation of executed components remains the adapter's responsibility.
The workflow travels in trace provenance and in persisted resolved configs, including cache
replay through the host controller. Direct RAG-CM callers must supply the
matching workflow explicitly. Quality-store identity includes the selected backend and backend options.
Version the external runner/judge and include those identities in options.

Registration does not implement new hardware cost models. A new unsupported
component, tokenizer/model mismatch, or an arbitrary parallel dependency graph
requires separate IR/CM work. The current trace is an ordered invocation, not a
timestamped runtime profile. `supports_measured` is descriptive metadata; a plugin
does not automatically receive the built-in physical measurement runner.
External backend integration currently uses `performance_source: cost_model`
and `rag_ir_mode: dynamic`.

## 5. Register and select a backend

```python
# my_eval/backend.py
from rag_stack_evaluator.factory import BackendSpec

BACKEND = BackendSpec(
    name="my_eval",
    factory=MyEvaluator,  # MyEvaluator(context) implements evaluate(...)
    metrics=("answer_quality",),
    pipeline_modes=("sequential",),
    pipeline_style="node_lines",
)
```

For scripts embedding RAG-Stack, call `register_backend(BACKEND)` before creating
the controller. For CLI and subprocess discovery, install an entry-point plugin:

```toml
[project.entry-points."rag_stack.evaluators"]
my_eval = "my_eval.backend:BACKEND"
```

The entry point resolves to a `BackendSpec` object with the same name. Duplicate
registrations and ambiguous installed names fail. Discovery lists package
metadata without loading all plugins; selecting a backend loads its descriptor.
Keep model imports inside its factory/runtime. `metrics` and `pipeline_modes`
can also be lazy zero-argument callables returning tuples.

```yaml
global:
  eval_backend: my_eval
  performance_source: cost_model
  rag_ir_mode: dynamic
eval_backend_setting:
  metrics: [answer_quality]
  backend_options:
    runner_revision: your-pinned-revision
    # Additional fields are defined by your adapter.
```

This is an overlay on a complete host config, not a standalone optimization
configuration. Keep the dataset, system, optimizer and algorithm search-space
sections. `pipeline_style: node_lines` accepts the normal node-line representation
(and method blocks for advertised modes); `methods` requires explicit method
blocks. Backend metadata drives validator and registry APIs. Use YAML for plugin
options; the visual editor has no custom plugin-specific controls. The generic
scaffold does not invent valid options for an external backend.

`flashrag_options` remains accepted by the factory for existing FlashRAG callers;
do not combine it with `backend_options`. Existing static/measured interfaces are
unchanged.

## 6. MT-RAG example and CPU validation

See [the runnable MT-RAG example](../examples/mtrag/README.md). It contains an
installable backend, an external pipeline callback, public tracing hooks,
multiple selected official score keys, a subprocess bridge to the official
MT-RAG generation-evaluation script, and a deterministic CPU fixture.

MT-RAG evaluates saved responses and contexts; tracing must be added to the
pipeline that produced them. The adapter sends only task ID and conversation
input to the runner. References and benchmark labels reach the judge afterward.

From the host root, with compatible host/evaluator packages installed:

```bash
python RAG-Stack-Evaluator/examples/mtrag/run_cpu.py \
  --output /tmp/mtrag-interface-example
python -m pytest RAG-Stack-Evaluator/tests/test_evaluator_plugins.py \
  RAG-Stack-Evaluator/tests/test_public_tracing.py \
  RAG-Stack-Evaluator/tests/test_mtrag_backend_example.py -q
```

These validate contracts, task alignment, hook behavior, workflow transport,
callback order, and score transport using fixtures. They do not establish formal
MT-RAG quality or RAG-CM accuracy. The example README separately documents the
real upstream command and dependencies for collaborators running experiments.
