# MT-RAG as an external evaluator

This example connects an independently implemented RAG pipeline and IBM's
**complete generation evaluator** to RAG-Stack. The pipeline produces answers
and a serving trace through public hooks. MT-RAG scores the answers afterward.
The adapter returns `EvaluationResult(metrics, trace, workflow)` through the
same interface as built-in evaluator backends.

RAG-Stack owns the interface, registry, and hooks. This example imports
`rag_stack.evaluation` directly; it does not import `rag_stack_evaluator`.
Its location in this repository is for distribution of the example only.
The host's [Evaluator API Reference](https://github.com/haiqiang-zhang/rag-stack/blob/main/docs/evaluator_api.md)
describes the contract shared by this plugin, `static_gt`, and `flashrag`.

## 1. What this example demonstrates

```text
Resolved candidate configuration + benchmark conversation history
    -> external pipeline, instrumented with TraceSession
    -> actual answers, actual final contexts, workflow, complete serving trace
    -> publish_trace(...): RAG-CM can begin consuming the serving trace
    -> IBM run_generation_eval.py in its own environment
    -> selected per-task quality scores -> means -> EvaluationResult
```

The adapter does not execute the benchmark's reference answers or reference
contexts as the tested pipeline. It joins those references to the actual
outputs only for judging. It uses each task's published conversation history;
it does not feed this run's previous generated answer into the next turn.

| Component | Responsibility |
| --- | --- |
| `rag_stack_mtrag/backend.py` | External backend descriptor, workload selection, pipeline invocation, publication of trace/workflow, and quality aggregation |
| Your `module:run_pipeline` callable | Interpret candidate knobs, run your own components, record their actual I/O, and declare the actual workflow |
| `rag_stack_mtrag.scoring.score_with_mtrag` | Hash-verified task mapping, subprocess transport, evidence/ID checks, and selected-score parsing |
| `rag_stack_mtrag/run_official.py` | Verify the upstream checkout/configuration and invoke the full official generation CLI |
| `rag_stack_mtrag/fixture.py` and `run_cpu.py` | Small, deterministic CPU contract fixtures; their scores and components are not benchmark results |

The full official generation entry point runs algorithmic metrics,
underspecified-question evaluation, IDK judging, RAGAS faithfulness, RAD-Bench
judging, and conditioned metrics. The plugin exposes these scalar selections:

| RAG-Stack metric | Official output key |
| --- | --- |
| `mt_rag_faithfulness` | `RL_F` |
| `mt_rag_radbench` | `RB_llm` |
| `mt_rag_aggregate` | `RB_agg` |
| `mt_rag_faithfulness_idk` | `RL_F_idk_underspecified` |
| `mt_rag_radbench_idk` | `RB_llm_idk_underspecified` |
| `mt_rag_aggregate_idk` | `RB_agg_idk_underspecified` |

Selecting fewer returned metrics does **not** prune the official full-suite
command. It only selects the metrics passed back to RAG-Stack. The optional
`run_official_faithfulness.py` script preserves the earlier single-judge example.
For the static backend's optional `mt_rag_faithfulness` metric, install this
example package too; the legacy `rag_stack_evaluator.integrations.mtrag` imports
forward to `rag_stack_mtrag.scoring`.

## 2. Run the CPU interface example

From a RAG-Stack checkout with its Python environment available:

```bash
PYTHONPATH="$PWD:$PWD/RAG-Stack-Evaluator/examples/mtrag" \
python RAG-Stack-Evaluator/examples/mtrag/run_cpu.py \
    --output /tmp/mtrag-interface-example
```

Choose a new output directory for each run. The command registers the backend
in its own Python process, runs a two-task lexical-retrieval/extractive-generation
fixture, invokes a synthetic scoring subprocess, and writes `report.json` plus
audited scoring inputs/outputs. No model, official judge, network service, or
cost-model simulation is started. The fixture's components have no calibration
and are not suitable for a performance claim.

The two fixture tasks yield three named fixture metrics and two serving calls
per task: `lexical_retrieval` and `generator`. The recorded generator input
includes history, final contexts, and the rendered answer prompt. There are no
judge calls in this trace.

Focused tests, from the evaluator checkout:

```bash
PYTHONPATH="$PWD/..:$PWD/examples/mtrag" \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest tests/test_mtrag_adapter.py tests/test_mtrag_backend_example.py -q
```

These tests cover the real plugin loader, host QA subsets, workflow publication
before scoring, stable-ID joins despite reordered scorer output, missing or
duplicate results, changed evidence, invalid scores, callback failures, and
the exact official CLI invocation using mocks. They never run the official
judge. `validate_cpu.py` separately checks mapping against pinned public MT-RAG
data and replays published responses; it performs no rescoring.

## 3. Implement your pipeline with hooks

The callable receives a resolved candidate configuration, a list of requests
containing only `task_id` and structured conversation `input`, a fresh recorder,
and backend-specific pipeline options. Return `PipelineRun`:

```python
from rag_stack_mtrag.backend import PipelineRun, Prediction

def run_pipeline(config, requests, recorder, *, options):
    components = build_my_components(config, options)
    outputs = []
    for request in requests:
        with recorder.query(request["task_id"]):
            query = components.prepare_query(request["input"])
            with recorder.stage("lexical_retrieval", model_id="bm25") as call:
                call.input(token_count=components.retrieval_token_count(query), text=query)
                contexts = components.retrieve(query)
                call.output(token_count=0)

            # Tokenize the fully rendered prompt, including the chat template,
            # history, and contexts actually supplied to the generator.
            prompt, prompt_ids = components.prepare_prompt(request["input"], contexts)
            with recorder.stage("generator", model_id=components.model_id) as call:
                call.input(token_ids=prompt_ids, text=prompt)
                response = components.generate(prompt_ids)
                call.output(token_ids=response.token_ids, text=response.text)
        outputs.append(Prediction(request["task_id"], response.text, contexts))
    return PipelineRun(outputs, workflow=components.workflow_schema())
```

`build_my_components` represents your framework-specific construction code.
The executable fixture provides a complete small implementation of this
contract. A production implementation must honor every applicable candidate
parameter (models, top-k, reranker/compressor choices, prompts, and so on).
Record actual model tokens or exact reported usage; fixture tokenization must
not be used to estimate a production model's token count. Dense retrieval needs
separate `semantic_retrieval_encode` and `semantic_retrieval_vectorsearch`
hooks. Optional reranking/compression hooks should capture their actual inputs.

The runner returns one prediction and one complete query trace per active host
QA ID. It must declare a workflow consistent with the execution and candidate
configuration. Registering this backend does not add support for new component
types to RAG-CM. Use already supported components for performance modeling, or
implement their cost models separately. This example declares sequential
quality evaluation only; it does not provide measured-performance execution.

## 4. Install the plugin for CLI and worker discovery

From the host checkout, with RAG-Stack available:

```bash
python -m pip install --no-deps -e RAG-Stack-Evaluator/examples/mtrag
```

The example's `pyproject.toml` declares:

```toml
[project.entry-points."rag_stack.evaluators"]
mtrag = "rag_stack_mtrag.backend:BACKEND"
```

Each new host process can now resolve `global.eval_backend: mtrag`. Explicit
`rag_stack.evaluation.registry.register_backend(BACKEND)` is an alternative for an embedded Python program;
that registration does not propagate to new CLI/worker processes.

## 5. Configure the official MT-RAG evaluator

Use a separate environment provisioned with MT-RAG's dependencies and the
desired judge resources. The wrapper targets upstream commit
[`53fef6f5917c7e10c436f697e69a955dbc795d81`](https://github.com/IBM/mt-rag-benchmark/tree/53fef6f5917c7e10c436f697e69a955dbc795d81).
This implementation was checked against that entry point; running its judges
requires an explicit experimental run beyond CPU interface validation.

The following is an **overlay for an existing valid six-section RAG-Stack
configuration**, not a complete optimization configuration. Keep the dataset,
search space, hardware, and optimizer settings appropriate to your pipeline:

```yaml
global:
  eval_backend: mtrag
  performance_source: cost_model
  rag_ir_mode: dynamic

eval_backend_setting:
  metrics: [mt_rag_faithfulness, mt_rag_radbench, mt_rag_aggregate]
  backend_options:
    tasks_path: /data/mt-rag/reference.jsonl
    tasks_sha256: b3dde4076f9f692a99513e424a0a39f57591dbe45600a4c1017aaa7a7edcfaac
    pipeline: my_rag_adapter:run_pipeline
    pipeline_identity: my-rag-adapter@EXACT_COMMIT-and-resource-configuration
    pipeline_options:
      corpus_path: /data/my-corpus
    upstream_revision: 53fef6f5917c7e10c436f697e69a955dbc795d81
    scorer_identity: mt-rag@53fef6f5;judge=YOUR_MODEL_AND_REVISION;env=YOUR_LOCKFILE_SHA256
    timeout_seconds: 3600
    scorer_command:
      - /environments/mt-rag/bin/python
      - /path/RAG-Stack-Evaluator/examples/mtrag/rag_stack_mtrag/run_official.py
      - --repo
      - /path/mt-rag-benchmark
      - --revision
      - 53fef6f5917c7e10c436f697e69a955dbc795d81
      - --provider
      - hf
      - --judge-model
      - YOUR_JUDGE_MODEL
      - --algorithmic-config-sha256
      - a9a9b1cca8d8cc843084e70479dfcea9e00fa628f5d2fa75da16001629858cc2
      - --input
      - "{input}"
      - --output
      - "{output}"
```

Update objective, combined-quality, and any stage-strategy metric references
to names exposed by this backend. Built-in metrics such as `rouge` or
`retrieval_f1` are not automatically implemented by the MT-RAG plugin. Use
cost-model performance evaluation; this example does not support measured mode.

`tasks_path` must identify the exact sidecar used to construct the host QA data.
Use `load_tasks` and `to_qa_rows` from `rag_stack_mtrag.scoring`
to preserve task IDs, full role-labelled histories, and reference answers.
When the host supplies `dataset_manager`, its active `qa_data.qid` values select
the subset and order to evaluate. Standalone construction without a manager
evaluates the whole sidecar.

The wrapper checks `git HEAD`, tracked changes under `scripts/evaluation`, and
the algorithmic configuration hash, then replaces itself with the official
`run_generation_eval.py` process. It exposes the `hf` and `vllm` providers.
Other provider/environment arrangements can use a different explicit argv
command with the same JSONL input/output contract. Commands are trusted
extension code, executed without a shell. Keep credentials out of YAML and
arguments; audited provenance records the command.

Keep task, pipeline/resource, upstream, judge/model, and dependency/configuration
identities explicit in the options. The host includes backend options in its
quality-cache identity. `pipeline_identity` and `scorer_identity` are declared
provenance; the adapter cannot verify arbitrary installed code or model files
from a label. Update these identities when their effective implementations or
resources change.

## 6. Result validation and limits

Scored records must preserve task IDs, histories, reference targets, final
contexts, and generated answers. Every selected metric must be a one-element
list containing a finite number for every requested task. Missing/duplicate
tasks, changed evidence, failed subprocesses, timeouts, null/NaN results, and
multi-value arrays fail the evaluation. No failure is replaced by zero and no
task is silently dropped. The bridge aligns scores by ID, then the backend
takes the mean over all requested tasks.

The generic bridge accepts any explicitly selected upstream scalar key and
optional metric-specific bounds. The example descriptor exposes six named
scalar metrics. To add a different metric, extend its metadata and choose an
explicit aggregation if the official output contains per-reference or
per-passage arrays.

Each run directory contains a unique `mtrag/mt_rag_*` audit directory with
judge inputs, scored outputs when available, stdout/stderr, command/provenance,
and success/failure status. MT-RAG judging is outside the serving trace even
when judging uses models or GPUs.

This is a generation-evaluator integration. It does not implement official
retrieval-qrels evaluation, benchmark corpus indexing, protocol switching, or
interactive rollout with newly generated prior answers. Reference contexts
are never substituted for pipeline retrieval. Generation records contain
context text without invented benchmark document IDs or ranking scores; they
must not be submitted as official retrieval runs.

## Sources

- [MT-RAG repository and benchmark](https://github.com/IBM/mt-rag-benchmark)
- [Official generation evaluator](https://github.com/IBM/mt-rag-benchmark/blob/53fef6f5917c7e10c436f697e69a955dbc795d81/scripts/evaluation/run_generation_eval.py)
- [Official generation-task protocol](https://github.com/IBM/mt-rag-benchmark/blob/53fef6f5917c7e10c436f697e69a955dbc795d81/mtrag-human/generation_tasks/README.md)
