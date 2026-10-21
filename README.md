# RAG-Stack Evaluator

RAG-Stack Evaluator provides the `static_gt` backend,
real-hardware measured execution, dataset adapters, scoring, and vLLM
instrumentation. RAG-Stack owns the evaluator interface, registry, and public
tracing hooks; other backend providers use that same contract. This package
exposes its implementations
through its own top-level Python package:

```python
from rag_stack_evaluator.static_rag_evaluator import (
    StaticRAGEvaluatorQualityOnly,
    MeasuredProvider,
)
```

This distribution owns the `rag_stack_evaluator` package, including
`rag_stack_evaluator.static_rag_evaluator`,
and
`rag_stack_evaluator.vllm_instrumentation`. It does not contribute modules to
the host's `rag_stack` namespace.

This is currently a RAG-Stack host subproject, not a standalone replacement
for the host package. The evaluator continues to consume RAG-Stack's shared
dataset, IR, layout, security, and cost-model contracts through `rag_stack.*`.
Install and run it through a compatible RAG-Stack checkout; the direct install
commands below are for developing the submodule in that host environment.
RAG-Stack owns optimization, search-space resolution, cost models, shared
contracts, evaluator registration, and public component hooks. Backend implementations and their unit tests are maintained here.
FlashRAG is a separate top-level submodule of RAG-Stack. Its backend and
component instrumentation live in `flashrag.integrations.rag_stack`; this
package does not own or register it.

## Backend selection

Every registered backend implements the host's `Evaluator` protocol and returns
an `EvaluationResult` containing scores and RAG-IR. RAG-Stack's `TraceSession`
supplies exact component input/output hooks, and its `BackendSpec` plus Python
entry points make providers discoverable. This distribution registers
`static_gt` through the same `rag_stack.evaluators` entry-point group as other
providers. The host's `docs/evaluator_api.md` documents the shared interface,
and `docs/flashrag.md` describes the separate FlashRAG integration.

Existing `rag_stack_evaluator.base.BaseEvaluator` subclasses remain supported.
The host registry constructs the selected backend:

```python
from rag_stack.evaluation.registry import create_quality_evaluator

evaluator = create_quality_evaluator(
    dataset_manager=dataset_manager,
    project_dir=dataset_manager.project_dir,
    backend="static_gt",
)
```

`create_quality_evaluator` also accepts `dataset`, `project_dir`, and
`backend_options` for every registered backend. The legacy
`rag_stack_evaluator.factory` forwarding module translates `flashrag_options`
for existing FlashRAG callers; new code uses `backend_options`. Direct static quality and measured
imports remain supported. `supported_metrics(backend)` and
`supported_pipeline_modes(backend)` in the same host registry expose backend
capabilities for host validation without importing optional FlashRAG runtime
modules.

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
uv pip install --torch-backend cu128 \
  -e 'RAG-Stack-Evaluator[cu12]' -e '.[cu12]'
# On NVIDIA driver >=580, use --torch-backend cu130 and cu13 in both
# editable requirements instead.

```

The commands below are only for developing this submodule after the compatible
host is already installed. Use Python 3.10 or newer and install exactly one
CUDA stack when local GPU models or measured evaluation are required:

```bash
# API/CPU components; add `faiss` unless faiss-cpu 1.14.1 is already present.
uv pip install -e '.[faiss]'

# NVIDIA driver 525-579.
uv pip install --torch-backend cu128 -e '.[cu12,faiss]'

# NVIDIA driver >=580.
uv pip install --torch-backend cu130 -e '.[cu13,faiss]'

```

Do not install the `cu12` and `cu13` extras together. Native benchmark
environments may provide `faiss-cpu=1.14.1` through conda and omit the `faiss`
extra. The supported integration checks this repository out as the
`RAG-Stack-Evaluator` submodule and installs both local projects as editable
distributions (or resolves them through the parent workspace), so the host's
shared `rag_stack` contracts and the evaluator package are available together.
The evaluator configures a matching CUDA toolkit at every vLLM launch boundary:
it validates an explicit `CUDA_HOME`, discovers system and Conda toolkits, and
uses the complete toolkit installed by the cu13 extra when needed. No activation
hook or toolkit-configuration shell script is required. CUDA 12's PyPI toolkit
components do not include an `nvcc` driver, so the cu12 extra installs matching
precompiled FlashInfer cubins. Any unsupported kernel that still triggers source
JIT requires a matching system or Conda CUDA 12 toolkit.

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

## [Examples](example/)

## Input contract

The authoritative [input and output contract](docs/input_contract.md) defines
resolved algorithm and system dictionaries, separate QA/corpus arguments,
quality and measured APIs, and their result types. The [examples](example/README.md)
follow that contract and include a concrete two-config pair.

The core API accepts deserialized dictionaries, not config paths or optimizer
search spaces. Dataset selection is passed by the caller and is absent from
both config files. Public quality calls return `EvaluationResult`; direct concrete-engine calls
retain the legacy flat result mapping. `MeasuredProvider` accepts the static
engine or its registry adapter and returns `ProviderResult`.

## Tests

Run `python -m pytest` from this repository root. The suite is organized by
component ownership and uses CPU fixtures instead of live models or judge
services. See [tests/README.md](tests/README.md) for the directory map and the
command for running it from a RAG-Stack host checkout.

## License and attribution

Most redistributed derived code is under Apache License 2.0; see
`LICENSE.autorag` and `NOTICE` for AutoRAG attribution. Two bundled TART model
and tokenizer implementation files are under CC BY-NC 4.0; see `LICENSE.tart`
and `NOTICE`. That TART-derived portion is restricted to non-commercial use, so
the combined distribution is not Apache-only and is not suitable for commercial
use without replacing those files or obtaining separate permission.
