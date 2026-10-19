# Evaluator API Reference has moved

The evaluator interface, backend registry, and public tracing hooks are owned by
RAG-Stack. Read the canonical
[Evaluator API Reference](https://github.com/haiqiang-zhang/rag-stack/blob/main/docs/evaluator_api.md)
and [Evaluator Extension Guide](https://github.com/haiqiang-zhang/rag-stack/blob/main/docs/evaluator_extensibility.md)
in the host repository (`docs/evaluator_api.md` and
`docs/evaluator_extensibility.md` in a local host checkout).

Use `rag_stack.evaluation.api`, `rag_stack.evaluation.registry`, and
`rag_stack.evaluation.tracing` for new integrations. `rag-stack-evaluator`
provides the `static_gt` and `flashrag` backend implementations, registered
through the same host entry-point mechanism as other providers.

The [MT-RAG example](../examples/mtrag/README.md) remains in this repository
as an independently installable plugin that imports the host API directly.
