"""Preserve the effective retrieval output before reranking or compression."""

from copy import deepcopy


def snapshot_retrieval_result(frame, stage):
    """Refresh the neutral snapshot after one actual retrieval/fusion node.

    Branch provenance remains in the semantic/lexical columns. Later nodes only
    overwrite unsuffixed columns, so this snapshot survives their mutations.
    """
    suffix = {
        "semantic_retrieval": "_semantic",
        "lexical_retrieval": "_lexical",
        "hybrid_retrieval": "",
    }[stage]
    for name in ("retrieved_contents", "retrieved_ids", "retrieve_scores"):
        source = name + suffix
        if source in frame.columns:
            frame[name + "_retrieval"] = frame[source].map(deepcopy)
    return frame


def validate_retrieval_capabilities(config):
    """Fail unsupported executions before loading datasets, models or indexes."""
    nodes = [node for line in config.get("node_lines", []) for node in line.get("nodes", [])]
    stages = {node.get("stage") for node in nodes}
    runtime = config.get("pipeline_runtime") or {}
    mode = str(runtime.get("rag_dataflow") or runtime.get("mode", "sequential")).lower()
    if mode == "react" and stages.intersection({"lexical_retrieval", "sparse_retrieval", "hybrid_retrieval"}):
        raise ValueError("static_gt ReAct currently supports dense retrieval only; use sequential for lexical or hybrid retrieval")
    for node in nodes:
        module = node.get("module")
        modules = [module] if module is not None else node.get("modules", [])
        if any(item.get("component") in {"hybrid_cc", "HybridCC"} for item in modules):
            raise ValueError("hybrid_cc execution tracing is not supported; use hybrid_rrf")
    if "hybrid_retrieval" in stages:
        from rag_stack.pipeline_contract import hybrid_structure_errors
        errors = hybrid_structure_errors({**config, "mode": mode, "rag_dataflow": mode})
        if errors:
            raise ValueError("; ".join(errors))
