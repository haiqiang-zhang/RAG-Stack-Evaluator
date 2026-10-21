"""Transport observed fusion I/O to the ordered static trace producers.

The private result column contains sizes and component identities and survives
process and batch boundaries. Producers consume it after encode/search and remove it before
saving results or passing them to the next serving stage.
"""

FUSION_TRACE_COLUMN = "__rag_stack_fusion_calls__"


def observed_embedding_model_id(observations, fallback=None):
    """Use the identity of the encoder actually owned by the VectorDB node."""
    identities = {call["embedding_model_id"] for calls in (observations or [])
                  for call in calls if call.get("embedding_model_id")}
    if len(identities) > 1:
        raise ValueError("One retrieval stage batch cannot contain multiple embedding model identities")
    return next(iter(identities), fallback)


def candidate_profiles(observations):
    """Actual search candidate collections consumed by each local fusion."""
    if observations is None:
        return None
    return [calls[0]["input_profile"] if calls else None for calls in observations]


def take_fusion_observations(result):
    """Remove and return per-row calls captured at real fusion invocations."""
    if FUSION_TRACE_COLUMN not in result.columns:
        return None
    return result.pop(FUSION_TRACE_COLUMN).tolist()


def record_fusion_observations(qids, observations):
    """Append observed fusion calls after the producer's retrieval calls."""
    if observations is None:
        return
    if len(qids) != len(observations):
        raise ValueError("fusion observations and question IDs must have equal length")
    from rag_stack.evaluation._recording import get_current_recorder

    recorder = get_current_recorder()
    if recorder is None:
        return
    for qid, calls in zip(qids, observations):
        for call in calls:
            node_id = call.get("node_id", "semantic_retrieval")
            if node_id == "hybrid_retrieval":
                input_refs = {
                    "dense": [recorder.latest_call_id(qid, node_id="semantic_retrieval")],
                    "lexical": [recorder.latest_call_id(qid, node_id="lexical_retrieval")],
                }
            else:
                input_refs = {"candidates": [recorder.latest_call_id(qid, node_id=node_id)]}
            recorder.record(
                qid, "retrieval_fusion", model_id="rrf_pure",
                node_id=node_id, input_refs=input_refs,
                input_tokens=0, output_tokens=0,
                input_bytes=None, output_bytes=None,
                input_profile=call["input_profile"],
                output_profile=call["output_profile"],
            )
