"""Deterministic CPU fixtures for interface checks, never benchmark results.

The tokenizer, retrieval rule, answer generator, and scores below are explicitly
test fixtures. None represents a production model, calibrated cost, or MT-RAG
quality result. Replace run_pipeline and the scorer command for a real study.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CORPUS = (
    "The example service supports private deployment.",
    "The example service exports JSON records.",
)


def fixture_config() -> dict:
    return {
        "pipeline_runtime": {"mode": "sequential"},
        "node_lines": [{"node_line_name": "fixture", "nodes": [
            {"stage": "lexical_retrieval", "modules": [
                {"component": "example_token_overlap", "top_k": 1}]},
            {"stage": "generator", "modules": [
                {"component": "example_extractive", "model": "cpu-fixture-extractive"}]},
        ]}],
    }


def fixture_tasks() -> list[dict]:
    return [{
        "task_id": f"fixture-conversation<::>{turn}",
        "conversation_id": "fixture-conversation", "turn": str(turn), "task_type": "rag",
        "Collection": "synthetic-cpu-fixture", "Answerability": ["ANSWERABLE"],
        "input": [
            {"speaker": "user", "text": "Tell me about the example service."},
            {"speaker": "agent", "text": "It is an example service."},
            {"speaker": "user", "text": question},
        ],
        "targets": [{"speaker": "agent", "text": "Reference reserved for judging."}],
        "contexts": [{"text": "Reference evidence reserved for judging."}],
    } for turn, question in ((2, "Does it support private deployment?"),
                             (3, "Can it export JSON records?"))]


def run_pipeline(config, requests, recorder, *, options):
    # Imports are local so the subprocess scoring fixture stays stdlib-only.
    from rag_stack_mtrag.backend import PipelineRun, Prediction

    nodes = config["node_lines"][0]["nodes"]
    top_k = nodes[0]["modules"][0]["top_k"]
    model_id = nodes[1]["modules"][0]["model"]
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= len(CORPUS):
        raise ValueError("Fixture top_k must be 1 or 2")
    if options:
        raise ValueError("The CPU fixture accepts no pipeline options")
    predictions = []
    for request in requests:
        # The fixture's defined tokenizer uses whitespace tokens. Production
        # hooks must use the actual component tokenizer or reported token usage.
        history = "\n".join(f"{turn['speaker']}: {turn['text']}" for turn in request["input"])
        query = request["input"][-1]["text"]
        with recorder.query(request["task_id"]):
            with recorder.stage("lexical_retrieval", model_id="cpu-fixture-token-overlap") as call:
                call.input(token_count=len(query.split()), text=query)
                terms = set(query.lower().replace("?", "").split())
                contexts = sorted(CORPUS, key=lambda text: -len(terms & set(text.lower().split())))[:top_k]
                call.output(token_count=0)
            prompt = history + "\nEvidence:\n" + "\n".join(contexts) + "\nAnswer:"
            with recorder.stage("generator", model_id=model_id) as call:
                call.input(token_count=len(prompt.split()), text=prompt)
                answer = contexts[0]
                call.output(token_count=len(answer.split()), text=answer)
        predictions.append(Prediction(request["task_id"], answer, list(contexts)))
    return PipelineRun(predictions, workflow={
        "dataflow_type": "sequential",
        "stages": ["lexical_retrieval", "generator_prefill", "generator_decode"],
        "has_query_expansion": False, "has_reranker": False,
        "has_encode": False, "has_compressor": False,
    })


def score_fixture(source: Path, destination: Path) -> None:
    records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    for index, record in enumerate(sorted(records, key=lambda row: row["task_id"])):
        value = 0.25 + index * 0.25
        record["metrics"] = {
            "RL_F": [value], "RB_llm": [value + 0.125], "RB_agg": [value / 2],
            "RL_F_idk_underspecified": [value],
            "RB_llm_idk_underspecified": [value + 0.125],
            "RB_agg_idk_underspecified": [value / 2],
        }
    destination.write_text("".join(json.dumps(row) + "\n" for row in reversed(records)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    score_fixture(args.input, args.output)


if __name__ == "__main__":
    main()
