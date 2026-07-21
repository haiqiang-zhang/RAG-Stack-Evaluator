"""Shared offline fixture for the executable evaluator examples."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from rag_stack_evaluator.static_rag_evaluator import (
    DatasetEvalManager,
    StaticRAGEvaluatorQualityOnly,
)


EVIDENCE = (
    "Paris is the capital of France. "
    "The Pacific Ocean is Earth's largest ocean."
)


def prepare_project(work_dir: str | None, *, prefix: str) -> Path:
    """Create a writable project directory and its inline Parquet inputs."""
    project_dir = (
        Path(work_dir).expanduser().resolve()
        if work_dir
        else Path(tempfile.mkdtemp(prefix=f"rag-stack-evaluator-{prefix}-"))
    )
    input_dir = project_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    qa_path = input_dir / "qa.parquet"
    corpus_path = input_dir / "corpus.parquet"

    pd.DataFrame(
        {
            "qid": ["capital", "ocean"],
            "query": [
                "What is the capital of France?",
                "Which ocean is Earth's largest?",
            ],
            "generation_gt": [["Paris"], ["Pacific Ocean"]],
            "references": [[EVIDENCE], [EVIDENCE]],
        }
    ).to_parquet(qa_path, index=False)

    pd.DataFrame(
        {
            "doc_id": ["facts-1"],
            "contents": [EVIDENCE],
            "metadata": [{"source": "inline-example"}],
            "start_end_idx": [[0, len(EVIDENCE)]],
        }
    ).to_parquet(corpus_path, index=False)

    return project_dir


def build_evaluator(project_dir: Path) -> StaticRAGEvaluatorQualityOnly:
    """Bind the generated data to the stable path-based evaluator API."""
    dataset = DatasetEvalManager(
        project_dir=str(project_dir),
        qa_data_path=str(project_dir / "inputs" / "qa.parquet"),
        corpus_data_path=str(project_dir / "inputs" / "corpus.parquet"),
    )
    return StaticRAGEvaluatorQualityOnly(
        dataset_manager=dataset,
        project_dir=str(project_dir),
    )


def resolved_pipeline_config() -> dict:
    """Return one fully resolved, offline, CPU-only pipeline configuration."""
    return {
        "dataset": {"dataset_name": "inline-offline-example"},
        "corpus_runtime": {"chunker": {}},
        "pipeline_runtime": {"mode": "sequential"},
        "vectordb": [
            {
                "name": "offline_hnsw",
                "db_type": "faiss_hnsw",
                "embedding_model": "mock",
                "embedding_dim": 8,
                "collection_name": "offline_facts",
                "path": "${PROJECT_DIR}/resources/faiss",
                "similarity_metric": "cosine",
                "M": 4,
                "ef_construction": 16,
                "ef_search": 8,
            }
        ],
        "node_lines": [
            {
                "node_line_name": "rag_pipeline",
                "nodes": [
                    {
                        "stage": "semantic_retrieval",
                        "strategy": {"metrics": [], "strategy": "mean"},
                        "top_k": 1,
                        "modules": [
                            {
                                "component": "vectordb",
                                "vectordb": "offline_hnsw",
                                "ef_search": 8,
                            }
                        ],
                    },
                    {
                        "stage": "prompt_maker",
                        "strategy": {"metrics": [], "strategy": "mean"},
                        "modules": [
                            {
                                "component": "fstring",
                                "prompt": (
                                    "Question: {query}\n"
                                    "Context: {retrieved_contents}\n"
                                    "Answer:"
                                ),
                            }
                        ],
                    },
                    {
                        "stage": "generator",
                        "strategy": {"metrics": [], "strategy": "mean"},
                        "modules": [
                            {
                                "component": "openai_llm",
                                "model": "gpt-5-nano",
                                "api_key": "mock_offline_example",
                                "max_tokens": 8,
                            }
                        ],
                    },
                ],
            }
        ],
        "eval_backend_setting": {
            "strategy": "mean",
            "metrics": [
                {"metric_name": "retrieval_token_recall"},
                {"metric_name": "retrieval_token_precision"},
                {"metric_name": "retrieval_token_f1"},
            ],
        },
    }
