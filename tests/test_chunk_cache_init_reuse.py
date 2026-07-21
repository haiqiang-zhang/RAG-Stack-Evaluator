"""CPU-only regressions for initialization/per-eval chunk-cache identity."""

from __future__ import annotations

import os
import shutil

import pandas as pd

import rag_stack.cost_model.token_stats as token_stats_module
from rag_stack_evaluator.static_rag_evaluator import chunk_cache
from rag_stack_evaluator.static_rag_evaluator import dataset as dataset_module
from rag_stack_evaluator.static_rag_evaluator.dataset import DatasetEvalManager


class _FakeTokenStats:
    """Tiny tokenizer-free stand-in; these tests exercise cache lifecycle only."""

    def __init__(self, qa_data, corpus_data, config):
        del config
        self.avg_chunk_tokens = int(corpus_data["contents"].str.len().mean())
        self.prompt_template_tokens = 1
        self.avg_output_tokens = int(
            qa_data["generation_gt"].map(lambda values: len(values[0])).mean()
        )
        self.avg_query_tokens = int(qa_data["query"].str.len().mean())
        self.bytes_per_token = 1.0
        self._tokenizer = object()
        self._compressor_tokenizer = None

    def with_chunked_corpus(self, corpus_data):
        new = object.__new__(type(self))
        new.avg_chunk_tokens = int(corpus_data["contents"].str.len().mean())
        new.prompt_template_tokens = self.prompt_template_tokens
        new.avg_output_tokens = self.avg_output_tokens
        new.avg_query_tokens = self.avg_query_tokens
        new.bytes_per_token = self.bytes_per_token
        new._tokenizer = self._tokenizer
        new._compressor_tokenizer = None
        return new

    def to_dict(self):
        return {
            "avg_chunk_tokens": self.avg_chunk_tokens,
            "prompt_template_tokens": self.prompt_template_tokens,
            "avg_output_tokens": self.avg_output_tokens,
            "avg_query_tokens": self.avg_query_tokens,
            "bytes_per_token": self.bytes_per_token,
        }

    @classmethod
    def from_dict(cls, values):
        new = object.__new__(cls)
        for key, value in values.items():
            setattr(new, key, value)
        # Match production TokenStats.from_dict: runtime tokenizers are not
        # serialized and are reattached from the manager's live base stats.
        return new


def _write_inputs(root):
    qa_path = root / "qa.parquet"
    raw_path = root / "raw.parquet"
    pd.DataFrame(
        {
            "qid": ["q0", "q1"],
            "query": ["first question", "second question"],
            "generation_gt": [["first answer"], ["second answer"]],
        }
    ).to_parquet(qa_path, index=False)
    pd.DataFrame(
        {
            "doc_id": ["raw-0", "raw-1"],
            "contents": ["alpha beta", "gamma delta"],
            "metadata": [{"source": "a"}, {"source": "b"}],
        }
    ).to_parquet(raw_path, index=False)
    return qa_path, raw_path


def _config():
    return {
        "dataset": {"dataset_name": "tiny"},
        "algo_search_space": {
            "corpus": {
                "chunker": {
                    "component": "character",
                    "chunk_size": 2048,
                    "chunk_overlap": 0,
                }
            }
        },
    }


def _patch_stats(monkeypatch):
    monkeypatch.setattr(dataset_module, "TokenStats", _FakeTokenStats)
    monkeypatch.setattr(token_stats_module, "TokenStats", _FakeTokenStats)


def _counting_chunker(counter):
    def run(raw_df, chunk_method, chunk_size, chunk_overlap):
        counter["calls"] += 1
        assert (chunk_method, chunk_size, chunk_overlap) == (
            "character", 2048, 0,
        )
        return pd.DataFrame(
            {
                "doc_id": raw_df["doc_id"],
                "contents": raw_df["contents"],
                "metadata": raw_df["metadata"],
                "start_end_idx": [
                    [0, len(text)] for text in raw_df["contents"]
                ],
            }
        )

    return run


def test_fresh_project_builds_default_chunk_once_across_init_and_eval0(
    tmp_path, monkeypatch,
):
    _patch_stats(monkeypatch)
    qa_path, raw_path = _write_inputs(tmp_path)
    calls = {"calls": 0}
    monkeypatch.setattr(
        chunk_cache, "_run_chunker", _counting_chunker(calls),
    )

    project = tmp_path / "project"
    manager = DatasetEvalManager(
        project_dir=str(project),
        config=_config(),
        qa_data_path=str(qa_path),
        corpus_data_path=str(raw_path),
    )
    assert calls["calls"] == 1
    assert not (project / "data" / "_init_chunked_corpus.parquet").exists()

    params = {"component": "character", "chunk_size": 2048, "chunk_overlap": 0}
    controller_view = manager.resolve_corpus(params)
    eval0_view = manager.resolve_corpus(params)
    assert calls["calls"] == 1
    assert controller_view.chunk_hash == eval0_view.chunk_hash
    assert os.path.samefile(controller_view.corpus_path, eval0_view.corpus_path)

    # The startup frame came from this exact immutable cache entry. Eval0 must
    # not copy or deserialize the same parquet a second time when activating it.
    def unexpected_copy(*args, **kwargs):
        raise AssertionError(f"eval0 copied an already-active chunk: {args!r}")

    monkeypatch.setattr(shutil, "copyfile", unexpected_copy)
    active = manager.activate(eval0_view)
    assert active is manager.corpus_data


def test_existing_complete_chunk_cache_hit_never_invokes_chunker(
    tmp_path, monkeypatch,
):
    _patch_stats(monkeypatch)
    qa_path, raw_path = _write_inputs(tmp_path)
    calls = {"calls": 0}
    monkeypatch.setattr(
        chunk_cache, "_run_chunker", _counting_chunker(calls),
    )
    project = tmp_path / "project"
    DatasetEvalManager(
        project_dir=str(project),
        config=_config(),
        qa_data_path=str(qa_path),
        corpus_data_path=str(raw_path),
    )
    assert calls["calls"] == 1

    def forbidden_build(*args, **kwargs):
        raise AssertionError("complete chunk cache unexpectedly rebuilt")

    monkeypatch.setattr(chunk_cache, "_run_chunker", forbidden_build)
    resumed = DatasetEvalManager(
        project_dir=str(project),
        config=_config(),
        qa_data_path=str(qa_path),
        corpus_data_path=str(raw_path),
    )
    view = resumed.resolve_corpus(
        {"component": "character", "chunk_size": 2048, "chunk_overlap": 0}
    )
    assert view.n_vectors == 2
