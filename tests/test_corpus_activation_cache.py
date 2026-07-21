from __future__ import annotations

import gc
import shutil
import weakref
from pathlib import Path

import pandas as pd
import pytest

import rag_stack_evaluator.static_rag_evaluator.dataset as dataset_module
import rag_stack_evaluator.static_rag_evaluator.nodes.retrieval.base as retrieval_base
from rag_stack_evaluator.static_rag_evaluator.dataset import (
    CorpusView,
    DatasetManager,
    get_active_corpus,
    register_active_corpus,
)


class _ConcreteRetrieval(retrieval_base.BaseRetrieval):
    def pure(self, previous_result, *args, **kwargs):
        return previous_result

    def _pure(self, queries, *args, **kwargs):
        return queries


def _manager(project_dir: Path) -> DatasetManager:
    manager = object.__new__(DatasetManager)
    manager.project_dir = str(project_dir)
    manager.corpus_data = pd.DataFrame(
        {"doc_id": ["canonical"], "contents": ["canonical text"]}
    )
    manager._active_view_key = None
    data_dir = project_dir / "data"
    data_dir.mkdir(parents=True)
    manager.corpus_data.to_parquet(data_dir / "corpus.parquet", index=False)
    return manager


def _view(path: Path, chunk_hash: str) -> CorpusView:
    return CorpusView(
        chunk_hash=chunk_hash,
        corpus_path=str(path),
        token_stats=None,
        n_vectors=len(pd.read_parquet(path)),
    )


def test_identical_view_activation_reuses_frame_and_active_file(
    monkeypatch, tmp_path
):
    manager = _manager(tmp_path / "project")
    source = tmp_path / "cached.parquet"
    expected = pd.DataFrame(
        {"doc_id": ["a", "b"], "contents": ["first", "second"]}
    )
    expected.to_parquet(source, index=False)
    view = _view(source, "same-hash")

    copy_calls = []
    read_calls = []
    real_copyfile = shutil.copyfile
    real_read_parquet = pd.read_parquet

    def tracked_copyfile(source_path, destination_path):
        copy_calls.append((source_path, destination_path))
        return real_copyfile(source_path, destination_path)

    def tracked_read_parquet(path, *args, **kwargs):
        read_calls.append(path)
        return real_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "copyfile", tracked_copyfile)
    monkeypatch.setattr(dataset_module.pd, "read_parquet", tracked_read_parquet)

    first = manager.activate(view)
    active_path = tmp_path / "project" / "data" / "corpus.parquet"
    first_stat = active_path.stat()
    second = manager.activate(view)
    second_stat = active_path.stat()

    assert first is second
    pd.testing.assert_frame_equal(first, expected)
    assert len(copy_calls) == 1
    assert len(read_calls) == 1
    assert first_stat.st_ino == second_stat.st_ino
    assert first_stat.st_mtime_ns == second_stat.st_mtime_ns
    assert get_active_corpus(str(tmp_path / "project")) is first


def test_changed_view_refreshes_active_corpus(monkeypatch, tmp_path):
    manager = _manager(tmp_path / "project")
    first_source = tmp_path / "first.parquet"
    second_source = tmp_path / "second.parquet"
    pd.DataFrame({"doc_id": ["a"], "contents": ["first"]}).to_parquet(
        first_source, index=False
    )
    pd.DataFrame({"doc_id": ["b"], "contents": ["second"]}).to_parquet(
        second_source, index=False
    )
    first_view = _view(first_source, "first-hash")
    second_view = _view(second_source, "second-hash")

    copy_calls = []
    read_calls = []
    real_copyfile = shutil.copyfile
    real_read_parquet = pd.read_parquet

    def tracked_copyfile(source_path, destination_path):
        copy_calls.append((source_path, destination_path))
        return real_copyfile(source_path, destination_path)

    def tracked_read_parquet(path, *args, **kwargs):
        read_calls.append(path)
        return real_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "copyfile", tracked_copyfile)
    monkeypatch.setattr(dataset_module.pd, "read_parquet", tracked_read_parquet)

    first = manager.activate(first_view)
    second = manager.activate(second_view)

    assert first is not second
    assert second["doc_id"].tolist() == ["b"]
    assert len(copy_calls) == 2
    assert len(read_calls) == 2
    assert get_active_corpus(str(tmp_path / "project")) is second


def test_base_retrieval_reuses_registered_frame(monkeypatch, tmp_path):
    project_dir = tmp_path / "registered-project"
    frame = pd.DataFrame({"doc_id": ["a"], "contents": ["text"]})
    register_active_corpus(str(project_dir), frame)
    monkeypatch.setattr(
        retrieval_base.pd,
        "read_parquet",
        lambda *_args, **_kwargs: pytest.fail("disk fallback was used"),
    )

    retrieval = _ConcreteRetrieval(str(project_dir))
    assert retrieval.corpus_df is frame


def test_base_retrieval_falls_back_to_project_parquet(tmp_path):
    project_dir = tmp_path / "standalone-project"
    data_dir = project_dir / "data"
    data_dir.mkdir(parents=True)
    expected = pd.DataFrame({"doc_id": ["a"], "contents": ["text"]})
    expected.to_parquet(data_dir / "corpus.parquet", index=False)

    retrieval = _ConcreteRetrieval(str(project_dir))
    pd.testing.assert_frame_equal(retrieval.corpus_df, expected)


def test_active_corpus_registry_is_project_scoped_and_weak(tmp_path):
    first_project = str(tmp_path / "first")
    second_project = str(tmp_path / "second")
    first = pd.DataFrame({"doc_id": ["a"]})
    second = pd.DataFrame({"doc_id": ["b"]})
    first_ref = weakref.ref(first)
    register_active_corpus(first_project, first)
    register_active_corpus(second_project, second)

    assert get_active_corpus(first_project) is first
    assert get_active_corpus(second_project) is second
    del first
    gc.collect()
    assert first_ref() is None
    assert get_active_corpus(first_project) is None
    assert get_active_corpus(second_project) is second
