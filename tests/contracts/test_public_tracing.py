"""Compatibility coverage for historical evaluator-owned tracing imports."""

import importlib.util
from pathlib import Path

import rag_stack_evaluator
import sys

import pytest

from rag_stack.evaluation import _recording
from rag_stack.evaluation.tracing import TraceSession


def test_old_public_trace_path_uses_the_host_implementation():
    from rag_stack.evaluation import tracing as host_tracing
    from rag_stack_evaluator import tracing as legacy_tracing

    assert legacy_tracing is host_tracing
    assert legacy_tracing.TraceSession is TraceSession


@pytest.mark.parametrize("relative_path", [
    "_recording.py", "static_rag_evaluator/recording.py",
])
def test_legacy_recorders_share_state_and_monkeypatches(relative_path, monkeypatch):
    # Execute the compatibility file directly so this test does not initialize
    # model-specific instrumentation from the legacy static backend package.
    path = Path(rag_stack_evaluator.__file__).resolve().parent.parent / "rag_stack_evaluator" / relative_path
    name = "_recording_compat_test"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules[name] = module
        spec.loader.exec_module(module)
        alias = sys.modules[name]
        assert alias is _recording
        assert alias._current_recorder is _recording._current_recorder
        assert alias._current_qids is _recording._current_qids
        assert alias._tokenizer_for is _recording._tokenizer_for
        sentinel = lambda *args: [7]
        monkeypatch.setattr(alias, "count_tokens_batch", sentinel)
        assert _recording.count_tokens_batch is sentinel
    finally:
        sys.modules.pop(name, None)


def test_legacy_trace_session_uses_host_recorder_context():
    from rag_stack_evaluator.tracing import TraceSession as LegacyTraceSession
    from rag_stack_evaluator import _recording as legacy_recording

    previous = _recording.get_current_recorder()
    session = LegacyTraceSession()
    with session.query("task-1"):
        assert legacy_recording.get_current_recorder() is session._recorder
        assert _recording.get_current_qids() == ["task-1"]
        with session.stage("generator", model_id="compatibility-model") as call:
            call.input(token_count=3)
            call.output(token_count=2)
    assert legacy_recording.get_current_recorder() is previous
    assert session.finish(expected_question_ids=["task-1"])["queries"][0]["calls"][0]["input_tokens"] == 3
