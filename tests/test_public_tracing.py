"""CPU-only contract tests for external pipeline instrumentation."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from rag_stack_evaluator import _recording
from rag_stack_evaluator.tracing import TraceSession


def _generate(session, qid, *, tokens=3):
    with session.query(qid):
        with session.stage("generator", model_id="external/model") as call:
            call.input(token_count=tokens)
            call.output(token_ids=[5, 7], text="answer")


def test_exact_counts_bytes_and_explicit_dataset_order():
    session = TraceSession()
    _generate(session, "task-b", tokens=11)
    with session.query("task-a"):
        with session.stage("semantic_retrieval_encode", model_id="embedder") as call:
            call.input(token_ids=[1, 2], text="café")
            call.output(token_count=0, byte_count=0)
        with session.stage("generator", model_id="generator") as call:
            call.input(token_count=15, text="history + retrieved context + question")
            call.output(token_ids=[], text="")

    trace = session.finish(expected_question_ids=["task-a", "task-b"])
    assert [q["question_id"] for q in trace["queries"]] == ["task-a", "task-b"]
    first, last = trace["queries"][0]["calls"]
    assert first == {
        "stage": "semantic_retrieval_encode", "input_tokens": 2,
        "output_tokens": 0, "input_bytes": 5, "output_bytes": 0,
        "step_idx": 0, "model_id": "embedder",
    }
    assert last["step_idx"] == 1 and last["output_tokens"] == 0
    assert "input_bytes" not in trace["queries"][1]["calls"][0]
    assert session.finish(expected_question_ids=["task-a", "task-b"]) is trace
    with pytest.raises(RuntimeError, match="finished"):
        _generate(session, "task-c")
    with pytest.raises(ValueError, match="differ"):
        session.finish(expected_question_ids=["task-b", "task-a"])


def test_batched_calls_preserve_alignment():
    session = TraceSession()
    with session.batch(["q1", "q2"]):
        with session.stage("generator", model_id="batch-model") as call:
            call.input(token_counts=[4, 7], texts=["one", "two"])
            call.output(token_ids=[[1], [2, 3]], byte_counts=[3, 8])
    trace = session.finish(expected_question_ids=["q1", "q2"])
    assert [q["calls"][0]["input_tokens"] for q in trace["queries"]] == [4, 7]
    assert [q["calls"][0]["output_tokens"] for q in trace["queries"]] == [1, 2]
    assert [q["calls"][0]["output_bytes"] for q in trace["queries"]] == [3, 8]


def test_text_requires_the_callers_exact_tokenizer():
    session = TraceSession()
    seen = []

    def tokenizer(text):
        seen.append(text)
        return [10, 20, 30, 40]

    with session.query("q"):
        with session.stage("generator", model_id="custom") as call:
            call.input(text="actual rendered prompt", tokenizer=tokenizer)
            call.output(text="yes", tokenizer=tokenizer)
    trace = session.finish(expected_question_ids=["q"])
    assert seen == ["actual rendered prompt", "yes"]
    assert trace["queries"][0]["calls"][0]["input_tokens"] == 4
    assert trace["queries"][0]["calls"][0]["output_bytes"] == 3


@pytest.mark.parametrize("payload, error", [
    ({"token_count": -1}, "non-negative"),
    ({"token_count": True}, "non-negative"),
    ({"token_count": 1.5}, "non-negative"),
    ({"token_ids": [True]}, "non-negative"),
    ({"token_ids": "1 2"}, "sequence"),
    ({"text": "never estimate from whitespace"}, "explicit tokenizer"),
    ({"token_ids": [1], "token_count": 1}, "exactly one"),
    ({"token_count": 1, "tokenizer": lambda text: [1]}, "cannot be combined"),
    ({"token_counts": [1, 2]}, "exactly one entry"),
    ({"token_count": 1, "texts": ["one", "two"]}, "one string"),
    ({"token_count": 1, "text": "é", "byte_count": 1}, "disagree"),
    ({"token_count": 1, "byte_counts": []}, "exactly one entry"),
])
def test_invalid_payload_fails_closed_even_if_the_hook_error_is_caught(payload, error):
    session = TraceSession()
    with pytest.raises(RuntimeError, match="successful input hook"):
        with session.query("q"):
            with session.stage("generator", model_id="model") as call:
                with pytest.raises((TypeError, ValueError), match=error):
                    call.input(**payload)
                # A retry must not turn a failed component scope into success.
                call.input(token_count=1)
                call.output(token_count=0)
    with pytest.raises(RuntimeError, match="failed scope"):
        session.finish(expected_question_ids=["q"])


@pytest.mark.parametrize("qids", [[], ["same", "same"], [""], [None], [42], "q"])
def test_invalid_question_ids(qids):
    session = TraceSession()
    with pytest.raises((ValueError, TypeError)):
        with session.batch(qids):
            pass


def test_batch_mismatch_is_rejected_without_truncating():
    session = TraceSession()
    with pytest.raises(ValueError, match="exactly one entry"):
        with session.batch(["q1", "q2"]):
            with session.stage("generator", model_id="model") as call:
                call.input(token_counts=[1])
    with pytest.raises(RuntimeError, match="failed scope"):
        session.finish(expected_question_ids=["q1", "q2"])


def test_missing_output_and_pipeline_exception_cannot_publish_partial_success():
    session = TraceSession()
    with pytest.raises(RuntimeError, match="output hook"):
        with session.query("q"):
            with session.stage("generator", model_id="model") as call:
                call.input(token_count=3)
    with pytest.raises(RuntimeError, match="failed scope"):
        session.finish(expected_question_ids=["q"])

    session = TraceSession()
    with pytest.raises(LookupError):
        with session.query("q"):
            with session.stage("generator", model_id="model") as call:
                call.input(token_count=3)
                raise LookupError("component failed")
    with pytest.raises(RuntimeError, match="failed scope"):
        session.finish(expected_question_ids=["q"])


def test_query_exception_after_completed_generator_still_fails():
    session = TraceSession()
    with pytest.raises(ValueError, match="after generation"):
        with session.query("q"):
            with session.stage("generator", model_id="model") as call:
                call.input(token_count=1)
                call.output(token_count=1)
            raise ValueError("failure after generation")
    with pytest.raises(RuntimeError, match="failed scope"):
        session.finish(expected_question_ids=["q"])


def test_coverage_and_terminal_generator_are_required():
    session = TraceSession()
    _generate(session, "q1")
    with pytest.raises(ValueError, match="missing=.*q2.*extra=.*q1"):
        session.finish(expected_question_ids=["q2"])
    session = TraceSession()
    with session.query("q"):
        with session.stage("lexical_retrieval", model_id="bm25") as call:
            call.input(token_count=2)
            call.output(token_count=0)
    with pytest.raises(ValueError, match="must end at a generator"):
        session.finish(expected_question_ids=["q"])


def test_scope_stage_and_model_requirements():
    session = TraceSession()
    with pytest.raises(RuntimeError, match="query.*batch"):
        with session.stage("generator", model_id="model"):
            pass
    with session.query("q"):
        with pytest.raises(ValueError, match="unsupported trace stage"):
            with session.stage("semantic_retrieval", model_id="model"):
                pass
        with pytest.raises(ValueError, match="model_id"):
            with session.stage("generator", model_id=""):
                pass
        with pytest.raises(RuntimeError, match="exit all query"):
            session.finish(expected_question_ids=["q"])
        with session.stage("generator", model_id="model") as call:
            with pytest.raises(RuntimeError, match="overlapping stages"):
                with session.stage("generator", model_id="model"):
                    pass
            call.input(token_count=1)
            call.output(token_count=1)
    session.finish(expected_question_ids=["q"])


def test_nested_sessions_restore_legacy_context_even_after_failure():
    previous_recorder = _recording.get_current_recorder()
    previous_qids = _recording.get_current_qids()
    outer, inner = TraceSession(), TraceSession()
    with outer.query("outer"):
        assert _recording.get_current_recorder() is outer._recorder
        assert _recording.get_current_qids() == ["outer"]
        with pytest.raises(ValueError):
            with inner.query("inner"):
                assert _recording.get_current_recorder() is inner._recorder
                assert _recording.get_current_qids() == ["inner"]
                raise ValueError("test scope cleanup")
        assert _recording.get_current_recorder() is outer._recorder
        assert _recording.get_current_qids() == ["outer"]
        with outer.stage("generator", model_id="model") as call:
            call.input(token_count=1)
            call.output(token_count=1)
    assert _recording.get_current_recorder() is previous_recorder
    assert _recording.get_current_qids() == previous_qids
    outer.finish(expected_question_ids=["outer"])


def test_async_requests_keep_query_attribution_separate():
    session = TraceSession()

    async def run():
        ready = asyncio.Event()

        async def worker(qid, count):
            with session.query(qid):
                with session.stage("generator", model_id="async-model") as call:
                    call.input(token_count=count)
                    if qid == "q2":
                        ready.set()
                    await ready.wait()
                    await asyncio.sleep(0)
                    assert _recording.get_current_qids() == [qid]
                    call.output(token_count=1)

        await asyncio.gather(worker("q1", 10), worker("q2", 20))

    asyncio.run(run())
    trace = session.finish(expected_question_ids=["q1", "q2"])
    assert [q["calls"][0]["input_tokens"] for q in trace["queries"]] == [10, 20]


def test_worker_threads_record_to_the_shared_explicit_session():
    session = TraceSession()
    barrier = threading.Barrier(2)

    def worker(qid):
        with session.query(qid):
            with session.stage("generator", model_id="thread-model") as call:
                call.input(token_count=1)
                barrier.wait(timeout=5)
                assert _recording.get_current_qids() == [qid]
                call.output(token_count=2)
        assert _recording.get_current_qids() is None

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(worker, ["q1", "q2"]))
    assert len(session.finish(expected_question_ids=["q1", "q2"])["queries"]) == 2


def test_public_import_does_not_load_host_or_model_runtimes():
    package_root = str(Path(__file__).resolve().parents[1])
    script = f"""
import importlib.abc
import sys
sys.path.insert(0, {package_root!r})
blocked = {{'torch', 'transformers', 'vllm', 'rag_stack', 'rag_stack_evaluator.static_rag_evaluator'}}
class BlockHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            raise AssertionError('Unexpected heavy import: ' + fullname)
sys.meta_path.insert(0, BlockHeavyImports())
from rag_stack_evaluator.tracing import TraceSession
session = TraceSession()
with session.query('q'):
    with session.stage('generator', model_id='model') as call:
        call.input(token_count=2)
        call.output(token_count=1)
"""
    subprocess.run([sys.executable, "-I", "-c", script], check=True, capture_output=True, text=True)


def test_legacy_module_alias_keeps_core_identity_and_monkeypatches(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "rag_stack_evaluator/static_rag_evaluator/recording.py"
    name = "_recording_compat_test"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules[name] = module
        spec.loader.exec_module(module)
        assert sys.modules[name] is _recording
        sentinel = lambda *args: [7]
        monkeypatch.setattr(sys.modules[name], "count_tokens_batch", sentinel)
        assert _recording.count_tokens_batch is sentinel
    finally:
        sys.modules.pop(name, None)


def test_deferred_legacy_records_are_rejected_without_tokenizer_loading(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("public hooks must not load a tokenizer implicitly")

    monkeypatch.setattr(_recording, "count_tokens_batch", forbidden)
    session = TraceSession()
    with session.query("q"):
        _recording.record_io("generator", ["q"], ["prompt"], out_texts=["answer"])
    with pytest.raises(ValueError, match="deferred legacy.*uninstrumented"):
        session.finish(expected_question_ids=["q"])


def test_cancelled_async_scope_cannot_publish():
    session = TraceSession()

    async def worker():
        with session.query("q"):
            with session.stage("generator", model_id="model") as call:
                call.input(token_count=1)
                raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(worker())
    with pytest.raises(RuntimeError, match="failed scope"):
        session.finish(expected_question_ids=["q"])
