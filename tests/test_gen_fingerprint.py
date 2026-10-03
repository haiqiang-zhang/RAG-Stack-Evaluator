"""Generation-fingerprint semantics and two-phase store contracts."""
import importlib.util
import json
from pathlib import Path

import pytest

# This store is stdlib-only; avoid importing the evaluator's eager ML stack.
_spec = importlib.util.spec_from_file_location(
    "generation_fingerprint_test_target",
    Path(__file__).resolve().parents[1]
    / "rag_stack_evaluator/static_rag_evaluator/gen_fingerprint.py",
)
gfp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gfp)


Q = ["what is x?", "why y?"]
CTX = [["passage a", "passage b"], ["passage c"]]
P = ["prompt 1", "prompt 2"]
PARAM = {"model": "Qwen/Qwen2.5-7B-Instruct", "temperature": 0.0,
         "max_tokens": 512, "uri": "http://host-a:8000", "batch": 100}


def test_fingerprint_deterministic_and_content_keyed():
    a = gfp.fingerprint(Q, CTX, P, "vllm_api", PARAM)
    b = gfp.fingerprint(list(Q), [list(c) for c in CTX], list(P), "vllm_api", dict(PARAM))
    assert a == b
    # content changes break it
    assert gfp.fingerprint(Q, CTX, ["prompt 1", "prompt 2!"], "vllm_api", PARAM) != a
    assert gfp.fingerprint(Q, [["passage a"], ["passage c"]], P, "vllm_api", PARAM) != a
    # generator SEMANTICS change breaks it
    assert gfp.fingerprint(Q, CTX, P, "vllm_api", {**PARAM, "temperature": 0.7}) != a
    assert gfp.fingerprint(Q, CTX, P, "vllm_api", {**PARAM, "model": "Qwen/Qwen2.5-14B-Instruct"}) != a


def test_fingerprint_ignores_transport_knobs():
    a = gfp.fingerprint(Q, CTX, P, "vllm_api", PARAM)
    b = gfp.fingerprint(Q, CTX, P, "vllm_api",
                        {**PARAM, "uri": "http://other-host:9999", "batch": 4,
                         "request_timeout": 3.0})
    assert a == b


@pytest.mark.parametrize(
    "left,right",
    [
        ("Vllm", "VllmAPI"),
        ("vllm", "vllm_api"),
    ],
)
def test_fingerprint_unifies_equivalent_v1_chat_transports(left, right):
    local = gfp.fingerprint(
        Q, CTX, P, left, {**PARAM, "uri": "in-process"},
    )
    api = gfp.fingerprint(
        Q, CTX, P, right, {**PARAM, "uri": "http://host-a:8000"},
    )
    assert local == api


def test_fingerprint_does_not_merge_unrelated_generator_modules():
    vllm = gfp.fingerprint(Q, CTX, P, "Vllm", PARAM)
    unrelated = gfp.fingerprint(Q, CTX, P, "OtherGenerator", PARAM)
    assert unrelated != vllm


def test_fingerprint_separates_historical_native_completion_from_chat():
    chat = gfp.fingerprint(Q, CTX, P, "Vllm", PARAM)
    legacy = gfp.fingerprint(
        Q, CTX, P, "Vllm", {**PARAM, "generation_protocol": "legacy_completion"},
    )
    assert legacy != chat
    assert legacy != gfp.fingerprint(Q, CTX, P, "VllmAPI", PARAM)
    with pytest.raises(ValueError, match="restricted to native"):
        gfp.fingerprint(Q, CTX, P, "VllmAPI", {
            **PARAM, "generation_protocol": "legacy_completion",
        })


def test_store_two_phase_roundtrip(tmp_path):
    proj = str(tmp_path)
    fp = gfp.fingerprint(Q, CTX, P, "vllm_api", PARAM)
    # incomplete record (answers only) is NOT a usable hit
    gfp.save_answers(proj, fp, ["ans 1", "ans 2"], [7, 9], donor="eval_0003")
    assert gfp.load_complete_record(proj, fp) is None
    # quality attach completes it
    gfp.attach_quality(proj, fp, {"deepeval_answer_correctness": 0.42,
                                  "retrieval_token_recall": 0.5,
                                  "__execution_dag__": [1, 2]})
    rec = gfp.load_complete_record(proj, fp)
    assert rec is not None
    assert rec["answers"] == ["ans 1", "ans 2"]
    assert rec["token_counts"] == [7, 9]
    assert rec["quality"] == {"deepeval_answer_correctness": 0.42,
                              "retrieval_token_recall": 0.5}  # dunders stripped
    # first complete donor wins — a second attach is a no-op
    gfp.attach_quality(proj, fp, {"deepeval_answer_correctness": 0.99})
    assert gfp.load_complete_record(proj, fp)["quality"][
        "deepeval_answer_correctness"] == 0.42


def test_markers_roundtrip(tmp_path):
    root = str(tmp_path)
    assert gfp.read_marker(root, "fp_hit.json") is None
    gfp.write_marker(root, "fp_hit.json", {"fp": "abc", "donor": "eval_0001"})
    assert gfp.read_marker(root, "fp_hit.json")["donor"] == "eval_0001"


def test_retry_quality_belongs_to_fresh_answers_after_failed_judging(tmp_path):
    proj = str(tmp_path)
    fp = gfp.fingerprint(Q, CTX, P, "vllm_api", PARAM)
    gfp.save_answers(proj, fp, ["failed first", "failed second"], [4, 5], donor="failed")
    assert gfp.load_complete_record(proj, fp) is None

    path = gfp.save_answers(proj, fp, ["retry first", "retry second"], [8, 11], donor="retry")
    assert gfp.load_complete_record(proj, fp) is None
    pending = json.loads(Path(path).read_text())
    assert pending["answers"] == ["retry first", "retry second"]
    assert pending["token_counts"] == [8, 11]
    assert pending["donor"] == "retry"
    assert "quality" not in pending

    gfp.attach_quality(proj, fp, {"deepeval_answer_correctness": 0.73})
    completed = gfp.load_complete_record(proj, fp)
    assert completed == {**pending, "quality": {"deepeval_answer_correctness": 0.73}}


@pytest.mark.parametrize("quality", [{"deepeval_answer_correctness": 0.42}, {}])
def test_fresh_answers_preserve_complete_donor_bytes(tmp_path, quality):
    proj = str(tmp_path)
    fp = gfp.fingerprint(Q, CTX, P, "vllm_api", PARAM)
    path = Path(gfp.save_answers(proj, fp, ["accepted"], [7], donor="accepted"))
    gfp.attach_quality(proj, fp, quality)
    accepted = path.read_bytes()

    assert gfp.save_answers(proj, fp, ["different"], [19], donor="later") == str(path)
    assert path.read_bytes() == accepted
    assert gfp.load_complete_record(proj, fp)["quality"] == quality


def test_failed_atomic_replacement_preserves_previous_incomplete_record(tmp_path, monkeypatch):
    proj = str(tmp_path)
    fp = gfp.fingerprint(Q, CTX, P, "vllm_api", PARAM)
    path = Path(gfp.save_answers(proj, fp, ["previous"], [7], donor="failed"))
    previous = path.read_bytes()

    def unavailable_replace(source, destination):
        assert Path(destination).read_bytes() == previous
        assert json.loads(Path(source).read_text())["answers"] == ["fresh"]
        raise OSError("replacement unavailable")

    monkeypatch.setattr(gfp.os, "replace", unavailable_replace)
    assert gfp.save_answers(proj, fp, ["fresh"], [13], donor="retry") is None
    assert path.read_bytes() == previous
    assert gfp.load_complete_record(proj, fp) is None


def test_enabled_env_gate(monkeypatch):
    monkeypatch.delenv("RAG_STACK_GEN_FP_DEDUP", raising=False)
    assert not gfp.enabled()
    monkeypatch.setenv("RAG_STACK_GEN_FP_DEDUP", "1")
    assert gfp.enabled()
