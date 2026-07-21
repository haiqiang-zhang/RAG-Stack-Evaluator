import asyncio

import pytest

from rag_stack_evaluator.static_rag_evaluator.measured.vllm_pd_pair import (
    VllmPdPair,
    VllmPdPairKey,
)
from rag_stack_evaluator.static_rag_evaluator.measured.vllm_subprocess import (
    MEASURED_REQUEST_FORMAT_KEY,
    REQUEST_FORMAT_CHAT_COMPLETIONS,
)


def _pair(*, prefill: int = 2, decode: int = 256) -> VllmPdPair:
    pair = object.__new__(VllmPdPair)
    pair.key = VllmPdPairKey(
        model="Qwen/Test",
        max_num_seqs=64,
        prefill_max_num_seqs=prefill,
        decode_max_num_seqs=decode,
    )
    pair._served_max_model_len = 32_768
    pair.prefill_port = 10_001
    pair.decode_port = 10_002
    pair._http_by_loop = {}
    pair._role_admission_by_loop = {}
    pair._last_role_admission_stats = None
    return pair


def _sampling_params() -> dict:
    return {
        "temperature": 0.0,
        "max_tokens": 8,
        MEASURED_REQUEST_FORMAT_KEY: REQUEST_FORMAT_CHAT_COMPLETIONS,
    }


def _decode_response(text: str = "answer") -> dict:
    return {
        "choices": [{"message": {"content": text}}],
        "usage": {"completion_tokens": 2},
    }


async def _wait_until(predicate, *, turns: int = 1_000) -> None:
    for _ in range(turns):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition did not become true")


def test_pd_role_and_outer_admission_limits_are_role_aware():
    pair = _pair(prefill=2, decode=256)

    assert pair.role_engine_limits() == {"prefill": 2, "decode": 256}
    assert pair.role_admission_limits() == {"prefill": 3, "decode": 256}
    assert pair.stage_admission_limit() == 259

    # The policy is derived from the configured P cap, not a literal client
    # admission of three that only works for the s44 point.
    larger_pair = _pair(prefill=7, decode=11)
    assert larger_pair.role_engine_limits() == {"prefill": 7, "decode": 11}
    assert larger_pair.role_admission_limits() == {"prefill": 8, "decode": 11}
    assert larger_pair.stage_admission_limit() == 19


@pytest.mark.parametrize("prefill_engine_cap", [2, 5])
def test_prefill_permit_is_held_until_decode_admission(prefill_engine_cap):
    pair = _pair(prefill=prefill_engine_cap, decode=1)
    prefill_admission = prefill_engine_cap + 1
    n_requests = prefill_admission + 3
    prefill_calls = 0
    decode_calls = 0

    async def fake_post(port, payload, request_id, *, request_format):
        nonlocal prefill_calls, decode_calls
        if port == pair.prefill_port:
            prefill_calls += 1
            return {"kv_transfer_params": {"remote_request_id": request_id}}
        decode_calls += 1
        await asyncio.sleep(0)
        return _decode_response()

    pair._raw_post = fake_post

    async def exercise():
        # Occupy D first. At most engine P + one feeder requests may create
        # producer handles, and all retain their P permits while waiting for D.
        # The remaining requests must therefore stay before prefill.
        decode_blocker = await pair._acquire_role("decode")
        tasks = [
            asyncio.create_task(pair.generate_one(f"q{i}", _sampling_params()))
            for i in range(n_requests)
        ]
        await _wait_until(
            lambda: pair.role_admission_stats()["prefill"]["current_inflight"]
            == prefill_admission
            and pair.role_admission_stats()["decode"]["current_waiting"]
            == prefill_admission
        )

        assert prefill_calls == prefill_admission
        assert prefill_calls - prefill_engine_cap == 1
        assert decode_calls == 0
        stats = pair.role_admission_stats()
        assert stats["prefill"]["engine_max_num_seqs"] == prefill_engine_cap
        assert stats["prefill"]["admission_limit"] == prefill_admission
        assert stats["prefill"]["current_inflight"] == prefill_admission
        assert stats["prefill"]["current_waiting"] == 3

        pair._release_role(decode_blocker)
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=2.0)
        return pair.role_admission_stats()

    stats = asyncio.run(exercise())

    assert prefill_calls == n_requests
    assert decode_calls == n_requests
    assert stats["prefill"]["max_inflight_observed"] == prefill_admission
    assert stats["decode"]["max_inflight_observed"] == 1
    assert stats["prefill"]["current_inflight"] == 0
    assert stats["decode"]["current_inflight"] == 0


def test_cancellation_while_waiting_for_decode_releases_prefill():
    pair = _pair(prefill=1, decode=1)

    async def fake_post(port, payload, request_id, *, request_format):
        if port == pair.prefill_port:
            return {"kv_transfer_params": {"remote_request_id": request_id}}
        return _decode_response()

    pair._raw_post = fake_post

    async def exercise():
        decode_blocker = await pair._acquire_role("decode")
        task = asyncio.create_task(pair.generate_one("cancel", _sampling_params()))
        await _wait_until(
            lambda: pair.role_admission_stats()["decode"]["current_waiting"] == 1
        )
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        after_cancel = pair.role_admission_stats()
        assert after_cancel["prefill"]["current_inflight"] == 0
        assert after_cancel["decode"]["current_waiting"] == 0
        assert after_cancel["prefill"]["cancelled"] == 1
        assert after_cancel["decode"]["cancelled"] == 1

        pair._release_role(decode_blocker)
        result = await asyncio.wait_for(
            pair.generate_one("next", _sampling_params()), timeout=1.0
        )
        return result, pair.role_admission_stats()

    (text, _perf), stats = asyncio.run(exercise())

    assert text == "answer"
    assert stats["prefill"]["current_inflight"] == 0
    assert stats["decode"]["current_inflight"] == 0


def test_decode_failure_releases_role_permit_for_next_request():
    pair = _pair(prefill=1, decode=1)
    decode_attempt = 0

    async def fake_post(port, payload, request_id, *, request_format):
        nonlocal decode_attempt
        if port == pair.prefill_port:
            return {"kv_transfer_params": {"remote_request_id": request_id}}
        decode_attempt += 1
        if decode_attempt == 1:
            raise RuntimeError("decode failed")
        return _decode_response("recovered")

    pair._raw_post = fake_post

    async def exercise():
        with pytest.raises(RuntimeError, match="decode failed"):
            await pair.generate_one("first", _sampling_params())
        result = await asyncio.wait_for(
            pair.generate_one("second", _sampling_params()), timeout=1.0
        )
        return result, pair.role_admission_stats()

    (text, _perf), stats = asyncio.run(exercise())

    assert text == "recovered"
    assert stats["decode"]["failed"] == 1
    assert stats["decode"]["released"] == 2
    assert stats["decode"]["completed"] == 1
    assert stats["decode"]["current_inflight"] == 0


def test_role_gates_are_loop_local_and_close_preserves_window_stats():
    pair = _pair(prefill=1, decode=1)

    async def fake_post(port, payload, request_id, *, request_format):
        await asyncio.sleep(0)
        if port == pair.prefill_port:
            return {"kv_transfer_params": {"remote_request_id": request_id}}
        return _decode_response()

    pair._raw_post = fake_post

    async def one_loop():
        pair.mark_role_admission_window_start()
        await asyncio.gather(
            *(pair.generate_one(f"q{i}", _sampling_params()) for i in range(3))
        )
        pair.mark_role_admission_window_end()
        before_close = pair.role_admission_stats()
        await pair.aclose_http()
        after_close = pair.role_admission_stats()
        return before_close, after_close

    first_before, first_after = asyncio.run(one_loop())
    second_before, second_after = asyncio.run(one_loop())

    for stats in (first_before, first_after, second_before, second_after):
        assert stats["prefill"]["submitted"] == 3
        assert stats["prefill"]["completed"] == 3
        assert stats["decode"]["submitted"] == 3
        assert stats["decode"]["completed"] == 3
        assert stats["prefill"]["max_inflight_observed"] == 2
        assert stats["decode"]["max_inflight_observed"] == 1
    assert pair._role_admission_by_loop == {}
