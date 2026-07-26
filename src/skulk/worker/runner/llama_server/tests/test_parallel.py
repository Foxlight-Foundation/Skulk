# pyright: reportAny=false, reportPrivateUsage=false
"""Tests for llama-server parallel-slot sizing."""

import threading
from unittest.mock import MagicMock

import pytest
from anyio import WouldBlock

from skulk.shared.types.tasks import TaskId
from skulk.worker.runner.llama_server.runner import (
    _LLAMA_SERVER_MAX_OUTPUT_TOKENS,
    Runner,
    _llama_server_parallel,
    _request_context_reservation,
    _slot_server_args,
)


def test_parallel_defaults_to_sixteen(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unset node serves the fleet-qualified concurrent width.

    The served engine exists to batch requests, and a unified KV buffer keeps
    the full stamped window available to every slot. Shipping one slot made a
    fresh install serial while the release fleet exercised sixteen.
    """
    monkeypatch.delenv("SKULK_LLAMA_SERVER_PARALLEL", raising=False)

    assert _llama_server_parallel() == 16
    assert _slot_server_args(_llama_server_parallel()) == [
        "--parallel",
        "16",
        "--kv-unified",
    ]


def test_parallel_rejects_invalid_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid and non-positive overrides fall back to the shipped width."""
    monkeypatch.setenv("SKULK_LLAMA_SERVER_PARALLEL", "not-an-integer")
    assert _llama_server_parallel() == 16

    monkeypatch.setenv("SKULK_LLAMA_SERVER_PARALLEL", "0")
    assert _llama_server_parallel() == 16

    monkeypatch.setenv("SKULK_LLAMA_SERVER_PARALLEL", "-4")
    assert _llama_server_parallel() == 16


def test_parallel_honors_the_declared_count_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The declared slot count is passed through without a context-derived cap.

    The old cap collapsed a high count to ``floor(n_ctx / 8192)`` because
    llama.cpp sliced the single ``-c`` window into fixed per-slot shares. The
    runner now launches with ``--kv-unified`` above one slot, which removes the
    slicing entirely, so capping would only take away concurrency the operator
    asked for without protecting anything (#689).
    """
    monkeypatch.setenv("SKULK_LLAMA_SERVER_PARALLEL", "8")
    assert _llama_server_parallel() == 8

    # Previously capped to 2 against a 16k window; now honored, because every
    # slot still sees all 16k rather than 8k.
    monkeypatch.setenv("SKULK_LLAMA_SERVER_PARALLEL", "64")
    assert _llama_server_parallel() == 64


def test_serial_slot_args_stay_byte_identical() -> None:
    """One slot launches with exactly the flags previous releases used.

    ``--kv-unified`` changes how slots share the KV buffer, and at one slot
    there is nothing to share, so adding it would only perturb the validated
    single-slot paths (draft-mtp speculation, the RPC driver) and every shipped
    install for no behavioral gain.
    """
    assert _slot_server_args(1) == ["--parallel", "1"]


def test_concurrent_slot_args_request_a_unified_kv_buffer() -> None:
    """Above one slot the runner asks for one shared window, not N slices.

    Without ``--kv-unified``, llama.cpp sets each slot's context to
    ``n_ctx / n_seq_max``, so N slots would serve a fraction of the
    ``context_token_limit`` placement stamped while the API kept admitting
    against the full one (#689).
    """
    assert _slot_server_args(4) == ["--parallel", "4", "--kv-unified"]
    assert _slot_server_args(2) == ["--parallel", "2", "--kv-unified"]


def test_context_reservation_bounds_omitted_output() -> None:
    """An omitted output limit becomes finite before aggregate admission."""
    max_output, reservation = _request_context_reservation(
        input_tokens=128,
        requested_max_output_tokens=None,
        context_pool_tokens=8192,
    )

    assert max_output == _LLAMA_SERVER_MAX_OUTPUT_TOKENS
    assert reservation == 128 + _LLAMA_SERVER_MAX_OUTPUT_TOKENS


def test_context_reservation_allows_bounded_concurrent_requests() -> None:
    """Small explicit requests reserve only their worst-case KV footprint."""
    max_output, reservation = _request_context_reservation(
        input_tokens=256,
        requested_max_output_tokens=128,
        context_pool_tokens=8192,
    )

    assert max_output == 128
    assert reservation == 384


def test_context_reservation_runs_an_over_window_request_alone() -> None:
    """A request the server must reject cannot overbook the shared pool."""
    max_output, reservation = _request_context_reservation(
        input_tokens=8000,
        requested_max_output_tokens=1024,
        context_pool_tokens=8192,
    )

    assert max_output == 1024
    assert reservation == 8192


def test_context_budget_tracks_only_resource_active_requests() -> None:
    """The weighted gate reserves shared KV and refines concurrency stamps."""
    runner = Runner.__new__(Runner)
    runner._serving_context_tokens = 8192
    runner._context_budget_condition = threading.Condition()
    runner._reserved_context_tokens = 0
    runner._context_active_requests = 0
    runner.server_proc = None
    runner._init_concurrent_dispatch(16, "test")
    runner.cancel_receiver = MagicMock()
    runner.cancel_receiver.receive_nowait.side_effect = WouldBlock
    runner.cancelled_tasks = set()
    first = TaskId("first")
    second = TaskId("second")
    with runner._admission_lock:
        runner._admission_inflight[first] = 1
        runner._admission_inflight[second] = 2

    assert runner._acquire_context_budget(first, 2048)
    assert runner._acquire_context_budget(second, 4096)
    assert runner._reserved_context_tokens == 6144
    assert runner._admission_concurrency(first) == 1
    assert runner._admission_concurrency(second) == 2

    runner._release_context_budget(2048)
    runner._release_context_budget(4096)
    assert runner._reserved_context_tokens == 0
    assert runner._context_active_requests == 0


def test_context_budget_queues_until_capacity_is_released() -> None:
    """A request cannot enter llama-server while its reservation overcommits KV."""
    runner = Runner.__new__(Runner)
    runner._serving_context_tokens = 8192
    runner._context_budget_condition = threading.Condition()
    runner._reserved_context_tokens = 0
    runner._context_active_requests = 0
    runner.server_proc = None
    runner._init_concurrent_dispatch(16, "test")
    runner.cancel_receiver = MagicMock()
    runner.cancel_receiver.receive_nowait.side_effect = WouldBlock
    runner.cancelled_tasks = set()
    first = TaskId("first")
    second = TaskId("second")
    with runner._admission_lock:
        runner._admission_inflight[first] = 1
        runner._admission_inflight[second] = 2

    assert runner._acquire_context_budget(first, 6144)
    waiting_started = threading.Event()
    admitted = threading.Event()

    def acquire_second() -> None:
        waiting_started.set()
        if runner._acquire_context_budget(second, 4096):
            admitted.set()

    waiter = threading.Thread(target=acquire_second)
    waiter.start()
    assert waiting_started.wait(timeout=1)
    assert not admitted.wait(timeout=0.1)

    runner._release_context_budget(6144)
    assert admitted.wait(timeout=1)
    waiter.join(timeout=1)
    runner._release_context_budget(4096)
    assert not waiter.is_alive()


def test_count_input_tokens_uses_llama_server_chat_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admission uses the server's exact rendered prompt token count."""
    response = MagicMock()
    response.json.return_value = {"input_tokens": 42}
    post = MagicMock(return_value=response)
    monkeypatch.setattr(
        "skulk.worker.runner.llama_server.runner.httpx.post",
        post,
    )
    runner = Runner.__new__(Runner)
    runner.base_url = "http://127.0.0.1:12345"
    body = {"messages": [{"role": "user", "content": "hello"}]}

    assert runner._count_input_tokens(body) == 42
    post.assert_called_once_with(
        "http://127.0.0.1:12345/v1/chat/completions/input_tokens",
        json=body,
        timeout=30.0,
    )
    response.raise_for_status.assert_called_once_with()


def test_count_input_tokens_fails_safe_on_malformed_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed token counts trigger whole-pool serialization, never a guess."""
    response = MagicMock()
    response.json.return_value = {"input_tokens": "42"}
    monkeypatch.setattr(
        "skulk.worker.runner.llama_server.runner.httpx.post",
        MagicMock(return_value=response),
    )
    runner = Runner.__new__(Runner)
    runner.base_url = "http://127.0.0.1:12345"

    assert runner._count_input_tokens({"messages": []}) is None
