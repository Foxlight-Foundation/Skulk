# pyright: reportPrivateUsage=false
"""Tests for llama-server parallel-slot sizing."""

import pytest

from skulk.shared.models.memory_estimate import KV_CONTEXT_BUDGET_TOKENS
from skulk.worker.runner.llama_server.runner import (
    _LLAMA_SERVER_PARALLEL_AUTO_CAP,
    _llama_server_parallel,
)


def test_parallel_derives_concurrency_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no override, a fresh node serves concurrent load out of the box."""
    monkeypatch.delenv("SKULK_LLAMA_SERVER_PARALLEL", raising=False)

    # A large context funds the full derived ceiling.
    assert (
        _llama_server_parallel(_LLAMA_SERVER_PARALLEL_AUTO_CAP * KV_CONTEXT_BUDGET_TOKENS)
        == _LLAMA_SERVER_PARALLEL_AUTO_CAP
    )
    # A smaller context shrinks the derived count to its funded slots.
    assert _llama_server_parallel(4 * KV_CONTEXT_BUDGET_TOKENS) == 4
    # A tiny context still gets one full-context slot.
    assert _llama_server_parallel(KV_CONTEXT_BUDGET_TOKENS // 2) == 1


def test_parallel_explicit_one_forces_serial(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit 1 remains the serial opt-out."""
    monkeypatch.setenv("SKULK_LLAMA_SERVER_PARALLEL", "1")

    assert _llama_server_parallel(131072) == 1


def test_parallel_rejects_invalid_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid and non-positive overrides fall back to one slot."""
    monkeypatch.setenv("SKULK_LLAMA_SERVER_PARALLEL", "not-an-integer")
    assert _llama_server_parallel(131072) == 1

    monkeypatch.setenv("SKULK_LLAMA_SERVER_PARALLEL", "0")
    assert _llama_server_parallel(131072) == 1


def test_parallel_keeps_requested_count_when_each_slot_retains_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configured concurrency is unchanged when the context can fund every slot."""
    monkeypatch.setenv("SKULK_LLAMA_SERVER_PARALLEL", "8")

    assert _llama_server_parallel(8 * KV_CONTEXT_BUDGET_TOKENS) == 8


def test_parallel_caps_count_to_context_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A large override cannot collapse per-request context below the floor."""
    monkeypatch.setenv("SKULK_LLAMA_SERVER_PARALLEL", "64")

    assert _llama_server_parallel(2 * KV_CONTEXT_BUDGET_TOKENS) == 2
    assert _llama_server_parallel(16 * KV_CONTEXT_BUDGET_TOKENS) == 16


def test_parallel_keeps_one_slot_for_a_small_model_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model below the shared floor still receives one full-context slot."""
    monkeypatch.setenv("SKULK_LLAMA_SERVER_PARALLEL", "64")

    assert _llama_server_parallel(KV_CONTEXT_BUDGET_TOKENS // 2) == 1
