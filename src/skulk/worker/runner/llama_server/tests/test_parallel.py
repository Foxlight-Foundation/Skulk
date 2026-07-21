# pyright: reportPrivateUsage=false
"""Tests for llama-server parallel-slot sizing."""

import pytest

from skulk.shared.models.memory_estimate import KV_CONTEXT_BUDGET_TOKENS
from skulk.worker.runner.llama_server.runner import _llama_server_parallel


def test_parallel_defaults_to_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unset node override preserves serial served-engine behavior."""
    monkeypatch.delenv("SKULK_LLAMA_SERVER_PARALLEL", raising=False)

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
