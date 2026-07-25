# pyright: reportPrivateUsage=false
"""Tests for llama-server parallel-slot sizing."""

import pytest

from skulk.worker.runner.llama_server.runner import (
    _llama_server_parallel,
    _slot_server_args,
)


def test_parallel_defaults_to_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unset node override preserves serial served-engine behavior.

    Deliberate, and unrelated to whether concurrency is safe: with a unified KV
    buffer every slot serves the full stamped window, so N > 1 no longer shrinks
    what a request gets. What it does share is a finite pool, and how much of it
    to hand out is a per-deployment judgement (#689).
    """
    monkeypatch.delenv("SKULK_LLAMA_SERVER_PARALLEL", raising=False)

    assert _llama_server_parallel() == 1


def test_parallel_rejects_invalid_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid and non-positive overrides fall back to one slot."""
    monkeypatch.setenv("SKULK_LLAMA_SERVER_PARALLEL", "not-an-integer")
    assert _llama_server_parallel() == 1

    monkeypatch.setenv("SKULK_LLAMA_SERVER_PARALLEL", "0")
    assert _llama_server_parallel() == 1

    monkeypatch.setenv("SKULK_LLAMA_SERVER_PARALLEL", "-4")
    assert _llama_server_parallel() == 1


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
