# pyright: reportPrivateUsage=false, reportUnusedFunction=false
"""Tests for the external-shipper mode of the structured JSON log sink.

Covers:
- ``SKULK_LOGGING_EXTERNAL=1`` enables the JSON sink without spawning
  Skulk's internal Vector subprocess.
- Toggling via ``set_structured_stdout`` honors the same env var.
- Falsy / unset values leave the legacy "internal subprocess" path in
  place (where ingest_url drives ``_start_vector``).
"""

from __future__ import annotations

import contextlib
from unittest.mock import patch

import pytest
from loguru import logger

from exo.shared import logging as skulk_logging


@pytest.fixture(autouse=True)
def _reset_logging_state() -> None:
    """Drop any sink the previous test left behind so each test starts clean."""
    if skulk_logging._json_sink_id is not None:
        with contextlib.suppress(ValueError):
            logger.remove(skulk_logging._json_sink_id)
        skulk_logging._json_sink_id = None
    skulk_logging._vector_pipe = None
    skulk_logging._vector_process = None


def test_external_flag_enables_json_sink_without_starting_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SKULK_LOGGING_EXTERNAL=1 should activate the JSON sink and skip _start_vector."""
    monkeypatch.setenv("SKULK_LOGGING_EXTERNAL", "1")

    with patch.object(skulk_logging, "_start_vector") as start_vector:
        skulk_logging.logger_setup(
            log_file=None,
            verbosity=0,
            structured_stdout=True,
            ingest_url="http://example.invalid/ignored",
        )

    assert skulk_logging._json_sink_id is not None, (
        "JSON sink should be active in external mode"
    )
    start_vector.assert_not_called()


def test_external_flag_works_with_empty_ingest_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In external mode, ingest_url is irrelevant — the agent owns transport."""
    monkeypatch.setenv("SKULK_LOGGING_EXTERNAL", "true")

    with patch.object(skulk_logging, "_start_vector") as start_vector:
        skulk_logging.logger_setup(
            log_file=None,
            verbosity=0,
            structured_stdout=True,
            ingest_url="",
        )

    assert skulk_logging._json_sink_id is not None
    start_vector.assert_not_called()


def test_external_unset_falls_back_to_internal_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the external flag, an ingest_url should drive _start_vector as before."""
    monkeypatch.delenv("SKULK_LOGGING_EXTERNAL", raising=False)

    with patch.object(skulk_logging, "_start_vector", return_value=True) as start_vector:
        skulk_logging.logger_setup(
            log_file=None,
            verbosity=0,
            structured_stdout=True,
            ingest_url="http://example.invalid/insert",
        )

    start_vector.assert_called_once_with("http://example.invalid/insert")


def test_external_falsy_values_disable_external_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Truthy detection should be strict — only 1/true/yes/on count."""
    monkeypatch.setenv("SKULK_LOGGING_EXTERNAL", "0")

    with patch.object(skulk_logging, "_start_vector", return_value=True) as start_vector:
        skulk_logging.logger_setup(
            log_file=None,
            verbosity=0,
            structured_stdout=True,
            ingest_url="http://example.invalid/insert",
        )

    start_vector.assert_called_once()


def test_set_structured_stdout_honors_external_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime toggle via cluster sync also respects the external flag."""
    monkeypatch.setenv("SKULK_LOGGING_EXTERNAL", "1")

    with patch.object(skulk_logging, "_start_vector") as start_vector:
        skulk_logging.set_structured_stdout(enabled=True, ingest_url="")

    assert skulk_logging._json_sink_id is not None
    start_vector.assert_not_called()

    with patch.object(skulk_logging, "_stop_vector") as stop_vector:
        skulk_logging.set_structured_stdout(enabled=False)

    assert skulk_logging._json_sink_id is None
    stop_vector.assert_called_once()


# ---------------------------------------------------------------------------
# Production call-site gating: callers must allow external mode to activate
# the JSON sink even when `logging.enabled=false` and `ingest_url=""` in
# skulk.yaml. These tests reproduce the gating logic from main.py,
# api/main.py, and download/coordinator.py to catch regressions of the
# "external mode is unreachable through the real startup path" bug.
# ---------------------------------------------------------------------------


def _main_gate(log_enabled: bool) -> bool:
    """Boot-time gate from src/exo/main.py."""
    return skulk_logging.external_log_pipe_enabled() or log_enabled


def _runtime_sync_gate(log_enabled: bool, ingest_url: str = "x") -> bool:
    """Runtime gate from api/main.py and download/coordinator.py.

    In internal-subprocess mode (env var off), the legacy contract is
    that both ``enabled`` and ``ingest_url`` must be set — clearing the
    URL at runtime disables shipping. The env var bypasses both because
    transport is owned by the external agent.
    """
    return skulk_logging.external_log_pipe_enabled() or (
        log_enabled and bool(ingest_url)
    )


def test_main_gate_activates_with_external_flag_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SKULK_LOGGING_EXTERNAL=1 alone must satisfy main.py's boot gate."""
    monkeypatch.setenv("SKULK_LOGGING_EXTERNAL", "1")
    assert _main_gate(log_enabled=False) is True


def test_main_gate_inactive_without_either(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No env var, no logging.enabled → no JSON sink (legacy behavior preserved)."""
    monkeypatch.delenv("SKULK_LOGGING_EXTERNAL", raising=False)
    assert _main_gate(log_enabled=False) is False


def test_runtime_sync_gate_activates_with_external_flag_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dashboard sync of `enabled=false` cannot disable an env-var-driven sink."""
    monkeypatch.setenv("SKULK_LOGGING_EXTERNAL", "1")
    assert _runtime_sync_gate(log_enabled=False) is True


def test_runtime_sync_gate_respects_legacy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the env var, runtime gating still honors logging.enabled."""
    monkeypatch.delenv("SKULK_LOGGING_EXTERNAL", raising=False)
    assert _runtime_sync_gate(log_enabled=True, ingest_url="http://x") is True
    assert _runtime_sync_gate(log_enabled=False, ingest_url="http://x") is False


def test_runtime_sync_clearing_ingest_url_disables_internal_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Internal mode: clearing ingest_url at runtime must disable shipping.

    Regression guard for the case where an operator removes ingest_url
    via a runtime config sync. The legacy contract requires shipping to
    stop; without this, the in-process Vector subprocess would keep
    shipping to the prior URL until Skulk is restarted.
    """
    monkeypatch.delenv("SKULK_LOGGING_EXTERNAL", raising=False)
    assert _runtime_sync_gate(log_enabled=True, ingest_url="") is False


def test_runtime_sync_external_mode_ignores_empty_ingest_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """External mode: ingest_url is owned by the agent, so empty is fine."""
    monkeypatch.setenv("SKULK_LOGGING_EXTERNAL", "1")
    assert _runtime_sync_gate(log_enabled=False, ingest_url="") is True
