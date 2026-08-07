"""Tests for the fabric's generic experimental-feature gate."""

import pytest

from skulk.shared.experimental import (
    EXPERIMENTAL_MODE_ENV_VAR,
    experimental_mode_enabled,
)


def test_disabled_when_unset() -> None:
    """Absent the env var, experimental mode is off (the default)."""
    assert experimental_mode_enabled(env={}) is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "Yes", "on", " on "])
def test_enabled_for_truthy_values(value: str) -> None:
    """Any accepted truthy spelling (case- and whitespace-insensitive) enables it."""
    assert experimental_mode_enabled(env={EXPERIMENTAL_MODE_ENV_VAR: value}) is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "maybe"])
def test_disabled_for_non_truthy_values(value: str) -> None:
    """Explicit falsey and unrecognized values keep experimental mode off."""
    assert experimental_mode_enabled(env={EXPERIMENTAL_MODE_ENV_VAR: value}) is False


def test_reads_process_env_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no injected mapping, the gate reads the real process environment."""
    monkeypatch.setenv(EXPERIMENTAL_MODE_ENV_VAR, "1")
    assert experimental_mode_enabled() is True
    monkeypatch.delenv(EXPERIMENTAL_MODE_ENV_VAR, raising=False)
    assert experimental_mode_enabled() is False
