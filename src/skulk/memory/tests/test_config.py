"""Tests for the memory feature flag: off by default, opt-in only."""

from __future__ import annotations

from skulk.memory.config import (
    ENABLE_ENV_VAR,
    MemorySettings,
    memory_enabled,
    resolve_memory_settings,
)


def test_disabled_by_default() -> None:
    # No env var set -> memory is off, so Skulk is unchanged.
    assert resolve_memory_settings(env={}).enabled is False
    assert memory_enabled(env={}) is False


def test_enable_flag_is_opt_in() -> None:
    for value in ("1", "true", "TRUE", "yes", "on", " On "):
        assert memory_enabled(env={ENABLE_ENV_VAR: value}) is True


def test_non_truthy_values_stay_disabled() -> None:
    for value in ("0", "false", "no", "off", "", "maybe"):
        assert memory_enabled(env={ENABLE_ENV_VAR: value}) is False


def test_defaults_carry_the_measured_tuning() -> None:
    settings = MemorySettings()
    assert settings.enabled is False
    assert settings.field_dim == 32768
    assert settings.separation_alpha == 0.5
    assert settings.confidence_gate == 0.15


def test_settings_are_immutable() -> None:
    settings = MemorySettings()
    try:
        settings.enabled = True
    except (ValueError, TypeError):
        return
    raise AssertionError("MemorySettings must be frozen")
