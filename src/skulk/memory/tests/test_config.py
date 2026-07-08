"""Tests for the memory feature flags: two gates, both off by default."""

from __future__ import annotations

from skulk.memory.config import (
    EXPERIMENTAL_MODE_ENV_VAR,
    MemorySettings,
    experimental_mode_enabled,
    memory_enabled,
    resolve_memory_settings,
)

ON = {EXPERIMENTAL_MODE_ENV_VAR: "1"}


def test_disabled_by_default() -> None:
    # Neither gate set -> memory is off, so Skulk is unchanged.
    assert experimental_mode_enabled(env={}) is False
    assert memory_enabled(feature_enabled=True, env={}) is False
    assert resolve_memory_settings(feature_enabled=True, env={}).enabled is False


def test_both_gates_required() -> None:
    # Experimental mode on but the feature toggle off -> still disabled.
    assert memory_enabled(feature_enabled=False, env=ON) is False
    # Feature toggle on but experimental mode off -> still disabled.
    assert memory_enabled(feature_enabled=True, env={}) is False
    # Both on -> enabled.
    assert memory_enabled(feature_enabled=True, env=ON) is True


def test_experimental_mode_truthy_parsing() -> None:
    for value in ("1", "true", "TRUE", "yes", "on", " On "):
        assert experimental_mode_enabled(env={EXPERIMENTAL_MODE_ENV_VAR: value}) is True
    for value in ("0", "false", "no", "off", "", "maybe"):
        assert experimental_mode_enabled(env={EXPERIMENTAL_MODE_ENV_VAR: value}) is False


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
