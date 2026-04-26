"""Tests for the per-model FAST_SYNCH resolver.

The resolver decides ``MLX_METAL_FAST_SYNCH`` for a runner by combining an
operator-supplied env override, a model-card preference, and the cluster
default. The kernel-panic failure mode that motivated this resolver
(gemma-4 + ring + multinode multimodal load) is severe enough that the
priority order is load-bearing — anything that downgrades the operator
override or silently ignores a card preference can re-introduce the
panic on a fresh deployment.
"""

import pytest

from exo.shared.models.model_cards import RuntimeCapabilityCardConfig
from exo.worker.runner.bootstrap import (
    FAST_SYNCH_CLUSTER_DEFAULT,
    resolve_metal_fast_synch,
)


def test_returns_cluster_default_when_no_override_or_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SKULK_FAST_SYNCH", raising=False)
    monkeypatch.delenv("EXO_FAST_SYNCH", raising=False)
    assert resolve_metal_fast_synch(None) is FAST_SYNCH_CLUSTER_DEFAULT


def test_card_preference_overrides_cluster_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SKULK_FAST_SYNCH", raising=False)
    monkeypatch.delenv("EXO_FAST_SYNCH", raising=False)
    runtime_off = RuntimeCapabilityCardConfig(metal_fast_synch=False)
    runtime_on = RuntimeCapabilityCardConfig(metal_fast_synch=True)
    assert resolve_metal_fast_synch(runtime_off) is False
    assert resolve_metal_fast_synch(runtime_on) is True


def test_card_none_falls_through_to_cluster_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SKULK_FAST_SYNCH", raising=False)
    monkeypatch.delenv("EXO_FAST_SYNCH", raising=False)
    runtime_silent = RuntimeCapabilityCardConfig(metal_fast_synch=None)
    assert resolve_metal_fast_synch(runtime_silent) is FAST_SYNCH_CLUSTER_DEFAULT


@pytest.mark.parametrize("value", ["on", "ON", " on ", "On"])
def test_skulk_env_on_forces_true(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("SKULK_FAST_SYNCH", value)
    monkeypatch.delenv("EXO_FAST_SYNCH", raising=False)
    runtime_off = RuntimeCapabilityCardConfig(metal_fast_synch=False)
    # Operator override beats card preference, even when the card says off.
    assert resolve_metal_fast_synch(runtime_off) is True


@pytest.mark.parametrize("value", ["off", "OFF", " off ", "Off"])
def test_skulk_env_off_forces_false(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("SKULK_FAST_SYNCH", value)
    monkeypatch.delenv("EXO_FAST_SYNCH", raising=False)
    runtime_on = RuntimeCapabilityCardConfig(metal_fast_synch=True)
    # Operator override beats card preference, even when the card says on.
    assert resolve_metal_fast_synch(runtime_on) is False


def test_legacy_exo_env_still_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SKULK_FAST_SYNCH", raising=False)
    monkeypatch.setenv("EXO_FAST_SYNCH", "off")
    assert resolve_metal_fast_synch(None) is False


def test_skulk_env_wins_over_legacy_exo_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SKULK_FAST_SYNCH", "on")
    monkeypatch.setenv("EXO_FAST_SYNCH", "off")
    assert resolve_metal_fast_synch(None) is True


def test_unknown_env_value_falls_through_to_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Garbage in env should not silently pick a value — fall through.

    This protects against a typo in operator scripts (e.g. ``FAST_SYNCH=no``
    or ``=true``) silently selecting the wrong behavior. We only honor
    explicit ``on`` / ``off``; anything else flows to the card layer.
    """
    monkeypatch.setenv("SKULK_FAST_SYNCH", "auto")
    monkeypatch.delenv("EXO_FAST_SYNCH", raising=False)
    runtime_off = RuntimeCapabilityCardConfig(metal_fast_synch=False)
    assert resolve_metal_fast_synch(runtime_off) is False


def test_unknown_env_value_with_no_card_falls_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SKULK_FAST_SYNCH", "yes")
    monkeypatch.delenv("EXO_FAST_SYNCH", raising=False)
    assert resolve_metal_fast_synch(None) is FAST_SYNCH_CLUSTER_DEFAULT
