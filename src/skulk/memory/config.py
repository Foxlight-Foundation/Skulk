"""Feature-flag and tuning config for the memory subsystem.

Memory is **off by default** and gated by two independent switches:

1. ``SKULK_ENABLE_EXPERIMENTAL_MODE`` (node-local env, off by default) is the
   master gate. It reveals the "Experiments" section in the dashboard Settings
   and is required for any experimental feature to take effect. A node without
   it behaves exactly as it does today.
2. ``experiments.memory_enabled`` (a fleet-synced Settings field) is the
   per-feature toggle inside that section.

Effective memory state is the AND of the two: the operator has opted the node
into experimental mode *and* turned the memory toggle on. The later phases'
capture/surfacing hooks all check :func:`memory_enabled` first, so with either
switch off Skulk is unchanged (implementation-plan invariant #5).

The subsystem's tuning defaults (dimension, alpha, gate, decay) live here too so
they have a single home before the Settings UI exposes them.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field

from skulk.memory.index import DEFAULT_CONFIDENCE_GATE
from skulk.memory.separation import DEFAULT_ALPHA

# Field dimension derived from the Phase 0 measurements (128 KB fp32 in RAM).
DEFAULT_FIELD_DIM = 32768

_TRUTHY = frozenset({"1", "true", "yes", "on"})

EXPERIMENTAL_MODE_ENV_VAR = "SKULK_ENABLE_EXPERIMENTAL_MODE"


class MemorySettings(BaseModel):
    """Resolved, effective memory configuration for a node.

    Immutable and strict like every Skulk config model. ``enabled`` is the
    effective state (experimental mode AND the memory toggle); the remaining
    fields are the tuning defaults derived from the Phase 0 experiments.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    enabled: bool = False
    field_dim: int = Field(default=DEFAULT_FIELD_DIM, gt=0)
    separation_alpha: float = Field(default=DEFAULT_ALPHA, ge=0.0, le=1.0)
    confidence_gate: float = Field(default=DEFAULT_CONFIDENCE_GATE, ge=0.0, le=1.0)
    decay_half_life_days: float = Field(default=3.0, gt=0.0)


def _is_truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in _TRUTHY


def experimental_mode_enabled(env: Mapping[str, str] | None = None) -> bool:
    """True when this node is opted into experimental features (the master gate).

    Driven by ``SKULK_ENABLE_EXPERIMENTAL_MODE``; off unless explicitly set. This
    both reveals the Experiments settings section and gates every experimental
    feature's effect.
    """
    source = os.environ if env is None else env
    return _is_truthy(source.get(EXPERIMENTAL_MODE_ENV_VAR))


def resolve_memory_settings(
    *, feature_enabled: bool = False, env: Mapping[str, str] | None = None
) -> MemorySettings:
    """Resolve effective memory settings.

    ``feature_enabled`` is the fleet-synced ``experiments.memory_enabled`` toggle
    (defaults off, and passed by the caller from the loaded cluster config).
    Memory is enabled only when experimental mode is on *and* the toggle is on.
    ``env`` is injectable so tests never touch real process state.
    """
    enabled = experimental_mode_enabled(env) and feature_enabled
    return MemorySettings(enabled=enabled)


def memory_enabled(
    *, feature_enabled: bool = False, env: Mapping[str, str] | None = None
) -> bool:
    """True only when memory is effectively active (both gates cleared)."""
    return resolve_memory_settings(feature_enabled=feature_enabled, env=env).enabled
