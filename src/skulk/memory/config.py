"""Feature-flag and tuning config for the memory subsystem.

Memory is **off by default** and gated by two independent switches:

1. ``SKULK_ENABLE_EXPERIMENTAL_MODE`` (node-local env, off by default) is the
   master gate. It reveals the "Experiments" section in the dashboard Settings
   and is required for any experimental feature to take effect. A node without
   it behaves exactly as it does today.
2. A fleet-synced per-feature toggle passed by the caller. Phase 1 landed
   while the legacy ``experiments`` config section was being deprecated
   (1.5.0 accepts and ignores it), so the toggle's durable home is a Phase 2
   decision under the configuration-taxonomy doctrine; callers pass the
   resolved boolean and this module stays agnostic about where it lives.

Effective memory state is the AND of the two: the operator has opted the node
into experimental mode *and* turned the memory toggle on. The later phases'
capture/surfacing hooks all check :func:`memory_enabled` first, so with either
switch off Skulk is unchanged (implementation-plan invariant #5).

The subsystem's tuning defaults (dimension, alpha, gate, decay) live here too so
they have a single home before the Settings UI exposes them.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field

from skulk.memory.index import DEFAULT_CONFIDENCE_GATE
from skulk.memory.separation import DEFAULT_ALPHA
from skulk.shared.experimental import (
    EXPERIMENTAL_MODE_ENV_VAR as EXPERIMENTAL_MODE_ENV_VAR,  # re-exported
)
from skulk.shared.experimental import (
    experimental_mode_enabled as experimental_mode_enabled,  # re-exported
)

# Field dimension derived from the Phase 0 measurements (128 KB fp32 in RAM).
DEFAULT_FIELD_DIM = 32768



class MemorySettings(BaseModel):
    """Resolved, effective memory configuration for a node.

    Immutable and strict like every Skulk config model. ``enabled`` is the
    effective state (experimental mode AND the memory toggle); the remaining
    fields are the tuning defaults derived from the Phase 0 experiments.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    enabled: bool = False
    field_dim: int = Field(
        default=DEFAULT_FIELD_DIM,
        gt=0,
        description=(
            "Holographic field dimension. The Phase 0 capacity ceiling is about "
            "dim/20 traces with cleanup, and the default costs 128 KB of fp32."
        ),
    )
    separation_alpha: float = Field(
        default=DEFAULT_ALPHA,
        ge=0.0,
        le=1.0,
        description=(
            "Fractional-ZCA whitening exponent for the pattern-separation "
            "stage: 0 is no whitening, 1 is full ZCA."
        ),
    )
    confidence_gate: float = Field(
        default=DEFAULT_CONFIDENCE_GATE,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum probe confidence before a trace surfaces. The Phase 0 "
            "experiments measured zero cross-episode false confidence at the "
            "default; surfacing, not recall, is the contract."
        ),
    )
    decay_half_life_days: float = Field(
        default=3.0,
        gt=0.0,
        description=(
            "Half-life for trace amplitude decay applied by the maintenance "
            "cadence; forgetting emerges from interference plus this decay."
        ),
    )




def resolve_memory_settings(
    *, feature_enabled: bool = False, env: Mapping[str, str] | None = None
) -> MemorySettings:
    """Resolve effective memory settings.

    ``feature_enabled`` is the fleet-synced per-feature toggle (defaults off;
    the caller passes the resolved boolean from wherever Phase 2 homes it,
    since the legacy experiments config section is deprecated).
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
