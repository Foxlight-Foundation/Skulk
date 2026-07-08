"""Feature-flag and tuning config for the memory subsystem.

Memory is **off by default**. Nothing in the Skulk runtime path imports the
memory package, and when the later phases add their capture/surfacing hooks
every one of them checks :func:`memory_enabled` first, so a cluster with the
flag off behaves exactly as it does today (implementation-plan invariant #5:
"Memory off = Skulk unchanged").

For Phase 1 the switch is the node-local env var ``SKULK_MEMORY_ENABLED`` (the
config-taxonomy doctrine allows env for node-local launch + dev/test). When the
MEMORY plane lands in Phase 3 this promotes to a gossipsub Settings field so the
whole fleet toggles together; call sites use :func:`resolve_memory_settings`, so
that swap does not touch them.
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

ENABLE_ENV_VAR = "SKULK_MEMORY_ENABLED"


class MemorySettings(BaseModel):
    """Resolved memory configuration for a node.

    Immutable and strict like every Skulk config model. ``enabled`` is the
    master switch; the remaining fields are the tuning defaults derived from the
    Phase 0 experiments, surfaced here so they have a single home before the
    Settings/UI plumbing arrives.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    enabled: bool = False
    field_dim: int = Field(default=DEFAULT_FIELD_DIM, gt=0)
    separation_alpha: float = Field(default=DEFAULT_ALPHA, ge=0.0, le=1.0)
    confidence_gate: float = Field(default=DEFAULT_CONFIDENCE_GATE, ge=0.0, le=1.0)
    decay_half_life_days: float = Field(default=3.0, gt=0.0)


def _is_truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in _TRUTHY


def resolve_memory_settings(env: Mapping[str, str] | None = None) -> MemorySettings:
    """Resolve memory settings from the environment (defaults to off).

    ``env`` defaults to ``os.environ``; it is injectable so tests never touch
    real process state. Only the enable flag is env-driven in Phase 1; the tuning
    fields keep their measured defaults until Settings exposes them.
    """
    source = os.environ if env is None else env
    return MemorySettings(enabled=_is_truthy(source.get(ENABLE_ENV_VAR)))


def memory_enabled(env: Mapping[str, str] | None = None) -> bool:
    """True only when the memory subsystem is explicitly enabled for this node."""
    return resolve_memory_settings(env).enabled
