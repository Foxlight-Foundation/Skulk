"""Skulk fabric memory: the ``skulk.memory`` core library (Phase 1).

Pure, dependency-light (numpy only) building blocks for the cluster's short-term
associative memory: HRR primitives, the fractional-ZCA pattern-separation stage,
and the :class:`MemoryIndex` interface with an exact baseline and a holographic
field implementation. No I/O and no fabric here; the content store, the MEMORY
plane, and the capture/surfacing hooks are later phases that compose these.
"""

from skulk.memory.config import (
    DEFAULT_FIELD_DIM,
    EXPERIMENTAL_MODE_ENV_VAR,
    MemorySettings,
    experimental_mode_enabled,
    memory_enabled,
    resolve_memory_settings,
)
from skulk.memory.hrr import (
    bind,
    cosine,
    make_unitary,
    normalize,
    random_vectors,
    superpose,
    unbind,
)
from skulk.memory.index import (
    DEFAULT_CONFIDENCE_GATE,
    ExactIndex,
    HolographicField,
    MemoryIndex,
    ProbeResult,
)
from skulk.memory.separation import (
    DEFAULT_ALPHA,
    DEFAULT_SHRINKAGE,
    Whitener,
)

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_CONFIDENCE_GATE",
    "DEFAULT_FIELD_DIM",
    "DEFAULT_SHRINKAGE",
    "EXPERIMENTAL_MODE_ENV_VAR",
    "ExactIndex",
    "HolographicField",
    "MemoryIndex",
    "MemorySettings",
    "ProbeResult",
    "Whitener",
    "experimental_mode_enabled",
    "memory_enabled",
    "resolve_memory_settings",
    "bind",
    "cosine",
    "make_unitary",
    "normalize",
    "random_vectors",
    "superpose",
    "unbind",
]
