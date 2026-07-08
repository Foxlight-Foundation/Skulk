"""Skulk fabric memory: the ``skulk.memory`` core library (Phase 1).

Pure, dependency-light (numpy only) building blocks for the cluster's short-term
associative memory: HRR primitives, the fractional-ZCA pattern-separation stage,
and the :class:`MemoryIndex` interface with an exact baseline and a holographic
field implementation. No I/O and no fabric here; the content store, the MEMORY
plane, and the capture/surfacing hooks are later phases that compose these.
"""

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

# Field dimension derived from the Phase 0 measurements: 128 KB fp32 in RAM,
# 64 KB fp16 on the wire, ~400-800 episode-level traces at 91-97% accuracy.
DEFAULT_FIELD_DIM = 32768

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_CONFIDENCE_GATE",
    "DEFAULT_FIELD_DIM",
    "DEFAULT_SHRINKAGE",
    "ExactIndex",
    "HolographicField",
    "MemoryIndex",
    "ProbeResult",
    "Whitener",
    "bind",
    "cosine",
    "make_unitary",
    "normalize",
    "random_vectors",
    "superpose",
    "unbind",
]
