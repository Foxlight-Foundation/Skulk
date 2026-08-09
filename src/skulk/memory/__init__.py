"""Skulk fabric memory: the ``skulk.memory`` core library.

Dependency-light (numpy for the math, pydantic for the models) building blocks
for the cluster's short-term associative memory: HRR primitives, the
fractional-ZCA pattern-separation stage, the :class:`MemoryIndex` interface
with an exact baseline and a holographic field implementation, and the durable
node-local :class:`ContentStore` that fields are rebuildable indexes over. The
store is the only module here that touches disk; nothing touches the fabric.
The MEMORY plane and the capture/surfacing hooks are later phases that compose
these, and nothing outside this package imports it yet.
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
from skulk.memory.store import (
    ContentStore,
    SpanRecord,
    SpanRole,
    StoreFullError,
    encode_vector,
)

__all__ = [
    "ContentStore",
    "SpanRecord",
    "SpanRole",
    "StoreFullError",
    "encode_vector",
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
