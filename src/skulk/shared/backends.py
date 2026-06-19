"""Backend capability tags: the shared vocabulary for engine + compute routing.

A *backend tag* names how a model is actually executed on a node, in the form
``<engine>-<compute>`` (for example ``mlx-metal``, ``llama_cpp-vulkan``,
``llama_cpp-rocm``). Two axes are deliberately folded into one self-describing
string:

- **engine** -- which inference runtime loads and runs the model (``mlx`` vs
  ``llama_cpp``). This is what selects the worker runner class.
- **compute** -- which compute backend that runtime drives on a given node
  (Apple ``metal``; for llama.cpp ``vulkan`` / ``rocm`` / ``cuda`` / ``cpu``).
  The *same* model file runs identically across compute backends, but their
  performance differs per model, so a card may prefer one (see
  ``PlacementCardConfig.backend_preference``).

Nodes advertise the set of tags they can actually serve (probed +
operator-declared) in ``NodeResources.backends``; cards declare the set they are
*allowed* on (``compatible_backends``, a hard placement filter) and an ordered
*preference* among them (soft, with graceful fallback). Keeping the filter and
the preference separate is what lets a model say "fastest on Vulkan, but ROCm is
fine" and still place on a ROCm-only node.

For backward compatibility a node also advertises the bare engine tag alongside
each compound tag (a Mac advertises both ``mlx`` and ``mlx-metal``), so cards
written against the original ``{"mlx"}`` vocabulary keep matching unchanged.
"""

from __future__ import annotations

import os
import sys
from typing import Final, Literal

EngineType = Literal["mlx", "llama_cpp"]
"""Inference runtime that loads and runs a model; selects the worker runner."""

ComputeBackend = Literal["metal", "vulkan", "rocm", "cuda", "cpu"]
"""Compute backend a runtime drives on a node."""

# Explicit typed tuples (rather than typing.get_args, which erases to Any) so the
# values stay narrowed to their Literal types where they are consumed.
_ENGINES: Final[tuple[EngineType, ...]] = ("mlx", "llama_cpp")
_COMPUTE_BACKENDS: Final[tuple[ComputeBackend, ...]] = (
    "metal",
    "vulkan",
    "rocm",
    "cuda",
    "cpu",
)

_TAG_SEPARATOR: Final = "-"

# Operator declaration of which llama.cpp compute backends a node was built with,
# as a comma-separated list (e.g. "vulkan" or "vulkan,rocm"). The compiled build
# -- not what libraries happen to be installed -- determines what llama.cpp can
# actually use, and the Python binding does not cleanly expose that, so we treat
# this as authoritative operator policy (mirroring SKULK_NODE_PARTICIPATION).
LLAMA_CPP_BACKENDS_ENV: Final = "SKULK_LLAMA_CPP_BACKENDS"


def make_backend_tag(engine: EngineType, compute: ComputeBackend) -> str:
    """Return the compound ``<engine>-<compute>`` tag for an engine + compute pair."""
    return f"{engine}{_TAG_SEPARATOR}{compute}"


def engine_of(tag: str) -> EngineType | None:
    """Return the engine a backend tag selects, or ``None`` if it names no known engine.

    Accepts both compound tags (``llama_cpp-vulkan``) and bare engine tags
    (``llama_cpp``). Returns ``None`` for unrecognized strings so callers can
    skip tags they do not understand rather than crash on forward-compat input.
    """
    for engine in _ENGINES:
        if tag == engine or tag.startswith(f"{engine}{_TAG_SEPARATOR}"):
            return engine
    return None


def resolve_node_engine(
    compatible_backends: frozenset[str],
    backend_preference: tuple[str, ...],
    node_backends: frozenset[str],
) -> EngineType | None:
    """Resolve which engine a node should use to serve a model.

    Intersects the card's ``compatible_backends`` (hard filter) with the node's
    advertised ``node_backends``, orders the result by ``backend_preference``
    (preferred tags first, then the rest deterministically), and returns the
    engine of the winning tag. Returns ``None`` when the node advertises none of
    the model's compatible backends (which placement should already have ruled
    out, so the caller treats it as "fall back to the default engine"). This is
    the single point that turns the backend-tag vocabulary into a runner choice.
    """
    intersection = compatible_backends & node_backends
    if not intersection:
        return None
    ordered = [tag for tag in backend_preference if tag in intersection]
    ordered += [tag for tag in sorted(intersection) if tag not in backend_preference]
    return engine_of(ordered[0])


def _probe_llama_cpp_backends() -> frozenset[str]:
    """Probe whether llama.cpp is usable here and which compute backends it offers.

    Returns the bare ``llama_cpp`` tag plus one compound tag per advertised
    compute backend. Compute backends come from ``SKULK_LLAMA_CPP_BACKENDS``
    when set (authoritative, since the compiled build decides); otherwise we fall
    back to the always-correct ``cpu`` tag so a node never over-claims GPU
    capability it might not have built. Returns an empty set when the binding is
    not importable, so a node without llama.cpp simply does not advertise it.
    """
    try:
        import llama_cpp  # noqa: F401  # pyright: ignore[reportMissingImports, reportUnusedImport]
    except ImportError:
        return frozenset()

    tags: set[str] = {"llama_cpp"}
    declared_tokens = {
        token
        for raw in os.environ.get(LLAMA_CPP_BACKENDS_ENV, "").split(",")
        if (token := raw.strip())
    }
    # Intersect with the known compute backends (preserving our canonical order;
    # order is irrelevant in the advertised set). ``metal`` is MLX-only, so it is
    # never a valid llama.cpp compute backend even if an operator declares it.
    computes: list[ComputeBackend] = [
        cb for cb in _COMPUTE_BACKENDS if cb != "metal" and cb in declared_tokens
    ]
    if not computes:
        # No operator declaration: claim only CPU, which any llama.cpp build can do.
        computes = ["cpu"]
    for compute in computes:
        tags.add(make_backend_tag("llama_cpp", compute))
    return frozenset(tags)


def probe_node_backends() -> frozenset[str]:
    """Probe the backend tags this node can actually serve.

    Apple-Silicon nodes advertise ``{"mlx", "mlx-metal"}`` (the bare engine tag
    is kept for backward compatibility with cards written against the original
    ``{"mlx"}`` vocabulary). Any node with an importable ``llama_cpp`` adds its
    llama.cpp tags. A bare Linux node with neither advertises an empty set and
    is therefore not a placement candidate, which is the pre-existing behavior.
    """
    tags: set[str] = set()
    if sys.platform == "darwin":
        tags |= {"mlx", make_backend_tag("mlx", "metal")}
    tags |= _probe_llama_cpp_backends()
    return frozenset(tags)
