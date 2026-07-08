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
from pathlib import Path
from typing import Final, Literal

from loguru import logger


def _is_executable_file(path: str) -> bool:
    """Whether ``path`` names an existing executable file."""
    return os.path.isfile(path) and os.access(path, os.X_OK)

EngineType = Literal["mlx", "mlx_audio", "llama_cpp", "llama_server"]
"""Inference runtime that loads and runs a model; selects the worker runner.

``llama_server`` is a *served-backend* engine: instead of loading the model
in-process (like ``mlx`` and ``llama_cpp``), the worker launches an external
``llama-server`` subprocess and proxies its OpenAI HTTP API. It is the only way
to reach llama.cpp's native multi-token-prediction speculative decoding
(``--spec-type draft-mtp``), whose orchestration lives in the server application
rather than ``libllama`` / the Python binding. It coexists with the in-process
``llama_cpp`` engine and is selected per model by the card's ``compatible_backends``.

``mlx_audio`` is the single-node speech engine backed by the upstream
``mlx-audio`` package. It is kept separate from ``mlx`` because TTS/STT model
loading, generation, and future realtime session contracts are not the same as
the text/vision MLX runner.
"""

ComputeBackend = Literal["metal", "vulkan", "rocm", "cuda", "cpu"]
"""Compute backend a runtime drives on a node."""

# Explicit typed tuples (rather than typing.get_args, which erases to Any) so the
# values stay narrowed to their Literal types where they are consumed.
_ENGINES: Final[tuple[EngineType, ...]] = (
    "mlx",
    "mlx_audio",
    "llama_cpp",
    "llama_server",
)
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

# Path to the external ``llama-server`` binary the served-backend engine launches.
# Set this on a node that should serve models via the ``llama_server`` engine
# (e.g. for native MTP). Absent => the node does not advertise ``llama_server``,
# so it is never a placement candidate for served-engine cards. The binary must
# be a build recent enough to expose ``--spec-type`` (>= b9196 for ``draft-mtp``).
LLAMA_SERVER_BIN_ENV: Final = "SKULK_LLAMA_SERVER_BIN"

# Compute backends the ``llama-server`` build was compiled with (comma-separated,
# e.g. "vulkan" or "vulkan,rocm"), same vocabulary as ``SKULK_LLAMA_CPP_BACKENDS``.
# When unset, the served engine falls back to the node's llama.cpp backend
# declaration (the GPU is the same regardless of which engine drives it), then to
# ``cpu``.
LLAMA_SERVER_BACKENDS_ENV: Final = "SKULK_LLAMA_SERVER_BACKENDS"

# Path to the ``ggml-rpc-server`` binary an RPC memory-donor runner launches
# (#328, multi-node GGUF pooling). Optional: when unset, the donor looks for
# ``ggml-rpc-server`` next to the node's ``SKULK_LLAMA_SERVER_BIN`` (the two are
# built together by ``cmake --build build --target ggml-rpc-server llama-server``
# with ``-DGGML_RPC=ON``; note the target was renamed upstream from
# ``rpc-server``).
RPC_SERVER_BIN_ENV: Final = "SKULK_RPC_SERVER_BIN"


def rpc_server_binary() -> str | None:
    """Resolve the ``ggml-rpc-server`` binary path for an RPC donor runner.

    Prefers the explicit ``SKULK_RPC_SERVER_BIN`` override; otherwise looks for
    a ``ggml-rpc-server`` sibling of ``SKULK_LLAMA_SERVER_BIN`` (they are built
    from the same llama.cpp tree). Returns ``None`` when neither yields an
    executable file, in which case the donor runner fails loudly at spawn.
    """
    explicit = os.environ.get(RPC_SERVER_BIN_ENV, "").strip()
    if explicit:
        return explicit if _is_executable_file(explicit) else None
    server = os.environ.get(LLAMA_SERVER_BIN_ENV, "").strip()
    if not server:
        return None
    sibling = str(Path(server).resolve().parent / "ggml-rpc-server")
    return sibling if _is_executable_file(sibling) else None


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


# Engines that can serve a model sharded across multiple nodes. MLX has the
# multi-node ring / jaccl path. The served ``llama_server`` engine pools memory
# across nodes via llama.cpp's RPC backend (#328): one driver node runs
# ``llama-server --rpc donor:port,...`` and each donor runs ``ggml-rpc-server``;
# llama.cpp splits weights/KV across the devices itself, so Skulk computes no
# GGUF layer math. The in-process ``llama_cpp`` engine stays single-node: the
# Python binding cannot drive the RPC backend, and its runner asserts
# ``world_size == 1``. This is the single place that capability lives.
_MULTI_NODE_ENGINES: Final[frozenset[EngineType]] = frozenset({"mlx", "llama_server"})


def engine_supports_multi_node(engine: EngineType) -> bool:
    """Whether an engine can serve a model sharded across more than one node.

    Placement uses this to pin a model to a single-node cycle when none of its
    compatible engines can shard across nodes (otherwise the placement would
    download and then crash at runner startup with ``world_size != 1``). MLX
    (ring/jaccl) and the served ``llama_server`` engine (RPC driver + donors,
    #328) are multi-node capable; the in-process ``llama_cpp`` engine is
    single-node (binding gap).
    """
    return engine in _MULTI_NODE_ENGINES


# Modalities each engine's RUNNER can currently exploit on THIS platform.
# This is PLATFORM truth, deliberately separate from the model card: a card's
# [vision] section declares what the MODEL can do (its projector artifact
# exists and is grounded); this table declares which of our runner
# implementations can actually serve it. The served ``llama_server`` engine is
# text-only until its runner stages and passes the mmproj projector (upstream
# llama-server supports --mmproj; the gap is ours). Keeping the limitation
# here rather than on cards means a platform capability landing lights up
# every affected card at once, with no card edits, and cards stay a clean
# description of the model.
_VISION_SERVING_ENGINES: Final[frozenset[EngineType]] = frozenset(
    {"mlx", "llama_cpp"}
)
_SPEECH_SERVING_ENGINES: Final[frozenset[EngineType]] = frozenset({"mlx_audio"})


def platform_compatible_backends(
    compatible_backends: frozenset[str],
    *,
    card_serves_vision: bool,
    card_serves_speech: bool = False,
) -> frozenset[str]:
    """Filter a card's declared backends down to what this platform can serve.

    The card's ``compatible_backends`` is MODEL truth (which engines the model's
    artifacts run on); this helper subtracts current PLATFORM limitations (which
    of our runners can exploit the card's declared capabilities) so the two are
    never conflated on the card itself. Today the platform gates are vision
    and speech: a card with a ``[vision]`` section is kept off engines whose
    runner cannot load its projector, and a TTS/STT card is kept off non-speech
    engines until the ``mlx_audio`` runner owns that contract. Placement and the
    worker's engine resolution both apply this filter, keeping master and
    worker in agreement.

    Args:
        compatible_backends: the card's declared backend tags.
        card_serves_vision: whether the card declares a vision capability.
        card_serves_speech: whether the card declares a speech capability.

    Returns:
        The subset of tags whose engine can serve everything the card declares.
    """
    filtered = compatible_backends
    if card_serves_vision:
        filtered = frozenset(
            tag
            for tag in filtered
            if (engine := engine_of(tag)) is None or engine in _VISION_SERVING_ENGINES
        )
    if card_serves_speech:
        filtered = frozenset(
            tag
            for tag in filtered
            if (engine := engine_of(tag)) is None or engine in _SPEECH_SERVING_ENGINES
        )
    return filtered


def resolve_node_backend(
    compatible_backends: frozenset[str],
    backend_preference: tuple[str, ...],
    node_backends: frozenset[str],
) -> str | None:
    """Resolve the winning backend TAG a node should use to serve a model.

    Intersects the card's ``compatible_backends`` (hard filter) with the node's
    advertised ``node_backends``, orders the result by ``backend_preference``
    (preferred tags first, then the rest deterministically), and returns the top
    tag (e.g. ``"llama_cpp-vulkan"``). Returns ``None`` when the node advertises
    none of the model's compatible backends. This is the single point that turns
    the backend-tag vocabulary into a concrete choice; both the master (to stamp
    ``resolved_backend`` on a shard at placement, #330) and the worker (to pick a
    runner) go through it so the two cannot disagree.
    """
    intersection = compatible_backends & node_backends
    if not intersection:
        return None
    ordered = [tag for tag in backend_preference if tag in intersection]
    # Fallback for tags outside the card's preference list: deterministic, but
    # never let a CPU compute tag beat a GPU tag on alphabetical accident --
    # GPU serving dominates CPU for every model class we ship, and a card
    # without an explicit llama_server preference would otherwise resolve
    # ``llama_server-cpu`` over ``llama_server-vulkan`` and run ``-ngl 0`` on a
    # GPU-admitted node. Platform default, not card policy.
    ordered += [
        tag
        for tag in sorted(
            intersection, key=lambda tag: (tag.endswith(f"{_TAG_SEPARATOR}cpu"), tag)
        )
        if tag not in backend_preference
    ]
    return ordered[0]


def resolve_node_engine(
    compatible_backends: frozenset[str],
    backend_preference: tuple[str, ...],
    node_backends: frozenset[str],
) -> EngineType | None:
    """Resolve which engine a node should use to serve a model.

    Thin wrapper over :func:`resolve_node_backend` that maps the winning tag to
    its engine. Returns ``None`` when the node advertises none of the model's
    compatible backends (which placement should already have ruled out, so the
    caller treats it as "fall back to the default engine").
    """
    tag = resolve_node_backend(compatible_backends, backend_preference, node_backends)
    return engine_of(tag) if tag is not None else None


def _llama_cpp_gpu_offload_supported() -> bool | None:
    """Whether the installed llama-cpp-python build has GPU offload compiled in.

    Returns the result of ``llama_cpp.llama_supports_gpu_offload()`` (a stable
    llama.cpp C API), or ``None`` when the binding cannot be introspected so the
    caller can treat it as "cannot verify" rather than "no GPU". A CPU-only wheel
    -- e.g. the PyPI prebuilt that ``uv sync`` reinstalls over a source-built
    Vulkan/ROCm wheel -- returns ``False`` here even on a GPU host, which is what
    lets a node notice its GPU build was clobbered and stop over-claiming.
    """
    try:
        import llama_cpp  # pyright: ignore[reportMissingImports]

        # llama_cpp ships no type stubs, so the member + its result are Unknown.
        return bool(llama_cpp.llama_supports_gpu_offload())  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    except Exception:  # noqa: BLE001 -- any binding/ABI quirk means "unverifiable"
        return None


def _probe_llama_cpp_backends() -> frozenset[str]:
    """Probe whether llama.cpp is usable here and which compute backends it offers.

    Returns the bare ``llama_cpp`` tag plus one compound tag per advertised
    compute backend. Compute backends come from ``SKULK_LLAMA_CPP_BACKENDS``
    when set (authoritative, since the compiled build decides); otherwise we fall
    back to the always-correct ``cpu`` tag so a node never over-claims GPU
    capability it might not have built. Returns an empty set when the binding is
    not importable, so a node without llama.cpp simply does not advertise it.

    Declared GPU backends are cross-checked against the actual build: if an
    operator declares e.g. ``vulkan`` but the installed wheel has no GPU offload
    compiled in (the classic failure where ``uv sync`` restores the CPU-only PyPI
    wheel over a source-built GPU wheel), the GPU tags are dropped and the node
    advertises only ``llama_cpp-cpu``. That keeps GPU GGUF work from being routed
    to a node that would silently run it on CPU or fail, until the build is
    rebuilt (see ``deployment/rocm``).
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

    gpu_computes = [cb for cb in computes if cb != "cpu"]
    if gpu_computes:
        supported = _llama_cpp_gpu_offload_supported()
        if supported is False:
            logger.warning(
                f"{LLAMA_CPP_BACKENDS_ENV} declares GPU backend(s) {gpu_computes} but the "
                "installed llama-cpp-python has no GPU offload compiled in (likely a "
                "CPU-only wheel that replaced a source-built GPU wheel, e.g. after "
                "`uv sync`). Advertising llama_cpp-cpu only so GPU GGUF work is not "
                "routed here; rebuild the GPU wheel (see deployment/rocm) to restore it."
            )
            computes = ["cpu"]
        elif supported is None:
            logger.warning(
                "could not verify llama.cpp GPU offload support; trusting "
                f"{LLAMA_CPP_BACKENDS_ENV}={gpu_computes}"
            )

    for compute in computes:
        tags.add(make_backend_tag("llama_cpp", compute))
    return frozenset(tags)


def _probe_served_backends() -> frozenset[str]:
    """Probe whether this node can serve models via the ``llama_server`` engine.

    Returns the bare ``llama_server`` tag plus one compound tag per advertised
    compute backend when ``SKULK_LLAMA_SERVER_BIN`` points at an existing
    executable; otherwise an empty set (the node does not advertise the served
    engine and is never a candidate for served-engine cards). Compute backends
    come from ``SKULK_LLAMA_SERVER_BACKENDS``, falling back to the node's
    ``SKULK_LLAMA_CPP_BACKENDS`` declaration (the GPU is the same whichever engine
    drives it), then to ``cpu``. ``metal`` is MLX-only and never valid here.
    """
    binary = os.environ.get(LLAMA_SERVER_BIN_ENV, "").strip()
    if not binary or not os.path.isfile(binary) or not os.access(binary, os.X_OK):
        return frozenset()

    declared = os.environ.get(LLAMA_SERVER_BACKENDS_ENV, "").strip() or os.environ.get(
        LLAMA_CPP_BACKENDS_ENV, ""
    )
    declared_tokens = {token for raw in declared.split(",") if (token := raw.strip())}
    computes: list[ComputeBackend] = [
        cb for cb in _COMPUTE_BACKENDS if cb != "metal" and cb in declared_tokens
    ]
    if not computes:
        computes = ["cpu"]

    tags: set[str] = {"llama_server"}
    for compute in computes:
        tags.add(make_backend_tag("llama_server", compute))
    return frozenset(tags)


def _probe_mlx_audio_backends() -> frozenset[str]:
    """Probe whether this macOS node can serve speech models via ``mlx_audio``.

    Returns the bare ``mlx_audio`` tag plus ``mlx_audio-metal`` only when the
    upstream ``mlx_audio`` package imports successfully on Darwin. Other
    platforms advertise nothing until a non-Metal speech engine is implemented.
    """
    if sys.platform != "darwin":
        return frozenset()
    try:
        import mlx_audio  # noqa: F401  # pyright: ignore[reportMissingTypeStubs, reportUnusedImport]
    except Exception:  # noqa: BLE001
        # Optional native wheels can fail during extension loading with OSError
        # or ABI errors; backend probing must degrade to "unavailable".
        return frozenset()
    return frozenset({"mlx_audio", make_backend_tag("mlx_audio", "metal")})


def probe_node_backends() -> frozenset[str]:
    """Probe the backend tags this node can actually serve.

    macOS nodes (``sys.platform == "darwin"``) advertise ``{"mlx", "mlx-metal"}`` (the bare engine tag
    is kept for backward compatibility with cards written against the original
    ``{"mlx"}`` vocabulary). Any node with an importable ``llama_cpp`` adds its
    llama.cpp tags; a node with ``SKULK_LLAMA_SERVER_BIN`` set adds its
    ``llama_server`` tags. A macOS node with importable ``mlx_audio`` adds
    ``{"mlx_audio", "mlx_audio-metal"}`` for single-node speech models. A bare
    Linux node with none advertises an empty set and is therefore not a placement
    candidate, which is the pre-existing behavior.
    """
    tags: set[str] = set()
    if sys.platform == "darwin":
        tags |= {"mlx", make_backend_tag("mlx", "metal")}
    tags |= _probe_llama_cpp_backends()
    tags |= _probe_served_backends()
    tags |= _probe_mlx_audio_backends()
    return frozenset(tags)
