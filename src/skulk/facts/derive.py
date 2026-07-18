"""Derivation side of Node Facts: observed facts -> advertised backends.

This is the principle inversion (#614): detection creates capability,
configuration overrides it, and disagreement between the two is always loud.
:func:`derive_node_backends` is a pure function over one
:class:`~skulk.shared.types.node_facts.NodeFacts` record, returning the backend
tags this node should advertise plus a :class:`CapabilityConflict` for every
place where observation and declaration disagree or where the derived result
leaves visible hardware unused. Purity is the point: the entire capability
pipeline is exercised in tests with synthetic facts, no hardware required.

Precedence per engine, most to least authoritative:

1. **Operator declaration** (``SKULK_*_BACKENDS``): honored even when it
   claims hardware the node cannot see (the conflict is loud, the override
   wins), with one exception -- a declared GPU backend for the in-process
   ``llama_cpp`` binding whose build positively reports no GPU offload is
   dropped, because honoring it would silently run GPU-placed work on CPU.
2. **The engine's own report** (``llama-server --list-devices``): ground truth
   for what the build can drive on this machine.
3. **Hardware vendor inference**: an observed NVIDIA device implies ``cuda``,
   an observed AMD device implies ``vulkan`` (llama.cpp's fleet-proven RADV
   path) for the served engine and ``rocm`` for vLLM.
4. **CPU floor**: only when nothing above yields a GPU backend -- and if a
   serving-capable GPU is visible at that point, the CPU floor is a loud
   ``gpu_serving_disabled`` conflict, never a silent default (#609).
"""

from __future__ import annotations

from typing import final

from pydantic import ConfigDict

from skulk.shared.types.node_facts import (
    CapabilityConflict,
    NodeFacts,
)
from skulk.utils.pydantic_ext import CamelCaseModel

# The compute vocabulary is owned by skulk.shared.backends; the names are
# duplicated here as plain strings to keep the module graph acyclic
# (backends.probe_node_backends delegates into this package). The vocabulary
# test suite pins the two in agreement.
_GPU_COMPUTES = ("vulkan", "rocm", "cuda")
_SERVED_COMPUTES = ("vulkan", "rocm", "cuda", "cpu")
_VLLM_COMPUTES = ("cuda", "rocm")

_INSTALL_DOCS_HINT = (
    "see website/docs (GPU node setup) or run `skulk doctor` for a full audit"
)


@final
class BackendDerivation(CamelCaseModel):
    """The advertised backends derived from one facts snapshot, plus fallout."""

    model_config = ConfigDict(frozen=True)

    backends: frozenset[str]
    """Backend tags this node should advertise in ``NodeResources.backends``."""

    conflicts: tuple[CapabilityConflict, ...] = ()
    """Loud observation-vs-declaration disagreements (ride node telemetry into
    ``nodeHealth``)."""

    notes: tuple[str, ...] = ()
    """Informational derivation notes worth one warning-level log line each
    (e.g. an unverifiable declaration that was trusted)."""


def _declared_tokens(raw: str | None) -> set[str]:
    """Split a comma-separated backends declaration into its raw tokens."""
    if raw is None:
        return set()
    return {token for piece in raw.split(",") if (token := piece.strip())}


def _hardware_supports(compute: str, facts: NodeFacts) -> bool:
    """Whether observed hardware can plausibly drive one GPU compute backend.

    ``cuda`` needs an NVIDIA device, ``rocm`` an AMD device, and ``vulkan``
    either (llama.cpp's Vulkan backend drives both vendors). Used only to
    detect declaration conflicts; the declaration is honored regardless.
    """
    nvidia = bool(facts.gpus_of("nvidia"))
    amd = bool(facts.gpus_of("amd"))
    if compute == "cuda":
        return nvidia
    if compute == "rocm":
        return amd
    if compute == "vulkan":
        return nvidia or amd
    return True


def _override_conflicts(
    computes: list[str], facts: NodeFacts, *, env_var: str, engine: str
) -> list[CapabilityConflict]:
    """Conflicts for declared GPU computes that no observed hardware supports."""
    unsupported = [
        compute
        for compute in computes
        if compute in _GPU_COMPUTES and not _hardware_supports(compute, facts)
    ]
    if not unsupported:
        return []
    return [
        CapabilityConflict(
            code="backend_override_conflict",
            message=(
                f"{env_var} declares {sorted(unsupported)} for {engine}, but this "
                "node observes no GPU that backend can drive. The declaration is "
                "honored, but serving will fail at engine startup if the "
                "hardware is truly absent."
            ),
            remediation=(
                f"Fix or remove {env_var} so it matches the node's hardware, "
                f"then restart skulk; {_INSTALL_DOCS_HINT}."
            ),
        )
    ]


def _derive_llama_cpp(
    facts: NodeFacts,
) -> tuple[set[str], list[CapabilityConflict], list[str]]:
    """Derive in-process ``llama_cpp`` tags from binding + declaration facts."""
    if not facts.llama_cpp_importable:
        return set(), [], []

    conflicts: list[CapabilityConflict] = []
    notes: list[str] = []
    declared = _declared_tokens(facts.declared_llama_cpp_backends)
    # ``metal`` is deliberately absent from _SERVED_COMPUTES: it is MLX-only,
    # never a valid llama.cpp compute backend, even if an operator declares it.
    computes = [cb for cb in _SERVED_COMPUTES if cb in declared]

    if computes:
        gpu_computes = [cb for cb in computes if cb != "cpu"]
        if gpu_computes:
            conflicts.extend(
                _override_conflicts(
                    computes,
                    facts,
                    env_var="SKULK_LLAMA_CPP_BACKENDS",
                    engine="llama_cpp",
                )
            )
            if facts.llama_cpp_gpu_offload is False:
                conflicts.append(
                    CapabilityConflict(
                        code="backend_override_conflict",
                        message=(
                            f"SKULK_LLAMA_CPP_BACKENDS declares GPU backend(s) "
                            f"{sorted(gpu_computes)} but the installed "
                            "llama-cpp-python has no GPU offload compiled in "
                            "(likely a CPU-only wheel that replaced a "
                            "source-built GPU wheel, e.g. after `uv sync`). "
                            "Advertising llama_cpp-cpu only so GPU GGUF work is "
                            "not routed here."
                        ),
                        remediation=(
                            "Rebuild the GPU llama-cpp-python wheel (see "
                            "deployment/rocm) and restart skulk."
                        ),
                    )
                )
                computes = ["cpu"]
            elif facts.llama_cpp_gpu_offload is None:
                notes.append(
                    "could not verify llama.cpp GPU offload support; trusting "
                    f"SKULK_LLAMA_CPP_BACKENDS={sorted(gpu_computes)}"
                )
    elif (
        facts.llama_cpp_gpu_offload is True
        and facts.platform != "darwin"
        and facts.has_serving_gpu
    ):
        # No declaration, but the build positively reports GPU offload and a
        # GPU is visible: derive. The binding runs whatever backend it was
        # compiled with regardless of the tag we advertise, so tag choice only
        # steers placement preference matching -- an NVIDIA device implies a
        # CUDA build; an AMD device could be a Vulkan or ROCm build, so
        # advertise both and let card preferences pick.
        computes = list[str]()
        if facts.gpus_of("nvidia"):
            computes.append("cuda")
        if facts.gpus_of("amd"):
            computes.extend(("vulkan", "rocm"))
        notes.append(
            "llama_cpp binding reports GPU offload with no "
            f"SKULK_LLAMA_CPP_BACKENDS declared; derived {sorted(computes)} "
            "from observed hardware"
        )
    else:
        # No declaration and no positive GPU evidence: the CPU floor, which
        # any llama.cpp build can serve.
        computes = ["cpu"]

    tags = {"llama_cpp"} | {f"llama_cpp-{compute}" for compute in computes}
    return tags, conflicts, notes


def _derive_llama_server(
    facts: NodeFacts,
) -> tuple[set[str], list[CapabilityConflict], list[str]]:
    """Derive served ``llama_server`` tags: declaration > binary probe > vendor."""
    binary = facts.llama_server_binary
    if binary.state == "not_configured":
        return set(), [], []
    if binary.state in ("missing", "not_executable"):
        return (
            set(),
            [
                CapabilityConflict(
                    code="invalid_engine_binary",
                    message=(
                        f"{binary.env_var} is set to {binary.configured_path!r} "
                        f"but that path is {binary.state.replace('_', ' ')}; the "
                        "llama_server engine is disabled on this node."
                    ),
                    remediation=(
                        f"Point {binary.env_var} at an executable llama-server "
                        "binary (or unset it) and restart skulk."
                    ),
                )
            ],
            [],
        )

    conflicts: list[CapabilityConflict] = []
    notes: list[str] = []
    declared = _declared_tokens(
        facts.declared_llama_server_backends
    ) or _declared_tokens(facts.declared_llama_cpp_backends)
    computes = [cb for cb in _SERVED_COMPUTES if cb in declared]

    if computes:
        conflicts.extend(
            _override_conflicts(
                computes,
                facts,
                env_var="SKULK_LLAMA_SERVER_BACKENDS",
                engine="llama_server",
            )
        )
    else:
        probe = facts.llama_server_device_probe
        if probe.outcome == "devices":
            computes = [cb for cb in _GPU_COMPUTES if cb in probe.computes]
            if computes:
                notes.append(
                    f"llama-server --list-devices reported {sorted(computes)}; "
                    "derived served backends from the binary's own device list"
                )
        else:
            if facts.gpus_of("nvidia"):
                computes.append("cuda")
            if facts.gpus_of("amd"):
                computes.append("vulkan")
            if computes:
                notes.append(
                    "derived served backend(s) "
                    f"{sorted(computes)} from observed GPU hardware "
                    f"(--list-devices probe: {probe.outcome})"
                )

    if not computes:
        computes = ["cpu"]
    if facts.has_serving_gpu and all(compute == "cpu" for compute in computes):
        # The #609 class, made loud: a GPU is visible but the served engine
        # would launch with -ngl 0 and crawl at CPU speed.
        probe = facts.llama_server_device_probe
        cause = (
            "the configured llama-server binary reports no GPU devices "
            "(a CPU-only build)"
            if probe.outcome == "devices" and not probe.computes
            else "the resolved served backend is cpu"
        )
        conflicts.append(
            CapabilityConflict(
                code="gpu_serving_disabled",
                message=(
                    f"A GPU is visible on this node but {cause}; llama_server "
                    "would serve on CPU at a fraction of hardware speed."
                ),
                remediation=(
                    "Use a GPU-enabled llama-server build (or set "
                    "SKULK_LLAMA_SERVER_BACKENDS to the build's GPU backend) "
                    f"and restart skulk; {_INSTALL_DOCS_HINT}."
                ),
            )
        )

    tags = {"llama_server"} | {f"llama_server-{compute}" for compute in computes}
    return tags, conflicts, notes


def _derive_vllm(
    facts: NodeFacts,
) -> tuple[set[str], list[CapabilityConflict], list[str]]:
    """Derive ``vllm`` tags: declaration chain > vendor inference; GPU-only."""
    binary = facts.vllm_binary
    if binary.state == "not_configured":
        return set(), [], []
    if binary.state in ("missing", "not_executable"):
        return (
            set(),
            [
                CapabilityConflict(
                    code="invalid_engine_binary",
                    message=(
                        f"{binary.env_var} is set to {binary.configured_path!r} "
                        f"but that path is {binary.state.replace('_', ' ')}; the "
                        "vllm engine is disabled on this node."
                    ),
                    remediation=(
                        f"Point {binary.env_var} at an executable vllm CLI (or "
                        "unset it) and restart skulk."
                    ),
                )
            ],
            [],
        )

    conflicts: list[CapabilityConflict] = []
    notes: list[str] = []
    declared = (
        _declared_tokens(facts.declared_vllm_backends)
        or _declared_tokens(facts.declared_llama_server_backends)
        or _declared_tokens(facts.declared_llama_cpp_backends)
    )
    computes = [cb for cb in _VLLM_COMPUTES if cb in declared]

    if computes:
        conflicts.extend(
            _override_conflicts(
                computes, facts, env_var="SKULK_VLLM_BACKENDS", engine="vllm"
            )
        )
    elif not declared:
        # No declaration anywhere in the chain: derive from hardware. vLLM is
        # GPU-only, so an NVIDIA device implies cuda and an AMD device rocm.
        if facts.gpus_of("nvidia"):
            computes.append("cuda")
        if facts.gpus_of("amd"):
            computes.append("rocm")
        if computes:
            notes.append(
                f"derived vllm backend(s) {sorted(computes)} from observed "
                "GPU hardware (no backends declared)"
            )

    if not computes:
        conflicts.append(
            CapabilityConflict(
                code="backend_override_conflict",
                message=(
                    f"{binary.env_var} configures the GPU-only vllm engine but "
                    "no usable GPU backend was declared or observed; vllm is "
                    "disabled on this node."
                ),
                remediation=(
                    "Set SKULK_VLLM_BACKENDS to cuda or rocm on a GPU node, or "
                    f"unset {binary.env_var}, then restart skulk."
                ),
            )
        )
        return set(), conflicts, notes

    tags = {"vllm"} | {f"vllm-{compute}" for compute in computes}
    return tags, conflicts, notes


def derive_node_backends(facts: NodeFacts) -> BackendDerivation:
    """Derive the backend tags a node advertises, plus every loud conflict.

    Pure function over one :class:`NodeFacts` record; see the module docstring
    for the per-engine precedence. The returned conflicts are self-contained
    (message + remediation composed against this node's own paths and env) so
    telemetry consumers only render them.

    Args:
        facts: The observed-and-declared record for this node.

    Returns:
        The advertised tags, conflicts, and informational derivation notes.
    """
    tags: set[str] = set()
    conflicts: list[CapabilityConflict] = []
    notes: list[str] = []

    if facts.platform == "darwin":
        tags |= {"mlx", "mlx-metal"}
        if facts.mlx_audio_importable:
            tags |= {"mlx_audio", "mlx_audio-metal"}

    for derive in (_derive_llama_cpp, _derive_llama_server, _derive_vllm):
        engine_tags, engine_conflicts, engine_notes = derive(facts)
        tags |= engine_tags
        conflicts.extend(engine_conflicts)
        notes.extend(engine_notes)

    if any(gpu.detection_source == "nvidia_device_node" for gpu in facts.gpus):
        # The #612 class: hardware is visibly present but the node cannot read
        # its name or VRAM, so served-context sizing and hardware attribution
        # silently degrade. The cause differs by whether the binding imports:
        # absent binding means an install gap; importable binding with no NVML
        # devices means the driver/library handshake failed.
        if facts.pynvml_importable:
            message = (
                "An NVIDIA GPU is present and nvidia-ml-py is installed, but "
                "NVML failed to enumerate it (likely a driver/library version "
                "mismatch); VRAM detection, served-context sizing, and "
                "hardware attribution are degraded."
            )
            remediation = (
                "Check `nvidia-smi` works and the NVIDIA driver matches the "
                "installed CUDA/NVML libraries, then restart skulk."
            )
        else:
            message = (
                "An NVIDIA GPU is present but nvidia-ml-py (pynvml) is not "
                "importable, so VRAM detection, served-context sizing, and "
                "hardware attribution are degraded."
            )
            remediation = (
                "Install nvidia-ml-py into skulk's environment "
                "(`uv pip install nvidia-ml-py`) and restart skulk."
            )
        conflicts.append(
            CapabilityConflict(
                code="gpu_detection_degraded",
                message=message,
                remediation=remediation,
            )
        )

    rpc = facts.rpc_server_binary
    if rpc.state in ("missing", "not_executable"):
        # #462: an explicit RPC override that cannot be used previously read
        # like the env var was never set when a donor spawn failed.
        conflicts.append(
            CapabilityConflict(
                code="invalid_engine_binary",
                message=(
                    f"{rpc.env_var} is set to {rpc.configured_path!r} but that "
                    f"path is {rpc.state.replace('_', ' ')}; RPC donor spawns "
                    "on this node will fail."
                ),
                remediation=(
                    f"Point {rpc.env_var} at an executable ggml-rpc-server "
                    "binary (or unset it to use the llama-server sibling) and "
                    "restart skulk."
                ),
            )
        )

    # A GPU node whose entire advertised set has no GPU compute tag is serving
    # everything on CPU: loud, unless an engine-specific conflict above already
    # named the cause (the served-engine #609 case emits its own).
    has_gpu_tag = any(
        tag.endswith(("-cuda", "-vulkan", "-rocm", "-metal")) for tag in tags
    )
    already_flagged = any(c.code == "gpu_serving_disabled" for c in conflicts)
    if tags and facts.has_serving_gpu and not has_gpu_tag and not already_flagged:
        conflicts.append(
            CapabilityConflict(
                code="gpu_serving_disabled",
                message=(
                    "A GPU is visible on this node but no inference engine "
                    "resolved a GPU backend; all serving would run on CPU at a "
                    "fraction of hardware speed."
                ),
                remediation=(
                    "Install or configure a GPU-enabled engine (GPU "
                    "llama-cpp-python build, SKULK_LLAMA_SERVER_BIN with a GPU "
                    f"llama-server build, or SKULK_VLLM_BIN); {_INSTALL_DOCS_HINT}."
                ),
            )
        )

    return BackendDerivation(
        backends=frozenset(tags),
        conflicts=tuple(conflicts),
        notes=tuple(notes),
    )
