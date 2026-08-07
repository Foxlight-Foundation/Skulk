"""Node Facts: the typed record of what a node observed about itself.

This is the keystone of the "detection creates capability" inversion (#614).
One probe pass produces a single :class:`NodeFacts` record with three strictly
separated ingredients:

- **Hardware observed**: every GPU the node can see (all of them, even while
  multi-GPU serving is deferred), including how it was detected.
- **Software observed**: importable dependencies and configured engine
  binaries, including whether a configured binary is actually usable.
- **Configuration declared**: the raw serving-relevant ``SKULK_*`` overrides,
  verbatim, so derivation can compare what the operator *said* against what
  the node *saw*.

Backend derivation (``skulk.facts.derive``) consumes this record and produces
the advertised backend tags plus :class:`CapabilityConflict` entries wherever
observation and declaration disagree, or where detection alone would leave the
node degraded. Conflicts ride ``NodeResources`` over the telemetry plane into
``nodeHealth`` so a misconfigured node is loud on every surface instead of
silently serving degraded (#609, #612, #462).

These are pure data types: no probing happens here.
"""

from __future__ import annotations

from typing import Final, Literal, final

from pydantic import ConfigDict

from skulk.utils.pydantic_ext import CamelCaseModel

GpuVendor = Literal["nvidia", "amd", "apple"]
"""Vendor of an observed GPU device."""

GpuDetectionSource = Literal[
    "nvml",
    "nvidia_device_node",
    "amdgpu_sysfs",
    "apple_platform",
]
"""How a GPU was observed.

``nvml`` is the full NVIDIA path (name, VRAM, compute capability all known).
``nvidia_device_node`` means an NVIDIA device is visibly present (driver proc
node, ``/dev/nvidia*``, or ``nvidia-smi`` on PATH) but NVML is not importable,
so everything beyond presence is unknown -- the #612 degraded state.
``amdgpu_sysfs`` is the passive AMD path; ``apple_platform`` is implied by
running on Darwin (Apple Silicon unified memory).
"""


@final
class GpuDeviceFact(CamelCaseModel):
    """One GPU device the node observed, with how much it could learn about it."""

    model_config = ConfigDict(frozen=True)

    vendor: GpuVendor
    """Vendor of the device."""

    name: str = "Unknown"
    """Marketing/device name when the detection source exposes it."""

    index: int = 0
    """Device index within its vendor's enumeration (NVML index, drm card order)."""

    detection_source: GpuDetectionSource
    """Which observation path found this device (see :data:`GpuDetectionSource`)."""

    vram_total_bytes: int | None = None
    """Total discrete VRAM in bytes, or ``None`` when the source cannot read it."""

    gtt_total_bytes: int | None = None
    """GPU-mappable host memory (GTT) for unified-memory APUs; ``None`` elsewhere."""

    compute_capability: str | None = None
    """NVIDIA SM level as ``"<major>.<minor>"``; ``None`` for other vendors."""


EngineBinaryState = Literal["not_configured", "ok", "missing", "not_executable"]
"""Usability of a configured engine binary path.

``not_configured`` means the env var is unset (the engine is simply absent,
which is not a conflict). ``missing`` / ``not_executable`` mean the operator
declared a path that cannot be used -- explicit intent that silently degrades
today (#462), which derivation turns into a loud conflict.
"""


@final
class EngineBinaryFact(CamelCaseModel):
    """State of one operator-configured engine binary (llama-server, vllm, rpc)."""

    model_config = ConfigDict(frozen=True)

    env_var: str
    """The environment variable that configures this binary."""

    configured_path: str | None = None
    """The declared path, verbatim, or ``None`` when unset."""

    state: EngineBinaryState = "not_configured"
    """Whether the declared path resolves to an executable file."""


ListDevicesOutcome = Literal["devices", "unsupported", "failed", "not_run"]
"""Outcome of probing a llama-server binary with ``--list-devices``.

``devices`` means the binary answered and its device list was parsed (possibly
empty for a CPU-only build). ``unsupported`` means the build predates the flag.
``failed`` covers crashes and timeouts. ``not_run`` means there was no usable
binary to probe or a declaration made the probe unnecessary.
"""


@final
class LlamaServerDeviceProbe(CamelCaseModel):
    """Result of asking the configured llama-server binary what it can drive.

    ``--list-devices`` is ground truth for which compute backends the *build*
    supports on *this* machine -- strictly better evidence than either the
    operator's declaration or hardware vendor inference, so derivation prefers
    it when no declaration overrides.
    """

    model_config = ConfigDict(frozen=True)

    outcome: ListDevicesOutcome = "not_run"
    """How the probe concluded (see :data:`ListDevicesOutcome`)."""

    computes: tuple[str, ...] = ()
    """Compute backends the binary reported devices for (``cuda``/``vulkan``/
    ``rocm``), deduplicated, in first-seen order. Empty with outcome
    ``devices`` means a CPU-only build."""

    detail: str = ""
    """Short diagnostic detail for ``failed`` outcomes (bounded, single line)."""


CapabilityConflictCode = Literal[
    "gpu_serving_disabled",
    "gpu_detection_degraded",
    "invalid_engine_binary",
    "backend_override_conflict",
]
"""Machine codes for observation-vs-declaration conflicts.

- ``gpu_serving_disabled``: a serving-capable GPU is visible but a configured
  engine would run CPU-only (#609's silent ``-ngl 0`` class). Severity: error.
- ``gpu_detection_degraded``: a GPU is visibly present but the node cannot
  fully detect it (e.g. NVIDIA without ``nvidia-ml-py``), so VRAM-derived
  behavior like the served-context lift silently degrades (#612). Severity: warn.
- ``invalid_engine_binary``: an engine binary override is set but unusable
  (#462). Severity: warn.
- ``backend_override_conflict``: a declared backend claims hardware the node
  cannot observe (e.g. ``cuda`` declared, no NVIDIA device). The declaration is
  still honored -- configuration overrides detection -- but the disagreement is
  loud. Severity: warn.
"""

CONFLICT_ERROR_CODES: Final[frozenset[CapabilityConflictCode]] = frozenset(
    {"gpu_serving_disabled"}
)
"""Conflict codes that surface at ``error`` severity in node health; all other
codes surface at ``warn``. Single source of truth shared by derivation logging
and ``node_health`` aggregation so the two can never disagree."""


@final
class CapabilityConflict(CamelCaseModel):
    """One loud disagreement between what a node observed and how it will serve.

    Produced by backend derivation, advertised on ``NodeResources``, and mapped
    one-to-one onto ``nodeHealth`` reasons. ``message`` and ``remediation`` are
    composed on the node that owns the facts (it knows its own paths and env),
    so consumers only render them.
    """

    model_config = ConfigDict(frozen=True)

    code: CapabilityConflictCode
    """Stable machine code for the conflict class."""

    message: str
    """What is wrong, with the concrete observed/declared values."""

    remediation: str
    """The operator's path to resolve it."""


@final
class NodeFacts(CamelCaseModel):
    """Everything one probe pass learned about this node.

    Observations and declarations only -- no derived capability. Derivation
    (``skulk.facts.derive``) is a pure function over this record, which is what
    makes the whole capability pipeline testable without hardware.
    """

    model_config = ConfigDict(frozen=True)

    platform: str
    """``sys.platform`` at probe time (``darwin``, ``linux``, ...)."""

    gpus: tuple[GpuDeviceFact, ...] = ()
    """Every GPU observed, all vendors, all devices (multi-GPU serving is
    deferred, but detection reports the full inventory from day one)."""

    pynvml_importable: bool = False
    """Whether ``pynvml`` (nvidia-ml-py) imports; gates full NVIDIA detection."""

    llama_cpp_importable: bool = False
    """Whether the in-process ``llama_cpp`` binding imports."""

    llama_cpp_gpu_offload: bool | None = None
    """``llama_cpp.llama_supports_gpu_offload()`` result; ``None`` when the
    binding is absent or cannot be introspected."""

    mlx_audio_importable: bool = False
    """Whether the upstream ``mlx_audio`` package imports (Darwin speech)."""

    llama_server_binary: EngineBinaryFact
    """State of the ``SKULK_LLAMA_SERVER_BIN`` declaration."""

    vllm_binary: EngineBinaryFact
    """State of the ``SKULK_VLLM_BIN`` declaration."""

    rpc_server_binary: EngineBinaryFact
    """State of the ``SKULK_RPC_SERVER_BIN`` declaration (explicit override
    only; the sibling fallback is resolved at donor spawn)."""

    llama_server_device_probe: LlamaServerDeviceProbe = LlamaServerDeviceProbe()
    """What the llama-server binary itself reported via ``--list-devices``."""

    declared_llama_cpp_backends: str | None = None
    """Raw ``SKULK_LLAMA_CPP_BACKENDS`` value, verbatim, or ``None`` when unset."""

    declared_llama_server_backends: str | None = None
    """Raw ``SKULK_LLAMA_SERVER_BACKENDS`` value, verbatim, or ``None`` when unset."""

    declared_vllm_backends: str | None = None
    """Raw ``SKULK_VLLM_BACKENDS`` value, verbatim, or ``None`` when unset."""

    def gpus_of(self, vendor: GpuVendor) -> tuple[GpuDeviceFact, ...]:
        """Return the observed GPUs of one vendor, preserving device order."""
        return tuple(gpu for gpu in self.gpus if gpu.vendor == vendor)

    @property
    def has_serving_gpu(self) -> bool:
        """Whether a discrete-serving-class GPU (NVIDIA/AMD) is visible.

        Apple GPUs are excluded deliberately: on Darwin the MLX engine owns the
        GPU, so a CPU compute tag on a non-MLX engine is not a degradation
        signal there.
        """
        return any(gpu.vendor in ("nvidia", "amd") for gpu in self.gpus)
