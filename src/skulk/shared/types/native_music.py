"""Closed native ACE process contracts, independent of model management and UI.

Only the worker creates these private paths and explicit device policies. Public
music requests never deserialize a native process configuration. This module and
the bounded WAV contract are copied into the separate Python 3.12 runtime.
"""

from __future__ import annotations

import re
from pathlib import PurePath, PurePosixPath
from typing import Annotated, Literal, Self, final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .audio_separation import SeparationSource

NativeMusicOperation = Literal["generate", "add_part", "repaint", "continue"]
NativeMusicDevice = Literal["cpu", "mps", "cuda", "rocm"]
NativeMusicPrecision = Literal["float32", "bfloat16", "float16"]
NativeMusicTensorPrecision = Literal["float32", "bfloat16", "float16"]
NativeMusicPart = Literal[
    "woodwinds",
    "brass",
    "fx",
    "synth",
    "strings",
    "percussion",
    "keyboard",
    "guitar",
    "bass",
    "drums",
    "backing_vocals",
    "vocals",
]
NATIVE_MUSIC_SAMPLE_RATE = 48_000
NATIVE_MUSIC_MAX_FRAMES = 120 * NATIVE_MUSIC_SAMPLE_RATE
NATIVE_MUSIC_LATENT_HOP_FRAMES = 1920
NATIVE_MUSIC_MIN_DECODE_FRAMES = 128 * NATIVE_MUSIC_LATENT_HOP_FRAMES
"""Pinned ACE minimum of 128 latent frames, each decoding to 1,920 samples."""
MAX_NATIVE_MUSIC_RECORD_BYTES = 256 * 1024


def native_music_conditioning_frames(target_frames: int) -> int:
    """Round an admitted target up to the pinned codec's 1,920-sample grid.

    The caller supplies a validated native frame count. Padding belongs after
    the requested timeline and must not change source samples or edit ranges.
    """
    return (
        (target_frames + NATIVE_MUSIC_LATENT_HOP_FRAMES - 1)
        // NATIVE_MUSIC_LATENT_HOP_FRAMES
        * NATIVE_MUSIC_LATENT_HOP_FRAMES
    )


class NativeMusicContract(BaseModel):
    """Immutable, strict, finite metadata shared by Core and the owned process."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


@final
class NativeMusicMemoryCapacity(NativeMusicContract):
    """Current measured capacity for host and the actual selected native device.

    The collector must supply device zero from the native runtime's own visible
    ordering. A maximum or aggregate across accelerators is not this contract.
    Shared capacity is a separate physical pool observation, not host plus
    device availability; omit it if unified memory has not been established.
    """

    sampled_at_unix_ms: int = Field(
        default=0,
        ge=0,
        le=2**53 - 1,
        description="Actual collection time; zero is an unsampled contract and cannot authorize production placement.",
    )
    host_total_bytes: int = Field(
        gt=0, description="Physical usable host-memory ceiling."
    )
    host_available_bytes: int = Field(
        ge=0,
        description="Current host availability, already net of resident allocations.",
    )
    device_index: Literal[0] | None = Field(
        default=None,
        description="Actual runtime-visible selected device; never an aggregate.",
    )
    device_memory_kind: Literal["discrete", "unified"] | None = Field(
        default=None,
        description="Established physical-memory topology; unknown cannot authorize GPU admission.",
    )
    device_total_bytes: int | None = Field(
        default=None, gt=0, description="Usable ceiling for that one selected device."
    )
    device_available_bytes: int | None = Field(
        default=None,
        ge=0,
        description="Current availability for that one selected device.",
    )
    shared_total_bytes: int | None = Field(
        default=None,
        gt=0,
        description="Established unified physical-memory ceiling, counted once.",
    )
    shared_available_bytes: int | None = Field(
        default=None,
        ge=0,
        description="Current combined physical availability, without summing shared host/device pages.",
    )

    @model_validator(mode="after")
    def validate_capacity(self) -> Self:
        """Keep complete observed domains, selected ordinal and physical ceilings bound."""
        if self.host_available_bytes > self.host_total_bytes:
            raise ValueError("native music host availability exceeds its ceiling")
        if (self.device_total_bytes is None) != (self.device_available_bytes is None):
            raise ValueError(
                "native music device capacity requires both measured values"
            )
        if (self.device_index is None) != (self.device_total_bytes is None):
            raise ValueError(
                "native music device capacity requires the actual selected ordinal"
            )
        if self.device_memory_kind is not None and self.device_index is None:
            raise ValueError("native music memory topology requires its actual device")
        if (
            self.device_available_bytes is not None
            and self.device_total_bytes is not None
            and self.device_available_bytes > self.device_total_bytes
        ):
            raise ValueError("native music device availability exceeds its ceiling")
        if (self.shared_total_bytes is None) != (self.shared_available_bytes is None):
            raise ValueError(
                "native music shared capacity requires both measured values"
            )
        if (self.device_memory_kind == "unified") != (
            self.shared_total_bytes is not None
        ):
            raise ValueError(
                "native music unified topology requires its measured physical pool"
            )
        if (
            self.shared_available_bytes is not None
            and self.shared_total_bytes is not None
            and (
                self.device_index is None
                or self.shared_available_bytes > self.shared_total_bytes
            )
        ):
            raise ValueError(
                "native music shared capacity requires its actual device and physical ceiling"
            )
        return self


@final
class NativeMusicMemoryBudget(NativeMusicContract):
    """The admitted phase peaks carried into the separately supervised process.

    Admission owns complete simultaneous reservations. The live guard enforces
    process allocation ceilings and actual physical-memory pressure; it does
    not pretend that RSS plus GPU allocations measure unique shared pages.
    """

    policy_version: Literal[1] = Field(
        default=1, description="Admitted memory-policy version."
    )
    host_peak_bytes: int = Field(
        ge=0, le=2**53 - 1, description="Largest admitted host phase."
    )
    device_peak_bytes: int = Field(
        ge=0,
        le=2**53 - 1,
        description="Largest admitted selected-device phase; zero for CPU.",
    )
    shared_peak_bytes: int = Field(
        ge=0,
        le=2**53 - 1,
        description="Largest simultaneous host/device physical phase.",
    )

    @model_validator(mode="after")
    def validate_phase_bounds(self) -> Self:
        """Keep a simultaneous phase within the possible independent peak bounds."""
        if (
            not max(self.host_peak_bytes, self.device_peak_bytes)
            <= self.shared_peak_bytes
            <= self.host_peak_bytes + self.device_peak_bytes
        ):
            raise ValueError("native music shared budget contradicts its phase peaks")
        return self


@final
class NativeMusicMemoryObservation(NativeMusicContract):
    """Bounded live facts from the owned native process, never State or event data."""

    process_id: int = Field(
        gt=1,
        le=2**31 - 1,
        description="Actual native leader collecting the observation.",
    )
    payload_sha256: str = Field(
        pattern=r"^[a-f0-9]{64}$",
        description="Independently verified native payload identity.",
    )
    device: NativeMusicDevice = Field(
        description="Actual explicitly selected native lane."
    )
    capacity: NativeMusicMemoryCapacity = Field(
        description="Fresh host and selected-device physical availability."
    )
    host_resident_bytes: int = Field(
        ge=0,
        le=2**53 - 1,
        description="Actual process RSS; not an exclusive-allocation credit.",
    )
    device_allocated_bytes: int = Field(
        ge=0,
        le=2**53 - 1,
        description="Actual Torch reserved device allocation, or MPS driver allocation; zero on CPU.",
    )

    @model_validator(mode="after")
    def validate_selected_device(self) -> Self:
        """Refuse a missing/foreign device and never let CPU borrow a GPU pool."""
        if self.device == "cpu":
            if self.capacity.device_index is not None or self.device_allocated_bytes:
                raise ValueError(
                    "native CPU observation cannot contain a GPU allocation"
                )
        elif (
            self.capacity.device_index != 0 or self.capacity.device_memory_kind is None
        ):
            raise ValueError(
                "native GPU observation requires actual selected device zero"
            )
        if self.device == "mps" and self.capacity.device_memory_kind != "unified":
            raise ValueError(
                "native Metal observation requires its physical shared pool"
            )
        return self


def native_music_admission_capacity(
    capacity: NativeMusicMemoryCapacity,
) -> NativeMusicMemoryCapacity:
    """Reserve physical pressure headroom once before pending/committed admission.

    Keep telemetry and live observations raw. All admission owners normalize
    those actual readings with this same policy before charging reservations.
    The live process guard enforces these floors rather than double-charging
    the complete future workspace every time resident memory is sampled.
    """
    host_floor = min(512 * 1024**2, capacity.host_total_bytes // 20)
    device_floor = (
        min(256 * 1024**2, capacity.device_total_bytes // 20)
        if capacity.device_total_bytes is not None
        else 0
    )
    shared_floor = (
        min(512 * 1024**2, capacity.shared_total_bytes // 20)
        if capacity.shared_total_bytes is not None
        else 0
    )
    return capacity.model_copy(
        update={
            "host_total_bytes": capacity.host_total_bytes - host_floor,
            "host_available_bytes": max(0, capacity.host_available_bytes - host_floor),
            "device_total_bytes": capacity.device_total_bytes - device_floor
            if capacity.device_total_bytes is not None
            else None,
            "device_available_bytes": max(
                0, capacity.device_available_bytes - device_floor
            )
            if capacity.device_available_bytes is not None
            else None,
            "shared_total_bytes": capacity.shared_total_bytes - shared_floor
            if capacity.shared_total_bytes is not None
            else None,
            "shared_available_bytes": max(
                0, capacity.shared_available_bytes - shared_floor
            )
            if capacity.shared_available_bytes is not None
            else None,
        }
    )


def native_music_host_available_bytes(
    total_bytes: int, fallback_available_bytes: int, mach_snapshot: str | None,
) -> int:
    """Use Core's reclaimable-cache Mach accounting in both native owners.

    The caller supplies one bounded ``vm_stat`` snapshot on macOS, or None on
    other systems/failure. Wired, anonymous and compressor pages are disjoint;
    file-backed cache remains reclaimable. Missing/changed counters use the
    actual psutil fallback rather than combining observations from two clocks.
    No process or system effect occurs in this portable contract function.
    """
    if total_bytes <= 0 or not 0 <= fallback_available_bytes <= total_bytes:
        raise ValueError("native host capacity requires valid physical observations")
    if mach_snapshot is None:
        return fallback_available_bytes
    size = re.search(r"page size of (\d+) bytes", mach_snapshot)
    if size is None or int(size.group(1)) <= 0:
        return fallback_available_bytes
    counters = {
        match.group(1).strip(): int(match.group(2))
        for match in re.finditer(r"^(.+?):\s+(\d+)\.?\s*$", mach_snapshot, re.MULTILINE)
    }
    names = ("Pages wired down", "Anonymous pages", "Pages occupied by compressor")
    if not all(name in counters for name in names):
        return fallback_available_bytes
    return max(0, total_bytes - sum(counters[name] for name in names) * int(size.group(1)))


def native_music_memory_budget_error(
    budget: NativeMusicMemoryBudget,
    capacity: NativeMusicMemoryCapacity,
    *,
    device: NativeMusicDevice,
    now_unix_ms: int,
) -> str | None:
    """Compare complete admitted peaks against one fresh normalized capacity.

    Core and the separately installed process execute this same pure fit policy.
    Callers reserve pressure headroom once with native_music_admission_capacity
    before charging pending and committed placements; no allocation is inferred
    from a model merely being loaded.
    """
    sampled = capacity.sampled_at_unix_ms
    if now_unix_ms <= 0 or sampled <= 0:
        return "Native music needs a timestamped current memory observation"
    if sampled - now_unix_ms > 30_000:
        return "Native music memory observation is ahead of the local clock; check node clocks and refresh resources"
    if now_unix_ms - sampled > 120_000:
        return "Native music memory observation is stale; refresh native resources before placement"
    if budget.host_peak_bytes > capacity.host_available_bytes:
        return "Native music needs more host memory for loading, media and forward execution"
    if (
        capacity.shared_available_bytes is not None
        and budget.shared_peak_bytes > capacity.shared_available_bytes
    ):
        return "Native music needs more unified physical memory for simultaneous host/device work"
    if device == "cpu":
        if budget.device_peak_bytes != 0:
            return "Native music CPU admission received a GPU execution estimate"
        return None
    if budget.device_peak_bytes == 0:
        return "Native music GPU admission received a CPU execution estimate"
    if capacity.device_index != 0 or capacity.device_available_bytes is None:
        return "Native music needs current memory for its actual selected device zero"
    if capacity.device_memory_kind is None:
        return (
            "Native music needs the actual device's discrete or unified memory topology"
        )
    if device == "mps" and capacity.device_memory_kind != "unified":
        return "Native music Metal requires its unified physical-memory observation"
    if budget.device_peak_bytes > capacity.device_available_bytes:
        return "Native music needs more memory on its actual selected device zero"
    return None


def native_music_live_memory_error(
    budget: NativeMusicMemoryBudget,
    observation: NativeMusicMemoryObservation,
    *,
    now_unix_ms: int,
) -> str | None:
    """Refuse stale sampling, allocation overruns or current physical pressure.

    The immutable budget comes from the same complete phase policy as admission.
    RSS may alias GPU pages on unified memory, so its ceiling is the simultaneous
    physical peak there. No RSS/allocator value earns an availability credit.
    Real unique physical peaks still require isolated workload qualification.
    """
    capacity = observation.capacity
    sampled = capacity.sampled_at_unix_ms
    if (
        sampled <= 0
        or now_unix_ms <= 0
        or now_unix_ms - sampled > 5000
        or sampled - now_unix_ms > 30000
    ):
        return "Native music live memory observation is missing, stale or clock-incompatible"
    if (observation.device == "cpu") != (budget.device_peak_bytes == 0):
        return "Native music live memory observation differs from its admitted compute policy"
    if budget.host_peak_bytes > capacity.host_total_bytes:
        return "Native music admitted host budget exceeds actual physical capacity"
    if (
        capacity.device_total_bytes is not None
        and budget.device_peak_bytes > capacity.device_total_bytes
    ):
        return "Native music admitted device budget exceeds actual physical capacity"
    if (
        capacity.shared_total_bytes is not None
        and budget.shared_peak_bytes > capacity.shared_total_bytes
    ):
        return "Native music admitted shared budget exceeds actual physical capacity"
    host_limit = (
        budget.shared_peak_bytes
        if capacity.device_memory_kind == "unified"
        else budget.host_peak_bytes
    )
    if observation.host_resident_bytes > host_limit:
        return (
            "Native music process exceeded its admitted host or shared memory ceiling"
        )
    if observation.device_allocated_bytes > budget.device_peak_bytes:
        return (
            "Native music process exceeded its admitted device allocation ceiling "
            f"(allocated={observation.device_allocated_bytes} bytes, "
            f"budget={budget.device_peak_bytes} bytes)"
        )
    host_floor = min(512 * 1024**2, capacity.host_total_bytes // 20)
    if capacity.host_available_bytes < host_floor:
        return "Native music stopped for current host memory pressure"
    if (
        capacity.shared_available_bytes is not None
        and capacity.shared_available_bytes
        < min(512 * 1024**2, (capacity.shared_total_bytes or 0) // 20)
    ):
        return "Native music stopped for current shared physical memory pressure"
    if (
        capacity.device_available_bytes is not None
        and capacity.device_total_bytes is not None
        and capacity.device_available_bytes
        < min(256 * 1024**2, capacity.device_total_bytes // 20)
    ):
        return "Native music stopped for current selected-device memory pressure"
    return None


@final
class NativeMusicLibraryProbe(NativeMusicContract):
    """Measured package/device execution without weights or model readiness.

    Core verifies the installed payload before running this private probe. The
    separate loaded-model receipt additionally binds actual component placement
    and model identity; this record cannot stand in for that receipt.
    """

    kind: Literal["library_probe"] = Field(
        default="library_probe", description="Weights-free native probe discriminator."
    )
    protocol_version: Literal[1] = Field(
        default=1, description="Native probe protocol."
    )
    payload_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="Actual installed runtime payload identity.",
    )
    device: NativeMusicDevice = Field(
        description="Explicit successfully executed compute lane."
    )
    precision: NativeMusicPrecision = Field(
        description="Executed diffusion/text-encoder precision policy."
    )
    autoencoder_precision: NativeMusicTensorPrecision = Field(
        description="Executed autoencoder precision, including Metal FP16."
    )
    cuda_compute_capability: (
        tuple[Annotated[int, Field(ge=0, le=255)], Annotated[int, Field(ge=0, le=255)]]
        | None
    ) = Field(
        default=None,
        description="Observed CUDA major/minor; absent on CPU, Metal and ROCm.",
    )
    torch_version: str = Field(
        min_length=1,
        max_length=128,
        description="Imported Torch build version, not a driver compatibility claim.",
    )
    memory_capacity: NativeMusicMemoryCapacity | None = Field(
        default=None,
        description="Timestamped host and actual runtime-visible device-zero capacity; a sample, not permanently ready memory or model qualification.",
    )

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        """Keep the executed device, native precision and capability consistent."""
        if self.device == "cuda":
            if self.cuda_compute_capability is None:
                raise ValueError(
                    "native CUDA library probe requires its actual compute capability"
                )
            expected: NativeMusicPrecision = (
                "bfloat16" if self.cuda_compute_capability[0] >= 8 else "float16"
            )
        else:
            if self.cuda_compute_capability is not None:
                raise ValueError(
                    "non-CUDA library probes cannot assert CUDA capability"
                )
            expected = "float32"
        autoencoder = "float16" if self.device == "mps" else expected
        if self.precision != expected or self.autoencoder_precision != autoencoder:
            raise ValueError(
                "native library probe precision differs from its compute policy"
            )
        memory = self.memory_capacity
        if memory is not None:
            if self.device == "cpu" and memory.device_index is not None:
                raise ValueError("native CPU probe cannot borrow GPU memory")
            if self.device != "cpu" and memory.device_index != 0:
                raise ValueError(
                    "native accelerator probe requires its actual device-zero memory"
                )
            if self.device == "mps" and memory.device_memory_kind != "unified":
                raise ValueError(
                    "native Metal probe requires its unified memory observation"
                )
        return self


def _relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or "\\" in value
        or "\x00" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or path.as_posix() != value
    ):
        raise ValueError("native music artifact path must be canonical and relative")
    return path


def _absolute(value: str) -> None:
    if not PurePath(value).is_absolute() or "\x00" in value:
        raise ValueError("native music private paths must be absolute")


@final
class NativeMusicFile(NativeMusicContract):
    """One exact selected component object; original model Python is excluded."""

    path: str = Field(
        min_length=1, max_length=512, description="Repository-relative selected file."
    )
    size_bytes: int = Field(
        gt=0, le=8 * 1024**3, description="Exact immutable object length."
    )
    object_id: str = Field(
        pattern=r"^(?:sha256:[0-9a-f]{64}|git-sha1:[0-9a-f]{40})$",
        description="SHA-256 tensor identity or Git blob identity for support metadata.",
    )

    @model_validator(mode="after")
    def validate_file(self) -> Self:
        """Refuse path escapes, executable model files and weak tensor identities."""
        path = _relative(self.path)
        if path.suffix in {".py", ".pyc", ".so", ".dylib", ".dll"}:
            raise ValueError(
                "native music model objects cannot contain executable code"
            )
        if path.suffix in {".safetensors", ".pt"} and not self.object_id.startswith(
            "sha256:"
        ):
            raise ValueError("native music tensor and latent objects require SHA-256")
        return self


@final
class NativeMusicComponent(NativeMusicContract):
    """Worker-selected immutable installation plus one complete loader component."""

    kind: Literal["base", "text_encoder", "autoencoder"] = Field(
        description="Native model role."
    )
    directory: str = Field(
        min_length=1, max_length=4096, description="Private staged installation root."
    )
    root: str | None = Field(
        default=None,
        max_length=512,
        description="Loader root beneath the installation.",
    )
    files: tuple[NativeMusicFile, ...] = Field(
        min_length=2, max_length=9, description="Complete selected role inventory."
    )

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        """Require exactly the files the pinned base, encoder and VAE loaders use."""
        _absolute(self.directory)
        prefix = _relative(self.root).as_posix() + "/" if self.root is not None else ""
        required = {
            "base": {"config.json", "model.safetensors", "silence_latent.pt"},
            "text_encoder": {
                "added_tokens.json",
                "chat_template.jinja",
                "config.json",
                "merges.txt",
                "model.safetensors",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
            },
            "autoencoder": {"config.json", "diffusion_pytorch_model.safetensors"},
        }[self.kind]
        paths = [item.path for item in self.files]
        if len(set(paths)) != len(paths) or set(paths) != {
            prefix + name for name in required
        }:
            raise ValueError(
                "native music component requires its exact complete file set"
            )
        return self


@final
class NativeMusicWeights(NativeMusicContract):
    """Three explicit model components, separate from the sealed runtime package."""

    base: NativeMusicComponent = Field(
        description="Native ACE base config, tensor and silence latent."
    )
    text_encoder: NativeMusicComponent = Field(
        description="Qwen text encoder and complete tokenizer."
    )
    autoencoder: NativeMusicComponent = Field(
        description="Audio autoencoder config and safetensors."
    )

    @model_validator(mode="after")
    def validate_roles(self) -> Self:
        """Do not permit a complete but incorrectly assigned component inventory."""
        if (self.base.kind, self.text_encoder.kind, self.autoencoder.kind) != (
            "base",
            "text_encoder",
            "autoencoder",
        ):
            raise ValueError("native music components must retain their exact roles")
        return self


@final
class NativeMusicDirectoryIdentity(NativeMusicContract):
    """Core-observed directory inode, recorded before the native process writes.

    A pathname alone cannot distinguish an owned partial result from a replaced
    directory after a failed process. These facts are private process metadata;
    they never enter the public request, replicated State or result manifest.
    """

    device: int = Field(
        ge=0, le=2**64 - 1, description="Observed directory filesystem device."
    )
    inode: int = Field(
        gt=0, le=2**64 - 1, description="Observed directory inode on that device."
    )


@final
class NativeMusicServeConfig(NativeMusicContract):
    """Private policy for one supervised loaded model; never a public API payload."""

    protocol_version: Literal[1] = Field(
        default=1, description="Native control protocol version."
    )
    model_identity: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 of the admitted immutable model selection, echoed by readiness.",
    )
    weights: NativeMusicWeights = Field(
        description="Complete independently verified model selections."
    )
    directory: str = Field(
        min_length=1,
        max_length=4096,
        description="Exclusive private native model lifetime root.",
    )
    directory_identity: NativeMusicDirectoryIdentity = Field(
        description="Core-owned lifetime root observed and journaled before spawn."
    )
    requests_identity: NativeMusicDirectoryIdentity = Field(
        description="Core-owned requests subdirectory observed before spawn."
    )
    loader_identity: NativeMusicDirectoryIdentity = Field(
        description="Core-owned empty loader view, separate from retained request output."
    )
    device: NativeMusicDevice = Field(
        description="Explicit requested compute lane, never auto."
    )
    precision: NativeMusicPrecision = Field(
        description="Actual loaded tensor precision required by admission."
    )
    memory_budget: NativeMusicMemoryBudget | None = Field(
        default=None,
        description="Admitted phase peaks. Production native serving requires this; an absent budget is reserved for controlled protocol fixtures.",
    )
    threads: int = Field(
        ge=1,
        le=32,
        default=2,
        description="Explicit bounded CPU intra-op thread policy.",
    )
    generation_timeout_seconds: int = Field(
        ge=30,
        le=7200,
        default=3600,
        description="Native cooperative render budget; the parent retains a separate hard process deadline.",
    )

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        """Keep proven FP32 non-CUDA policy explicit rather than deriving it from weights."""
        _absolute(self.directory)
        if self.device != "cuda" and self.precision != "float32":
            raise ValueError("native music CPU, Metal and ROCm policy requires float32")
        if self.memory_budget is not None and (self.device == "cpu") != (
            self.memory_budget.device_peak_bytes == 0
        ):
            raise ValueError(
                "native music budget differs from the configured compute lane"
            )
        return self


@final
class NativeMusicRequest(NativeMusicContract):
    """One generation or edit with coordinates in the native 48 kHz frame grid."""

    request_id: str = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$",
        description="Owner-correlated bounded request identity.",
    )
    operation: NativeMusicOperation = Field(
        description="Closed generation or arrangement operation."
    )
    prompt: str = Field(
        min_length=1,
        max_length=8000,
        description="Musical caption; not native options.",
    )
    lyrics: str = Field(
        default="[Instrumental]",
        max_length=32000,
        description="Explicit resolved lyrics or instrumental marker.",
    )
    seed: int = Field(
        ge=0,
        le=2**32 - 1,
        description="Resolved reproducible native seed; never random sentinel.",
    )
    target_frames: int = Field(
        ge=48000,
        le=NATIVE_MUSIC_MAX_FRAMES,
        description="Bounded total native output duration in 48 kHz frames.",
    )
    source: SeparationSource | None = Field(
        default=None, description="Original verified WAV identity; no raw media."
    )
    source_path: str | None = Field(
        default=None,
        max_length=4096,
        description="Worker-private original context WAV, required only for edits.",
    )
    region_start_frame: int = Field(
        default=0,
        ge=0,
        le=NATIVE_MUSIC_MAX_FRAMES,
        description="Inclusive native edit region start.",
    )
    region_end_frame: int | None = Field(
        default=None,
        ge=1,
        le=NATIVE_MUSIC_MAX_FRAMES,
        description="Exclusive native edit region end.",
    )
    part_class: NativeMusicPart | None = Field(
        default=None, description="Required native part class only for Add a part."
    )
    output_directory: str = Field(
        min_length=1,
        max_length=4096,
        description="Exclusive worker-owned request output root.",
    )
    output_directory_identity: NativeMusicDirectoryIdentity = Field(
        description="Core-owned empty result directory observed before admission."
    )

    @model_validator(mode="after")
    def validate_operation(self) -> Self:
        """Bind operation, original geometry, output length and region together."""
        _absolute(self.output_directory)
        if (self.operation == "add_part") != (self.part_class is not None):
            raise ValueError("native music part class belongs only to Add a part")
        if self.operation == "generate":
            if (
                self.source is not None
                or self.source_path is not None
                or self.region_end_frame is not None
                or self.region_start_frame
            ):
                raise ValueError(
                    "native music generation cannot contain editing context"
                )
            return self
        if self.source is None or self.source_path is None:
            raise ValueError("native music editing requires its verified context WAV")
        _absolute(self.source_path)
        mapped_frames = (
            self.source.frames * 48000 + self.source.sample_rate - 1
        ) // self.source.sample_rate
        end = self.region_end_frame
        if end is None or self.region_start_frame >= end:
            raise ValueError("native music editing requires a nonempty frame region")
        if self.operation == "continue":
            if (
                self.target_frames <= mapped_frames
                or self.region_start_frame != mapped_frames
                or end != self.target_frames
            ):
                raise ValueError(
                    "native music continuation must extend exactly beyond the original context"
                )
        elif self.target_frames != mapped_frames or end > mapped_frames:
            raise ValueError(
                "native music edit must remain within its original context"
            )
        return self


@final
class NativeMusicExecution(NativeMusicContract):
    """Observed module placement and dtype; not a complete operator execution trace."""

    device: NativeMusicDevice = Field(
        description="Observed diffusion transformer compute lane."
    )
    precision: NativeMusicPrecision = Field(
        description="Observed diffusion transformer tensor precision."
    )
    text_encoder_device: NativeMusicDevice = Field(
        description="Observed text encoder placement."
    )
    text_encoder_precision: NativeMusicTensorPrecision = Field(
        description="Observed text encoder tensor precision."
    )
    autoencoder_device: NativeMusicDevice = Field(
        description="Observed audio autoencoder placement."
    )
    autoencoder_precision: NativeMusicTensorPrecision = Field(
        description="Observed audio autoencoder precision, independently of the diffusion transformer."
    )


@final
class NativeMusicReady(NativeMusicContract):
    """Observed loaded model identity and actual execution policy, not a cache hint."""

    protocol_version: Literal[1] = Field(
        default=1, description="Native control protocol version."
    )
    model_identity: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="Exact worker-selected artifact identity that was loaded.",
    )
    payload_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", description="Actual sealed runtime payload identity."
    )
    execution: NativeMusicExecution = Field(
        description="Actual loaded component placement/precision after initialization."
    )
    load_seconds: float = Field(
        ge=0,
        allow_inf_nan=False,
        description="Measured complete model initialization time.",
    )


@final
class NativeMusicWav(NativeMusicContract):
    """Measured bounded PCM24 stereo output; no server-local file path or media."""

    content_type: Literal["audio/wav"] = Field(
        default="audio/wav", description="PCM WAV result encoding."
    )
    sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", description="SHA-256 of every encoded output byte."
    )
    size_bytes: int = Field(
        ge=44, le=64 * 1024**2, description="Complete encoded result, capped at 64 MiB."
    )
    frames: int = Field(
        ge=48000,
        le=NATIVE_MUSIC_MAX_FRAMES,
        description="Actual aligned native output frames.",
    )
    sample_rate: Literal[48000] = Field(
        default=48000, description="Preserved native ACE output sampling rate."
    )
    channels: Literal[2] = Field(default=2, description="Native stereo output.")
    bit_depth: Literal[24] = Field(
        default=24, description="Signed integer PCM production export."
    )
    peak: float = Field(
        ge=0,
        le=1,
        allow_inf_nan=False,
        description="Measured encoded absolute sample peak.",
    )
    gain: float = Field(
        gt=0,
        le=1,
        allow_inf_nan=False,
        description="Only attenuation needed to avoid integer PCM clipping; no loudness normalization.",
    )

    @model_validator(mode="after")
    def validate_size(self) -> Self:
        """Require enough bytes to contain the declared complete PCM24 stereo audio."""
        if self.size_bytes < 44 + self.frames * 6:
            raise ValueError("native music WAV cannot contain its declared frames")
        return self


@final
class NativeMusicResult(NativeMusicContract):
    """One correlated completed result whose bytes remain in the private worker."""

    request_id: str = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$",
        description="Exactly the accepted native request identity.",
    )
    model_identity: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="Actual loaded immutable model selection.",
    )
    payload_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", description="Actual sealed native runtime payload."
    )
    operation: NativeMusicOperation = Field(
        description="Operation that produced the result."
    )
    seed: int = Field(
        ge=0, le=2**32 - 1, description="Actual resolved seed used for this request."
    )
    source_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
        description="Original context identity; absent only for generation.",
    )
    conditioning_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
        description="Actual canonical/resampled and, for Continue, padded context WAV consumed by the model.",
    )
    conditioning_gain: float | None = Field(
        default=None,
        gt=0,
        le=1,
        allow_inf_nan=False,
        description="Explicit shared context attenuation, when needed to avoid upstream input clipping; original bytes remain unchanged.",
    )
    conditioning_padding_frames: int = Field(
        default=0,
        ge=0,
        le=NATIVE_MUSIC_LATENT_HOP_FRAMES - 1,
        description="Zero samples appended beyond the requested edit extent solely for codec alignment; excludes Continue's requested extension.",
    )
    padding_frames_removed: int = Field(
        default=0,
        ge=0,
        le=NATIVE_MUSIC_MIN_DECODE_FRAMES - NATIVE_MUSIC_SAMPLE_RATE,
        description="Only the pinned decoder minimum or explicitly aligned conditioning tail removed from an edit; never an arbitrary length repair.",
    )
    execution: NativeMusicExecution = Field(
        description="Observed loaded component placement/precision after this request."
    )
    output: NativeMusicWav = Field(
        description="Measured WAV facts, verified again by Core before transport."
    )
    generation_seconds: float = Field(
        gt=0, allow_inf_nan=False, description="Measured model execution time."
    )

    @model_validator(mode="after")
    def validate_lineage(self) -> Self:
        """Source identity and native conditioning must accompany every edit."""
        if (
            (self.source_sha256 is not None) != (self.operation != "generate")
            or (self.conditioning_sha256 is not None) != (self.operation != "generate")
            or (self.conditioning_gain is not None) != (self.operation != "generate")
        ):
            raise ValueError(
                "native music result lineage must match generation or editing"
            )
        conditioning_frames = native_music_conditioning_frames(self.output.frames)
        expected_conditioning_padding = (
            conditioning_frames - self.output.frames
            if self.operation != "generate"
            else 0
        )
        if self.conditioning_padding_frames != expected_conditioning_padding:
            raise ValueError(
                "native music conditioning padding must match its codec grid"
            )
        if self.padding_frames_removed and (
            self.operation == "generate"
            or self.output.frames + self.padding_frames_removed
            != max(NATIVE_MUSIC_MIN_DECODE_FRAMES, conditioning_frames)
        ):
            raise ValueError(
                "native music may remove only its verified codec edit padding"
            )
        return self


@final
class NativeMusicServeRequest(NativeMusicContract):
    """Contiguous private control request; the owner never pipelines work."""

    sequence: int = Field(
        ge=1,
        le=2**63 - 1,
        description="Monotonic process-local admission sequence; rejects replay without retained IDs.",
    )
    request: NativeMusicRequest = Field(
        description="One closed generation or editing operation."
    )


@final
class NativeMusicProgress(NativeMusicContract):
    """Measured lifecycle transition, never an estimated generation percentage."""

    kind: Literal["progress"] = Field(
        default="progress", description="Private control frame discriminator."
    )
    request_id: str | None = Field(
        default=None,
        max_length=160,
        description="Active request, absent during model initialization.",
    )
    stage: Literal[
        "verifying_weights", "loading", "conditioning", "generating", "writing"
    ] = Field(description="Actual operation currently entered.")


@final
class NativeMusicReadyFrame(NativeMusicContract):
    """Model initialization is complete and its actual component policy is known."""

    kind: Literal["ready"] = Field(
        default="ready", description="Private control frame discriminator."
    )
    ready: NativeMusicReady = Field(
        description="Observed loaded model, runtime and precision."
    )


@final
class NativeMusicMemoryFrame(NativeMusicContract):
    """Latest-value resource heartbeat; never a generation terminal or media."""

    kind: Literal["memory"] = Field(
        default="memory", description="Private resource discriminator."
    )
    sequence: int = Field(
        ge=1,
        le=2**53 - 1,
        description="Strictly increasing observation sequence within this process.",
    )
    observation: NativeMusicMemoryObservation = Field(
        description="Measured native leader and actual device resources."
    )


@final
class NativeMusicCompleted(NativeMusicContract):
    """One complete durable private WAV and result record are ready for custody."""

    kind: Literal["completed"] = Field(
        default="completed", description="Private control frame discriminator."
    )
    result: NativeMusicResult = Field(
        description="Exact request-correlated facts; no audio bytes or paths."
    )


@final
class NativeMusicFailed(NativeMusicContract):
    """A terminal native failure; its supervisor must retire this process."""

    kind: Literal["failed"] = Field(
        default="failed", description="Private control frame discriminator."
    )
    request_id: str | None = Field(
        default=None,
        max_length=160,
        description="Failed request when its complete framing was accepted.",
    )
    message: Literal[
        "Native music process failed; inspect bounded runner diagnostics."
    ] = Field(
        default="Native music process failed; inspect bounded runner diagnostics.",
        description="No private paths, prompts or raw upstream diagnostics enter control events.",
    )


NativeMusicFrame = Annotated[
    NativeMusicProgress
    | NativeMusicMemoryFrame
    | NativeMusicReadyFrame
    | NativeMusicCompleted
    | NativeMusicFailed,
    Field(discriminator="kind"),
]
