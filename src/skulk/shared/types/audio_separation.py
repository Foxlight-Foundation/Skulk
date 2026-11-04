"""Bounded source and four-stem contracts, independent of speech and generation.

Only verified WAV facts travel in commands and replicated State. Original and
separated bytes use the addressed media plane; local paths are never public
request fields. Runtime and API independently compare these facts with WAVs.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Iterator
from typing import Annotated, Final, Literal, Self, cast, final

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_SEPARATION_SOURCE_BYTES: Final = 64 * 1024 * 1024
MAX_SEPARATION_DECODED_BYTES: Final = 128 * 1024 * 1024
MAX_SEPARATION_STEM_BYTES: Final = 64 * 1024 * 1024
MAX_SEPARATION_SET_BYTES: Final = 256 * 1024 * 1024
SEPARATION_STEMS: Final = ("drums", "bass", "other", "vocals")
SeparationStemName = Literal["drums", "bass", "other", "vocals"]
"""Canonical complete-set order, not a promise of perfect source isolation."""


class SeparationContract(BaseModel):
    """Immutable strict records at the command and native runtime boundaries."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


@final
class AudioSeparationRequest(SeparationContract):
    """Public metadata beside one bounded multipart original WAV upload."""

    model: str = Field(
        min_length=1,
        max_length=256,
        description="Mounted source-separation model alias; not a download URL or module.",
    )


@final
class SeparationSource(SeparationContract):
    """Measured native WAV identity, constructed after input verification."""

    sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 of every original encoded WAV byte.",
    )
    size_bytes: int = Field(
        ge=44,
        le=MAX_SEPARATION_SOURCE_BYTES,
        description="Complete original WAV length, capped at 64 MiB.",
    )
    sample_rate: int = Field(
        ge=8000, le=192000, description="Verified native samples per second."
    )
    channels: int = Field(
        ge=1, le=2, description="Verified mono or stereo channel count."
    )
    frames: int = Field(
        ge=1,
        description="Complete interleaved native source frames, before resampling.",
    )
    encoding: Literal["pcm", "float"] = Field(
        description="Validated uncompressed WAV sample encoding."
    )
    bit_depth: Literal[8, 16, 24, 32, 64] = Field(
        description="Native precision; float accepts 32/64 bits and integer PCM accepts 8/16/24/32."
    )

    @model_validator(mode="after")
    def validate_geometry(self) -> Self:
        """Bound duration, decoded float32 allocation and encoded geometry together."""
        if not self.sample_rate <= self.frames <= self.sample_rate * 120:
            raise ValueError(
                "separation source duration must be between one and 120 seconds"
            )
        if self.frames * self.channels * 4 > MAX_SEPARATION_DECODED_BYTES:
            raise ValueError("separation source exceeds the 128 MiB decoded bound")
        if self.encoding == "float" and self.bit_depth not in (32, 64):
            raise ValueError("floating WAV sources require 32 or 64 bits")
        if self.encoding == "pcm" and self.bit_depth == 64:
            raise ValueError("64-bit integer PCM is not supported")
        if 44 + self.frames * self.channels * (self.bit_depth // 8) > self.size_bytes:
            raise ValueError(
                "separation source byte length cannot contain its declared samples"
            )
        return self


@final
class AudioSeparationTaskParams(SeparationContract):
    """Admitted model/source facts; raw input arrives separately before execution."""

    model: str = Field(
        min_length=1,
        max_length=256,
        description="Exact selected separation model alias.",
    )
    source: SeparationSource = Field(
        description="Original verified by the API and independently by the selected worker."
    )
    total_input_chunks: int = Field(
        ge=1,
        le=4096,
        description="Bounded raw-media chunk count acknowledged before starting the runner.",
    )


@final
class SeparatedStem(SeparationContract):
    """One measured PCM24 stereo WAV, identified without an output file path."""

    name: SeparationStemName = Field(
        description="Estimated musical source carried by this WAV."
    )
    content_type: Literal["audio/wav"] = Field(
        default="audio/wav", description="Uncompressed PCM WAV output."
    )
    size_bytes: int = Field(
        ge=44,
        le=MAX_SEPARATION_STEM_BYTES,
        description="Complete encoded size, capped at 64 MiB per stem.",
    )
    sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 of the complete retained stem WAV.",
    )
    sample_rate: int = Field(
        ge=8000,
        le=192000,
        description="Common measured samples per second for this stem set.",
    )
    frames: int = Field(
        ge=1, description="Complete frame count; identical for every stem in the set."
    )
    channels: Literal[2] = Field(
        default=2,
        description="Stereo output, including an explicitly duplicated mono source.",
    )
    bit_depth: Literal[24] = Field(
        default=24,
        description="PCM24 export precision; it does not restore source detail.",
    )
    peak: float = Field(
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        description="Measured absolute encoded sample peak after common gain.",
    )

    @model_validator(mode="after")
    def validate_geometry(self) -> Self:
        """Require a bounded PCM24 duration and enough bytes for complete samples."""
        if self.frames > self.sample_rate * 120:
            raise ValueError("separated stem duration exceeds 120 seconds")
        if 44 + self.frames * 6 > self.size_bytes:
            raise ValueError(
                "stem byte length cannot contain its declared PCM24 stereo frames"
            )
        return self


@final
class AudioSeparationOutput(SeparationContract):
    """A complete time-aligned set; partial results can never satisfy completion."""

    source: SeparationSource = Field(
        description="Exact original whose verified bytes were separated."
    )
    stems: tuple[SeparatedStem, ...] = Field(
        min_length=4,
        max_length=4,
        description="Complete drums, bass, other, vocals set in canonical order.",
    )
    common_gain: float = Field(
        gt=0.0,
        le=1.0,
        allow_inf_nan=False,
        description="One linear attenuation applied equally to all stems to avoid PCM clipping; no per-stem normalization.",
    )
    pre_gain_peak: float = Field(
        ge=0.0,
        allow_inf_nan=False,
        description="Maximum absolute native model sample across the whole set, before shared attenuation.",
    )

    @model_validator(mode="after")
    def validate_complete_set(self) -> Self:
        """Reject missing/reordered sources, different origins and over-limit sets."""
        if tuple(stem.name for stem in self.stems) != SEPARATION_STEMS:
            raise ValueError(
                "separation requires drums, bass, other and vocals exactly once in canonical order"
            )
        rate, frames = self.stems[0].sample_rate, self.stems[0].frames
        if any(
            (stem.sample_rate, stem.frames) != (rate, frames) for stem in self.stems
        ):
            raise ValueError(
                "all separated stems must share one sample rate and frame count"
            )
        expected_frames = (2 * self.source.frames * rate + self.source.sample_rate) // (
            2 * self.source.sample_rate
        )
        if frames != expected_frames:
            raise ValueError(
                "separated frames must preserve the complete source timebase"
            )
        if sum(stem.size_bytes for stem in self.stems) > MAX_SEPARATION_SET_BYTES:
            raise ValueError("complete separated stem set exceeds 256 MiB")
        if self.pre_gain_peak * self.common_gain > 1.0 + 1e-7:
            raise ValueError("shared stem gain does not prevent PCM clipping")
        return self


def source_from_wav(data: bytes) -> SeparationSource:
    """Verify a complete bounded uncompressed WAV before admitting its metadata.

    Accept integer PCM and IEEE float, including their standard extensible
    subtypes. Reject duplicate sample/format chunks, inconsistent container or
    sample geometry and nonfinite float samples. Callers must bound input reads
    before constructing ``data``; this function performs no file or network I/O.
    """
    return _source_and_samples(data)[0]


def _source_and_samples(data: bytes) -> tuple[SeparationSource, memoryview]:
    """Validate once and retain a view of sample bytes without another allocation."""
    if not 44 <= len(data) <= MAX_SEPARATION_SOURCE_BYTES:
        raise ValueError("separation WAV must fit the 64 MiB encoded bound")
    if (
        data[:4] != b"RIFF"
        or data[8:12] != b"WAVE"
        or int.from_bytes(data[4:8], "little") + 8 != len(data)
    ):
        raise ValueError("separation requires one complete RIFF WAVE file")
    format_chunk: memoryview | None = None
    sample_chunk: memoryview | None = None
    view = memoryview(data)
    offset = 12
    while offset < len(data):
        if offset + 8 > len(data):
            raise ValueError("truncated WAV chunk header")
        length = int.from_bytes(data[offset + 4 : offset + 8], "little")
        end = offset + 8 + length
        padded_end = end + (length % 2)
        if padded_end > len(data):
            raise ValueError("truncated WAV chunk or padding")
        name = data[offset : offset + 4]
        if name == b"fmt ":
            if format_chunk is not None:
                raise ValueError("duplicate WAV format chunk")
            format_chunk = view[offset + 8 : end]
        elif name == b"data":
            if sample_chunk is not None:
                raise ValueError("duplicate WAV sample chunk")
            sample_chunk = view[offset + 8 : end]
        offset = padded_end
    if format_chunk is None or len(format_chunk) < 16 or sample_chunk is None:
        raise ValueError("WAV requires format and sample chunks")
    code, channels, rate, byte_rate, block_align, bits = cast(
        tuple[int, int, int, int, int, int], struct.unpack_from("<HHIIHH", format_chunk)
    )
    if code == 0xFFFE:
        # Extensible WAV is common for production PCM24. Only uncompressed
        # PCM/float subtypes with full container precision are implemented.
        if (
            len(format_chunk) != 40
            or int.from_bytes(format_chunk[16:18], "little") != 22
        ):
            raise ValueError("unsupported extensible WAV format")
        valid_bits = int.from_bytes(format_chunk[18:20], "little")
        channel_mask = int.from_bytes(format_chunk[20:24], "little")
        subtype = bytes(format_chunk[24:40])
        if valid_bits != bits or subtype[4:] != bytes.fromhex(
            "00001000800000aa00389b71"
        ):
            raise ValueError("unsupported extensible WAV precision or subtype")
        code = int.from_bytes(subtype[:4], "little")
        if channel_mask not in (0, 4 if channels == 1 else 3):
            raise ValueError("extensible WAV must use mono or stereo channel placement")
    elif len(format_chunk) not in (16, 18) or (
        len(format_chunk) == 18 and bytes(format_chunk[16:18]) != b"\0\0"
    ):
        raise ValueError("unsupported WAV format extension")
    if code not in (1, 3) or channels not in (1, 2) or bits not in (8, 16, 24, 32, 64):
        raise ValueError("separation accepts uncompressed mono/stereo PCM or float WAV")
    if (
        block_align != channels * (bits // 8)
        or byte_rate != rate * block_align
        or len(sample_chunk) % block_align
    ):
        raise ValueError("WAV sample geometry is inconsistent")
    source = SeparationSource.model_validate(
        {
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
            "sample_rate": rate,
            "channels": channels,
            "frames": len(sample_chunk) // block_align,
            "encoding": "pcm" if code == 1 else "float",
            "bit_depth": bits,
        }
    )
    if code == 3:
        for (sample,) in cast(
            Iterator[tuple[float]],
            struct.iter_unpack("<f" if bits == 32 else "<d", sample_chunk),
        ):
            if not math.isfinite(sample) or abs(sample) > 3.4028234663852886e38:
                raise ValueError("floating WAV samples must remain finite in float32")
    return source, sample_chunk


def pcm24_from_wav(data: bytes) -> tuple[SeparationSource, float]:
    """Measure bounded PCM24 stereo geometry/digest and encoded peak without a role.

    Hash every encoded byte and scan actual signed PCM samples for the peak.
    Music editing and separation share the same container validator. This makes
    no file, network or native-library calls. Bound reads before constructing
    ``data`` and run large scans outside the API event loop.
    """
    source, samples = _source_and_samples(data)
    if (source.encoding, source.bit_depth, source.channels) != ("pcm", 24, 2):
        raise ValueError("output requires PCM24 stereo WAV")
    peak_integer = 0
    for offset in range(0, len(samples), 3):
        peak_integer = max(
            peak_integer,
            abs(int.from_bytes(samples[offset : offset + 3], "little", signed=True)),
        )
    return source, peak_integer / 8388608


def stem_from_wav(name: SeparationStemName, data: bytes) -> SeparatedStem:
    """Measure one role's complete PCM24 stereo WAV independently of its producer.

    The caller compares all returned facts with the authoritative manifest and
    gates complete-set content visibility. The encoded sample peak uses the
    same role-independent WAV inspection as native music editing.
    """
    source, peak = pcm24_from_wav(data)
    return SeparatedStem(
        name=name,
        size_bytes=source.size_bytes,
        sha256=source.sha256,
        sample_rate=source.sample_rate,
        frames=source.frames,
        peak=peak,
    )


SeparationDevice = Literal["cpu", "mps", "cuda", "rocm"]


@final
class NativeSeparationJob(SeparationContract):
    """Worker-owned verified paths and admitted execution policy for one process."""

    weights_path: str = Field(
        min_length=1,
        max_length=4096,
        description="Exact worker-staged HTDemucs safetensors file; never a public request field.",
    )
    source_path: str = Field(
        min_length=1,
        max_length=4096,
        description="Command-private verified source WAV path.",
    )
    output_dir: str = Field(
        min_length=1,
        max_length=4096,
        description="New command-private directory for this complete stem set.",
    )
    source: SeparationSource = Field(
        description="Original identity independently checked before decoding."
    )
    device: SeparationDevice = Field(
        description="Explicit probed compute lane; no implicit fallback."
    )
    threads: int = Field(
        ge=1,
        le=32,
        description="Admitted CPU thread budget, not Demucs parallel chunk workers.",
    )
    segment_seconds: float = Field(
        default=7.8,
        ge=2.0,
        le=7.8,
        allow_inf_nan=False,
        description="Admitted model segment size; no larger than the training window.",
    )


@final
class NativeSeparationProgress(SeparationContract):
    """Small replaceable stage observations; completion uses a separate manifest."""

    stage: Literal["verifying", "loading", "separating", "encoding"]
    completed_frames: int = Field(
        default=0,
        ge=0,
        le=44100 * 120,
        description="Monotonic end of completed model segments, not a quality estimate.",
    )


@final
class NativeSeparationResult(SeparationContract):
    """Complete measured output and scoped resource/timing observations."""

    output: AudioSeparationOutput
    device: SeparationDevice
    load_seconds: float = Field(
        ge=0.0,
        allow_inf_nan=False,
        description="Observed model construction/load wall time; zero for resident reuse, whose startup is reported by NativeSeparationReady.",
    )
    separation_seconds: float = Field(
        ge=0.0,
        allow_inf_nan=False,
        description="Observed normalized model inference and synchronization wall time.",
    )
    total_seconds: float = Field(
        ge=0.0,
        allow_inf_nan=False,
        description="Native pipeline wall time after engine import, including verification, decode, resampling and all four WAV exports; excludes interpreter/import startup and parent verification.",
    )
    process_peak_rss_bytes: int = Field(
        ge=0,
        description="Process lifetime RSS high water, including prior resident jobs; excludes unreported device/driver allocations.",
    )
    device_peak_allocated_bytes: int | None = Field(
        default=None,
        ge=0,
        description="CUDA/HIP allocator high water where measurable; not whole-device consumption.",
    )


@final
class NativeSeparationServeConfig(SeparationContract):
    """Fixed worker-owned model/device policy for one resident native lifetime."""

    weights_path: str = Field(
        min_length=1,
        max_length=4096,
        description="Exact worker-staged safetensors path, never a public request field.",
    )
    device: SeparationDevice = Field(
        description="Explicit native lane selected before model construction; no fallback."
    )
    threads: int = Field(
        ge=1, le=32, description="Fixed admitted CPU thread budget for this lifetime."
    )


@final
class NativeSeparationServeRequest(SeparationContract):
    """One private job correlated within a single resident-model process."""

    request_id: str = Field(
        pattern=r"^[a-f0-9]{32}$", description="Supervisor-owned request identity."
    )
    sequence: int = Field(
        ge=1,
        le=2**63 - 1,
        description="Contiguous request sequence within this one native lifetime; retries do not replay execution.",
    )
    job: NativeSeparationJob = Field(
        description="Bounded worker paths and verified original for this execution."
    )


@final
class NativeSeparationReady(SeparationContract):
    """A resident model actually loaded on the explicit device, not a library probe."""

    kind: Literal["ready"] = "ready"
    protocol: Literal[1] = Field(
        default=1, description="Private resident protocol version."
    )
    payload_sha256: str = Field(
        pattern=r"^[a-f0-9]{64}$",
        description="Actual installed native manifest identity.",
    )
    weights_sha256: str = Field(
        pattern=r"^[a-f0-9]{64}$",
        description="Complete verified weights loaded in memory.",
    )
    device: SeparationDevice = Field(
        description="Explicit loaded lane, without fallback."
    )
    threads: int = Field(
        ge=1, le=32, description="Actual configured CPU thread budget."
    )
    load_seconds: float = Field(
        ge=0,
        allow_inf_nan=False,
        description="Weight verification, model construction and device placement wall time; excludes interpreter imports.",
    )
    sample_rate: Literal[44100] = Field(
        description="Verified native model sample rate."
    )
    audio_channels: Literal[2] = Field(description="Verified native model channels.")
    stems: tuple[SeparationStemName, ...] = Field(
        min_length=4, max_length=4, description="Actual ordered model output roles."
    )

    @model_validator(mode="after")
    def validate_stem_order(self) -> NativeSeparationReady:
        """Refuse loaded models with a different ordered output contract."""
        if self.stems != SEPARATION_STEMS:
            raise ValueError("resident separation model has different output roles")
        return self


@final
class NativeSeparationResidentProgress(SeparationContract):
    """A bounded progress observation correlated to its resident request."""

    kind: Literal["progress"] = "progress"
    request_id: str = Field(
        pattern=r"^[a-f0-9]{32}$", description="Owning native request."
    )
    progress: NativeSeparationProgress = Field(
        description="Replaceable stage progress."
    )


@final
class NativeSeparationCompleted(SeparationContract):
    """Complete native result facts, independently verified again by the supervisor."""

    kind: Literal["completed"] = "completed"
    request_id: str = Field(
        pattern=r"^[a-f0-9]{32}$", description="Owning native request."
    )
    result: NativeSeparationResult = Field(
        description="Complete measured four-stem set."
    )


@final
class NativeSeparationFailed(SeparationContract):
    """Sanitized terminal failure; the resident lifetime ends after this frame."""

    kind: Literal["failed"] = "failed"
    request_id: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{32}$",
        description="Owning request, absent when request framing itself was invalid.",
    )
    code: Literal["request_failed"] = Field(
        default="request_failed",
        description="Private diagnostics retain actual details.",
    )


type NativeSeparationFrame = Annotated[
    NativeSeparationReady
    | NativeSeparationResidentProgress
    | NativeSeparationCompleted
    | NativeSeparationFailed,
    Field(discriminator="kind"),
]
