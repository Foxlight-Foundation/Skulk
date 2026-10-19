"""Strict local-job protocol, independent of the base Skulk environment."""

from typing import Literal, final

from pydantic import BaseModel, ConfigDict, Field


class ProtocolModel(BaseModel):
    """Reject unknown fields and coercions at the isolated process boundary."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


@final
class MediaSource(ProtocolModel):
    """Worker-prepared source facts; no upstream probing or duration guesses."""

    path: str = Field(max_length=4096, description="Worker-verified source clip or normalized PCM path.")
    frames: int = Field(default=0, ge=0, description="Fully decoded video frames, zero for source audio.")
    has_audio: bool = Field(description="Whether preparation verified a finite audio track.")
    audio_sample_rate: int | None = Field(default=None, description="Verified audio sample rate, normalized source audio uses 48 kHz.")
    audio_channels: int | None = Field(default=None, description="Verified audio channels, normalized source audio is stereo.")


@final
class Guidance(ProtocolModel):
    """Exact video/audio CFG, block-28 STG and cross-modality guidance."""

    video_cfg: float
    audio_cfg: float
    stg_scale: float
    modality_scale: float


@final
class ImageAnchor(ProtocolModel):
    """A worker-staged image at a pixel frame, with -1 meaning the final frame."""

    path: str
    frame: int


@final
class AudioConditioning(ProtocolModel):
    """Normalized source waveform conditioning; offset was applied by preparation."""

    start_seconds: float = Field(default=0.0, ge=0.0, le=120.0, description="Original offset already applied by worker normalization.")


@final
class Editing(ProtocolModel):
    """Bounded source timeline controls validated by the owning worker."""

    operation: Literal["retake", "extend"] = Field(description="Regenerate a region or append to the complete source timeline.")
    start_seconds: float | None = Field(default=None, description="Inclusive retake start before latent-grid rounding.")
    end_seconds: float | None = Field(default=None, description="Exclusive retake end before latent-grid rounding.")
    regenerate_audio: bool = Field(default=True, description="False remuxes the original retake source waveform.")


@final
class PromptSegment(ProtocolModel):
    """One scene's text and relative duration."""

    prompt: str = Field(min_length=1, max_length=8000, description="Local scene description beside the global prompt.")
    length: int = Field(ge=1, le=120, description="Positive relative scene duration.")


@final
class PromptRelay(ProtocolModel):
    """Bounded local scene sequence sharing the global prompt."""

    segments: tuple[PromptSegment, ...] = Field(min_length=2, max_length=8, description="Ordered bounded local scenes.")


@final
class Job(ProtocolModel):
    """One validated core request with exact local component/output paths."""

    model_dir: str
    output_dir: str
    prompt: str
    width: int = Field(ge=128, le=65535)
    height: int = Field(ge=128, le=65535)
    fps: int = Field(ge=1)
    frame_count: int = Field(ge=1)
    seed: int = Field(ge=0, le=2**64 - 1)
    audio: bool
    pipeline: Literal["distilled_two_stage", "dev_two_stage", "hq_two_stage", "dfr", "retake", "extend"]
    decoder: Literal["convolutional", "diffusion"]
    codec: str
    base_steps: int = Field(ge=1, le=200)
    negative_prompt: str | None
    guidance: Guidance | None
    generated_keyframe_interval: int | None = Field(ge=1, le=1024)
    duration: Literal["fixed", "auto"]
    minimum_seconds: int
    maximum_seconds: int
    refinement_strength: float
    images: tuple[ImageAnchor, ...]
    audio_conditioning: AudioConditioning | None = None
    editing: Editing | None = None
    prompt_relay: PromptRelay | None = None
    source_media: MediaSource | None = None
    transformer_variant: Literal["distilled", "dev"] | None = None


@final
class BuildMetadata(ProtocolModel):
    """Immutable package identity and member digests shipped in the owned wheel."""

    source_revision: str
    payload_sha256: str
    members: dict[str, str]
    features: tuple[Literal["audio_conditioning", "retake", "extend", "prompt_relay"], ...] = ()
