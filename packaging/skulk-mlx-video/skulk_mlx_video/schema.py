"""Strict local-job protocol, independent of the base Skulk environment."""

from typing import Literal, final

from pydantic import BaseModel, ConfigDict, Field


class ProtocolModel(BaseModel):
    """Reject unknown fields and coercions at the isolated process boundary."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


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
    pipeline: Literal["distilled_two_stage", "dev_two_stage", "hq_two_stage", "dfr"]
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


@final
class BuildMetadata(ProtocolModel):
    """Immutable package identity and member digests shipped in the owned wheel."""

    source_revision: str
    payload_sha256: str
    members: dict[str, str]
