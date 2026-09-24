"""Bounded, separate text-to-music command and output contracts."""

from __future__ import annotations

from typing import Literal, final

from pydantic import BaseModel, ConfigDict, Field


@final
class MusicGenerationTaskParams(BaseModel):
    """Publicly validated inputs that travel to one music model instance."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    model: str = Field(min_length=1, description="Mounted text-to-music card alias.")
    prompt: str = Field(min_length=1, max_length=8000, description="Musical description.")
    lyrics: str | None = Field(
        default=None, min_length=1, max_length=20_000,
        description="Caller-authored lyrics when allowed by the card.",
    )
    seconds: int = Field(ge=1, le=120, description="Generation target or budget in seconds.")
    seed: int | None = Field(
        default=None, ge=0, le=2**32 - 1,
        description="Optional unsigned 32-bit generation seed.",
    )


@final
class MusicOutputManifest(BaseModel):
    """Measured WAV metadata sent separately from the output media bytes."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    content_type: Literal["audio/wav"] = "audio/wav"
    size_bytes: int = Field(ge=1, le=64 * 1024 * 1024, description="Exact WAV size.")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$", description="SHA-256 of the WAV.")
    duration_seconds: float = Field(gt=0, description="Actual measured WAV duration.")
    sample_rate: int = Field(ge=1, description="Samples per second in the WAV.")
    channels: int = Field(ge=1, le=2, description="PCM channel count in the WAV.")


MusicJobStatus = Literal["queued", "in_progress", "completed", "failed", "cancelled"]
"""Caller-visible asynchronous music job state."""
