"""Bounded trajectory media, kept outside replicated video task parameters."""

from __future__ import annotations

from typing import Literal, cast, final

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_MOTION_TRACK_BYTES = 4 * 1024 * 1024
"""Maximum encoded trajectory attachment, independently of the video budget."""
MAX_MOTION_TRACKS = 16
"""Maximum simultaneous paths in one motion guide."""


@final
class VideoMotionPoint(BaseModel):
    """One normalized canvas position on one output frame."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    x: float = Field(ge=0.0, le=1.0, description="Horizontal position, left zero to right one.")
    y: float = Field(ge=0.0, le=1.0, description="Vertical position, top zero to bottom one.")


@final
class VideoMotionTrackData(BaseModel):
    """Versioned JSON attachment containing one position per path per frame."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    version: Literal[1] = Field(description="Trajectory media contract version.")
    frame_count: int = Field(ge=9, le=2881, description="Exact requested pixel timeline on the model's 8n+1 frame grid.")
    tracks: tuple[tuple[VideoMotionPoint, ...], ...] = Field(min_length=1, max_length=MAX_MOTION_TRACKS, description="Ordered paths with a normalized position on every output frame; no interpolation or held-frame fallback.")

    @field_validator("version", mode="before")
    @classmethod
    def validate_version(cls, value: object) -> object:
        """Reject boolean coercion at this versioned media boundary."""
        if type(value) is not int:
            raise ValueError("motion track version must be integer one")
        return value

    @field_validator("tracks", mode="before")
    @classmethod
    def coerce_tracks(cls, value: object) -> object:
        """Accept JSON arrays while retaining immutable nested path tuples."""
        if isinstance(value, list):
            return tuple(tuple(cast("list[object]", track)) if isinstance(track, list) else track for track in cast("list[object]", value))
        return value

    @model_validator(mode="after")
    def validate_timeline(self) -> VideoMotionTrackData:
        """Refuse incomplete trajectories before any image batch is allocated."""
        if (self.frame_count - 1) % 8 or any(len(track) != self.frame_count for track in self.tracks):
            raise ValueError("motion tracks must each cover the complete 8n+1 output timeline")
        return self


@final
class VideoMotionTracks(BaseModel):
    """Motion IC-LoRA controls; coordinates arrive as separately verified media."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    adapter_strength: float = Field(default=1.0, gt=0.0, le=2.0, description="Motion Track adapter strength for the single full-resolution sampling stage.")
    guide_strength: float = Field(default=1.0, ge=0.0, le=1.0, description="Attention strength of the rendered track guide.")
