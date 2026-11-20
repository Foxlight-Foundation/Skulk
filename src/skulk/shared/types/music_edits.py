"""Closed source-conditioned music requests, independent of private native paths.

Public frame coordinates use the uploaded WAV's declared sample clock. Core
verifies that clock against measured audio before resolving one native 48 kHz
request. Input bytes remain on the addressed media plane, never in these types.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self, final

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from skulk.shared.models.model_cards import (
    ModelCard,
    ModelTask,
    MusicLyricRequirement,
)
from skulk.shared.types.audio_separation import SeparationSource
from skulk.shared.types.native_music import (
    NATIVE_MUSIC_MAX_FRAMES,
    NATIVE_MUSIC_SAMPLE_RATE,
    NativeMusicPart,
    NativeMusicResult,
)

MusicEditMode = Literal["add_part", "repaint", "continue"]
"""Implemented direct-conditioning tasks; native options are adapter-owned."""


class MusicEditContract(BaseModel):
    """Immutable finite metadata at public, command and durable job boundaries."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


class _MusicEditInput(MusicEditContract):
    """Shared closed caption, source-clock and seed metadata beside an uploaded WAV."""

    model: str = Field(
        min_length=1, max_length=256, description="Selected native music model alias."
    )
    prompt: str = Field(
        min_length=1, max_length=8000, description="Creator's original musical caption."
    )
    lyrics: str | None = Field(
        default=None, min_length=1, max_length=20_000,
        description="Original lyrics when the selected card permits them. Omission is instrumental.",
    )
    source_sample_rate: int = Field(
        ge=8000, le=192000,
        description="Clock for request frames; must equal the independently measured WAV rate.",
    )
    seed: int | None = Field(
        default=None, ge=0, le=2**32 - 1,
        description="Optional seed; Core freezes one unsigned 32-bit seed before dispatch.",
    )

    @field_validator("prompt", "lyrics")
    @classmethod
    def nonblank_text(cls, value: str | None) -> str | None:
        """Reject blank input without rewriting the creator's text or line breaks."""
        if value is not None and not value.strip():
            raise ValueError("Music edit text must contain non-whitespace characters")
        return value


@final
class MusicAddPartRequest(_MusicEditInput):
    """Add one native track class, defaulting to the complete uploaded context."""

    task: Literal["add_part"] = Field(description="Direct native LEGO addition.")
    part_class: NativeMusicPart = Field(description="Closed native musical track class.")
    region_start_frame: int = Field(default=0, ge=0, le=192000 * 120, description="Inclusive source-clock frame.")
    region_end_frame: int | None = Field(default=None, ge=1, le=192000 * 120, description="Exclusive source-clock frame; omission selects the complete original.")

    @model_validator(mode="after")
    def nonempty_region(self) -> Self:
        """Reject a reversed explicit range before receiving media."""
        if self.region_end_frame is not None and self.region_start_frame >= self.region_end_frame:
            raise ValueError("Music edit region must be nonempty")
        return self


@final
class MusicRepaintRequest(_MusicEditInput):
    """Rework one explicit half-open source-clock region; preserve context extent."""

    task: Literal["repaint"] = Field(description="Native source-conditioned repaint.")
    region_start_frame: int = Field(default=0, ge=0, le=192000 * 120, description="Inclusive source-clock frame.")
    region_end_frame: int = Field(ge=1, le=192000 * 120, description="Required exclusive source-clock frame.")

    @model_validator(mode="after")
    def nonempty_region(self) -> Self:
        """Refuse empty or reversed repaint ranges before accepting media."""
        if self.region_start_frame >= self.region_end_frame:
            raise ValueError("Music edit region must be nonempty")
        return self


@final
class MusicContinueRequest(_MusicEditInput):
    """Extend after the complete original; total duration uses the source clock."""

    task: Literal["continue"] = Field(description="Native repaint of only the appended tail.")
    target_frames: int = Field(ge=1, le=192000 * 120, description="Total source-clock output extent, including the original context.")

    @model_validator(mode="after")
    def bounded_target(self) -> Self:
        """Refuse a target above the global bound on its declared sample clock."""
        if self.target_frames > self.source_sample_rate * 120:
            raise ValueError("Music edit target exceeds the global 120-second bound")
        return self


MusicEditRequest = Annotated[
    MusicAddPartRequest | MusicRepaintRequest | MusicContinueRequest,
    Field(discriminator="task"),
]
"""Closed public union; task-specific forbidden fields are visible in JSON Schema."""


@final
class MusicEditingTaskParams(MusicEditContract):
    """Resolved immutable work; all edit/output coordinates use native 48 kHz.

    The worker receives original encoded audio separately and independently
    verifies its source manifest. No runtime path or arbitrary native option is
    accepted here. The seed is already fixed and must never be drawn again.
    """

    model: str = Field(min_length=1, max_length=256, description="Exact selected model alias.")
    task: MusicEditMode = Field(description="Admitted source-conditioned operation.")
    prompt: str = Field(min_length=1, max_length=8000, description="Unmodified musical caption.")
    lyrics: str = Field(min_length=1, max_length=20_000, description="Resolved lyrics or instrumental marker.")
    seed: int = Field(ge=0, le=2**32 - 1, description="Once-resolved native seed.")
    source: SeparationSource = Field(description="Independently measured unchanged original WAV.")
    total_input_chunks: int = Field(ge=1, le=4096, description="Addressed input chunks requiring acknowledgement.")
    target_frames: int = Field(ge=NATIVE_MUSIC_SAMPLE_RATE, le=NATIVE_MUSIC_MAX_FRAMES, description="Total native-rate output extent.")
    region_start_frame: int = Field(ge=0, le=NATIVE_MUSIC_MAX_FRAMES, description="Inclusive native-rate edit boundary.")
    region_end_frame: int = Field(ge=1, le=NATIVE_MUSIC_MAX_FRAMES, description="Exclusive native-rate edit boundary.")
    part_class: NativeMusicPart | None = Field(default=None, description="Native part class only for Add a part.")

    @model_validator(mode="after")
    def validate_native_extent(self) -> Self:
        """Preserve context length and distinguish a continuation from an in-place edit."""
        original = _native_end(self.source.frames, self.source.sample_rate)
        if self.region_start_frame >= self.region_end_frame:
            raise ValueError("Resolved music edit region must be nonempty")
        if (self.task == "add_part") != (self.part_class is not None):
            raise ValueError("Resolved music part class disagrees with its task")
        if self.task == "continue":
            if (
                self.target_frames <= original
                or self.region_start_frame != original
                or self.region_end_frame != self.target_frames
            ):
                raise ValueError("Resolved Continue must cover only the complete appended tail")
        elif self.target_frames != original or self.region_end_frame > original:
            raise ValueError("Resolved edit must remain inside its complete original context")
        return self


def _native_end(frames: int, sample_rate: int) -> int:
    return (frames * NATIVE_MUSIC_SAMPLE_RATE + sample_rate - 1) // sample_rate


def resolve_music_edit(
    request: MusicEditRequest,
    source: SeparationSource,
    card: ModelCard,
    *,
    resolved_seed: int,
    total_input_chunks: int,
) -> MusicEditingTaskParams:
    """Validate card, measured source and source-rate range before any dispatch.

    The accepting job owner supplies and persists ``resolved_seed`` once. Starts
    floor and ends ceil onto the native grid, covering the requested source
    interval without floats. No model, package, random generator or I/O is used.
    """
    profile = card.music
    native = profile.native if profile is not None else None
    if (
        card.model_id != request.model or ModelTask.MusicEditing not in card.tasks
        or native is None or request.task not in native.editing_modes
    ):
        raise ValueError("Selected model does not implement this native music edit")
    assert profile is not None
    if source.sample_rate != request.source_sample_rate:
        raise ValueError("Music edit frame clock differs from the measured WAV")
    if request.seed is not None and request.seed != resolved_seed:
        raise ValueError("Resolved music edit seed differs from the caller's explicit seed")
    if profile.lyrics == MusicLyricRequirement.Required and request.lyrics is None:
        raise ValueError("Selected music model requires lyrics")
    if profile.lyrics == MusicLyricRequirement.Unsupported and request.lyrics is not None:
        raise ValueError("Selected music model does not accept lyrics")
    part = request.part_class if request.task == "add_part" else None
    if part is not None and part not in native.part_classes:
        raise ValueError("Selected music model does not implement this native part class")
    original = _native_end(source.frames, source.sample_rate)
    if request.task == "continue":
        target = _native_end(request.target_frames, source.sample_rate)
        if request.target_frames <= source.frames or target <= original:
            raise ValueError("Continue must extend beyond the complete original context")
        start, end = original, target
    else:
        end_source = request.region_end_frame if request.region_end_frame is not None else source.frames
        if request.region_start_frame >= end_source or end_source > source.frames:
            raise ValueError("Music edit region lies outside the uploaded original")
        start = request.region_start_frame * NATIVE_MUSIC_SAMPLE_RATE // source.sample_rate
        end = _native_end(end_source, source.sample_rate)
        target = original
    if target > min(120, profile.max_seconds) * NATIVE_MUSIC_SAMPLE_RATE:
        raise ValueError("Music edit exceeds the selected card's duration bound")
    return MusicEditingTaskParams(
        model=request.model, task=request.task, prompt=request.prompt,
        lyrics=request.lyrics if request.lyrics is not None else "[Instrumental]",
        seed=resolved_seed, source=source, total_input_chunks=total_input_chunks,
        target_frames=target, region_start_frame=start, region_end_frame=end,
        part_class=part,
    )


@final
class MusicEditingOutput(MusicEditContract):
    """Path-free native edit provenance; content is ready only after API custody."""

    source: SeparationSource = Field(description="Verified unmodified original WAV.")
    result: NativeMusicResult = Field(description="Measured complete native result and executed policy.")

    @model_validator(mode="after")
    def validate_source_lineage(self) -> Self:
        """Refuse generation or a foreign original masquerading as a conditioned edit."""
        if self.result.operation == "generate" or self.result.source_sha256 != self.source.sha256:
            raise ValueError("Native music edit output must bind the exact source original")
        original = _native_end(self.source.frames, self.source.sample_rate)
        if (
            self.result.operation == "continue" and self.result.output.frames <= original
            or self.result.operation != "continue" and self.result.output.frames != original
        ):
            raise ValueError("Native edit output differs from its source timebase")
        return self
