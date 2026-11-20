"""Public edit resources expose accepted intent and only fully settled output."""

from __future__ import annotations

from typing import Literal, Self, final

from pydantic import Field, model_validator

from skulk.api.music_edit_jobs import MusicEditJob
from skulk.shared.types.audio_separation import SeparationSource
from skulk.shared.types.music import MusicJobStatus
from skulk.shared.types.music_edits import (
    MusicEditContract,
    MusicEditingOutput,
    MusicEditMode,
)
from skulk.shared.types.native_music import NativeMusicPart


@final
class MusicEditResource(MusicEditContract):
    """One API-local immutable edit with explicit native frame coordinates."""

    id: str = Field(
        min_length=1,
        max_length=128,
        description="Accepting node's music edit identity.",
    )
    object: Literal["music.edit"] = Field(
        default="music.edit", description="Resource kind."
    )
    model: str = Field(description="Selected native music model alias.")
    task: MusicEditMode = Field(
        description="Accepted Add a part, Repaint or Continue operation."
    )
    prompt: str = Field(description="Creator's unmodified musical caption.")
    lyrics: str = Field(description="Accepted lyrics or explicit instrumental marker.")
    seed: int = Field(
        ge=0, le=2**32 - 1, description="Fixed seed persisted before dispatch."
    )
    source: SeparationSource = Field(
        description="Measured unchanged original WAV facts."
    )
    frame_sample_rate: Literal[48000] = Field(
        default=48000,
        description="Clock for the resolved frame fields in this resource.",
    )
    target_frames: int = Field(
        description="Complete output extent in resolved 48 kHz frames."
    )
    region_start_frame: int = Field(
        description="Inclusive edit boundary in resolved 48 kHz frames."
    )
    region_end_frame: int = Field(
        description="Exclusive edit boundary in resolved 48 kHz frames."
    )
    part_class: NativeMusicPart | None = Field(
        description="Native track class for Add a part."
    )
    status: MusicJobStatus = Field(
        description="Queued, in-progress or terminal job lifecycle."
    )
    created_at: int = Field(description="Accepted Unix time.")
    completed_at: int | None = Field(description="Terminal Unix time.")
    expires_at: int | None = Field(description="Completed content expiry Unix time.")
    error: str | None = Field(description="Bounded terminal failure detail.")
    output: MusicEditingOutput | None = Field(
        description="Native provenance and verified WAV facts, available only after completion."
    )

    @model_validator(mode="after")
    def complete_receipt(self) -> Self:
        """Refuse partial/foreign output even when the projector is bypassed."""
        if (self.status == "completed") != (self.output is not None):
            raise ValueError("Only completed music edits expose output")
        if self.status == "completed":
            if (
                self.output is None
                or self.output.source != self.source
                or self.output.result.request_id != self.id
                or self.output.result.operation != self.task
                or self.output.result.seed != self.seed
                or self.output.result.output.frames != self.target_frames
                or self.completed_at is None
                or self.expires_at is None
                or self.expires_at <= self.completed_at
            ):
                raise ValueError(
                    "Completed music edit receipt differs from accepted intent or retention"
                )
        elif self.expires_at is not None:
            raise ValueError("Unsettled music edits have no public content expiry")
        return self


@final
class MusicEditListResponse(MusicEditContract):
    """A bounded newest-first page from the accepting API node."""

    object: Literal["list"] = Field(default="list", description="List resource kind.")
    data: tuple[MusicEditResource, ...] = Field(
        max_length=64, description="Committed music edit resources."
    )
    first_id: str | None = Field(description="First resource on this page.")
    last_id: str | None = Field(
        description="Last resource; use as the next-page cursor."
    )
    has_more: bool = Field(description="Whether another retained page is available.")


@final
class MusicEditDeletedResponse(MusicEditContract):
    """Receipt after cancellation and removal of owned metadata/content."""

    id: str = Field(description="Deleted music edit identity.")
    object: Literal["music.edit.deleted"] = Field(
        default="music.edit.deleted", description="Deletion resource kind."
    )
    deleted: Literal[True] = Field(default=True, description="Deletion completed.")


def music_edit_resource(job: MusicEditJob) -> MusicEditResource:
    """Project accepted edit facts without worker identities or partial manifests."""
    params = job.params
    return MusicEditResource(
        id=str(job.id),
        model=params.model,
        task=params.task,
        prompt=params.prompt,
        lyrics=params.lyrics,
        seed=params.seed,
        source=params.source,
        target_frames=params.target_frames,
        region_start_frame=params.region_start_frame,
        region_end_frame=params.region_end_frame,
        part_class=params.part_class,
        status=job.status,
        created_at=job.created_at,
        completed_at=job.completed_at,
        expires_at=job.expires_at if job.status == "completed" else None,
        error=job.error,
        output=job.output,
    )
