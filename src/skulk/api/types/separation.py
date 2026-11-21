"""Public separation job views omit internal placement and partial receipts."""

from __future__ import annotations

from typing import Literal, final

from pydantic import Field, model_validator

from skulk.api.separation_jobs import SeparationJob, SeparationJobStatus
from skulk.shared.types.audio_separation import (
    AudioSeparationOutput,
    SeparationContract,
    SeparationSource,
)


@final
class SeparationResource(SeparationContract):
    """Status and complete verified output facts for one source-separation job."""

    id: str = Field(
        min_length=1, max_length=128, description="Separation job identifier."
    )
    object: Literal["audio.separation"] = Field(
        default="audio.separation", description="Resource kind."
    )
    model: str = Field(description="Selected separation model card.")
    source: SeparationSource = Field(
        description="Original verified WAV identity and geometry."
    )
    status: SeparationJobStatus = Field(
        description="Queued, in-progress or terminal lifecycle state."
    )
    created_at: int = Field(description="Creation Unix time.")
    completed_at: int | None = Field(description="Terminal Unix time, when available.")
    expires_at: int | None = Field(
        description="Complete-set expiry Unix time, when available."
    )
    error: str | None = Field(description="Bounded failure detail.")
    output: AudioSeparationOutput | None = Field(
        description="All four verified stems; absent before completion."
    )

    @model_validator(mode="after")
    def complete_receipt(self) -> SeparationResource:
        """Refuse public partial results even if a caller bypasses the projector."""
        if (self.status == "completed") != (self.output is not None):
            raise ValueError("only complete separation jobs expose output")
        if self.status == "completed":
            if (
                self.output is None
                or self.output.source != self.source
                or self.completed_at is None
                or self.expires_at is None
                or self.expires_at <= self.completed_at
            ):
                raise ValueError(
                    "complete separation receipt has inconsistent source or retention"
                )
        elif self.expires_at is not None:
            raise ValueError("partial separation jobs have no public content expiry")
        return self


@final
class SeparationListResponse(SeparationContract):
    """A bounded page of this API node's separation jobs, newest first."""

    object: Literal["list"] = Field(default="list", description="Resource list kind.")
    data: tuple[SeparationResource, ...] = Field(
        max_length=64, description="Job resources on this page."
    )
    first_id: str | None = Field(description="First job identity on this page.")
    last_id: str | None = Field(
        description="Last identity, usable as the next page cursor."
    )
    has_more: bool = Field(description="Whether another retained page is available.")


@final
class SeparationDeletedResponse(SeparationContract):
    """Confirmation after terminal job metadata and all owned content are removed."""

    id: str = Field(description="Deleted separation job identity.")
    object: Literal["audio.separation.deleted"] = Field(
        default="audio.separation.deleted", description="Deletion resource kind."
    )
    deleted: Literal[True] = Field(default=True, description="Deletion completed.")


def separation_resource(job: SeparationJob) -> SeparationResource:
    """Project one immutable record without exposing pending manifests or node IDs."""
    completed = job.status == "completed"
    return SeparationResource(
        id=str(job.id),
        model=str(job.model),
        source=job.source,
        status=job.status,
        created_at=job.created_at,
        completed_at=job.completed_at,
        expires_at=job.expires_at if completed else None,
        error=job.error,
        output=job.output,
    )
