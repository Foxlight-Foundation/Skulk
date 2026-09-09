"""Job records for audio-video generation on the owning API node.

A job is the caller-facing view of one ``VideoGeneration`` command. It moves
``queued`` to ``in_progress`` on the first progress frame, and reaches a
terminal state once the render's terminal frame and the finished container
have both arrived (the two travel on different planes and may land in either
order). The registry is bounded and mirrored to a JSON index so a restarted
API still lists recent jobs, with anything that was in flight marked failed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Final, cast

from pydantic import Field

from skulk.shared.types.common import CommandId
from skulk.shared.types.video import (
    VideoGenerationStats,
    VideoJobStatus,
    VideoOutputManifest,
    VideoStage,
)
from skulk.utils.pydantic_ext import CamelCaseModel

MAX_RETAINED_JOBS: Final[int] = 256
"""Oldest terminal jobs are evicted beyond this bound."""


class VideoJob(CamelCaseModel):
    """One audio-video generation as seen by the job API."""

    id: CommandId
    """Job identifier; equal to the generation command id."""
    model: str
    """Card the job was placed on."""
    prompt: str
    """Prompt as submitted."""
    mode: str
    """Resolved generation mode."""
    seconds: int
    """Requested duration."""
    size: str | None = None
    """Requested canvas, if any."""
    status: VideoJobStatus = "queued"
    """Lifecycle state."""
    progress: int = Field(default=0, ge=0, le=100)
    """Approximate completion percentage."""
    stage: VideoStage | None = None
    """Latest reported render phase."""
    created_at: int
    """Creation time, unix seconds."""
    completed_at: int | None = None
    """Terminal time, unix seconds."""
    expires_at: int | None = None
    """When the downloadable artifacts expire."""
    error: str | None = None
    """Sanitized failure detail."""
    output: VideoOutputManifest | None = None
    """Manifest of the finished container once delivered."""
    stats: VideoGenerationStats | None = None
    """Runner timing once the render finished."""
    render_finished: bool = False
    """Whether the render's terminal frame arrived."""
    media_delivered: bool = False
    """Whether the container was received and verified."""

    @property
    def is_terminal(self) -> bool:
        """Whether no further transitions can occur."""
        return self.status in ("completed", "failed", "cancelled")


class VideoJobRegistry:
    """Bounded in-memory job table with a JSON mirror."""

    def __init__(self, index_path: Path | None) -> None:
        self._index_path = index_path
        self._jobs: dict[CommandId, VideoJob] = {}
        self._load()

    def _load(self) -> None:
        if self._index_path is None or not self._index_path.exists():
            return
        try:
            raw = cast("object", json.loads(self._index_path.read_text()))
        except (OSError, ValueError):
            return
        if not isinstance(raw, list):
            return
        for item in cast("list[object]", raw):
            try:
                job = VideoJob.model_validate(item)
            except ValueError:
                continue
            if not job.is_terminal:
                # The process that owned the live stream is gone; the render
                # may still finish on the worker, but nobody can receive it.
                job = job.model_copy(
                    update={
                        "status": "failed",
                        "error": "API node restarted while the job was in flight",
                        "completed_at": int(time.time()),
                    }
                )
            self._jobs[job.id] = job

    def _persist(self) -> None:
        if self._index_path is None:
            return
        try:
            self._index_path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(
                [job.model_dump(mode="json") for job in self._jobs.values()]
            )
            staging = self._index_path.with_suffix(".tmp")
            staging.write_text(payload)
            staging.replace(self._index_path)
        except OSError:
            # The mirror is a convenience for restarts, never the source of
            # truth for a live job; a failed write must not fail the job.
            return

    def create(self, job: VideoJob) -> VideoJob:
        """Register a new job, evicting the oldest terminal jobs beyond the bound."""

        self._jobs[job.id] = job
        overflow = len(self._jobs) - MAX_RETAINED_JOBS
        if overflow > 0:
            terminal = sorted(
                (candidate for candidate in self._jobs.values() if candidate.is_terminal),
                key=lambda candidate: candidate.created_at,
            )
            for stale in terminal[:overflow]:
                self._jobs.pop(stale.id, None)
        self._persist()
        return job

    def get(self, job_id: CommandId) -> VideoJob | None:
        """Return one job."""
        return self._jobs.get(job_id)

    def list(self, *, limit: int = 20, after: CommandId | None = None, descending: bool = True) -> list[VideoJob]:
        """Return jobs ordered by creation, newest first by default."""

        ordered = sorted(
            self._jobs.values(),
            key=lambda job: (job.created_at, str(job.id)),
            reverse=descending,
        )
        if after is not None:
            ids = [job.id for job in ordered]
            if after in ids:
                ordered = ordered[ids.index(after) + 1 :]
        return ordered[:limit]

    def update(self, job_id: CommandId, **changes: object) -> VideoJob | None:
        """Apply field changes to a live job; terminal jobs are immutable."""

        job = self._jobs.get(job_id)
        if job is None or job.is_terminal:
            return job
        updated = job.model_copy(update=changes)
        self._jobs[job_id] = updated
        self._persist()
        return updated

    def delete(self, job_id: CommandId) -> VideoJob | None:
        """Forget a job."""

        job = self._jobs.pop(job_id, None)
        if job is not None:
            self._persist()
        return job

    def settle(self, job_id: CommandId) -> VideoJob | None:
        """Complete a job once both the render terminal and the media arrived."""

        job = self._jobs.get(job_id)
        if job is None or job.is_terminal:
            return job
        if job.render_finished and job.media_delivered:
            now = int(time.time())
            return self.update(
                job_id,
                status="completed",
                progress=100,
                completed_at=now,
            )
        return job

    def fail(self, job_id: CommandId, error: str, *, cancelled: bool = False) -> VideoJob | None:
        """Move a live job to a terminal failure or cancellation."""

        job = self._jobs.get(job_id)
        if job is None or job.is_terminal:
            return job
        return self.update(
            job_id,
            status="cancelled" if cancelled else "failed",
            error=error,
            completed_at=int(time.time()),
        )
