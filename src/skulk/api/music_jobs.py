"""Bounded, restart-safe job records for text-to-music generation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import cast, final

from pydantic import BaseModel, ConfigDict, Field

from skulk.shared.types.common import CommandId
from skulk.shared.types.music import MusicJobStatus, MusicOutputManifest

MAX_ACTIVE_MUSIC_JOBS = 32
MAX_RETAINED_MUSIC_JOBS = 256


@final
class MusicJob(BaseModel):
    """One asynchronous generation and its verified output facts."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    id: CommandId = Field(description="Job and command identifier.")
    model: str = Field(description="Selected music card.")
    prompt: str = Field(description="Submitted musical description.")
    seconds: int = Field(description="Requested generation target or budget.")
    status: MusicJobStatus = Field(description="Current job lifecycle state.")
    created_at: int = Field(description="Creation time in Unix seconds.")
    completed_at: int | None = Field(default=None, description="Terminal time in Unix seconds.")
    expires_at: int | None = Field(default=None, description="Stored WAV expiration time.")
    error: str | None = Field(default=None, description="Bounded failure detail.")
    output: MusicOutputManifest | None = Field(default=None, description="Verified WAV facts.")
    render_finished: bool = Field(default=False, description="Whether the terminal DATA frame arrived.")
    media_delivered: bool = Field(default=False, description="Whether the WAV was verified and stored.")

    @property
    def is_terminal(self) -> bool:
        """Whether the job can no longer change state."""
        return self.status in ("completed", "failed", "cancelled")


@final
class MusicJobRegistry:
    """Persist a small local job index and fail interrupted jobs on restart."""

    def __init__(self, index_path: Path) -> None:
        self._index_path = index_path
        self._jobs: dict[CommandId, MusicJob] = {}
        self._load()

    def _load(self) -> None:
        if not self._index_path.is_file():
            return
        try:
            raw = cast("object", json.loads(self._index_path.read_text()))
        except (OSError, ValueError):
            return
        if not isinstance(raw, list):
            return
        for item in cast("list[object]", raw):
            try:
                job = MusicJob.model_validate(item)
            except ValueError:
                continue
            if not job.is_terminal:
                job = job.model_copy(update={
                    "status": "failed",
                    "error": "API node restarted while the job was in flight",
                    "completed_at": int(time.time()),
                })
            self._jobs[job.id] = job

    def _persist(self) -> None:
        try:
            self._index_path.parent.mkdir(parents=True, exist_ok=True)
            staging = self._index_path.with_suffix(".tmp")
            staging.write_text(json.dumps([job.model_dump(mode="json") for job in self._jobs.values()]))
            staging.replace(self._index_path)
        except OSError:
            # Disk failure must not erase the live in-memory outcome.
            return

    def create(self, job: MusicJob) -> MusicJob:
        """Add a job, evicting only oldest terminal records when bounded."""
        self._jobs[job.id] = job
        overflow = len(self._jobs) - MAX_RETAINED_MUSIC_JOBS
        if overflow > 0:
            terminal = sorted(
                (candidate for candidate in self._jobs.values() if candidate.is_terminal),
                key=lambda candidate: candidate.created_at,
            )
            for stale in terminal[:overflow]:
                self._jobs.pop(stale.id, None)
        self._persist()
        return job

    def get(self, job_id: CommandId) -> MusicJob | None:
        """Return one job or none."""
        return self._jobs.get(job_id)

    def list(self, *, limit: int = 20, after: CommandId | None = None) -> list[MusicJob]:
        """Page newest first through retained jobs."""
        ordered = sorted(self._jobs.values(), key=lambda job: (job.created_at, str(job.id)), reverse=True)
        if after is not None:
            ids = [job.id for job in ordered]
            if after in ids:
                ordered = ordered[ids.index(after) + 1:]
        return ordered[:limit]

    def active_count(self) -> int:
        """Count jobs still consuming admission."""
        return sum(not job.is_terminal for job in self._jobs.values())

    def active_ids(self) -> frozenset[CommandId]:
        """Return jobs the output store must not evict."""
        return frozenset(job.id for job in self._jobs.values() if not job.is_terminal)

    def update(self, job_id: CommandId, **changes: object) -> MusicJob | None:
        """Apply a transition to a live job, then persist it."""
        job = self._jobs.get(job_id)
        if job is None or job.is_terminal:
            return job
        updated = job.model_copy(update=changes)
        self._jobs[job_id] = updated
        self._persist()
        return updated

    def settle(self, job_id: CommandId) -> MusicJob | None:
        """Complete only after the terminal frame and verified WAV both arrive."""
        job = self._jobs.get(job_id)
        if job is not None and not job.is_terminal and job.render_finished and job.media_delivered:
            return self.update(job_id, status="completed", completed_at=int(time.time()))
        return job

    def fail(self, job_id: CommandId, error: str, *, cancelled: bool = False) -> MusicJob | None:
        """End a live job after failure or cancellation."""
        return self.update(
            job_id,
            status="cancelled" if cancelled else "failed",
            error=error[:1024],
            completed_at=int(time.time()),
        )

    def invalidate(self, job_id: CommandId, error: str) -> None:
        """Demote completed output that did not survive process restart."""
        job = self._jobs.get(job_id)
        if job is None:
            return
        self._jobs[job_id] = job.model_copy(update={"status": "failed", "error": error, "expires_at": None})
        self._persist()

    def mark_expired(self, job_id: CommandId) -> None:
        """Mark content evicted from the bounded store."""
        job = self._jobs.get(job_id)
        if job is None or job.status != "completed":
            return
        self._jobs[job_id] = job.model_copy(update={"expires_at": int(time.time())})
        self._persist()

    def delete(self, job_id: CommandId) -> MusicJob | None:
        """Forget one job record."""
        job = self._jobs.pop(job_id, None)
        if job is not None:
            self._persist()
        return job
