"""Durable source-bound music editing with independent render and WAV custody.

This registry shares the separation service's private atomic-file ownership and
uncertain-write fence. It never stores WAV bytes, starts inference, retries a
submission, or treats a native completion frame as delivered content.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Self, final

from pydantic import Field, model_validator

from skulk.api.media_job_storage import PrivateMediaJobFile
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.music import MusicJobStatus
from skulk.shared.types.music_edits import (
    MusicEditContract,
    MusicEditingOutput,
    MusicEditingTaskParams,
)
from skulk.utils.runtime_files import write_private

MAX_ACTIVE_MUSIC_EDITS = 4
MAX_RETAINED_MUSIC_EDITS = 64
MAX_MUSIC_EDIT_INDEX_BYTES = 4 * 1024**2


@final
class MusicEditJob(MusicEditContract):
    """One accepted immutable native edit and its independently verified outcome."""

    id: CommandId = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$", description="Node-local accepted command/job identity.")
    params: MusicEditingTaskParams = Field(description="Frozen source, caption, lyrics, seed and native-grid extent.")
    status: MusicJobStatus = Field(default="queued", description="Current asynchronous lifecycle.")
    created_at: int = Field(ge=0, description="Accepted Unix time.")
    producer_node: NodeId | None = Field(default=None, min_length=1, max_length=256, description="One authoritative placed worker.")
    completed_at: int | None = Field(default=None, ge=0, description="Terminal Unix time.")
    expires_at: int | None = Field(default=None, ge=0, description="Verified local WAV expiry; absent before custody.")
    error: str | None = Field(default=None, max_length=1024, description="Bounded terminal failure without raw engine diagnostics.")
    candidate_output: MusicEditingOutput | None = Field(default=None, description="Authoritative native metadata; not a public content receipt.")
    render_finished: bool = Field(default=False, description="Successful render DATA termination observed.")
    verified_output: bool = Field(default=False, description="Complete local WAV independently verified and synced.")
    return_acknowledged: bool = Field(default=False, description="Reverse acceptance control has drained after local custody.")

    @property
    def is_terminal(self) -> bool:
        """Return whether later native/media frames must leave this outcome unchanged."""
        return self.status in {"completed", "failed", "cancelled"}

    @property
    def output(self) -> MusicEditingOutput | None:
        """Expose result facts only after render, byte custody and return acknowledgement."""
        return self.candidate_output if self.status == "completed" else None

    @model_validator(mode="after")
    def validate_lifecycle(self) -> Self:
        """Refuse impossible terminals and results for another source, seed or extent."""
        output = self.candidate_output
        if output is not None and (
            output.result.request_id != str(self.id)
            or output.source != self.params.source
            or output.result.operation != self.params.task
            or output.result.seed != self.params.seed
            or output.result.output.frames != self.params.target_frames
        ):
            raise ValueError("Music edit result differs from the accepted immutable input")
        if output is not None and self.producer_node is None:
            raise ValueError("Music edit manifest requires an authoritative placed worker")
        if self.status == "queued" and self.producer_node is not None or self.status == "in_progress" and self.producer_node is None:
            raise ValueError("Music edit placement must agree with its live status")
        if (self.render_finished or self.verified_output) and (output is None or self.producer_node is None):
            raise ValueError("Music edit completion facts require a placed worker and manifest")
        if (self.expires_at is not None) != self.verified_output:
            raise ValueError("Music edit retention expiry requires exactly one verified WAV")
        if self.return_acknowledged and not self.verified_output:
            raise ValueError("Music edit return acknowledgement requires verified local bytes")
        if self.is_terminal != (self.completed_at is not None):
            raise ValueError("Terminal music edits require exactly one completion time")
        if self.status == "completed":
            if (
                not self.render_finished or not self.verified_output
                or not self.return_acknowledged or self.expires_at is None
                or self.completed_at is None or self.expires_at <= self.completed_at
                or self.error is not None
            ):
                raise ValueError("Completed music edits require rendering, live local WAV and drained return acknowledgement")
        elif self.status in {"failed", "cancelled"}:
            if (
                output is not None or self.render_finished or self.verified_output
                or self.return_acknowledged or self.expires_at is not None
                or self.error is None
            ):
                raise ValueError("Failed music edits cannot expose partial output")
        elif self.error is not None:
            raise ValueError("Active music edits cannot contain a terminal failure")
        return self


class _Index(MusicEditContract):
    """Strict bounded edit index; no bytes, arbitrary fields or duplicate identities."""

    version: Literal[1] = 1
    jobs: tuple[MusicEditJob, ...] = Field(max_length=MAX_RETAINED_MUSIC_EDITS)

    @model_validator(mode="after")
    def unique_jobs(self) -> Self:
        """Refuse duplicate identities rather than dropping a retained result."""
        if len({job.id for job in self.jobs}) != len(self.jobs):
            raise ValueError("Music edit index contains duplicate identities")
        return self


class MusicEditCapacityError(ValueError):
    """The bounded active, retained or encoded-metadata budget is full."""


class MusicEditStorageError(RuntimeError):
    """Durable write outcome is uncertain; reopen before using further job state."""


@final
class MusicEditJobRegistry:
    """Publish typed edit transitions only after shared durable storage commits.

    Restart fails interrupted work without resubmission. Existing completed
    records retain their exact source/seed/result, subject to separate local
    content adoption before any API download. One OS fence excludes other owners.
    """

    def __init__(
        self, path: Path, *, clock: Callable[[], int] = lambda: int(time.time()),
        write: Callable[[Path, bytes], None] = write_private,
    ) -> None:
        """Read one private index and durably fail interrupted jobs without replay."""
        self.clock = clock
        self._mutex = threading.RLock()
        self._file = PrivateMediaJobFile(
            path, limit=MAX_MUSIC_EDIT_INDEX_BYTES, kind="music edit",
            capacity_error=MusicEditCapacityError, storage_error=MusicEditStorageError,
            write=write,
        )
        try:
            retained = self._file.read()
            index = _Index(jobs=()) if retained is None else _Index.model_validate_json(retained)
            self._jobs = {job.id: job for job in index.jobs}
            recovered = {
                identifier: job if job.is_terminal else self._failed(job, "API restarted during music editing")
                for identifier, job in self._jobs.items()
            }
            if recovered != self._jobs:
                self._commit(recovered)
        except BaseException:
            self._file.close()
            raise

    def close(self) -> None:
        """Release the file fence after active API and media effects have stopped."""
        with self._mutex:
            self._file.close()

    def _commit(self, jobs: dict[CommandId, MusicEditJob]) -> None:
        self._file.replace(_Index(jobs=tuple(jobs.values())).model_dump_json().encode())
        self._jobs = jobs

    def _replace(self, job: MusicEditJob, **changes: object) -> MusicEditJob:
        updated = MusicEditJob.model_validate(job.model_dump() | changes)
        self._commit(self._jobs | {job.id: updated})
        return updated

    def _failed(self, job: MusicEditJob, error: str, *, cancelled: bool = False) -> MusicEditJob:
        return MusicEditJob.model_validate(job.model_dump() | {
            "status": "cancelled" if cancelled else "failed", "completed_at": self.clock(),
            "error": (error or "Music edit failed")[:1024], "candidate_output": None,
            "render_finished": False, "verified_output": False,
            "return_acknowledged": False, "expires_at": None,
        })

    def get(self, identifier: CommandId) -> MusicEditJob | None:
        """Read immutable metadata without starting model, media or recovery work."""
        with self._mutex:
            self._file.check()
            return self._jobs.get(identifier)

    def active_ids(self) -> frozenset[CommandId]:
        """Protect accepted in-flight jobs from media expiry and eviction."""
        with self._mutex:
            self._file.check()
            return frozenset(job.id for job in self._jobs.values() if not job.is_terminal)

    def list(self, *, limit: int = 20, after: CommandId | None = None) -> tuple[MusicEditJob, ...]:
        """Page newest-first with a bounded count and retained last-ID cursor."""
        if not 1 <= limit <= MAX_RETAINED_MUSIC_EDITS:
            raise ValueError("Music edit page limit must be between one and 64")
        with self._mutex:
            self._file.check()
            ordered = sorted(self._jobs.values(), key=lambda job: (job.created_at, str(job.id)), reverse=True)
            if after is not None:
                identifiers = [job.id for job in ordered]
                if after not in identifiers:
                    raise ValueError("Music edit page cursor is no longer retained")
                ordered = ordered[identifiers.index(after) + 1:]
            return tuple(ordered[:limit])

    def create(self, job: MusicEditJob) -> tuple[CommandId, ...]:
        """Retain an exact queued input; only expired/failed terminal jobs may evict."""
        if job.status != "queued" or job.producer_node is not None or job.candidate_output is not None:
            raise ValueError("Only a fresh queued music edit may be created")
        with self._mutex:
            self._file.check()
            existing = self._jobs.get(job.id)
            if existing is not None:
                if (existing.params, existing.created_at) != (job.params, job.created_at):
                    raise ValueError("Music edit identity already belongs to another immutable input")
                return ()
            if len(self.active_ids()) >= MAX_ACTIVE_MUSIC_EDITS:
                raise MusicEditCapacityError("Four music edits are already active on this API")
            jobs = self._jobs.copy()
            evicted: list[CommandId] = []
            candidates = sorted((item for item in jobs.values() if item.is_terminal and (
                item.status != "completed" or item.expires_at is not None and item.expires_at <= self.clock()
            )), key=lambda item: (item.created_at, str(item.id)))
            while len(jobs) >= MAX_RETAINED_MUSIC_EDITS and candidates:
                stale = candidates.pop(0)
                del jobs[stale.id]
                evicted.append(stale.id)
            if len(jobs) >= MAX_RETAINED_MUSIC_EDITS:
                raise MusicEditCapacityError("Delete a retained music edit or wait for content expiry")
            jobs[job.id] = job
            self._commit(jobs)
            return tuple(evicted)

    def bind_worker(self, identifier: CommandId, producer: NodeId) -> MusicEditJob | None:
        """Stamp one authoritative worker; another producer cannot adopt this job."""
        with self._mutex:
            self._file.check()
            job = self._jobs.get(identifier)
            if job is None or job.is_terminal:
                return job
            if job.producer_node is not None:
                if job.producer_node != producer:
                    raise ValueError("Music edit is bound to another worker")
                return job
            return self._replace(job, status="in_progress", producer_node=producer)

    def record_manifest(self, identifier: CommandId, producer: NodeId, output: MusicEditingOutput) -> MusicEditJob | None:
        """Freeze authoritative native provenance before accepting result media."""
        with self._mutex:
            self._file.check()
            job = self._jobs.get(identifier)
            if job is None or job.is_terminal:
                return job
            if job.producer_node != producer:
                raise ValueError("Music edit manifest belongs to another worker")
            if job.candidate_output is not None:
                if job.candidate_output != output:
                    raise ValueError("Music edit manifest changed after declaration")
                return job
            return self._replace(job, candidate_output=output)

    def record_verified_output(self, identifier: CommandId, producer: NodeId, output: MusicEditingOutput, *, expires_at: int) -> MusicEditJob | None:
        """Record independently checked and synced local bytes, never a partial WAV."""
        with self._mutex:
            self._file.check()
            job = self._jobs.get(identifier)
            if job is None or job.is_terminal:
                return job
            if job.producer_node != producer or job.candidate_output != output:
                raise ValueError("Verified music edit media differs from its authoritative manifest")
            if expires_at <= self.clock():
                raise ValueError("Verified music edit content is already expired")
            return self._replace(job, verified_output=True, expires_at=min(job.expires_at or expires_at, expires_at))

    def _mark(self, identifier: CommandId, producer: NodeId, *, acknowledged: bool) -> MusicEditJob | None:
        with self._mutex:
            self._file.check()
            job = self._jobs.get(identifier)
            if job is None or job.is_terminal:
                return job
            if job.producer_node != producer or job.candidate_output is None:
                raise ValueError("Music edit completion requires its authoritative worker and manifest")
            return self._replace(job, **({"return_acknowledged": True} if acknowledged else {"render_finished": True}))

    def finish_render(self, identifier: CommandId, producer: NodeId) -> MusicEditJob | None:
        """Record successful DATA termination independently of result delivery."""
        return self._mark(identifier, producer, acknowledged=False)

    def acknowledge_return(self, identifier: CommandId, producer: NodeId) -> MusicEditJob | None:
        """Record drained reverse acceptance control after independently verified custody."""
        return self._mark(identifier, producer, acknowledged=True)

    def settle(self, identifier: CommandId) -> MusicEditJob | None:
        """Complete only after rendering, verified local WAV and return-path drain."""
        with self._mutex:
            self._file.check()
            job = self._jobs.get(identifier)
            if job is None or job.is_terminal or not (job.render_finished and job.verified_output and job.return_acknowledged):
                return job
            if job.expires_at is None or job.expires_at <= self.clock():
                raise ValueError("Music edit content expired before complete delivery")
            return self._replace(job, status="completed", completed_at=self.clock())

    def fail(self, identifier: CommandId, error: str, *, cancelled: bool = False) -> MusicEditJob | None:
        """Retain one terminal failure; late frames cannot revive or overwrite it."""
        with self._mutex:
            self._file.check()
            job = self._jobs.get(identifier)
            if job is None or job.is_terminal:
                return job
            failed = self._failed(job, error, cancelled=cancelled)
            self._commit(self._jobs | {identifier: failed})
            return failed

    def invalidate(self, identifier: CommandId, error: str) -> MusicEditJob | None:
        """Demote completed metadata whose content failed independent startup adoption."""
        with self._mutex:
            self._file.check()
            job = self._jobs.get(identifier)
            if job is None or job.status != "completed":
                return job
            failed = self._failed(job, error)
            self._commit(self._jobs | {identifier: failed})
            return failed

    def delete(self, identifier: CommandId) -> MusicEditJob | None:
        """Forget terminal metadata after cancellation and owned media cleanup."""
        with self._mutex:
            self._file.check()
            job = self._jobs.get(identifier)
            if job is None:
                return None
            if not job.is_terminal:
                raise ValueError("Cancel an active music edit before deleting it")
            jobs = self._jobs.copy()
            del jobs[identifier]
            self._commit(jobs)
            return job
