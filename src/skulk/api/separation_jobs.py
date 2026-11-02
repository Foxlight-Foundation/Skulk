"""Durable separation metadata; completion requires one verified four-stem set.

This index never stores audio or paths. The owning API separately verifies and
retains addressed media, then records those facts here. It must gate every
content read on completed status, including reads of individually verified stems.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal, final

from pydantic import Field, model_validator

from skulk.shared.models.model_cards import ModelId
from skulk.shared.types.audio_separation import (
    SEPARATION_STEMS,
    AudioSeparationOutput,
    SeparatedStem,
    SeparationContract,
    SeparationSource,
    SeparationStemName,
)
from skulk.shared.types.common import CommandId, NodeId
from skulk.utils.runtime_files import RuntimeLock, read_private, write_private

MAX_ACTIVE_SEPARATION_JOBS = 4
MAX_RETAINED_SEPARATION_JOBS = 64
MAX_SEPARATION_INDEX_BYTES = 1024 * 1024
SeparationJobStatus = Literal[
    "queued", "in_progress", "completed", "failed", "cancelled"
]


@final
class SeparationJob(SeparationContract):
    """A node-local job record containing facts, never source or result bytes."""

    id: CommandId = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$",
        description="Bounded job/command identity, safe as an owned artifact basename.",
    )
    model: ModelId = Field(
        min_length=1, max_length=256, description="Selected model card."
    )
    source: SeparationSource = Field(
        description="Verified original WAV identity and geometry."
    )
    status: SeparationJobStatus = Field(
        default="queued", description="Current lifecycle."
    )
    created_at: int = Field(ge=0, description="Creation time in Unix seconds.")
    producer_node: NodeId | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        description="Authoritatively placed worker.",
    )
    completed_at: int | None = Field(
        default=None, ge=0, description="Terminal Unix time."
    )
    expires_at: int | None = Field(
        default=None, ge=0, description="Earliest verified stem expiry."
    )
    error: str | None = Field(
        default=None, max_length=1024, description="Bounded terminal failure."
    )
    candidate_output: AudioSeparationOutput | None = Field(
        default=None,
        description="Authoritative complete manifest; not a public content receipt.",
    )
    render_finished: bool = Field(
        default=False, description="Successful terminal render frame received."
    )
    verified_stems: frozenset[SeparationStemName] = Field(
        default=frozenset(),
        max_length=4,
        description="Locally verified and retained artifacts.",
    )

    @property
    def is_terminal(self) -> bool:
        """Return whether later render/media frames must leave this record unchanged."""
        return self.status in ("completed", "failed", "cancelled")

    @property
    def output(self) -> AudioSeparationOutput | None:
        """Expose an output receipt only after all four stems and rendering settle."""
        return self.candidate_output if self.status == "completed" else None

    @model_validator(mode="after")
    def validate_lifecycle(self) -> SeparationJob:
        """Refuse impossible completion, foreign-source and partial-result records."""
        if (
            self.candidate_output is not None
            and self.candidate_output.source != self.source
        ):
            raise ValueError("separation result belongs to a different original")
        if self.verified_stems and self.candidate_output is None:
            raise ValueError("verified stems require the authoritative output manifest")
        if self.render_finished and (
            self.candidate_output is None or self.producer_node is None
        ):
            raise ValueError(
                "render completion requires a placed worker and output manifest"
            )
        if self.is_terminal != (self.completed_at is not None):
            raise ValueError(
                "terminal separation jobs require exactly one completion time"
            )
        if self.status == "completed":
            if (
                not self.render_finished
                or self.verified_stems != frozenset(SEPARATION_STEMS)
                or self.expires_at is None
                or self.completed_at is None
                or self.expires_at <= self.completed_at
                or self.error is not None
            ):
                raise ValueError(
                    "separation completion requires all four live verified stems"
                )
        elif self.status in ("failed", "cancelled"):
            if (
                self.candidate_output is not None
                or self.verified_stems
                or self.expires_at is not None
            ):
                raise ValueError("failed separation jobs cannot expose partial results")
            if self.error is None or self.render_finished:
                raise ValueError(
                    "failed separation jobs require a failure without render success"
                )
        elif self.error is not None:
            raise ValueError("live separation jobs cannot contain terminal failures")
        if self.expires_at is not None and not self.verified_stems:
            raise ValueError("retention expiry requires locally verified content")
        return self


class _Index(SeparationContract):
    """Versioned bounded index; unknown fields and duplicate IDs fail closed."""

    version: Literal[1] = 1
    jobs: tuple[SeparationJob, ...] = Field(max_length=MAX_RETAINED_SEPARATION_JOBS)

    @model_validator(mode="after")
    def unique_jobs(self) -> _Index:
        """Reject duplicate identities rather than silently dropping durable records."""
        if len({job.id for job in self.jobs}) != len(self.jobs):
            raise ValueError("separation index contains duplicate job identities")
        return self


class SeparationJobCapacityError(ValueError):
    """Admission or retained metadata capacity is full; existing content survives."""


class SeparationJobStorageError(RuntimeError):
    """Persistence is uncertain; this index is fenced until reopened by its owner."""


@final
class SeparationJobRegistry:
    """Own one private job index and publish transitions only after durable writes.

    A lifetime OS fence prevents overlapping API owners; an in-process lock also
    serializes synchronous callers. Any write/fsync failure fences this object,
    because replacement may already have happened. Reopen after closing to read
    the actual durable outcome. Existing complete jobs survive restart; every
    interrupted job fails without exposing its partially received content.
    """

    def __init__(
        self, index_path: Path, *, clock: Callable[[], int] = lambda: int(time.time())
    ) -> None:
        """Take exclusive ownership, load bounded records and fail interrupted work."""
        self._path = index_path
        self._clock = clock
        self._mutex = threading.RLock()
        self._fenced = False
        self._closed = False
        self._lock = RuntimeLock(index_path.parent, name="service.lock")
        try:
            try:
                index = _Index.model_validate_json(
                    read_private(index_path, MAX_SEPARATION_INDEX_BYTES)
                )
            except FileNotFoundError:
                index = _Index(jobs=())
            self._jobs = {job.id: job for job in index.jobs}
            recovered = {
                identifier: job
                if job.is_terminal
                else self._failed(job, "API restarted during separation")
                for identifier, job in self._jobs.items()
            }
            if recovered != self._jobs:
                self._commit(recovered)
        except BaseException:
            self._lock.close()
            raise

    def close(self) -> None:
        """Release ownership after the API has stopped its job and media handlers."""
        with self._mutex:
            self._closed = True
            self._lock.close()

    def _check(self) -> None:
        if self._closed or self._fenced:
            raise SeparationJobStorageError(
                "separation index must be reopened before use"
            )

    def _commit(self, jobs: dict[CommandId, SeparationJob]) -> None:
        self._check()
        record = _Index(jobs=tuple(jobs.values())).model_dump_json().encode()
        if len(record) > MAX_SEPARATION_INDEX_BYTES:
            raise SeparationJobCapacityError(
                "separation metadata exceeds its bounded index"
            )
        try:
            write_private(self._path, record)
        except OSError as error:
            # Directory fsync can fail after replacement. Never continue from
            # potentially older in-memory state after an uncertain durable write.
            self._fenced = True
            raise SeparationJobStorageError(
                "could not durably retain separation state; reopen its index"
            ) from error
        self._jobs = jobs

    def _replace(self, job: SeparationJob, **changes: object) -> SeparationJob:
        updated = SeparationJob.model_validate(job.model_dump() | changes)
        self._commit(self._jobs | {job.id: updated})
        return updated

    def _failed(
        self, job: SeparationJob, error: str, *, cancelled: bool = False
    ) -> SeparationJob:
        return SeparationJob.model_validate(
            job.model_dump()
            | {
                "status": "cancelled" if cancelled else "failed",
                "completed_at": self._clock(),
                "error": (error or "separation failed")[:1024],
                "candidate_output": None,
                "render_finished": False,
                "verified_stems": frozenset(),
                "expires_at": None,
            }
        )

    def get(self, job_id: CommandId) -> SeparationJob | None:
        """Read immutable metadata; a missing or terminal job never starts work."""
        with self._mutex:
            self._check()
            return self._jobs.get(job_id)

    def active_ids(self) -> frozenset[CommandId]:
        """Protect in-flight job content from expiry or capacity eviction."""
        with self._mutex:
            self._check()
            return frozenset(
                job.id for job in self._jobs.values() if not job.is_terminal
            )

    def list(
        self, *, limit: int = 20, after: CommandId | None = None
    ) -> tuple[SeparationJob, ...]:
        """Page newest-first with strict bounded limits and retained cursor identity."""
        if not 1 <= limit <= MAX_RETAINED_SEPARATION_JOBS:
            raise ValueError("separation page limit must be between one and 64")
        with self._mutex:
            self._check()
            ordered = sorted(
                self._jobs.values(),
                key=lambda job: (job.created_at, str(job.id)),
                reverse=True,
            )
            if after is not None:
                identifiers = [job.id for job in ordered]
                if after not in identifiers:
                    raise ValueError("separation page cursor is no longer retained")
                ordered = ordered[identifiers.index(after) + 1 :]
            return tuple(ordered[:limit])

    def create(self, job: SeparationJob) -> tuple[CommandId, ...]:
        """Persist a fresh queued job and return only safely evicted terminal IDs.

        Unexpired complete results are never evicted to admit work. An exact
        duplicate is idempotent; another intent using the same ID is refused.
        Capacity refusal leaves every existing record and reservation unchanged.
        """
        if (
            job.status != "queued"
            or job.producer_node is not None
            or job.candidate_output is not None
            or job.verified_stems
        ):
            raise ValueError("only a fresh queued separation job may be created")
        with self._mutex:
            self._check()
            existing = self._jobs.get(job.id)
            if existing is not None:
                if (existing.model, existing.source, existing.created_at) != (
                    job.model,
                    job.source,
                    job.created_at,
                ):
                    raise ValueError(
                        "separation identity already belongs to another intent"
                    )
                return ()
            if (
                sum(not item.is_terminal for item in self._jobs.values())
                >= MAX_ACTIVE_SEPARATION_JOBS
            ):
                raise SeparationJobCapacityError(
                    "four separations are already active on this API"
                )
            jobs = self._jobs.copy()
            evicted: list[CommandId] = []
            candidates = sorted(
                (
                    item
                    for item in jobs.values()
                    if item.is_terminal
                    and (
                        item.status != "completed"
                        or (
                            item.expires_at is not None
                            and item.expires_at <= self._clock()
                        )
                    )
                ),
                key=lambda item: (item.created_at, str(item.id)),
            )
            while len(jobs) >= MAX_RETAINED_SEPARATION_JOBS and candidates:
                stale = candidates.pop(0)
                del jobs[stale.id]
                evicted.append(stale.id)
            if len(jobs) >= MAX_RETAINED_SEPARATION_JOBS:
                raise SeparationJobCapacityError(
                    "delete a retained separation or wait for content expiry"
                )
            jobs[job.id] = job
            self._commit(jobs)
            return tuple(evicted)

    def bind_worker(
        self, job_id: CommandId, producer_node: NodeId
    ) -> SeparationJob | None:
        """Stamp authoritative placement once; a later worker cannot adopt the job."""
        with self._mutex:
            self._check()
            job = self._jobs.get(job_id)
            if job is None or job.is_terminal:
                return job
            if job.producer_node is not None:
                if job.producer_node != producer_node:
                    raise ValueError("separation already belongs to a different worker")
                return job
            return self._replace(job, producer_node=producer_node, status="in_progress")

    def record_manifest(
        self, job_id: CommandId, producer_node: NodeId, output: AudioSeparationOutput
    ) -> SeparationJob | None:
        """Record authoritative source-bound output before accepting media receipts."""
        with self._mutex:
            self._check()
            job = self._jobs.get(job_id)
            if job is None or job.is_terminal:
                return job
            if job.producer_node != producer_node or output.source != job.source:
                raise ValueError(
                    "separation manifest belongs to a different worker or source"
                )
            if job.candidate_output is not None:
                if job.candidate_output != output:
                    raise ValueError("separation manifest changed after declaration")
                return job
            return self._replace(job, candidate_output=output)

    def record_verified_stem(
        self,
        job_id: CommandId,
        producer_node: NodeId,
        stem: SeparatedStem,
        *,
        expires_at: int,
    ) -> SeparationJob | None:
        """Record already-verified local bytes; never expose an individual partial result."""
        with self._mutex:
            self._check()
            job = self._jobs.get(job_id)
            if job is None or job.is_terminal:
                return job
            if job.producer_node != producer_node or job.candidate_output is None:
                raise ValueError(
                    "verified separation media requires its authoritative manifest"
                )
            if stem != next(
                item for item in job.candidate_output.stems if item.name == stem.name
            ):
                raise ValueError("verified separation stem differs from its manifest")
            if expires_at <= self._clock():
                raise ValueError("verified separation content is already expired")
            return self._replace(
                job,
                verified_stems=job.verified_stems | {stem.name},
                expires_at=min(job.expires_at or expires_at, expires_at),
            )

    def finish_render(
        self, job_id: CommandId, producer_node: NodeId
    ) -> SeparationJob | None:
        """Record successful DATA termination; media completion remains independent."""
        with self._mutex:
            self._check()
            job = self._jobs.get(job_id)
            if job is None or job.is_terminal:
                return job
            if job.producer_node != producer_node or job.candidate_output is None:
                raise ValueError(
                    "render completion requires its authoritative worker and manifest"
                )
            if job.render_finished:
                return job
            return self._replace(job, render_finished=True)

    def settle(self, job_id: CommandId) -> SeparationJob | None:
        """Publish completion only after the render and all four verified artifacts."""
        with self._mutex:
            self._check()
            job = self._jobs.get(job_id)
            if (
                job is None
                or job.is_terminal
                or not job.render_finished
                or job.verified_stems != frozenset(SEPARATION_STEMS)
            ):
                return job
            if job.expires_at is None or job.expires_at <= self._clock():
                raise ValueError("separation content expired before complete delivery")
            return self._replace(job, status="completed", completed_at=self._clock())

    def fail(
        self, job_id: CommandId, error: str, *, cancelled: bool = False
    ) -> SeparationJob | None:
        """Persist one terminal failure; late frames cannot revive or replace it."""
        with self._mutex:
            self._check()
            job = self._jobs.get(job_id)
            if job is None or job.is_terminal:
                return job
            failed = self._failed(job, error, cancelled=cancelled)
            self._commit(self._jobs | {job.id: failed})
            return failed

    def invalidate(self, job_id: CommandId, error: str) -> SeparationJob | None:
        """Demote a complete set whose bytes failed startup adoption or verification."""
        with self._mutex:
            self._check()
            job = self._jobs.get(job_id)
            if job is None or job.status != "completed":
                return job
            failed = self._failed(job, error)
            self._commit(self._jobs | {job.id: failed})
            return failed

    def delete(self, job_id: CommandId) -> SeparationJob | None:
        """Forget terminal metadata after cancellation and owned-media cleanup."""
        with self._mutex:
            self._check()
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if not job.is_terminal:
                raise ValueError("cancel an active separation before deleting it")
            jobs = self._jobs.copy()
            del jobs[job_id]
            self._commit(jobs)
            return job
