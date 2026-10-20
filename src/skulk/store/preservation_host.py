"""Durable preservation orchestration belonging to the actual model-store owner."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Literal, cast, final

from foxlight_model_preservation.models import TrustProfile
from foxlight_model_preservation.signing import create_key
from pydantic import BaseModel, ConfigDict, JsonValue

from skulk.store.installed_cards import read_installed_card
from skulk.store.model_store import ModelStore
from skulk.store.preservation_contracts import (
    PreservationCopy,
    PreservationJob,
    PreservationLocation,
    PreservationModel,
    PreservationPage,
    PreservationPlan,
    PreservationPlanRequest,
    PreservationStart,
)
from skulk.store.preservation_export import plan_preservation, preservation_generation
from skulk.store.preservation_journal import (
    PreservationHostError,
    PreservationJournal,
    safe_json,
    secure_directory,
)
from skulk.store.preservation_worker import WorkerRequest


class _Location(BaseModel):
    """Private enrollment; path and custody never cross the operator API."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    location_id: str
    label: str
    path: Path
    device: int
    inode: int
    trust: TrustProfile
    key_path: Path


class _Observation(BaseModel):
    """Bounded safe protocol from an exclusively owned known byte worker."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    event: Literal["progress", "expected", "result", "error"]
    stage: str | None = None
    bytes_hashed: int = 0
    bytes_copied: int = 0
    manifest_id: str | None = None
    size_bytes: int | None = None
    member_count: int | None = None
    audience: Literal["private", "public"] | None = None
    error_code: str | None = None


def preservation_worker_command(request: Path) -> tuple[str, ...]:
    """Launch the known source or frozen worker without choosing arbitrary code."""
    if cast(bool, getattr(sys, "frozen", False)):
        return (sys.executable, "_preservation-worker", str(request.absolute()))
    return (
        sys.executable,
        "-m",
        "skulk.store.preservation_worker",
        str(request.absolute()),
    )


@final
class PreservationHost:
    """One journal and queue, independent of browser and capability generations.

    Construct only around the authoritative running ModelStore. Copy work runs
    in owned subprocesses; the same store publication fence excludes eviction,
    downloads and peer imports for a Preserve source lease.
    """

    def __init__(self, store: ModelStore, root: Path) -> None:
        self.store = store
        self.root = root
        self.journal = PreservationJournal(root)
        self.journal.interrupt_active()
        self._queue = asyncio.Lock()
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._closed = False
        self._setup_lock = asyncio.Lock()

    def _location(self, location_id: str) -> _Location:
        value = self.journal.setting("location:" + location_id)
        if value is None:
            raise PreservationHostError("archive_not_configured")
        return _Location.model_validate_json(safe_json(value))

    @asynccontextmanager
    async def _archive(self, location: _Location) -> AsyncIterator[int]:
        try:
            descriptor = os.open(
                location.path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            )
        except OSError as error:
            raise PreservationHostError("archive_unavailable") from error
        try:
            info = os.fstat(descriptor)
            if (info.st_dev, info.st_ino) != (location.device, location.inode):
                raise PreservationHostError("archive_identity_changed")
            yield descriptor
        finally:
            os.close(descriptor)

    async def setup_device(self) -> PreservationLocation:
        """Explicitly enroll the built-in private archive with protected capture custody."""
        async with self._setup_lock:
            if self._closed:
                raise PreservationHostError("preservation_host_closed")
            if self.journal.setting("location:device") is None:
                archive = self.root / "device-archive"
                secure_directory(archive)
                custody = self.root / "custody"
                secure_directory(custody)
                key = custody / "capture.pem"
                # Existing unjournaled keys are not overwritten or regenerated.
                if key.exists():
                    raise PreservationHostError("capture_identity_needs_reconciliation")
                trust = create_key(key, None)
                info = archive.stat()
                location = _Location(
                    location_id="device",
                    label="This device",
                    path=archive.absolute(),
                    device=info.st_dev,
                    inode=info.st_ino,
                    trust=trust,
                    key_path=key.absolute(),
                )
                self.journal.set_setting(
                    "location:device",
                    cast(dict[str, JsonValue], location.model_dump(mode="json")),
                )
            return await self.location_view("device")

    async def location_view(self, location_id: str) -> PreservationLocation:
        """Observe actual enrolled storage without creating any archive pathname."""
        location = self._location(location_id)
        available = False
        free: int | None = None
        try:
            async with self._archive(location) as descriptor:
                free = shutil.disk_usage(descriptor).free
                available = True
        except (OSError, PreservationHostError):
            pass
        return PreservationLocation(
            location_id=location.location_id,
            label=location.label,
            purpose="private",
            reachable=available,
            free_bytes=free,
            publisher_key_id=location.trust.keys[0].key_id,
        )

    def _selected(self, model_id: str) -> Path:
        path = self.store.get_store_path(model_id)
        if path is None:
            raise PreservationHostError("installed_model_unavailable")
        return path.absolute()

    async def page(self, offset: int = 0) -> PreservationPage:
        """Read one bounded installed/library/activity page; never submit byte work."""
        if not 0 <= offset <= 2**31:
            raise PreservationHostError("invalid_offset")
        locations = (
            ()
            if self.journal.setting("location:device") is None
            else (await self.location_view("device"),)
        )
        models: list[PreservationModel] = []
        entries = self.store.list_models()[offset : offset + 5]
        for entry in entries[:4]:
            path = self.store.get_store_path(entry.model_id)
            record = read_installed_card(path) if path is not None else None
            if record is None or record.artifact_role != "base":
                continue
            try:
                generation = (
                    preservation_generation(
                        path, artifact_roots=(self.store.store_path,)
                    )
                    if path is not None
                    else ""
                )
            except (ValueError, OSError):
                continue
            models.append(
                PreservationModel(
                    model_id=entry.model_id,
                    generation_id=generation,
                    size_bytes=entry.total_bytes,
                    verification=record.verification,
                )
            )
        copies = self.journal.copies(offset)[:4]
        jobs = self.journal.jobs(offset)[:4]
        more = (
            len(entries) > 4
            or len(self.journal.copies(offset + 4)) > 0
            or len(self.journal.jobs(offset + 4)) > 0
        )
        return PreservationPage(
            locations=locations,
            models=tuple(models),
            copies=copies,
            jobs=jobs,
            next_offset=offset + 4 if more else None,
        )

    async def plan(self, request: PreservationPlanRequest) -> PreservationPlan:
        """Bind a no-byte-write plan to the exact source generation and enrolled root."""
        if self._closed:
            raise PreservationHostError("preservation_host_closed")
        if request.kind not in {"preserve", "check"}:
            raise PreservationHostError("operation_not_available_in_this_host_version")
        location = self._location(request.location_id)
        async with self._archive(location):
            pass
        payload: dict[str, JsonValue] = {
            "location": cast(dict[str, JsonValue], location.model_dump(mode="json"))
        }
        manifest: str | None = None
        copy_id = request.copy_id
        if request.kind == "preserve":
            if request.model_id is None or request.generation_id is None:
                raise PreservationHostError("exact_installed_selection_required")
            async with self.store.preservation_lease():
                path = self._selected(request.model_id)
                generation = preservation_generation(
                    path, artifact_roots=(self.store.store_path,)
                )
                if generation != request.generation_id:
                    raise PreservationHostError("installed_selection_changed")
                selection = plan_preservation(
                    path, artifact_roots=(self.store.store_path,)
                )
                payload.update(
                    {"model_directory": str(path), "generation_id": generation}
                )
                model_id, size, members = (
                    request.model_id,
                    selection.source_bytes,
                    selection.artifacts,
                )
        else:
            if copy_id is None:
                raise PreservationHostError("selected_copy_required")
            try:
                copy = self.journal.copy(copy_id)
                if copy.location_id != request.location_id:
                    raise PreservationHostError("copy_location_changed")
                manifest, model_id, size = (
                    copy.manifest_id,
                    copy.model_id,
                    copy.size_bytes,
                )
                members = (model_id,)
            except PreservationHostError as error:
                if error.code != "copy_not_found":
                    raise
                previous = self.journal.job(copy_id)
                prior_plan, _ = self.journal.plan(previous.plan_id)
                if (
                    previous.state != "interrupted"
                    or previous.expected_manifest_id is None
                    or prior_plan.location_id != request.location_id
                ):
                    raise PreservationHostError("no_reconcilable_outcome") from error
                manifest, model_id, size = (
                    previous.expected_manifest_id,
                    previous.model_id,
                    prior_plan.size_bytes,
                )
                members = prior_plan.members
            payload["manifest_id"] = manifest
        if len(members) > 16:
            raise PreservationHostError("selection_needs_paged_review")
        expires = int(time.time()) + 600
        identity = hashlib.sha256(
            safe_json(
                {
                    "request": request.model_dump(mode="json"),
                    "payload": payload,
                    "expires": expires,
                }
            ).encode()
        ).hexdigest()
        plan = PreservationPlan(
            plan_id=identity,
            kind=request.kind,
            model_id=model_id,
            location_id=request.location_id,
            copy_id=copy_id,
            manifest_id=manifest,
            generation_id=request.generation_id,
            size_bytes=size,
            members=members[:16],
            member_count=len(members),
            expires_at=expires,
        )
        if (
            len(json.dumps(plan.model_dump(mode="json"), ensure_ascii=True).encode())
            > 48000
        ):
            raise PreservationHostError("plan_summary_too_large")
        self.journal.add_plan(plan, payload)
        return plan

    def start(self, request: PreservationStart) -> PreservationJob:
        """Reserve exact idempotency and return promptly; the host owns completion."""
        if self._closed:
            raise PreservationHostError("preservation_host_closed")
        job, created = self.journal.reserve(request)
        if created:
            self._tasks[job.job_id] = asyncio.create_task(self._execute(job.job_id))
        return job

    async def cancel(self, job_id: str) -> PreservationJob:
        """Persist stop intent; only an observed stopped process confirms cancellation."""
        job = self.journal.job(job_id)
        if job.state not in {"queued", "running", "cancelling"}:
            return job
        job = job.model_copy(
            update={"state": "cancelling", "updated_at": int(time.time())}
        )
        self.journal.update_job(job)
        process = self._processes.get(job_id)
        if process is not None and process.stdin is not None:
            with suppress(BrokenPipeError, ConnectionResetError):
                process.stdin.write(b"cancel\n")
                await process.stdin.drain()
            try:
                await asyncio.wait_for(asyncio.shield(process.wait()), 5)
            except TimeoutError:
                process.terminate()
                try:
                    await asyncio.wait_for(asyncio.shield(process.wait()), 2)
                except TimeoutError:
                    process.kill()
                    await process.wait()
        return self.journal.job(job_id)

    async def close(self) -> None:
        """Drain owned workers before releasing the exclusive host journal owner."""
        self._closed = True
        for job_id in tuple(self._tasks):
            await self.cancel(job_id)
        if self._tasks:
            await asyncio.gather(*tuple(self._tasks.values()), return_exceptions=True)
        self.journal.close()

    async def _execute(self, job_id: str) -> None:
        try:
            async with self._queue:
                job = self.journal.job(job_id)
                if job.state == "cancelling":
                    self._finish(job_id, "cancelled", "cancelled")
                    return
                self.journal.update_job(
                    job.model_copy(
                        update={
                            "state": "running",
                            "stage": "starting",
                            "attempts": job.attempts + 1,
                        }
                    )
                )
                async with self.store.preservation_lease():
                    await self._run(job_id)
        except Exception:
            job = self.journal.job(job_id)
            if job.state in {"queued", "running", "cancelling"}:
                self._finish(
                    job_id,
                    "interrupted" if job.expected_manifest_id else "failed",
                    "worker_unavailable",
                )
        finally:
            self._tasks.pop(job_id, None)

    def _finish(
        self,
        job_id: str,
        state: Literal["failed", "cancelled", "interrupted"],
        code: str,
    ) -> None:
        job = self.journal.job(job_id)
        self.journal.update_job(
            job.model_copy(
                update={
                    "state": state,
                    "stage": "check_outcome" if state == "interrupted" else state,
                    "error_code": code,
                    "updated_at": int(time.time()),
                    "finished_at": int(time.time()),
                }
            )
        )

    async def _run(self, job_id: str) -> None:
        job = self.journal.job(job_id)
        if job.state == "cancelling":
            self._finish(job_id, "cancelled", "cancelled")
            return
        plan, payload = self.journal.plan(job.plan_id)
        location = _Location.model_validate_json(safe_json(payload["location"]))
        work = self.root / "work" / job_id
        secure_directory(work)
        async with self._archive(location) as descriptor:
            model_path = (
                self._selected(plan.model_id) if plan.kind == "preserve" else None
            )
            if model_path is not None and preservation_generation(
                model_path, artifact_roots=(self.store.store_path,)
            ) != payload.get("generation_id"):
                raise PreservationHostError("installed_selection_changed")
            request = WorkerRequest(
                kind=cast(Literal["preserve", "check"], plan.kind),
                archive_descriptor=descriptor,
                archive_device=location.device,
                archive_inode=location.inode,
                model_directory=model_path,
                source_root=self.store.store_path.absolute(),
                generation_id=cast(str | None, payload.get("generation_id")),
                snapshot_parent=work.absolute(),
                key_path=location.key_path,
                trust=location.trust,
                manifest_id=plan.manifest_id,
            )
            request_path = work / "request.json"
            fd = os.open(
                request_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
                0o600,
            )
            with os.fdopen(fd, "w") as stream:
                _ = stream.write(request.model_dump_json())
                stream.flush()
                os.fsync(stream.fileno())
            process = await asyncio.create_subprocess_exec(
                *preservation_worker_command(request_path),
                pass_fds=(descriptor,),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                limit=32768,
            )
            self._processes[job_id] = process
            if (
                self.journal.job(job_id).state == "cancelling"
                and process.stdin is not None
            ):
                process.stdin.write(b"cancel\n")
                await process.stdin.drain()
            result: _Observation | None = None
            error_code = "worker_interrupted"
            try:
                if process.stdout is None or process.stdin is None:
                    raise PreservationHostError("worker_control_unavailable")
                while data := await process.stdout.readline():
                    if len(data) > 16384:
                        raise PreservationHostError("worker_observation_too_large")
                    observation = _Observation.model_validate_json(data)
                    current = self.journal.job(job_id)
                    if observation.event == "expected":
                        if observation.manifest_id is None:
                            raise PreservationHostError("invalid_expected_outcome")
                        self.journal.update_job(
                            current.model_copy(
                                update={
                                    "expected_manifest_id": observation.manifest_id,
                                    "updated_at": int(time.time()),
                                }
                            )
                        )
                        process.stdin.write(
                            ("publish:" + observation.manifest_id + "\n").encode()
                        )
                        await process.stdin.drain()
                    elif observation.event == "progress":
                        self.journal.update_job(
                            current.model_copy(
                                update={
                                    "stage": observation.stage or "working",
                                    "updated_at": int(time.time()),
                                    "bytes_hashed": max(
                                        current.bytes_hashed, observation.bytes_hashed
                                    ),
                                    "bytes_copied": max(
                                        current.bytes_copied, observation.bytes_copied
                                    ),
                                }
                            )
                        )
                    elif observation.event == "result":
                        result = observation
                    else:
                        error_code = observation.error_code or "operation_failed"
                exit_code = await process.wait()
                current = self.journal.job(job_id)
                if (
                    exit_code == 0
                    and result is not None
                    and result.manifest_id is not None
                ):
                    expected = (
                        current.expected_manifest_id
                        if plan.kind == "preserve"
                        else plan.manifest_id
                    )
                    if (
                        result.manifest_id != expected
                        or result.size_bytes is None
                        or result.member_count is None
                        or result.audience != "private"
                    ):
                        raise PreservationHostError("worker_outcome_mismatch")
                    now = int(time.time())
                    copy_id = hashlib.sha256(
                        (plan.location_id + result.manifest_id).encode()
                    ).hexdigest()
                    self.journal.add_copy(
                        PreservationCopy(
                            copy_id=copy_id,
                            model_id=plan.model_id,
                            location_id=plan.location_id,
                            manifest_id=result.manifest_id,
                            audience="private",
                            size_bytes=result.size_bytes,
                            created_at=now,
                            checked_at=now,
                            member_count=result.member_count,
                        )
                    )
                    self.journal.update_job(
                        current.model_copy(
                            update={
                                "state": "succeeded",
                                "stage": "complete",
                                "copy_id": copy_id,
                                "updated_at": now,
                                "finished_at": now,
                                "result": {
                                    "manifest_id": result.manifest_id,
                                    "copy_id": copy_id,
                                },
                            }
                        )
                    )
                else:
                    state = (
                        "interrupted"
                        if current.expected_manifest_id is not None
                        else "cancelled"
                        if current.state == "cancelling" or error_code == "cancelled"
                        else "failed"
                    )
                    self._finish(job_id, state, error_code)
            finally:
                if process.stdin is not None:
                    process.stdin.close()
                if process.returncode is None:
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), 2)
                    except TimeoutError:
                        process.kill()
                        await process.wait()
                self._processes.pop(job_id, None)
                request_path.unlink(missing_ok=True)
                shutil.rmtree(work)
