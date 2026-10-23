"""Finite native setup jobs owned by Core, with separate software acquisition consent."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import final

from foxlight_model_preservation.peer_install import (
    UnsupportedRuntimePlatform,
    managed_executable,
    runtime_plan,
)
from foxlight_model_preservation.peer_runtime import executable
from foxlight_model_preservation.signing import digest, encode_record, strict_json
from foxlight_model_preservation.storage import atomic_bytes, directory

from skulk.store.peer_install_contracts import (
    RuntimeSetupHistory,
    RuntimeSetupJob,
    RuntimeSetupRecord,
    RuntimeSetupRequest,
    RuntimeSetupView,
)
from skulk.store.peer_install_worker import RuntimeSetupWorkerRequest
from skulk.store.preservation_io import PreservationIO
from skulk.store.preservation_journal import (
    PreservationHostError,
    PreservationJournal,
    safe_json,
)

_ACTIVE = {"queued", "running", "cancelling"}


@final
class RuntimeSetupHost:
    """Keep permanent bounded replay records and drain known children before terminal state."""

    def __init__(
        self,
        root: Path,
        journal: PreservationJournal,
        storage: PreservationIO,
        gate: asyncio.Lock,
        command: Callable[[Path], tuple[str, ...]],
    ) -> None:
        self.root, self.journal, self.storage, self.gate, self.command = (
            root,
            journal,
            storage,
            gate,
            command,
        )
        self.runtime_directory = root / "peer-runtime"
        self._admission = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._closed = False
        self._closing: asyncio.Task[None] | None = None
        for record in self._history().records:
            if record.job.state in _ACTIVE:
                self._update(
                    record.job.model_copy(
                        update={
                            "state": "interrupted",
                            "updated_at": int(time.time()),
                            "error_code": "core_restarted",
                        }
                    )
                )

    def _history(self) -> RuntimeSetupHistory:
        value = self.journal.setting("peer-runtime-history")
        return (
            RuntimeSetupHistory.model_validate_json(safe_json(value))
            if value is not None
            else RuntimeSetupHistory()
        )

    def _save(self, history: RuntimeSetupHistory) -> None:
        self.journal.set_setting(
            "peer-runtime-history", history.model_dump(mode="json")
        )

    def _update(self, job: RuntimeSetupJob) -> None:
        records = tuple(
            record.model_copy(update={"job": job})
            if record.job.job_id == job.job_id
            else record
            for record in self._history().records
        )
        self._save(RuntimeSetupHistory(records=records))

    def job(self, job_id: str, owner_id: str) -> RuntimeSetupJob:
        """Read one owner-scoped receipt without networking or replay."""
        for record in self._history().records:
            if record.job.job_id == job_id and record.request.owner_id == owner_id:
                return record.job
        raise PreservationHostError("peer_runtime_job_not_found")

    async def view(self, owner_id: str) -> RuntimeSetupView:
        """Review packaged native identities and this client's latest local receipt offline."""
        try:
            plan = await self.storage.run(runtime_plan)
        except UnsupportedRuntimePlatform:
            plan = None
        jobs = [
            record.job
            for record in self._history().records
            if record.request.owner_id == owner_id
        ]
        return RuntimeSetupView(
            supported=plan is not None,
            plan=plan,
            plan_id=digest(encode_record(plan)) if plan else None,
            job=jobs[-1] if jobs else None,
        )

    async def start(self, request: RuntimeSetupRequest) -> RuntimeSetupJob:
        """Reserve one exact software grant; retries observe old work and never restart it."""
        if self._closed:
            raise PreservationHostError("preservation_host_closed")
        async with self._admission:
            history = self._history()
            for record in history.records:
                if (record.request.owner_id, record.request.operation_id) == (
                    request.owner_id,
                    request.operation_id,
                ):
                    if record.request != request:
                        raise PreservationHostError("request_changed")
                    return record.job
            if len(history.records) >= 64:
                raise PreservationHostError("peer_runtime_job_capacity_reached")
            if self.gate.locked() or any(
                record.job.state in _ACTIVE for record in history.records
            ):
                raise PreservationHostError("peer_busy_try_again")
            plan = await self.storage.run(runtime_plan)
            if self._closed:
                raise PreservationHostError("preservation_host_closed")
            if request.plan_id != digest(encode_record(plan)):
                raise PreservationHostError("peer_runtime_plan_changed")
            now = int(time.time())
            job = RuntimeSetupJob(
                job_id=digest(
                    (
                        "peer-runtime\x00"
                        + request.owner_id
                        + "\x00"
                        + request.operation_id
                    ).encode()
                ),
                owner_id=request.owner_id,
                state="queued",
                created_at=now,
                updated_at=now,
            )
            self._save(
                RuntimeSetupHistory(
                    records=(
                        *history.records,
                        RuntimeSetupRecord(request=request, job=job),
                    )
                )
            )
            self._task = asyncio.create_task(self._execute(request, job.job_id))
            return job

    async def cancel(self, job_id: str, owner_id: str) -> RuntimeSetupJob:
        """Retain stop intent and await actual child termination before a terminal receipt."""
        job = self.job(job_id, owner_id)
        if job.state in _ACTIVE:
            self._update(
                job.model_copy(
                    update={"state": "cancelling", "updated_at": int(time.time())}
                )
            )
            if self._process is not None:
                await self._stop(self._process)
        return self.job(job_id, owner_id)

    @staticmethod
    async def _stop(process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        if process.stdin is not None:
            with suppress(BrokenPipeError, ConnectionResetError):
                process.stdin.write(b"cancel\n")
                await process.stdin.drain()
        try:
            await asyncio.wait_for(asyncio.shield(process.wait()), 5)
        except TimeoutError:
            with suppress(ProcessLookupError):
                process.terminate()
            try:
                await asyncio.wait_for(asyncio.shield(process.wait()), 2)
            except TimeoutError:
                with suppress(ProcessLookupError):
                    process.kill()
                await process.wait()

    async def _execute(self, request: RuntimeSetupRequest, job_id: str) -> None:
        try:
            async with self.gate:
                await self._run(request, job_id)
        except asyncio.CancelledError:
            current = self.job(job_id, request.owner_id)
            self._update(
                current.model_copy(
                    update={
                        "state": "interrupted",
                        "error_code": "core_restarted",
                        "updated_at": int(time.time()),
                    }
                )
            )

    async def _spawn(self, path: Path) -> asyncio.subprocess.Process:
        # Shield process creation so cancellation cannot abandon a newly
        # returned child before it is assigned to the owning job.
        pending = asyncio.create_task(
            asyncio.create_subprocess_exec(
                *self.command(path),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        )
        cancelled = False
        while True:
            try:
                process = await asyncio.shield(pending)
                break
            except asyncio.CancelledError:
                cancelled = True
        if cancelled:
            await self._stop(process)
            raise asyncio.CancelledError
        return process

    async def _run(self, request: RuntimeSetupRequest, job_id: str) -> None:
        path = self.root / "peer-runtime-request.json"
        terminal = "failed"
        error = "peer_runtime_setup_failed"
        try:
            job = self.job(job_id, request.owner_id)
            if self._closed or job.state == "cancelling":
                terminal, error = "cancelled", "cancelled"
                return
            self._update(
                job.model_copy(
                    update={"state": "running", "updated_at": int(time.time())}
                )
            )
            private_request = RuntimeSetupWorkerRequest(
                runtime_directory=self.runtime_directory.absolute(),
                plan_id=request.plan_id,
            )

            def write() -> None:
                with directory(self.root) as root:
                    atomic_bytes(
                        root, path.name, private_request.model_dump_json().encode()
                    )

            await self.storage.run(write)
            async with asyncio.timeout(330):
                process = await self._spawn(path)
                self._process = process
                if (
                    self._closed
                    or self.job(job_id, request.owner_id).state == "cancelling"
                ):
                    await self._stop(process)
                if process.stdout is None:
                    raise PreservationHostError("peer_runtime_worker_unavailable")
                count = 0
                while line := await process.stdout.readline():
                    count += len(line)
                    if count > 512 * 1024 or len(line) > 1024:
                        raise PreservationHostError("peer_runtime_response_too_large")
                    value = strict_json(line)
                    if isinstance(value, dict) and value.get("event") == "progress":
                        copied = value.get("bytes_copied")
                        if type(copied) is int and 0 <= copied <= 256 * 1024**2:
                            current = self.job(job_id, request.owner_id)
                            self._update(
                                current.model_copy(
                                    update={
                                        "bytes_copied": max(
                                            current.bytes_copied, copied
                                        )
                                    }
                                )
                            )
                await process.wait()
            # Publication may have won a late cancel/EOF race. Actual pinned
            # managed bytes, not a worker's claimed success, decide readiness.
            installed = await self.storage.run(
                lambda: managed_executable(self.runtime_directory)
            )
            if installed is not None:
                await self.storage.run(lambda: executable(installed))
                terminal, error = "succeeded", ""
            elif self.job(job_id, request.owner_id).state == "cancelling":
                terminal, error = "cancelled", "cancelled"
        except asyncio.CancelledError:
            terminal, error = "interrupted", "core_restarted"
        except (ValueError, OSError, TimeoutError, RuntimeError):
            if self._closed:
                terminal, error = "interrupted", "core_restarted"
            elif self.job(job_id, request.owner_id).state == "cancelling":
                terminal, error = "cancelled", "cancelled"
        finally:
            if self._process is not None:
                await self._stop(self._process)
                self._process = None
            await self.storage.run(lambda: path.unlink(missing_ok=True))
            current = self.job(job_id, request.owner_id)
            self._update(
                current.model_copy(
                    update={
                        "state": terminal,
                        "error_code": error or None,
                        "updated_at": int(time.time()),
                    }
                )
            )

    async def close(self) -> None:
        """Drain acquisition before releasing journal/IO ownership; never replay on restart."""
        self._closed = True
        if self._closing is None:
            self._closing = asyncio.create_task(self._drain())
        while not self._closing.done():
            try:
                await asyncio.shield(self._closing)
            except asyncio.CancelledError:
                continue
        self._closing.result()

    async def _drain(self) -> None:
        # Shutdown wins admission before draining the job. In-flight plan IO
        # cannot reserve a new worker after the journal owner starts closing.
        async with self._admission:
            if self._task is not None and not self._task.done():
                self._task.cancel()
                while not self._task.done():
                    try:
                        await asyncio.shield(self._task)
                    except asyncio.CancelledError:
                        continue
                with suppress(asyncio.CancelledError):
                    self._task.result()
            for record in self._history().records:
                if record.job.state in _ACTIVE:
                    self._update(
                        record.job.model_copy(
                            update={
                                "state": "interrupted",
                                "error_code": "core_restarted",
                                "updated_at": int(time.time()),
                            }
                        )
                    )
