"""One managed public mirror under the actual preservation owner's lifetime."""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import threading
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager, suppress
from pathlib import Path
from typing import final

from foxlight_model_preservation.mirror_models import HeldModel, MirrorState
from foxlight_model_preservation.mirror_recovery import (
    inspect_selection,
    selection_status,
)
from foxlight_model_preservation.mirror_state import (
    MirrorLedger,
    policy_id,
    restart_allowed,
)
from foxlight_model_preservation.operations import failure_code
from foxlight_model_preservation.storage import atomic_bytes, directory
from pydantic import BaseModel, ConfigDict

from skulk.store.mirror_contracts import (
    MirrorConfiguration,
    MirrorJob,
    MirrorModel,
    MirrorPage,
    MirrorRequest,
    MirrorReview,
    MirrorSelection,
)
from skulk.store.mirror_kits import KitSeal, MirrorKitStore, kit_fingerprint
from skulk.store.mirror_process import MirrorProcess, remove_mirror_request
from skulk.store.mirror_worker import (
    MirrorKitTarget,
    MirrorNativeRuntime,
    MirrorRetainSource,
    MirrorWorkerEvent,
    MirrorWorkerRequest,
)
from skulk.store.preservation_delivery import publication_members
from skulk.store.preservation_io import PreservationIO
from skulk.store.preservation_journal import (
    PreservationHostError,
    PreservationJournal,
    safe_json,
    secure_directory,
)

SourceProvider = Callable[[str, str], AbstractAsyncContextManager[MirrorRetainSource]]
RuntimeProvider = Callable[[str], Awaitable[MirrorNativeRuntime]]


class _Enrollment(BaseModel):
    """Protected filesystem identity, never an operator path or network grant."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    identity: tuple[int, int]


@final
class MirrorHost:
    """Persistent controller and bounded byte jobs; construction starts no sharing."""

    def __init__(
        self,
        root: Path,
        journal: PreservationJournal,
        storage: PreservationIO,
        command: Callable[[Path], tuple[str, ...]],
        source: SourceProvider,
        runtime: RuntimeProvider,
    ) -> None:
        self.root, self.journal, self.storage = root, journal, storage
        self.command, self.source, self.runtime = command, source, runtime
        self.path = root / "managed-mirror"
        self.requests = root / "mirror-requests"
        secure_directory(self.requests)
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._processes: dict[str, MirrorProcess] = {}
        self._cancellations: dict[str, asyncio.Task[MirrorJob]] = {}
        self._closing: asyncio.Task[None] | None = None
        self._closed = False
        self._restored = False
        self.kits = MirrorKitStore(root, journal)
        self._downloads: set[asyncio.Task[object]] = set()
        self._kit_removals: dict[str, asyncio.Task[MirrorJob]] = {}
        self._labels: dict[str, tuple[str, int, int]] = {}
        self._label_lock = threading.RLock()
        journal.interrupt_mirror_jobs()

    def _enrollment(self) -> _Enrollment | None:
        value = self.journal.setting("mirror-root")
        return (
            _Enrollment.model_validate_json(safe_json(value))
            if value is not None
            else None
        )

    def _ledger(self) -> MirrorLedger:
        selected = self._enrollment()
        if selected is None:
            raise PreservationHostError("mirror_not_configured")
        ledger = MirrorLedger(self.path)
        if ledger.read().identity["."] != selected.identity:
            raise PreservationHostError("mirror_directory_changed")
        return ledger

    def _configuration(self, state: MirrorState) -> MirrorConfiguration | None:
        value = self.journal.setting("mirror-configuration")
        old = (
            MirrorConfiguration.model_validate_json(safe_json(value))
            if value is not None
            else None
        )
        native = state.native_policy
        ipfs = bool(native and native.ipfs_binary)
        if ipfs and (old is None or old.peer_setup_id is None):
            return None
        return MirrorConfiguration(
            name=state.name,
            disk_bytes=state.disk_bytes,
            ingress_daily=state.ingress_daily,
            egress_daily=state.egress_daily,
            egress_monthly=state.egress_monthly,
            http=state.http_enabled,
            exposure="network" if state.bind == "0.0.0.0" else "loopback",
            port=state.port,
            ipfs=ipfs,
            torrent=native.torrent if native else False,
            peer_setup_id=old.peer_setup_id if old and ipfs else None,
            max_active=native.max_active if native else 2,
            session_seconds=native.session_seconds if native else 3600,
            session_bytes=native.session_bytes if native else 64 * 1024**2,
            upload_rate=native.upload_rate if native else 1024**2,
        )

    def _view(self, state: MirrorState, offset: int) -> MirrorPage:
        selections = sorted(state.models.items())
        models = tuple(
            self._model(identity, held)
            for identity, held in selections[offset : offset + 20]
        )
        return MirrorPage(
            configured=True,
            available=True,
            policy_id=policy_id(state),
            configuration=self._configuration(state),
            paused=state.paused or self.journal.setting("mirror-stopped") is True,
            restart_expires=state.restart.expires if state.restart else None,
            restart_allowed=restart_allowed(state),
            models=models,
            model_count=len(selections),
            next_offset=offset + 20 if offset + 20 < len(selections) else None,
        )

    def _model(self, identity: str, held: HeldModel) -> MirrorModel:
        with self._label_lock:
            if held.state == "removed":
                self._labels.pop(identity, None)
            metadata = self._labels.get(identity)
        return MirrorModel(
            manifest_id=identity,
            state=held.state,
            pinned=held.pinned,
            blocked=held.blocked,
            error_code=held.error
            if held.error is None or re.fullmatch(r"[a-z_]{1,80}", held.error)
            else "invalid_or_unsafe",
            name=metadata[0] if metadata is not None else None,
            member_count=metadata[1] if metadata is not None else None,
            size_bytes=metadata[2] if metadata is not None else None,
        )

    def _load_labels(self, state: MirrorState, offset: int) -> None:
        ledger = self._ledger()
        for identity, held in sorted(state.models.items())[offset : offset + 20]:
            with self._label_lock:
                if held.state == "removed":
                    self._labels.pop(identity, None)
                if held.state == "removed" or identity in self._labels:
                    continue
            try:
                metadata = inspect_selection(ledger, identity)
            except (OSError, ValueError):
                continue
            with self._label_lock:
                if len(self._labels) < 1000:
                    self._labels[identity] = (
                        metadata.name,
                        len(metadata.members),
                        metadata.total_bytes,
                    )

    async def selection(self, manifest_id: str) -> MirrorSelection:
        """Observe an exact selection without pagination, IO replay or authority changes."""
        configured = self._enrollment() is not None
        fallback = MirrorSelection(
            manifest_id=manifest_id, configured=configured, available=not configured
        )
        if not configured:
            return fallback

        def observe() -> MirrorSelection:
            ledger = self._ledger()
            held = selection_status(ledger, manifest_id)
            return MirrorSelection(
                manifest_id=manifest_id,
                configured=True,
                available=True,
                policy_id=policy_id(ledger.read()),
                model=self._model(manifest_id, held) if held is not None else None,
            )

        try:
            return await self.storage.probe(
                "mirror-selection:" + manifest_id, observe, fallback
            )
        except (OSError, ValueError):
            return fallback

    async def review(self, manifest_id: str) -> MirrorReview:
        """Authenticate complete bounded terms locally without tensor reads or signing."""

        def inspect() -> MirrorReview:
            ledger = self._ledger()
            manifest = inspect_selection(ledger, manifest_id)
            held = selection_status(ledger, manifest_id)
            if held is None or held.state == "removed":
                raise PreservationHostError("mirror_selection_changed")
            trust = held.grant.trust
            result = MirrorReview(
                manifest_id=manifest_id,
                name=manifest.name,
                size_bytes=manifest.total_bytes,
                workloads=manifest.workloads,
                runtime_requirements=manifest.runtime_requirements,
                members=publication_members(manifest, private_capture=False),
                capture_key_ids=tuple(
                    key.key_id
                    for key in trust.keys
                    if key.key_id not in trust.revoked_keys
                ),
                capture_threshold=trust.threshold,
            )
            if (
                len(
                    json.dumps(
                        result.model_dump(mode="json"), ensure_ascii=True
                    ).encode()
                )
                > 48000
            ):
                raise PreservationHostError("metadata_preview_too_large")
            with self._label_lock:
                if len(self._labels) < 1000:
                    self._labels[manifest_id] = (
                        manifest.name,
                        len(manifest.members),
                        manifest.total_bytes,
                    )
            return result

        return await self.storage.run(inspect)

    async def page(self, offset: int = 0) -> MirrorPage:
        """Read bounded local metadata and owned activity without starting work."""
        if not 0 <= offset <= 1000:
            raise PreservationHostError("invalid_offset")
        selected = self._enrollment() is not None
        fallback = MirrorPage(configured=selected, available=not selected)
        try:

            def snapshot() -> MirrorState | None:
                return self._ledger().read()

            state = (
                await self.storage.probe("mirror-state", snapshot, None)
                if selected
                else None
            )
            if state is not None:
                # Labels contain only previously authenticated immutable payload
                # facts. A bounded/coalesced probe fills the small cache even if
                # this request times out; a slow large card cannot hide custody
                # state forever or enqueue repeated unbounded metadata reads.
                await self.storage.probe(
                    "mirror-labels", lambda: self._load_labels(state, offset), None
                )
            page = self._view(state, offset) if state is not None else fallback
        except (ValueError, OSError):
            page = fallback
        for job_id in self._tasks.keys() | self._processes.keys():
            job, _ = self.journal.mirror_job(job_id)
            if job.kind != "run":
                continue
            owned = self._processes.get(job_id)
            activity = job.activity if owned is not None and owned.alive() else None
            return page.model_copy(
                update={"controller_job_id": job_id, "activity": activity}
            )
        return page

    def job(self, job_id: str) -> MirrorJob:
        """Observe one durable receipt; historical ports never become live again."""
        job, _ = self.journal.mirror_job(job_id)
        process = self._processes.get(job_id)
        if (
            job.activity is not None
            and job.activity.running
            and (process is None or not process.alive())
        ):
            return job.model_copy(update={"activity": None})
        return job

    @asynccontextmanager
    async def kit_file(self, job_id: str) -> AsyncIterator[tuple[int, KitSeal]]:
        """Lease an unchanged prepared file until response/disconnect cleanup finishes."""
        if self._closed:
            raise PreservationHostError("preservation_host_closed")
        job = self.job(job_id)
        if (
            job.kind != "prepare_kit"
            or job.state != "succeeded"
            or job.kit is None
            or job.kit_removed
        ):
            raise PreservationHostError("mirror_kit_not_ready")

        sealed: list[KitSeal] = []

        def open_prepared() -> int:
            descriptor, seal = self.kits.open(job_id)
            if seal.public() != job.kit:
                os.close(descriptor)
                raise PreservationHostError("mirror_kit_generation_changed")
            sealed.append(seal)
            return descriptor

        owner = asyncio.current_task()
        if owner is None:
            raise PreservationHostError("mirror_kit_owner_unavailable")
        self._downloads.add(owner)
        try:
            async with self.storage.directory(open_prepared) as descriptor:
                yield descriptor, sealed[0]
        finally:
            self._downloads.discard(owner)

    async def kit_chunk(
        self, descriptor: int, seal: KitSeal, offset: int, size: int
    ) -> bytes:
        """Read one bounded unchanged range from an already leased descriptor."""
        if (
            offset < 0
            or not 1 <= size <= 1024**2
            or offset + size > seal.fingerprint.size_bytes
        ):
            raise PreservationHostError("invalid_kit_range")

        def read() -> bytes:
            if kit_fingerprint(descriptor) != seal.fingerprint:
                raise PreservationHostError("mirror_kit_file_changed")
            result = os.pread(descriptor, size, offset)
            if len(result) != size or kit_fingerprint(descriptor) != seal.fingerprint:
                raise PreservationHostError("mirror_kit_file_changed")
            return result

        return await self.storage.run(read)

    async def remove_kit(self, job_id: str, confirmation: str) -> MirrorJob:
        """Own explicit generation removal through actual IO completion, never model deletion."""
        if confirmation != job_id:
            raise PreservationHostError("mirror_kit_confirmation_required")
        if self._closed:
            raise PreservationHostError("preservation_host_closed")
        job = self.job(job_id)
        if (
            job.kind != "prepare_kit"
            or job_id in self._tasks
            or job_id in self._processes
            or job_id in self._cancellations
        ):
            raise PreservationHostError("mirror_kit_busy_or_unknown")
        existing = self._kit_removals.get(job_id)
        if existing is None:
            existing = asyncio.create_task(self._remove_kit(job_id, confirmation))
            self._kit_removals[job_id] = existing
            existing.add_done_callback(self._observe_task)
        return await self._await_control(existing)

    async def _remove_kit(self, job_id: str, confirmation: str) -> MirrorJob:
        try:
            artifact = await self.storage.run(
                lambda: self.kits.remove(job_id, confirmation)
            )
            job, _ = self.journal.mirror_job(job_id)
            result = job.model_copy(
                update={
                    "kit": artifact,
                    "kit_removed": True,
                    "updated_at": int(time.time()),
                }
            )
            self.journal.update_mirror_job(result)
            return result
        finally:
            self._kit_removals.pop(job_id, None)

    async def start(self, decision: MirrorRequest) -> MirrorJob:
        """Reserve once and own the known operation independently of its client."""
        if self._closed:
            raise PreservationHostError("preservation_host_closed")
        job, created = self.journal.reserve_mirror(decision)
        if not created:
            return self.job(job.job_id)
        active = [self.journal.mirror_job(identity)[0] for identity in self._tasks]
        compatible: set[str] = (
            {"run", "prepare_kit"}
            if decision.kind == "model"
            else {"run"}
            if decision.kind == "prepare_kit"
            else {"prepare_kit"}
            if decision.kind == "run"
            else set()
        )
        if self._cancellations or any(other.kind not in compatible for other in active):
            failed = job.model_copy(
                update={
                    "state": "failed",
                    "error_code": "mirror_busy_stop_first",
                    "updated_at": int(time.time()),
                }
            )
            self.journal.update_mirror_job(failed)
            return failed
        self._tasks[job.job_id] = asyncio.create_task(self._execute(job.job_id))
        self._tasks[job.job_id].add_done_callback(self._observe_task)
        return job

    @staticmethod
    def _observe_task[T](task: asyncio.Task[T]) -> None:
        if not task.cancelled():
            task.exception()

    def _checkpoint(self, job_id: str) -> None:
        if self._closed or self.journal.mirror_job(job_id)[0].state == "cancelling":
            raise asyncio.CancelledError

    async def _execute(self, job_id: str) -> None:
        job, decision = self.journal.mirror_job(job_id)
        error: str | None = "mirror_worker_failed"
        result: MirrorWorkerEvent | None = None
        completed = False
        try:
            if job.state == "cancelling" or self._closed:
                raise asyncio.CancelledError
            selected = self._enrollment()
            if selected is not None:
                current = await self.storage.run(lambda: self._ledger().read())
                if policy_id(current) != decision.policy_id:
                    raise PreservationHostError("mirror_policy_changed")
            self._checkpoint(job_id)
            if decision.kind == "configure":
                if selected is not None and decision.policy_id is None:
                    raise PreservationHostError("mirror_policy_review_required")
                self.journal.set_setting("mirror-approved-resume", None)
                self.journal.set_setting("mirror-running-intent", None)
                self.journal.set_setting("mirror-stopped", True)
            elif selected is None:
                raise PreservationHostError("mirror_not_configured")
            runtime = None
            if decision.configuration is not None and decision.configuration.ipfs:
                setup = decision.configuration.peer_setup_id
                if setup is None:
                    raise PreservationHostError("peer_setup_required")
                runtime = await self.runtime(setup)
            self._checkpoint(job_id)
            if decision.kind == "run":
                await self._run_intent(job_id, decision)
            self.journal.update_mirror_job(
                job.model_copy(
                    update={"state": "running", "updated_at": int(time.time())}
                )
            )
            worker = MirrorWorkerRequest(
                mirror_path=self.path,
                mirror_identity=selected.identity if selected else None,
                decision=decision,
                runtime=runtime,
            )
            if decision.kind == "retain":
                if decision.copy_id is None or decision.manifest_id is None:
                    raise PreservationHostError("public_copy_required")
                self.journal.set_setting("mirror-approved-resume", None)
                async with self.source(
                    decision.copy_id, decision.manifest_id
                ) as source:
                    result, error = await self._worker(
                        job_id, worker.model_copy(update={"source": source})
                    )
            elif decision.kind == "prepare_kit":
                if decision.manifest_id is None:
                    raise PreservationHostError("mirror_selection_required")
                manifest_id = decision.manifest_id
                await self.storage.run(lambda: self.kits.prepare(job_id, manifest_id))
                async with self.storage.directory(
                    lambda: self.kits.target(job_id)
                ) as target:
                    info = os.fstat(target)
                    result, error = await self._worker(
                        job_id,
                        worker.model_copy(
                            update={
                                "kit_target": MirrorKitTarget(
                                    descriptor=target,
                                    device=info.st_dev,
                                    inode=info.st_ino,
                                )
                            }
                        ),
                    )
            else:
                result, error = await self._worker(job_id, worker)
            if result is None:
                error = error or "mirror_worker_failed"
            elif decision.kind == "configure":
                if result.mirror_identity is None:
                    raise PreservationHostError("mirror_worker_protocol")
                self.journal.set_setting(
                    "mirror-root",
                    _Enrollment(identity=result.mirror_identity).model_dump(
                        mode="json"
                    ),
                )
                if decision.configuration is not None:
                    self.journal.set_setting(
                        "mirror-configuration",
                        decision.configuration.model_dump(mode="json"),
                    )
            elif decision.kind == "grant_resume":
                self.journal.set_setting(
                    "mirror-approved-resume",
                    {
                        "policy_id": decision.policy_id,
                        "expires": decision.resume_expires,
                    },
                )
            elif decision.kind == "revoke_resume":
                self.journal.set_setting("mirror-approved-resume", None)
                self.journal.set_setting("mirror-running-intent", None)
            elif decision.kind == "prepare_kit" and error is None:
                if (
                    result.kit_seal is None
                    or result.manifest_id != decision.manifest_id
                ):
                    raise PreservationHostError("mirror_worker_protocol")
                sealed = result.kit_seal
                artifact = await self.storage.run(
                    lambda: self.kits.record(job_id, sealed)
                )
                current, _ = self.journal.mirror_job(job_id)
                self.journal.update_mirror_job(
                    current.model_copy(update={"kit": artifact})
                )
            completed = result is not None and error is None
        except asyncio.CancelledError:
            error = "core_stopped" if self._closed else "cancelled"
        except (ValueError, OSError) as failure:
            error = (
                failure.code
                if isinstance(failure, PreservationHostError)
                else failure_code(failure)
            )
        finally:
            if not completed and error is None:
                error = "mirror_worker_failed"
            current, _ = self.journal.mirror_job(job_id)
            if job_id not in self._processes:
                state = (
                    "interrupted"
                    if self._closed
                    else "cancelled"
                    if current.state == "cancelling" or error == "cancelled"
                    else "failed"
                    if error
                    else "succeeded"
                )
                self.journal.update_mirror_job(
                    current.model_copy(
                        update={
                            "state": state,
                            "activity": None,
                            "error_code": error,
                            "updated_at": int(time.time()),
                            "kit": current.kit if state == "succeeded" else None,
                        }
                    )
                )
                self._tasks.pop(job_id, None)
                if decision.kind == "run" and not self._closed:
                    self.journal.set_setting("mirror-running-intent", None)
                    self.journal.set_setting("mirror-stopped", True)
            else:
                # A real group-drain failure retains ownership and blocks new
                # control. Never convert it into a successful stopped receipt.
                self.journal.update_mirror_job(
                    current.model_copy(
                        update={
                            "state": "cancelling",
                            "error_code": "mirror_stop_incomplete",
                            "updated_at": int(time.time()),
                        }
                    )
                )

    async def _run_intent(self, job_id: str, decision: MirrorRequest) -> None:
        state = await self.storage.run(lambda: self._ledger().read())
        self._checkpoint(job_id)
        if policy_id(state) != decision.policy_id:
            raise PreservationHostError("mirror_policy_changed")
        if decision.mode == "resume":
            approved = self.journal.setting("mirror-approved-resume")
            if approved != {
                "policy_id": decision.policy_id,
                "expires": state.restart.expires if state.restart else None,
            }:
                raise PreservationHostError("mirror_resume_permission_required")
            if not restart_allowed(state.model_copy(update={"paused": False})):
                raise PreservationHostError("mirror_resume_permission_required")
            self.journal.set_setting(
                "mirror-running-intent",
                {
                    "job_id": job_id,
                    "policy_id": decision.policy_id,
                    "expires": state.restart.expires if state.restart else None,
                },
            )
        else:
            self.journal.set_setting("mirror-running-intent", None)
        self.journal.set_setting("mirror-stopped", False)

    async def _worker(
        self, job_id: str, request: MirrorWorkerRequest
    ) -> tuple[MirrorWorkerEvent | None, str | None]:
        path = self.requests / (job_id + ".json")

        def save() -> None:
            with directory(self.requests) as parent:
                atomic_bytes(parent, path.name, request.model_dump_json().encode())

        await self.storage.run(save)
        process = await MirrorProcess.start(
            self.command(path),
            request.source.descriptor
            if request.source
            else request.kit_target.descriptor
            if request.kit_target
            else None,
            self.storage,
        )
        self._processes[job_id] = process
        result: MirrorWorkerEvent | None = None
        error: str | None = None

        def receive(event: MirrorWorkerEvent) -> None:
            nonlocal result, error
            job, _ = self.journal.mirror_job(job_id)
            if event.event == "error":
                error = event.error_code or "mirror_worker_failed"
            elif event.event == "result":
                if result is not None:
                    raise PreservationHostError("mirror_worker_protocol")
                result = event
            elif event.event in {"ready", "activity"}:
                if request.decision.kind != "run" or event.activity is None:
                    raise PreservationHostError("mirror_worker_protocol")
                self.journal.update_mirror_job(
                    job.model_copy(
                        update={
                            "activity": event.activity,
                            "updated_at": int(time.time()),
                        }
                    )
                )
            else:
                self.journal.update_mirror_job(
                    job.model_copy(
                        update={
                            "bytes_hashed": max(job.bytes_hashed, event.bytes_hashed),
                            "bytes_copied": max(job.bytes_copied, event.bytes_copied),
                            "bytes_received": max(
                                job.bytes_received, event.bytes_received
                            ),
                            "updated_at": int(time.time()),
                        }
                    )
                )

        try:
            await process.observations(receive)
        finally:
            try:
                await process.stop()
            finally:
                await self._release_process(job_id, process)
        if process.process.returncode != 0:
            error = error or "mirror_worker_failed"
        return result, error

    async def _release_process(self, job_id: str, process: MirrorProcess) -> None:
        """Release only this drained owner, after private request cleanup ends."""
        if (
            process.process.returncode is None
            or self._processes.get(job_id) is not process
        ):
            return
        await remove_mirror_request(self.requests / (job_id + ".json"), self.storage)
        if self._processes.get(job_id) is process:
            self._processes.pop(job_id, None)

    async def cancel(self, job_id: str) -> MirrorJob:
        """Record Stop before signalling; return only after actual worker drain."""
        job, _ = self.journal.mirror_job(job_id)
        if job.state not in {"queued", "running", "cancelling"}:
            return self.job(job_id)
        operation = self._cancellations.get(job_id)
        if operation is None:
            self.journal.update_mirror_job(
                job.model_copy(
                    update={"state": "cancelling", "updated_at": int(time.time())}
                )
            )
            if job.kind == "run":
                self.journal.set_setting("mirror-running-intent", None)
                self.journal.set_setting("mirror-stopped", True)
            operation = asyncio.create_task(self._cancel(job_id))
            self._cancellations[job_id] = operation
            operation.add_done_callback(lambda _: self._cancellations.pop(job_id, None))
        return await self._await_control(operation)

    @staticmethod
    async def _await_control[T](operation: asyncio.Task[T]) -> T:
        interrupted = False
        while not operation.done():
            try:
                await asyncio.shield(operation)
            except asyncio.CancelledError:
                interrupted = True
        result = operation.result()
        if interrupted:
            raise asyncio.CancelledError
        return result

    async def _cancel(self, job_id: str) -> MirrorJob:
        job, _ = self.journal.mirror_job(job_id)
        if job.state in {"queued", "running", "cancelling"}:
            self.journal.update_mirror_job(
                job.model_copy(
                    update={"state": "cancelling", "updated_at": int(time.time())}
                )
            )
            task = self._tasks.get(job_id)
            process = self._processes.get(job_id)
            if process is not None:
                await process.stop()
            elif task is not None:
                task.cancel()
            if task is not None:
                with suppress(asyncio.CancelledError):
                    try:
                        await asyncio.shield(task)
                    except subprocess.TimeoutExpired:
                        # A retained task can have failed an earlier drain.
                        # Its historical timeout is resolved only after this
                        # exact owned process has now successfully drained.
                        if process is None or process.process.returncode is None:
                            raise
                if task.done() and process is not None:
                    await self._release_process(job_id, process)
                if task.done() and job_id not in self._processes:
                    current, _ = self.journal.mirror_job(job_id)
                    if current.state == "cancelling":
                        self.journal.update_mirror_job(
                            current.model_copy(
                                update={
                                    "state": "cancelled",
                                    "activity": None,
                                    "error_code": None
                                    if current.error_code == "mirror_stop_incomplete"
                                    else current.error_code,
                                    "updated_at": int(time.time()),
                                }
                            )
                        )
                    self._tasks.pop(job_id, None)
        if job.kind == "run":
            # Signal/drain the role before disk work that can stall. The private
            # Core stop fence was committed before any signal and still blocks
            # restart when an unavailable disk cannot record its own pause.
            with suppress(ValueError, OSError):
                await self.storage.run(
                    lambda: self._ledger().update(
                        lambda state: state.model_copy(update={"paused": True})
                    )
                )
        return self.job(job_id)

    async def resume_at_startup(self) -> MirrorJob | None:
        """Called only by explicit store startup; never from construction or GET."""
        if self._restored or self._closed:
            return None
        self._restored = True
        intent = self.journal.setting("mirror-running-intent")
        if (
            not isinstance(intent, dict)
            or self.journal.setting("mirror-stopped") is True
        ):
            return None
        state = await self.storage.run(lambda: self._ledger().read())
        if (
            not restart_allowed(state)
            or intent.get("policy_id") != policy_id(state)
            or state.restart is None
            or intent.get("expires") != state.restart.expires
        ):
            return None
        # A prior EOF-fenced worker may still be draining after its parent died.
        # Bound the handover and never run beside an unowned controller.
        deadline = time.monotonic() + 20
        while await self.storage.run(lambda: self._ledger().running()):
            if time.monotonic() >= deadline:
                raise PreservationHostError("previous_mirror_still_draining")
            await asyncio.sleep(0.1)
        return await self.start(
            MirrorRequest(
                owner_id="core-restart",
                operation_id=uuid.uuid4().hex,
                kind="run",
                policy_id=policy_id(state),
                mode="resume",
                public_participation=True,
            )
        )

    async def close(self) -> None:
        """Drain owned work while preserving only previously approved restart intent."""
        self._closed = True
        if self._closing is None or (
            self._closing.done()
            and (self._closing.cancelled() or self._closing.exception() is not None)
        ):
            self._closing = asyncio.create_task(self._close())
        await self._await_control(self._closing)

    async def _close(self) -> None:
        downloads = tuple(self._downloads)
        for task in downloads:
            task.cancel()
        tasks = tuple(self._tasks.values())

        async def withdraw(job_id: str, task: asyncio.Task[None]) -> None:
            job, _ = self.journal.mirror_job(job_id)
            if job.state in {"queued", "running"}:
                self.journal.update_mirror_job(
                    job.model_copy(
                        update={"state": "cancelling", "updated_at": int(time.time())}
                    )
                )
            process = self._processes.get(job_id)
            if process is not None:
                await process.stop()
            else:
                task.cancel()

        # A preparation may precede its controller. Start every withdrawal
        # before awaiting any drain, and do not cancel siblings on failure.
        results = await asyncio.gather(
            *(withdraw(job_id, task) for job_id, task in tuple(self._tasks.items())),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result
        # Unrelated cancelled model work can still own stalled filesystem
        # cleanup. Stop actual public controllers before waiting for that work.
        if self._cancellations:
            await asyncio.gather(*tuple(self._cancellations.values()))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if downloads:
            await asyncio.gather(*downloads, return_exceptions=True)
        if self._kit_removals:
            await asyncio.gather(
                *tuple(self._kit_removals.values()), return_exceptions=True
            )
        for job_id, task in tuple(self._tasks.items()):
            if task.done() and (process := self._processes.get(job_id)) is not None:
                await self._release_process(job_id, process)
            if task.done() and job_id not in self._processes:
                current, _ = self.journal.mirror_job(job_id)
                if current.state in {"queued", "running", "cancelling"}:
                    self.journal.update_mirror_job(
                        current.model_copy(
                            update={
                                "state": "interrupted",
                                "activity": None,
                                "error_code": "core_stopped",
                                "updated_at": int(time.time()),
                            }
                        )
                    )
                self._tasks.pop(job_id, None)
        if self._processes:
            raise PreservationHostError("mirror_stop_incomplete")
