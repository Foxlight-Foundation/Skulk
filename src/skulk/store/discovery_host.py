"""Canonical Core ownership of followed sources and finite metadata workers."""

from __future__ import annotations

import asyncio
import json
import secrets
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Literal, cast, final
from urllib.parse import urlsplit

from foxlight_model_preservation.discovery import (
    bootstrap_identity,
    follow_source,
    list_sources,
    read_source_document,
    remove_source,
    search_sources,
    select_entry,
    source_identity,
)
from foxlight_model_preservation.discovery_models import SourceBootstrap, SourceStatus
from foxlight_model_preservation.invitations import Invitation, read_invitation
from foxlight_model_preservation.models import MAX_METADATA_BYTES
from foxlight_model_preservation.signing import encode_record
from foxlight_model_preservation.storage import atomic_bytes, directory
from pydantic import BaseModel

from skulk.store.discovery_contracts import (
    DiscoveryJob,
    DiscoveryPage,
    DiscoveryQuery,
    DiscoveryRequest,
    DiscoveryResult,
    DiscoveryResults,
    DiscoverySource,
    DiscoverySupplier,
    SourceConfirmation,
    SourceReview,
)
from skulk.store.discovery_worker import (
    DiscoveryWorkerRequest,
    DiscoveryWorkerResult,
    public_model_mirrors,
)
from skulk.store.preservation_io import PreservationIO
from skulk.store.preservation_journal import (
    PreservationHostError,
    PreservationJournal,
    safe_json,
    secure_directory,
)
from skulk.store.preservation_uploads import KitUploads

METADATA_WORKER_SECONDS = 150

MetadataReader = Callable[[str, Literal["invitation", "publication", "source"]], bytes]


def _bounded[T: BaseModel](value: T) -> T:
    if (
        len(json.dumps(value.model_dump(mode="json"), ensure_ascii=True).encode())
        > 48000
    ):
        raise PreservationHostError("discovery_preview_too_large")
    return value


def _projection(value: str, budget: int) -> str:
    if len(json.dumps(value, ensure_ascii=True)) <= budget:
        return value
    lower, upper = 0, len(value)
    while lower < upper:
        midpoint = (lower + upper + 1) // 2
        if len(json.dumps(value[:midpoint] + "…", ensure_ascii=True)) <= budget:
            lower = midpoint
        else:
            upper = midpoint - 1
    return value[:lower] + "…"


@final
class DiscoveryHost:
    """One metadata queue under the existing host journal and shutdown owner.

    Browser/capability closure owns no worker. Network effects use known child
    processes with a lifetime pipe and hard parent deadline; local reads use
    drained IO. Source enrollment grants metadata authority only.
    """

    def __init__(
        self,
        root: Path,
        journal: PreservationJournal,
        storage: PreservationIO,
        uploads: KitUploads,
        upload_lock: asyncio.Lock,
        metadata: MetadataReader,
        command: Callable[[Path], tuple[str, ...]],
    ) -> None:
        self.root = root
        self.cache = root / "discovery-cache"
        self.journal = journal
        self.storage = storage
        self.uploads = uploads
        self.upload_lock = upload_lock
        self.metadata = metadata
        self.command = command
        self._queue = asyncio.Lock()
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._closed = False
        for job in journal.discovery_jobs():
            self._finish(job.job_id, "interrupted", "core_restarted")

    def _open(self) -> None:
        if self._closed:
            raise PreservationHostError("preservation_host_closed")

    async def review(self, upload_id: str) -> SourceReview:
        """Inspect a bootstrap offline and retain an expiring identity-bound choice."""
        self._open()

        def inspect() -> SourceReview:
            data = self.metadata(upload_id, "source")
            if len(data) > 256 * 1024:
                raise PreservationHostError("source_document_too_large")
            bootstrap = read_source_document(data)
            if not isinstance(bootstrap, SourceBootstrap):
                raise PreservationHostError("source_bootstrap_required")
            if any(
                urlsplit(endpoint).scheme not in {"https", "http"}
                for endpoint in bootstrap.endpoints
            ):
                raise PreservationHostError("local_sources_require_standalone_tools")
            expires = int(time.time()) + 600
            view = SourceReview(
                review_id=secrets.token_hex(32),
                source_id=source_identity(bootstrap),
                bootstrap_id=bootstrap_identity(bootstrap),
                name=bootstrap.name,
                endpoints=bootstrap.endpoints,
                expires_at=expires,
            )
            self.journal.add_discovery_review(
                view.review_id,
                {
                    "view": view.model_dump(mode="json"),
                    "bootstrap": bootstrap.model_dump(mode="json"),
                    "expires_at": expires,
                },
            )
            return _bounded(view)

        async with self.upload_lock:
            return await self.storage.run(inspect)

    async def follow(self, request: SourceConfirmation) -> SourceStatus:
        """Confirm only the reviewed discovery identity; perform no network IO."""
        self._open()
        if not request.confirmed:
            raise PreservationHostError("source_trust_confirmation_required")
        if self._queue.locked():
            raise PreservationHostError("discovery_busy_try_again")
        async with self._queue:
            self._open()

            def confirm() -> SourceStatus:
                value = self.journal.setting("discovery-review:" + request.review_id)
                if not isinstance(value, dict):
                    raise PreservationHostError("source_review_missing_or_expired")
                view = SourceReview.model_validate_json(safe_json(value["view"]))
                if view.expires_at < int(time.time()):
                    raise PreservationHostError("source_review_missing_or_expired")
                bootstrap = SourceBootstrap.model_validate_json(
                    safe_json(value["bootstrap"])
                )
                status = follow_source(
                    self.cache,
                    encode_record(bootstrap),
                    view.bootstrap_id,
                    allow_http=request.allow_http,
                    allow_private=request.allow_private,
                )
                if status.source_id != view.source_id:
                    raise PreservationHostError("source_identity_changed")
                return status

            return await self.storage.run(confirm)

    async def remove(self, source_id: str) -> None:
        """Unfollow without deleting roots/floors or touching received model copies."""
        self._open()
        if self._queue.locked():
            raise PreservationHostError("discovery_busy_try_again")
        async with self._queue:
            self._open()
            await self.storage.run(lambda: remove_source(self.cache, source_id))

    async def page(self, owner_id: str, offset: int) -> DiscoveryPage:
        """Read source and metadata-job states locally without probes or replay."""
        sources = await self.storage.run(lambda: list_sources(self.cache))
        jobs = self.journal.discovery_jobs(owner_id, offset)
        return _bounded(
            DiscoveryPage(
                sources=tuple(
                    DiscoverySource.model_validate_json(
                        source.model_copy(
                            update={
                                "name": _projection(source.name, 512),
                                "last_error": _projection(source.last_error, 512)
                                if source.last_error
                                else None,
                            }
                        ).model_dump_json()
                    )
                    for source in sources
                ),
                jobs=jobs[:4],
                next_offset=offset + 4 if len(jobs) > 4 else None,
            )
        )

    async def search(self, query: DiscoveryQuery) -> DiscoveryResults:
        """Search cached catalogs with bounded model/supplier pagination, off-loop."""

        def find() -> DiscoveryResults:
            matches = search_sources(self.cache, query.query, limit=1000)
            results: list[DiscoveryResult] = []
            for match in matches[query.offset : query.offset + 4]:
                suppliers = tuple(
                    DiscoverySupplier(
                        source_id=item.source_id,
                        source_name=_projection(item.source_name, 256),
                        entry_id=item.entry_id,
                        name=_projection(
                            (
                                item.description.name
                                if item.description
                                else item.entry_id
                            )[:200],
                            512,
                        ),
                        summary=_projection(
                            (item.description.summary if item.description else "")[
                                :800
                            ],
                            768,
                        ),
                        freshness=item.freshness,
                        model_keys_available=item.description is not None
                        and item.description.candidate_bundle_trust is not None,
                    )
                    for item in match.locations[
                        query.supplier_offset : query.supplier_offset + 4
                    ]
                )
                results.append(
                    DiscoveryResult(
                        manifest_id=match.manifest,
                        suppliers=suppliers,
                        supplier_count=len(match.locations),
                        next_supplier_offset=query.supplier_offset + 4
                        if len(match.locations) > query.supplier_offset + 4
                        else None,
                    )
                )
            return _bounded(
                DiscoveryResults(
                    results=tuple(results),
                    next_offset=query.offset + 4
                    if len(matches) > query.offset + 4
                    else None,
                )
            )

        return await self.storage.run(find)

    def start(self, request: DiscoveryRequest) -> DiscoveryJob:
        """Reserve an exact metadata request; return before any network operation."""
        self._open()
        job, created = self.journal.reserve_discovery(request)
        if created:
            self._tasks[job.job_id] = asyncio.create_task(self._execute(job.job_id))
        return job

    async def cancel(self, job_id: str) -> DiscoveryJob:
        """Retain stop intent promptly; only a stopped owned child is terminal."""
        job = self.journal.discovery_job(job_id)[0]
        if job.state in {"queued", "running", "cancelling"}:
            self.journal.update_discovery_job(
                job.model_copy(
                    update={"state": "cancelling", "updated_at": int(time.time())}
                )
            )
            process = self._processes.get(job_id)
            if process is not None:
                await self._stop(process)
        return self.journal.discovery_job(job_id)[0]

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

    def _finish(
        self,
        job_id: str,
        state: Literal["failed", "interrupted", "cancelled", "succeeded"],
        error: str | None = None,
        upload_id: str | None = None,
    ) -> None:
        job = self.journal.discovery_job(job_id)[0]
        self.journal.update_discovery_job(
            job.model_copy(
                update={
                    "state": state,
                    "updated_at": int(time.time()),
                    "error_code": error,
                    "upload_id": upload_id,
                }
            )
        )

    def _request(self, job_id: str, selection: DiscoveryRequest) -> Path:
        path = self.root / "discovery-requests"
        secure_directory(path)
        request = DiscoveryWorkerRequest(
            cache=self.cache.absolute(), selection=selection
        )
        with directory(path) as root:
            atomic_bytes(root, job_id + ".json", request.model_dump_json().encode())
        return path / (job_id + ".json")

    def _accept_metadata(
        self, selection: DiscoveryRequest, invitation: Invitation
    ) -> str:
        entry = select_entry(
            self.cache,
            selection.source_id,
            selection.entry_id or "",
            selection.manifest_id or "",
        )
        checked = read_invitation(encode_record(invitation))
        matches = search_sources(self.cache, entry.manifest, limit=1000)
        description = next(
            (
                location.description
                for match in matches
                for location in match.locations
                if location.source_id == selection.source_id
                and location.entry_id == selection.entry_id
            ),
            None,
        )
        if (
            description is None
            or description.candidate_bundle_trust != checked.candidate_trust
            or checked.manifest_id != entry.manifest
            or checked.mirrors != public_model_mirrors(entry.mirrors)
        ):
            raise PreservationHostError("source_model_reference_changed")
        encoded = encode_record(checked)
        status = self.uploads.create(len(encoded), "invitation")
        upload_id = cast(str, status["upload_id"])
        # Existing ingress flushes at most 4 MiB at a time and keeps its quota.
        for offset in range(0, len(encoded), 4 * 1024 * 1024):
            self.uploads.append(
                upload_id, offset, encoded[offset : offset + 4 * 1024 * 1024]
            )
        return upload_id

    async def _execute(self, job_id: str) -> None:
        try:
            # Retain serialization through actual process stop and IO cleanup.
            async with self._queue:
                await self._run(job_id)
        except asyncio.CancelledError:
            self._finish(job_id, "interrupted", "core_restarted")
            raise
        except Exception:
            if self.journal.discovery_job(job_id)[0].state in {
                "queued",
                "running",
                "cancelling",
            }:
                self._finish(job_id, "interrupted", "discovery_worker_unavailable")
        finally:
            self._tasks.pop(job_id, None)

    async def _run(self, job_id: str) -> None:
        process: asyncio.subprocess.Process | None = None
        request_path: Path | None = None
        try:
            job, selection = self.journal.discovery_job(job_id)
            if job.state == "cancelling" or self._closed:
                self._finish(job_id, "cancelled", "cancelled")
                return
            self.journal.update_discovery_job(
                job.model_copy(
                    update={"state": "running", "updated_at": int(time.time())}
                )
            )
            request_path = await self.storage.run(
                lambda: self._request(job_id, selection)
            )
            async with asyncio.timeout(METADATA_WORKER_SECONDS):
                process = await asyncio.create_subprocess_exec(
                    *self.command(request_path),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                self._processes[job_id] = process
                if self.journal.discovery_job(job_id)[0].state == "cancelling":
                    await self._stop(process)
                if process.stdout is None:
                    raise PreservationHostError("discovery_worker_unavailable")
                encoded = bytearray()
                while chunk := await process.stdout.read(65536):
                    encoded.extend(chunk)
                    if len(encoded) > MAX_METADATA_BYTES + 65536:
                        raise PreservationHostError("discovery_response_too_large")
                await process.wait()
                result_bytes = bytes(encoded)
                result = await self.storage.run(
                    lambda: DiscoveryWorkerResult.model_validate_json(result_bytes)
                )
                if process.returncode != 0 or result.outcome != "succeeded":
                    code = result.error_code or "discovery_update_failed"
                    self._finish(
                        job_id,
                        "cancelled" if code == "cancelled" else "failed",
                        code,
                    )
                    return
                if selection.kind == "refresh":
                    sources = await self.storage.run(lambda: list_sources(self.cache))
                    if result.catalog_id is None or not any(
                        source.source_id == selection.source_id
                        and source.catalog_id == result.catalog_id
                        for source in sources
                    ):
                        raise PreservationHostError("discovery_receipt_changed")
                    self._finish(job_id, "succeeded")
                elif result.invitation is not None:
                    if self.journal.discovery_job(job_id)[0].state == "cancelling":
                        self._finish(job_id, "cancelled", "cancelled")
                        return
                    invitation = result.invitation
                    async with self.upload_lock:
                        upload_id = await self.storage.run(
                            lambda: self._accept_metadata(selection, invitation)
                        )
                    self._finish(job_id, "succeeded", upload_id=upload_id)
                else:
                    raise PreservationHostError("discovery_receipt_changed")
        except (ValueError, OSError, TimeoutError) as error:
            if process is not None and process.returncode is None:
                await self._stop(process)
            current = self.journal.discovery_job(job_id)[0]
            self._finish(
                job_id,
                "interrupted"
                if process is not None
                else "cancelled"
                if current.state == "cancelling"
                else "failed",
                "discovery_deadline_exceeded"
                if isinstance(error, TimeoutError)
                else error.code
                if isinstance(error, PreservationHostError)
                else "discovery_outcome_uncertain"
                if process is not None
                else "discovery_worker_unavailable",
            )
        finally:
            if process is not None:
                if process.returncode is None:
                    await self._stop(process)
                if process.stdin is not None:
                    process.stdin.close()
            self._processes.pop(job_id, None)
            if request_path is not None:
                retained_path = request_path
                await self.storage.run(lambda: retained_path.unlink(missing_ok=True))

    async def close(self) -> None:
        """Stop and drain actual metadata effects before journal ownership releases."""
        self._closed = True
        for job_id in tuple(self._tasks):
            await self.cancel(job_id)
        if self._tasks:
            await asyncio.gather(*tuple(self._tasks.values()), return_exceptions=True)
