"""Canonical Core ownership of followed sources and finite metadata workers."""

from __future__ import annotations

import asyncio
import json
import secrets
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Literal, NamedTuple, cast, final
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
from foxlight_model_preservation.handoffs import read_handoff
from foxlight_model_preservation.i2p_http import i2p_http_endpoint
from foxlight_model_preservation.models import MAX_METADATA_BYTES
from foxlight_model_preservation.peer import (
    PeerSource,
    follow_peer_source,
    inspect_peer_source,
    peer_source_identity,
    read_peer_source,
)
from foxlight_model_preservation.signing import encode_record, strict_json
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
    I2PSourcePolicy,
    SourceConfirmation,
    SourceReview,
)
from skulk.store.discovery_worker import (
    DiscoveryWorkerRequest,
    DiscoveryWorkerResult,
    public_model_mirrors,
)
from skulk.store.peer_host import PeerHost, PeerLease
from skulk.store.preservation_io import PreservationIO
from skulk.store.preservation_journal import (
    PreservationHostError,
    PreservationJournal,
    safe_json,
    secure_directory,
)
from skulk.store.preservation_references import ModelInvitation as Invitation
from skulk.store.preservation_references import read_model_reference as read_invitation
from skulk.store.preservation_uploads import KitUploads

METADATA_WORKER_SECONDS = 150

MetadataReader = Callable[[str, Literal["invitation", "publication", "source"]], bytes]


@final
class _DiscoveryOutcome(NamedTuple):
    """Pending terminal result, published only after owned cleanup and unlock."""

    state: Literal["failed", "interrupted", "cancelled", "succeeded"]
    error: str | None = None
    upload_id: str | None = None


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
        peers: PeerHost | None = None,
    ) -> None:
        self.root = root
        self.cache = root / "discovery-cache"
        self.journal = journal
        self.storage = storage
        self.uploads = uploads
        self.upload_lock = upload_lock
        self.metadata = metadata
        self.command = command
        self.peers = peers
        self._peer_connections: dict[str, PeerLease] = {}
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
            raw = strict_json(data)
            handoff = (
                read_handoff(data)
                if isinstance(raw, dict)
                and raw.get("schema_version")
                == "foxlight.model-preservation.handoff.v1"
                else None
            )
            peer = (
                handoff.reference
                if handoff is not None and isinstance(handoff.reference, PeerSource)
                else inspect_peer_source(data)
                if isinstance(raw, dict)
                and raw.get("schema_version") == "foxlight.peer-source.v1"
                else None
            )
            if handoff is not None and peer is None:
                raise PreservationHostError("source_bootstrap_required")
            bootstrap = peer.source if peer is not None else read_source_document(data)
            if not isinstance(bootstrap, SourceBootstrap):
                raise PreservationHostError("source_bootstrap_required")
            if peer is None and any(
                urlsplit(endpoint).scheme not in {"https", "http"}
                for endpoint in bootstrap.endpoints
            ):
                raise PreservationHostError("local_sources_require_standalone_tools")
            expires = int(time.time()) + 600
            view = SourceReview(
                review_id=secrets.token_hex(32),
                source_id=source_identity(bootstrap),
                bootstrap_id=peer_source_identity(peer)
                if peer
                else bootstrap_identity(bootstrap),
                name=bootstrap.name,
                endpoints=() if peer else bootstrap.endpoints,
                contacts=handoff.contacts if handoff else (),
                transport="peer" if peer else "http",
                ipns_name=peer.ipns if peer else None,
                snapshot_count=len(peer.snapshots) if peer else 0,
                expires_at=expires,
            )
            self.journal.add_discovery_review(
                view.review_id,
                {
                    "view": view.model_dump(mode="json"),
                    "bootstrap": bootstrap.model_dump(mode="json"),
                    "peer": peer.model_dump(mode="json") if peer else None,
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
                if value.get("peer") is not None:
                    if request.allow_http or request.allow_private or request.allow_i2p:
                        raise PreservationHostError("peer_sources_use_peer_settings")
                    peer = PeerSource.model_validate_json(safe_json(value["peer"]))
                    status = follow_peer_source(
                        self.cache, encode_record(peer), view.bootstrap_id
                    )
                else:
                    if self._transport(view.source_id) == "peer":
                        raise PreservationHostError(
                            "source_transport_switch_unavailable"
                        )
                    if request.allow_i2p:
                        if request.i2p_proxy is None:
                            raise PreservationHostError("i2p_proxy_required")
                        endpoints = request.i2p_endpoints
                        if not endpoints:
                            supported: list[str] = []
                            for endpoint in bootstrap.endpoints:
                                try:
                                    supported.append(i2p_http_endpoint(endpoint))
                                except ValueError:
                                    continue
                            endpoints = tuple(supported)
                        if not endpoints:
                            raise PreservationHostError("i2p_metadata_mirror_required")
                        policy = I2PSourcePolicy(
                            source_id=view.source_id,
                            proxy=request.i2p_proxy,
                            endpoints=endpoints,
                        )
                        # Persist protected routing first. A crash may leave a
                        # harmless unenrolled policy, never direct-only membership
                        # that loses the operator's explicit I2P selection.
                        self.journal.set_setting(
                            "discovery-i2p:" + view.source_id,
                            policy.model_dump(mode="json"),
                        )
                    elif self._transport(view.source_id) == "i2p":
                        raise PreservationHostError(
                            "source_transport_switch_unavailable"
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

    def _i2p_policy(self, source_id: str) -> I2PSourcePolicy | None:
        value = self.journal.setting("discovery-i2p:" + source_id)
        if value is None:
            return None
        try:
            policy = I2PSourcePolicy.model_validate_json(safe_json(value))
            if policy.source_id != source_id:
                raise ValueError("source binding changed")
            return policy
        except ValueError as error:
            raise PreservationHostError("source_i2p_policy_invalid") from error

    def _transport(self, source_id: str) -> Literal["http", "peer", "i2p"]:
        # Even a malformed protected record refuses; it must never downgrade.
        if self.journal.setting("discovery-i2p:" + source_id) is not None:
            self._i2p_policy(source_id)
            return "i2p"
        return (
            "peer"
            if (self.cache / "peer-sources" / (source_id + ".json")).exists()
            else "http"
        )

    def _snapshot_count(self, source_id: str) -> int:
        try:
            return len(read_peer_source(self.cache, source_id).snapshots)
        except (ValueError, OSError):
            return 0

    async def page(self, owner_id: str, offset: int) -> DiscoveryPage:
        """Read source and metadata-job states locally without probes or replay."""
        sources = await self.storage.run(
            lambda: tuple(
                source.model_copy(update={"name": _projection(source.name, 512)})
                for source in list_sources(self.cache)
            )
        )
        transports = await self.storage.run(
            lambda: {
                source.source_id: self._transport(source.source_id)
                for source in sources
            }
        )
        snapshots = await self.storage.run(
            lambda: {
                source.source_id: self._snapshot_count(source.source_id)
                for source in sources
            }
        )
        i2p_policies = await self.storage.run(
            lambda: {
                source.source_id: self._i2p_policy(source.source_id)
                for source in sources
            }
        )
        jobs = self.journal.discovery_jobs(owner_id, offset)
        return _bounded(
            DiscoveryPage(
                sources=tuple(
                    DiscoverySource.model_validate_json(
                        safe_json(
                            {
                                **source.model_dump(mode="json"),
                                "name": _projection(source.name, 512),
                                "last_error": _projection(source.last_error, 512)
                                if source.last_error
                                else None,
                                "transport": transports[source.source_id],
                                "snapshot_count": snapshots[source.source_id],
                                **(
                                    {
                                        "i2p_policy_id": policy.policy_id,
                                        "i2p_proxy": policy.proxy.model_dump(
                                            mode="json"
                                        ),
                                    }
                                    if (policy := i2p_policies[source.source_id])
                                    is not None
                                    else {}
                                ),
                            }
                        )
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
                        transport=self._transport(item.source_id),
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
        peer = self._peer_connections.get(job_id)
        request = DiscoveryWorkerRequest(
            cache=self.cache.absolute(),
            selection=selection,
            i2p_policy=self._i2p_policy(selection.source_id)
            if selection.transport == "i2p"
            else None,
            peer_repository=peer.repository if peer else None,
            peer_binary=peer.binary if peer else None,
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
            or checked.mirrors
            != public_model_mirrors(entry.mirrors, selection.transport)
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
                outcome = await self._run(job_id)
            # No await separates lock release from the terminal receipt. Readers
            # may act on completion immediately, including source withdrawal.
            self._finish(job_id, outcome.state, outcome.error, outcome.upload_id)
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

    async def _run(self, job_id: str) -> _DiscoveryOutcome:
        process: asyncio.subprocess.Process | None = None
        request_path: Path | None = None
        peer: PeerLease | None = None
        try:
            job, selection = self.journal.discovery_job(job_id)
            if job.state == "cancelling" or self._closed:
                return _DiscoveryOutcome("cancelled", "cancelled")
            self.journal.update_discovery_job(
                job.model_copy(
                    update={"state": "running", "updated_at": int(time.time())}
                )
            )
            enrolled_transport = await self.storage.run(
                lambda: self._transport(selection.source_id)
            )
            if selection.transport != enrolled_transport:
                raise PreservationHostError("source_transport_changed")
            if selection.transport == "i2p":
                policy = await self.storage.run(
                    lambda: self._i2p_policy(selection.source_id)
                )
                if (
                    policy is None
                    or policy.policy_id != selection.i2p_policy_id
                    or not selection.allow_i2p
                ):
                    raise PreservationHostError("source_i2p_policy_changed")
            if selection.transport == "peer":
                if self.peers is None or selection.peer_setup_id is None:
                    raise PreservationHostError("peer_setup_required")

                def preflight() -> None:
                    if not any(
                        s.source_id == selection.source_id and s.enabled
                        for s in list_sources(self.cache)
                    ):
                        raise PreservationHostError("source_no_longer_followed")
                    if selection.kind == "refresh":
                        record = read_peer_source(self.cache, selection.source_id)
                        if selection.historical and not record.snapshots:
                            raise PreservationHostError("peer_pinned_snapshot_required")
                    else:
                        entry = select_entry(
                            self.cache,
                            selection.source_id,
                            selection.entry_id or "",
                            selection.manifest_id or "",
                        )
                        if not public_model_mirrors(entry.mirrors, "peer"):
                            raise PreservationHostError("public_model_mirror_required")

                        matches = search_sources(self.cache, entry.manifest, limit=1000)
                        if not any(
                            location.source_id == selection.source_id
                            and location.entry_id == selection.entry_id
                            and location.description is not None
                            and location.description.candidate_bundle_trust is not None
                            for match in matches
                            for location in match.locations
                        ):
                            raise PreservationHostError(
                                "candidate_bundle_keys_required"
                            )

                await self.storage.run(preflight)
                peer = await self.peers.open(
                    selection.peer_setup_id,
                    METADATA_WORKER_SECONDS,
                    lambda: self._closed
                    or self.journal.discovery_job(job_id)[0].state == "cancelling",
                )
                self._peer_connections[job_id] = peer
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
                if peer is not None:
                    await peer.close()
                result_bytes = bytes(encoded)
                result = await self.storage.run(
                    lambda: DiscoveryWorkerResult.model_validate_json(result_bytes)
                )
                if process.returncode != 0 or result.outcome != "succeeded":
                    code = result.error_code or "discovery_update_failed"
                    return _DiscoveryOutcome(
                        "cancelled" if code == "cancelled" else "failed",
                        code,
                    )
                if selection.kind == "refresh":
                    sources = await self.storage.run(lambda: list_sources(self.cache))
                    if result.catalog_id is None or not any(
                        source.source_id == selection.source_id
                        and source.catalog_id == result.catalog_id
                        for source in sources
                    ):
                        raise PreservationHostError("discovery_receipt_changed")
                    return _DiscoveryOutcome("succeeded")
                elif result.invitation is not None:
                    if self.journal.discovery_job(job_id)[0].state == "cancelling":
                        return _DiscoveryOutcome("cancelled", "cancelled")
                    invitation = result.invitation
                    async with self.upload_lock:
                        upload_id = await self.storage.run(
                            lambda: self._accept_metadata(selection, invitation)
                        )
                    return _DiscoveryOutcome("succeeded", upload_id=upload_id)
                else:
                    raise PreservationHostError("discovery_receipt_changed")
        except (ValueError, OSError, TimeoutError) as error:
            if process is not None and process.returncode is None:
                await self._stop(process)
            if peer is not None:
                await peer.close()
            current = self.journal.discovery_job(job_id)[0]
            return _DiscoveryOutcome(
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
            if peer is not None:
                await peer.close()
            self._peer_connections.pop(job_id, None)
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
