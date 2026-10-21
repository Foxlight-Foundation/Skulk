"""Trusted-operator controls for the local authoritative preservation host."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse
from foxlight_model_preservation.models import TrustProfile

from skulk.store.preservation import RecoveryCompatibilityError
from skulk.store.preservation_contracts import (
    Handle,
    KitUploadRequest,
    KitUploadStatus,
    PreservationJob,
    PreservationLocation,
    PreservationPage,
    PreservationPlan,
    PreservationPlanRequest,
    PreservationStart,
    RecoveryCleanup,
)
from skulk.store.preservation_host import PreservationHost
from skulk.store.preservation_journal import PreservationHostError
from skulk.store.preservation_uploads import KIT_CHUNK_BYTES

HostProvider = Callable[[], Awaitable[PreservationHost]]


def create_preservation_router(
    provider: HostProvider, authorize: Callable[[Request], None]
) -> APIRouter:
    """Mount bounded controls; private reads use the same operator gate as writes.

    No archive/key paths or arbitrary commands are accepted. The provider binds
    the actual configured store owner; absence is actionable 503, never a fake
    empty library. No admin route is added to the unauthenticated store listener.
    """
    router = APIRouter(prefix="/preservation", tags=["Model Preservation"])

    async def host(request: Request) -> PreservationHost:
        authorize(request)
        try:
            return await provider()
        except PreservationHostError as error:
            raise HTTPException(status_code=503, detail=error.code) from error

    def refusal(error: Exception) -> HTTPException:
        code = (
            error.code
            if isinstance(error, PreservationHostError)
            else "unsupported_model_card"
            if isinstance(error, RecoveryCompatibilityError)
            else "storage_unavailable"
            if isinstance(error, OSError)
            else "invalid_or_unsafe"
        )
        return HTTPException(status_code=409, detail=code)

    @router.get(
        "/library",
        response_model=PreservationPage,
        summary="Read the local preservation library",
        description="Trusted operators receive a bounded page of installed selections, enrolled storage, retained copies and durable jobs. Reading never queues, replays or starts sharing. Use next_offset for another page; paths and key material are withheld.",
    )
    async def library(request: Request, offset: int = 0) -> PreservationPage:
        owner = await host(request)
        try:
            return await owner.page(offset)
        except (ValueError, OSError) as error:
            raise refusal(error) from error

    @router.post(
        "/setup/device",
        response_model=PreservationLocation,
        summary="Enroll this device's private archive",
        description="An explicit operator action creates the built-in private archive and owner-only capture key. Repeated setup returns its enrolled identity without replacing keys or recreating disconnected storage.",
    )
    async def setup_device(request: Request) -> PreservationLocation:
        owner = await host(request)
        try:
            return await owner.setup_device()
        except (ValueError, OSError) as error:
            raise refusal(error) from error

    @router.post(
        "/plans",
        response_model=PreservationPlan,
        summary="Review an exact preservation effect",
        description="Bind an expiring review to an installed generation, retained copy or completed kit upload. Supports Preserve, Check, recovery kit export, whole-closure recovery, public-bundle mirror import, bounded HTTP mirrors and opt-in BitTorrent. Public mirroring preserves opaque cards and grants no runtime trust. Serving requires separate public_participation consent; review never starts a listener. Recovery metadata is inspected before start; actual weights are verified by the owned worker before publication. No byte job starts at review.",
    )
    async def plan(
        request: Request, selection: PreservationPlanRequest
    ) -> PreservationPlan:
        owner = await host(request)
        try:
            async with asyncio.timeout(5):
                return await owner.plan(selection)
        except TimeoutError as error:
            raise HTTPException(
                status_code=409, detail="store_busy_try_again"
            ) from error
        except (ValueError, OSError) as error:
            raise refusal(error) from error

    @router.get(
        "/plans/{plan_id}",
        response_model=PreservationPlan,
        summary="Read one retained review plan",
        description="Return only the exact safe review summary. This does not refresh expiry, execute, or expose private paths/keys. A caller can bind its start input to the actual retained selection.",
    )
    async def read_plan(request: Request, plan_id: Handle) -> PreservationPlan:
        owner = await host(request)
        try:
            return owner.journal.plan(plan_id)[0]
        except (ValueError, OSError) as error:
            raise refusal(error) from error

    @router.post(
        "/jobs",
        response_model=PreservationJob,
        status_code=202,
        summary="Start an owned preservation job",
        description="Deliberately start an exact retained plan with a stable caller operation ID and explicit consent. Repeating the same request returns its original durable job; changed input refuses. Closing the page or capability does not cancel Core's owned work.",
    )
    async def start(request: Request, submission: PreservationStart) -> PreservationJob:
        owner = await host(request)
        try:
            return owner.start(submission)
        except (ValueError, OSError) as error:
            raise refusal(error) from error

    @router.get(
        "/jobs/{job_id}",
        response_model=PreservationJob,
        summary="Observe one durable preservation job",
        description="Return the retained exact job and bounded receipt. A read never restarts interrupted work; the caller can explicitly review a Check plan for its expected outcome.",
    )
    async def observe(request: Request, job_id: Handle) -> PreservationJob:
        owner = await host(request)
        try:
            return owner.journal.job(job_id)
        except (ValueError, OSError) as error:
            raise refusal(error) from error

    @router.post(
        "/jobs/{job_id}/cancel",
        response_model=PreservationJob,
        summary="Request an owned preservation stop",
        description="Persist cancellation intent and request cooperative stop. After a bounded grace period Core may terminate only its own byte worker. An already committed result can win; cancellation is confirmed only by observed worker completion. Retained verified copies are never deleted.",
    )
    async def cancel(request: Request, job_id: Handle) -> PreservationJob:
        owner = await host(request)
        try:
            return await owner.cancel(job_id)
        except (ValueError, OSError) as error:
            raise refusal(error) from error

    @router.post(
        "/jobs/{job_id}/reclaim",
        response_model=RecoveryCleanup,
        summary="Reclaim stopped recovery temporary files",
        description="Explicitly inspect one stopped recovery job under the live store lease. Authenticate its known generation against the retained signed manifest and require a readable canonical index before removing unreferenced files. Indexed models, saved kits, cards and permissions remain intact. Interrupted cleanup is safely retryable; this never resumes or admits a model.",
    )
    async def reclaim(request: Request, job_id: Handle) -> RecoveryCleanup:
        owner = await host(request)
        try:
            return await owner.reclaim_recovery(job_id)
        except (ValueError, OSError) as error:
            raise refusal(error) from error

    @router.post(
        "/kit-uploads",
        response_model=KitUploadStatus,
        summary="Reserve a recovery kit upload",
        description="Trusted operators declare the exact kit length. Core checks bounded ingress slots and free space and returns an opaque resumable handle. This grants neither publisher trust nor model admission.",
    )
    async def create_upload(
        request: Request, selection: KitUploadRequest
    ) -> KitUploadStatus:
        owner = await host(request)
        try:
            async with owner.upload_lock:
                value = await asyncio.to_thread(
                    owner.uploads.create, selection.size_bytes
                )
            return KitUploadStatus.model_validate(value)
        except (ValueError, OSError) as error:
            raise refusal(error) from error

    @router.get(
        "/kit-uploads/{upload_id}",
        response_model=KitUploadStatus,
        summary="Observe a kit transfer offset",
        description="Read the flushed byte offset for explicit resume. A complete transfer is not a verified recovery; no byte worker starts.",
    )
    async def upload_status(request: Request, upload_id: Handle) -> KitUploadStatus:
        owner = await host(request)
        try:
            return KitUploadStatus.model_validate(owner.uploads.status(upload_id))
        except (ValueError, OSError) as error:
            raise refusal(error) from error

    @router.put(
        "/kit-uploads/{upload_id}",
        response_model=KitUploadStatus,
        summary="Transfer one recovery kit chunk",
        description="Append at most 4 MiB of application/octet-stream at the exact durable offset. Identical completed-chunk retries are safe. This separate byte plane never places weights in SDK JSON controls; signature/hash verification and consent are required by recovery.",
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "application/octet-stream": {
                        "schema": {
                            "type": "string",
                            "format": "binary",
                            "maxLength": KIT_CHUNK_BYTES,
                        }
                    }
                },
            }
        },
    )
    async def upload_chunk(
        request: Request, upload_id: Handle, offset: int
    ) -> KitUploadStatus:
        owner = await host(request)
        if request.headers.get("content-type") != "application/octet-stream":
            raise HTTPException(status_code=415, detail="binary_kit_chunk_required")
        data = bytearray()
        async for chunk in request.stream():
            if len(data) + len(chunk) > KIT_CHUNK_BYTES:
                raise HTTPException(status_code=413, detail="kit_chunk_too_large")
            data.extend(chunk)
        try:
            async with owner.upload_lock:
                value = await asyncio.to_thread(
                    owner.uploads.append, upload_id, offset, bytes(data)
                )
            return KitUploadStatus.model_validate(value)
        except (ValueError, OSError) as error:
            raise refusal(error) from error

    @router.delete(
        "/kit-uploads/{upload_id}",
        status_code=204,
        response_class=Response,
        summary="Remove an unused kit upload",
        description="Release ingress storage only when no queued/running recovery references the handle. Preserved copies and completed recovered models remain intact.",
    )
    async def remove_upload(request: Request, upload_id: Handle) -> Response:
        owner = await host(request)
        try:
            async with owner.upload_lock:
                for offset in range(0, 512, 16):
                    jobs = owner.journal.jobs(offset)
                    if not jobs:
                        break
                    if any(
                        job.state in {"queued", "running", "cancelling"}
                        and owner.journal.plan(job.plan_id)[0].upload_id == upload_id
                        for job in jobs
                    ):
                        raise PreservationHostError("kit_upload_in_use")
                owner.uploads.remove(upload_id)
            return Response(status_code=204)
        except (ValueError, OSError) as error:
            raise refusal(error) from error

    @router.get(
        "/kits/{job_id}",
        response_class=FileResponse,
        summary="Download a completed recovery kit",
        description="Stream a completed Core-owned kit as a file, with ordinary HTTP range support. Save it on separate storage so it survives loss of this Core. No keys, arbitrary host paths or incomplete jobs can be selected.",
    )
    async def download_kit(request: Request, job_id: Handle) -> FileResponse:
        owner = await host(request)
        try:
            path = owner.kit_path(job_id)
            return FileResponse(
                path,
                media_type="application/octet-stream",
                filename="recovery-" + job_id + ".foxkit",
                headers={
                    "Cache-Control": "no-store",
                    "X-Content-Type-Options": "nosniff",
                },
            )
        except (ValueError, OSError) as error:
            raise refusal(error) from error

    @router.get(
        "/torrents/{job_id}",
        response_class=FileResponse,
        summary="Download standard public torrent metadata",
        description="Trusted operators download the exact hash-checked metainfo from an explicitly started torrent job, including while it is serving. This route joins no swarm and exposes no private key or arbitrary path. It remains available after the session stops.",
    )
    async def download_torrent(request: Request, job_id: Handle) -> FileResponse:
        owner = await host(request)
        try:
            return FileResponse(
                owner.torrent_path(job_id),
                media_type="application/x-bittorrent",
                filename="model-" + job_id + ".torrent",
                headers={
                    "Cache-Control": "no-store",
                    "X-Content-Type-Options": "nosniff",
                },
            )
        except (ValueError, OSError) as error:
            raise refusal(error) from error

    @router.get(
        "/public-copies/{copy_id}/trust",
        response_model=TrustProfile,
        summary="Download accepted public-copy verification keys",
        description="Return only the public verification profile accepted for this exact public copy. Recipients must independently authenticate its fingerprints and selected manifest ID. Self-signatures are not registry authority or a new redistribution grant. Reading starts no sharing or model admission.",
    )
    async def download_trust(request: Request, copy_id: Handle) -> TrustProfile:
        owner = await host(request)
        try:
            return owner.public_trust(copy_id)
        except (ValueError, OSError) as error:
            raise refusal(error) from error

    # Keep explicit references to the registered handlers for strict static tooling.
    _ = (
        library,
        setup_device,
        plan,
        read_plan,
        start,
        observe,
        cancel,
        reclaim,
        create_upload,
        upload_status,
        upload_chunk,
        remove_upload,
        download_kit,
        download_torrent,
        download_trust,
    )
    return router
