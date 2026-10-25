"""Operator-gated persistent public mirrors under the preservation owner."""

from collections.abc import Awaitable, Callable

from fastapi import APIRouter, HTTPException, Request
from starlette.responses import Response

from skulk.api.mirror_kits import MirrorKitDownload
from skulk.store.mirror_contracts import (
    MirrorJob,
    MirrorJobsPage,
    MirrorKitRemoval,
    MirrorPage,
    MirrorRequest,
    MirrorReview,
    MirrorSelection,
)
from skulk.store.preservation_contracts import Digest, Handle
from skulk.store.preservation_host import PreservationHost


def install_mirror_routes(
    router: APIRouter,
    host: Callable[[Request], Awaitable[PreservationHost]],
    refusal: Callable[[Exception], HTTPException],
) -> None:
    """Use the canonical operator gate; no path, command or signer is accepted."""

    @router.get(
        "/mirror",
        response_model=MirrorPage,
        summary="Read the managed mirror without starting sharing",
        description="Read one bounded page of selected public copies, actual resource/transport settings, standing permission and live owned activity. Ports and native worker counts describe local ownership, not public reachability. Reading never starts a controller, follows a source or deletes data; private paths, executable selection and signing custody stay on Core.",
    )
    async def mirror(request: Request, offset: int = 0) -> MirrorPage:
        owner = await host(request)
        try:
            return await owner.mirrors.page(offset)
        except (ValueError, OSError) as error:
            raise refusal(error) from error

    @router.get(
        "/mirror/models/{manifest_id}",
        response_model=MirrorSelection,
        summary="Look up one exact retained mirror selection",
        description="Read exact state independently of list pagination and original application import records. Unknown selections are distinct from unavailable or corrupt storage. Authenticated labels use bounded metadata only; this starts no network, hashes no model payload and grants no trust or deletion.",
    )
    async def selection(request: Request, manifest_id: Digest) -> MirrorSelection:
        owner = await host(request)
        try:
            return await owner.mirrors.selection(manifest_id)
        except (ValueError, OSError) as error:
            raise refusal(error) from error

    @router.get(
        "/mirror/models/{manifest_id}/review",
        response_model=MirrorReview,
        summary="Review complete retained-copy metadata without the original import",
        description="Authenticate the original signed identity and all required member terms/notices using locally accepted mirror trust. Return a complete bounded review or refuse, without scanning weights, signing new provenance, following sources or admitting models. Capture keys remain independent of recipient, registry and runtime authority.",
    )
    async def review(request: Request, manifest_id: Digest) -> MirrorReview:
        owner = await host(request)
        try:
            return await owner.mirrors.review(manifest_id)
        except (ValueError, OSError) as error:
            raise refusal(error) from error

    @router.post(
        "/mirror/jobs",
        response_model=MirrorJob,
        status_code=202,
        summary="Start one explicit permanent mirror decision",
        description="Configure offline, retain an exact already accepted public copy with reviewed every-member terms, recover a complete pending selection offline, prepare its recovery kit, select a finite public run or separately approved standing resume, control pin/block/retry, or confirm exact stopped-owner deletion. Recovery requires a stopped controller and original model authority, verifies every actual object and clears Core restart intent; it never fetches or deletes partial bytes. The immutable client operation observes its original job on replay. Every new effect binds current policy; installation, metadata reads, receiving, or a source file grant no sharing. Configuration and local admission clear restart authority as appropriate; no policy evicts retained copies. Direct public PMF transport remains unavailable.",
    )
    async def start(request: Request, body: MirrorRequest) -> MirrorJob:
        owner = await host(request)
        try:
            return await owner.mirrors.start(body)
        except (ValueError, OSError) as error:
            raise refusal(error) from error

    @router.get(
        "/mirror/jobs",
        response_model=MirrorJobsPage,
        summary="Read permanent mirror job history",
        description="Read up to twenty owned control/byte/controller receipts, optionally filtered by client namespace. Network, filesystem and worker effects are never replayed. Old process observations are not substituted for live listeners after restart; existing finite preservation/sharing job contracts are unchanged.",
    )
    async def jobs(
        request: Request, owner_id: Handle | None = None, offset: int = 0
    ) -> MirrorJobsPage:
        owner = await host(request)
        try:
            rows = owner.journal.mirror_jobs(owner_id, offset)
            return MirrorJobsPage(
                jobs=tuple(owner.mirrors.job(job.job_id) for job in rows[:20]),
                next_offset=offset + 20 if len(rows) > 20 else None,
            )
        except (ValueError, OSError) as error:
            raise refusal(error) from error

    @router.get(
        "/mirror/jobs/{job_id}",
        response_model=MirrorJob,
        summary="Observe one durable mirror operation",
        description="Read an exact known job without replaying work, refreshing metadata or extending its grant. Activity is live only while Core owns that worker; worker counts include preparation/drain and are not reachable replica counts.",
    )
    async def job(request: Request, job_id: Digest) -> MirrorJob:
        owner = await host(request)
        try:
            return owner.mirrors.job(job_id)
        except (ValueError, OSError) as error:
            raise refusal(error) from error

    @router.delete(
        "/mirror/jobs/{job_id}",
        response_model=MirrorJob,
        summary="Stop the exact owned mirror operation and retain its copies",
        description="Record cancellation before signalling and drain the actual process group. Stopping an active controller also clears Core's running intent and durably pauses its role; saved copies, private partial metadata, keys, resource reservations and deletion tombstones remain. Stopping an old terminal receipt does not stop a newer controller. Closing a client does not release ongoing worker ownership or authorize a retry.",
    )
    async def stop(request: Request, job_id: Digest) -> MirrorJob:
        owner = await host(request)
        try:
            return await owner.mirrors.cancel(job_id)
        except (ValueError, OSError) as error:
            raise refusal(error) from error

    @router.get(
        "/mirror/kits/{job_id}",
        response_class=Response,
        summary="Download one prepared retained-copy recovery kit",
        description="Download only a successfully sealed, unchanged private preparation artifact through an owned descriptor. Supports bounded single byte ranges and strong If-Range; no full-byte hash runs in a GET. A kit retains original signed evidence and candidate keys; recipients still independently verify trust and all bytes. Reading starts no sharing and never deletes the retained model.",
        responses={
            200: {
                "content": {
                    "application/octet-stream": {
                        "schema": {"type": "string", "format": "binary"}
                    }
                }
            },
            206: {
                "description": "Selected byte range",
                "content": {
                    "application/octet-stream": {
                        "schema": {"type": "string", "format": "binary"}
                    }
                },
            },
            416: {"description": "Unsatisfiable or unsupported range"},
        },
    )
    @router.head(
        "/mirror/kits/{job_id}",
        response_class=Response,
        summary="Read prepared recovery-kit download headers",
        description="Check the enrolled artifact fingerprint and return prepared length/digest headers without reading payload bytes or replaying preparation. Model trust remains separate.",
    )
    async def kit(request: Request, job_id: Digest) -> Response:
        owner = await host(request)
        return MirrorKitDownload(
            owner.mirrors,
            job_id,
            request.headers.get("range"),
            request.headers.get("if-range"),
            refusal,
        )

    @router.delete(
        "/mirror/kits/{job_id}",
        response_model=MirrorJob,
        summary="Remove only explicitly confirmed kit preparation files",
        description="Delete one stopped job's enrolled complete/incomplete kit generation after exact job confirmation. Retained model weights, cards, provenance, sharing policy and other kits are untouched. Removal intent survives interruption; only explicit retry completes it. Unexpected/replaced files refuse deletion, and restart performs no automatic cleanup.",
    )
    async def remove_kit(
        request: Request, job_id: Digest, body: MirrorKitRemoval
    ) -> MirrorJob:
        owner = await host(request)
        try:
            return await owner.mirrors.remove_kit(job_id, body.confirm_job_id)
        except (ValueError, OSError) as error:
            raise refusal(error) from error

    _ = mirror, selection, review, start, jobs, job, stop, kit, remove_kit
