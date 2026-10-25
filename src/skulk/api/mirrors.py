"""Operator-gated persistent public mirrors under the preservation owner."""

from collections.abc import Awaitable, Callable

from fastapi import APIRouter, HTTPException, Request

from skulk.store.mirror_contracts import (
    MirrorJob,
    MirrorJobsPage,
    MirrorPage,
    MirrorRequest,
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

    @router.post(
        "/mirror/jobs",
        response_model=MirrorJob,
        status_code=202,
        summary="Start one explicit permanent mirror decision",
        description="Configure offline, retain an exact already accepted public copy with reviewed every-member terms, select a finite public run or separately approved standing resume, control pin/block/retry, or confirm exact stopped-owner deletion. The immutable client operation observes its original job on replay. Every new effect binds current policy; installation, metadata reads, receiving, or a source file grant no sharing. Configuration and local admission clear restart authority as appropriate; no policy evicts retained copies. Direct public PMF transport remains unavailable.",
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

    _ = mirror, start, jobs, job, stop
