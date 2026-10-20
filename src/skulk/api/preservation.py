"""Trusted-operator controls for the local authoritative preservation host."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from fastapi import APIRouter, HTTPException, Request

from skulk.store.preservation_contracts import (
    Handle,
    PreservationJob,
    PreservationLocation,
    PreservationPage,
    PreservationPlan,
    PreservationPlanRequest,
    PreservationStart,
)
from skulk.store.preservation_host import PreservationHost
from skulk.store.preservation_journal import PreservationHostError

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
        description="Bind an expiring review plan to an installed generation or retained copy and an enrolled location. The current host supports Preserve and Check, including checking an interrupted job's recorded expected outcome. No byte job starts. Other effect kinds refuse explicitly.",
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

    # Keep explicit references to the registered handlers for strict static tooling.
    _ = (library, setup_device, plan, read_plan, start, observe, cancel)
    return router
