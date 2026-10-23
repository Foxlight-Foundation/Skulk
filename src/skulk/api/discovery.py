"""Operator-gated, bounded local discovery controls under preservation ownership."""

from collections.abc import Awaitable, Callable

from fastapi import APIRouter, HTTPException, Request, Response
from foxlight_model_preservation.discovery_models import SourceStatus
from tuf.api.exceptions import RepositoryError

from skulk.store.discovery_contracts import (
    DiscoveryJob,
    DiscoveryPage,
    DiscoveryQuery,
    DiscoveryRequest,
    DiscoveryResults,
    SourceConfirmation,
    SourceReview,
    SourceReviewRequest,
)
from skulk.store.preservation_contracts import Digest, Handle
from skulk.store.preservation_host import PreservationHost


def install_discovery_routes(
    router: APIRouter,
    host: Callable[[Request], Awaitable[PreservationHost]],
    refusal: Callable[[Exception], HTTPException],
) -> None:
    """Use exactly the canonical host and operator gate for every private route."""

    @router.post(
        "/discovery/reviews",
        response_model=SourceReview,
        summary="Review a discovery source file offline",
        description="Inspect a completed source-purpose bootstrap upload of at most 256 KiB. Verify public root metadata and retain a ten-minute review without DNS or supplier requests. Source/cache paths and root bytes remain private; historical cache imports and local-file endpoints use standalone tools.",
    )
    async def review(request: Request, body: SourceReviewRequest) -> SourceReview:
        owner = await host(request)
        try:
            return await owner.discovery.review(body.upload_id)
        except (ValueError, OSError, RepositoryError) as error:
            raise refusal(error) from error

    @router.post(
        "/discovery/sources",
        response_model=SourceStatus,
        summary="Follow a reviewed model source",
        description="Explicitly trust the reviewed file for discovery metadata only. Keep separate local plaintext/private-address metadata grants and all existing verifier floors. Following performs no network request, enrolls no bundle signer and grants no registry or code authority.",
    )
    async def follow(request: Request, body: SourceConfirmation) -> SourceStatus:
        owner = await host(request)
        try:
            return await owner.discovery.follow(body)
        except (ValueError, OSError, RepositoryError) as error:
            raise refusal(error) from error

    @router.delete(
        "/discovery/sources/{source_id}",
        status_code=204,
        summary="Unfollow a source while retaining verifier floors",
        description="Disable a known source without deleting root history, cached evidence or any received model. Refuse while metadata work owns the queue; stop that work before changing enrollment. This causes no network IO.",
    )
    async def remove(request: Request, source_id: Digest) -> Response:
        owner = await host(request)
        try:
            await owner.discovery.remove(source_id)
            return Response(status_code=204)
        except (ValueError, OSError, RepositoryError) as error:
            raise refusal(error) from error

    @router.get(
        "/discovery",
        response_model=DiscoveryPage,
        summary="Read cached sources and durable metadata activity",
        description="Return up to sixteen retained source states and four metadata jobs for owner_id, with next_offset pagination. No endpoint probe, refresh, replay, query transmission or model discovery occurs on read.",
    )
    async def page(
        request: Request, owner_id: Handle, offset: int = 0
    ) -> DiscoveryPage:
        owner = await host(request)
        try:
            return await owner.discovery.page(owner_id, offset)
        except (ValueError, OSError, RepositoryError) as error:
            raise refusal(error) from error

    @router.post(
        "/discovery/search",
        response_model=DiscoveryResults,
        summary="Search cached model listings offline",
        description="Search only this Core's authenticated retained catalogs. Keep query text in the body, off URLs and supplier requests. Return four exact manifest identities and four attributed suppliers each. Use offset for models and an exact-manifest query plus supplier_offset for more suppliers. Source statements never modify archived cards or assert registry trust.",
    )
    async def search(request: Request, body: DiscoveryQuery) -> DiscoveryResults:
        owner = await host(request)
        try:
            return await owner.discovery.search(body)
        except (ValueError, OSError, RepositoryError) as error:
            raise refusal(error) from error

    @router.post(
        "/discovery/jobs",
        response_model=DiscoveryJob,
        status_code=202,
        summary="Start a finite owned metadata job",
        description="Reserve exact owner/operation idempotency before starting a bounded source refresh or selected signed model-metadata handoff. Refresh uses enrolled metadata grants; model metadata requires separate exact-source/entry/manifest key acceptance and separate network grants. No weights, signing, model admission or serving occurs. At most four pending jobs and 1024 retained reservations. Closing Library does not stop Core work; restart records interruption without replay.",
    )
    async def start(request: Request, body: DiscoveryRequest) -> DiscoveryJob:
        owner = await host(request)
        try:
            return owner.discovery.start(body)
        except (ValueError, OSError, RepositoryError) as error:
            raise refusal(error) from error

    @router.get(
        "/discovery/jobs/{job_id}",
        response_model=DiscoveryJob,
        summary="Observe a retained metadata job",
        description="Return one exact metadata receipt without replay. A successful model_metadata job exposes only a validated invitation upload handle; use existing offline invitation review and separately authorized receiving/Restore. Metadata jobs do not use the model-job or governed copy contract.",
    )
    async def observe(request: Request, job_id: Digest) -> DiscoveryJob:
        owner = await host(request)
        try:
            return owner.journal.discovery_job(job_id)[0]
        except (ValueError, OSError, RepositoryError) as error:
            raise refusal(error) from error

    @router.post(
        "/discovery/jobs/{job_id}/cancel",
        response_model=DiscoveryJob,
        summary="Stop an owned metadata update",
        description="Persist cancellation intent, signal the known child, then use a bounded terminate/kill ladder if needed. A completed atomic cache commit wins a late stop. Do not report cancellation before the actual worker stops; accepted catalogs and verifier floors remain retained.",
    )
    async def cancel(request: Request, job_id: Digest) -> DiscoveryJob:
        owner = await host(request)
        try:
            return await owner.discovery.cancel(job_id)
        except (ValueError, OSError, RepositoryError) as error:
            raise refusal(error) from error

    _ = review, follow, remove, page, search, start, observe, cancel
