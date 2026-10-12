# pyright: reportUnusedFunction=false
"""Remote capacity offers for a model the cluster cannot run, from installed providers."""

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

import anyio
from fastapi import APIRouter, HTTPException, Request, Response

from skulk.api.managed_plugins import ProtectedManagementRoute
from skulk.api.operator_auth import TailnetPeerVerifier, authorize_plugin_request
from skulk.connectivity.tailscale import is_tailscale_peer
from skulk.extensions.burst import (
    BurstNodeOffer,
    BurstProvider,
    BurstProviders,
    BurstRecommendation,
    BurstRecommendationRequest,
    BurstRecommendationState,
    NodeBurstProvider,
)
from skulk.extensions.loader import LoadedExtensions
from skulk.extensions.proposal_review import NodeProposalReviewProvider
from skulk.operator.pairing import OperatorPairingService
from skulk.shared.models.inference_assessment import InferenceAssessment

_Result = TypeVar("_Result")

type Assess = Callable[[str, int], Awaitable[InferenceAssessment]]
"""Run the cluster's own inference assessment for one model and context."""

_NOT_REQUIRED = {
    "local": (
        "This model fits on the cluster as it is, so no remote capacity is "
        "offered. Place it normally."
    ),
    "insufficient_evidence": (
        "The cluster cannot yet tell whether this model fits, so no remote "
        "capacity is offered. Try placing it normally first."
    ),
}
_BLOCKED = (
    "This model cannot be served as requested, here or on remote capacity. "
    "See the assessment's reasons."
)
_NO_PROVIDER = (
    "No burst provider is installed on this computer. Install a capability "
    "that provides remote capacity from the Plugins page."
)
_NO_NAMED_PROVIDER = "The named burst provider is not installed on this computer."
_PROVIDER_REQUIRED = "More than one burst provider is available. Name the one to ask."
_NOT_APPROVABLE = (
    "The provider prepared an offer but cannot take approval for it right now. "
    "Nothing was rented. Ask for a new offer."
)
_OFFER_STATES: dict[str, BurstRecommendationState] = {
    "offer": "offer",
    "not_required": "not_required",
    "no_eligible_capacity": "no_eligible_capacity",
    "skulk_build_unavailable": "skulk_build_unavailable",
    "refused": "refused",
    "unavailable": "provider_unavailable",
}


def create_burst_router(
    extensions: LoadedExtensions,
    pairing_service: OperatorPairingService | None,
    assess: Assess,
    *,
    tailnet_peer_verifier: TailnetPeerVerifier = is_tailscale_peer,
) -> APIRouter:
    """Expose installed burst providers and their offers under plugin scopes.

    Reads need ``plugins:read`` and an offer needs ``plugins:manage``, since the
    provider retains an inert proposal. Approving an offer stays the owner
    proposal approval under ``plugins:approve``; nothing here can rent.
    """
    router = APIRouter(
        prefix="/v1/burst", tags=["Burst"], route_class=ProtectedManagementRoute
    )
    capacity = anyio.CapacityLimiter(4)

    async def invoke(call: Callable[[], Awaitable[_Result]]) -> _Result:
        try:
            capacity.acquire_nowait()
        except anyio.WouldBlock:
            raise HTTPException(
                status_code=429, detail="burst offers are busy"
            ) from None
        try:
            async with asyncio.timeout(30):
                return await call()
        except LookupError:
            raise HTTPException(
                status_code=404, detail="burst provider not found"
            ) from None
        except ValueError:
            # The provider's own refusal carries no text to this host. Its
            # single-flight guard is the usual cause, so say how to recover.
            raise HTTPException(
                status_code=409,
                detail=(
                    "the burst provider refused this request; it may be preparing "
                    "another offer, so try again"
                ),
            ) from None
        except TimeoutError:
            raise HTTPException(
                status_code=504, detail="the burst provider timed out"
            ) from None
        except Exception:  # noqa: BLE001 - provider failures must not expose payloads
            raise HTTPException(
                status_code=503, detail="burst providers unavailable"
            ) from None
        finally:
            capacity.release()

    def installations() -> dict[str, NodeBurstProvider]:
        selected: dict[str, NodeBurstProvider] = {}
        for plugin_id, provider in extensions.configuration_providers.items():
            if isinstance(provider, NodeBurstProvider):
                selected[plugin_id] = provider
        if len(selected) > 32:
            raise ValueError("too many burst provider installations")
        return selected

    async def read_providers() -> tuple[BurstProvider, ...]:
        async def read(provider: NodeBurstProvider) -> tuple[BurstProvider, ...]:
            try:
                async with asyncio.timeout(5):
                    return await provider.burst_providers()
            except Exception:  # noqa: BLE001 - one installation must not hide the rest
                return ()

        found = await asyncio.gather(
            *(read(provider) for provider in installations().values())
        )
        return tuple(item for group in found for item in group)[:32]

    @router.get(
        "/providers",
        response_model=BurstProviders,
        summary="List installed burst providers",
        description=(
            "List every installed capability node on this API host that declares "
            "itself a burst provider (descriptor annotation skulk.burst.provider), "
            "with its readiness: available, configuration_required (the message "
            "names the missing setup step), unavailable, or disabled. Includes the "
            "owner's ceilings on one rental's total cost (US dollars) and lifetime. "
            "Requires direct owner authority or plugins:read. Prepares nothing and "
            "contacts no provider service; an empty list makes no statement about "
            "other hosts."
        ),
    )
    async def list_providers(request: Request, response: Response) -> BurstProviders:
        """Authorize, then read each declared provider's own readiness answer."""
        await authorize_plugin_request(
            request, pairing_service, "plugins:read", tailnet_peer_verifier
        )
        response.headers["Cache-Control"] = "no-store"

        async def read() -> BurstProviders:
            return BurstProviders(
                observed_at=int(time.time()), providers=await read_providers()
            )

        return await invoke(read)

    @router.post(
        "/recommendations",
        response_model=BurstRecommendation,
        summary="Ask for remote capacity for a model the cluster cannot run",
        description=(
            "Run the cluster's own assessment for the model and context first; only "
            "an external_required verdict asks a provider. Otherwise the response "
            "says why with state not_required or blocked and the assessment's "
            "reasons. With one available provider (or the one named by plugin_id "
            "and node_id), the provider prepares its cheapest fitting offer within "
            "the owner's limits and retains it as an inert, unsigned proposal. An "
            "offer returns its terms in US dollars, the proposal reference, the "
            "approval revision and the provider's review facts; approving the "
            "reference through the plugin proposal approval (plugins:approve) is "
            "what rents. Every other state carries an operator-readable message. "
            "Requires direct owner authority or plugins:manage. Unknown models "
            "return 404."
        ),
    )
    async def recommend(
        payload: BurstRecommendationRequest, request: Request, response: Response
    ) -> BurstRecommendation:
        """Check the verdict, choose the provider, then ask it for one offer."""
        await authorize_plugin_request(
            request, pairing_service, "plugins:manage", tailnet_peer_verifier
        )
        response.headers["Cache-Control"] = "no-store"
        # The assessment raises its own 404 for an unknown model, so it runs
        # outside the provider failure mapping.
        assessment = await assess(payload.model_id, payload.context_tokens)

        def answer(
            state: BurstRecommendationState,
            message: str,
            provider: BurstProvider | None = None,
        ) -> BurstRecommendation:
            return BurstRecommendation(
                state=state,
                message=message,
                model_id=payload.model_id,
                context_tokens=payload.context_tokens,
                assessment_status=assessment.status,
                assessment_reasons=assessment.reasons[:16],
                provider=provider,
            )

        if assessment.status == "blocked":
            return answer("blocked", _BLOCKED)
        if assessment.status != "external_required":
            return answer("not_required", _NOT_REQUIRED[assessment.status])

        async def offer() -> BurstRecommendation:
            providers = await read_providers()
            if payload.plugin_id is not None:
                named = [
                    item
                    for item in providers
                    if item.plugin_id == payload.plugin_id
                    and item.node_id == payload.node_id
                ]
                if not named:
                    return answer("no_provider", _NO_NAMED_PROVIDER)
                chosen = named[0]
            else:
                if not providers:
                    return answer("no_provider", _NO_PROVIDER)
                available = [item for item in providers if item.state == "available"]
                if len(available) > 1:
                    return answer("provider_required", _PROVIDER_REQUIRED)
                chosen = available[0] if available else providers[0]
            if chosen.state != "available":
                return answer("provider_unavailable", chosen.message, chosen)
            installation = installations().get(chosen.plugin_id)
            if installation is None:
                raise LookupError("burst provider left during the request")
            prepared: BurstNodeOffer = await installation.burst_offer(
                chosen.node_id, payload.model_id, payload.context_tokens
            )
            state = _OFFER_STATES[prepared.state]
            if prepared.reference is None:
                return answer(state, prepared.message, chosen)
            if not isinstance(installation, NodeProposalReviewProvider):
                raise LookupError("burst provider offers no proposal review")
            # The review supplies the approval fence and the provider's own
            # plain-text facts, so the offer can be approved as returned.
            review = await installation.node_proposal(prepared.reference)
            if review.proposal.reference != prepared.reference:
                raise ValueError("proposal review differs from the offer")
            # An offer must be approvable as returned. A proposal that expired,
            # was decided, or cannot take approval right now is not one.
            if (
                review.proposal.state != "pending_approval"
                or review.approval_revision is None
            ):
                return answer("provider_unavailable", _NOT_APPROVABLE, chosen)
            return BurstRecommendation(
                state=state,
                message=prepared.message,
                model_id=payload.model_id,
                context_tokens=payload.context_tokens,
                assessment_status=assessment.status,
                assessment_reasons=assessment.reasons[:16],
                provider=chosen,
                reference=prepared.reference,
                offer=prepared.offer,
                approval_revision=review.approval_revision,
                review=review.fields,
            )

        return await invoke(offer)

    return router
