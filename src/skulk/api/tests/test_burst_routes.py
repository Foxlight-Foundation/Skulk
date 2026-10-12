"""Burst offers ask a provider only for a model the cluster itself cannot run."""

from pathlib import Path
from typing import Literal, final

import httpx
import pytest
from fastapi import FastAPI

from skulk.api.burst import create_burst_router
from skulk.api.operator_gateway import OperatorGatewayAuthorization
from skulk.api.tests.test_operator_gateway import paired_service
from skulk.api.tests.test_plugins import ManagedExtension
from skulk.extensions import LoadedExtensions
from skulk.extensions.burst import (
    BurstNodeOffer,
    BurstOffer,
    BurstProvider,
    BurstProviders,
    BurstRecommendation,
)
from skulk.extensions.proposal_review import (
    ProposalField,
    ProposalPage,
    ProposalReference,
    ProposalReview,
    ProposalSummary,
)
from skulk.operator.pairing import PluginGrantUpdate
from skulk.operator.plugin_scopes import required_plugin_scope
from skulk.shared.models.inference_assessment import (
    InferenceAssessment,
    InferenceWorkload,
)

_HEADERS = {"X-Skulk-Dashboard": "pairing-v1", "Origin": "http://localhost"}

type Verdict = Literal["local", "external_required", "blocked", "insufficient_evidence"]


def assessment(status: Verdict) -> InferenceAssessment:
    """The cluster's own verdict, with only the fields a recommendation reads."""
    return InferenceAssessment(
        model_id="example/large",
        card_digest="c" * 64,
        source_repository="example/large",
        source_revision=None,
        selected_file=None,
        quantization="q4",
        workload=InferenceWorkload(context_tokens=8192),
        status=status,
        read_at="2026-10-12T00:00:00Z",
        reasons=("No node has enough memory.",),
        assumptions=(),
        uncertainty=(),
        weight_bytes=1,
    )


@final
class Provider(ManagedExtension):
    """A burst provider installation whose node answers from fixed state."""

    def __init__(self, name: str = "managed.fixture") -> None:
        super().__init__()
        self.name = name
        self.state: Literal["available", "configuration_required"] = "available"
        self.offers: list[tuple[str, str, int]] = []
        self.refuse = False
        self.review_state: Literal["pending_approval", "expired"] = "pending_approval"
        self.approvable = True
        self.reference = ProposalReference(
            plugin_id=name,
            node_id="gpu",
            proposal_id="a" * 64,
            proposal_digest="a" * 64,
        )

    async def burst_providers(self) -> tuple[BurstProvider, ...]:
        """One provider node, ready or waiting on its settings."""
        return (
            BurstProvider(
                plugin_id=self.name,
                node_id="gpu",
                bundle_id="bundle",
                version="0.4.0",
                memory_pools=("vram",),
                state=self.state,
                message="Ready to prepare offers."
                if self.state == "available"
                else "Turn on proposals in this burst provider's settings.",
                maximum_total_usd="10.000000",
                maximum_lifetime_seconds=3600,
            ),
        )

    async def burst_offer(
        self, node_id: str, model_id: str, context_tokens: int
    ) -> BurstNodeOffer:
        """Retain one inert offer, or refuse without saying why to this host."""
        if self.refuse:
            raise ValueError("sensitive provider refusal")
        self.offers.append((node_id, model_id, context_tokens))
        return BurstNodeOffer(
            state="offer",
            message="Remote capacity is available.",
            reference=self.reference,
            offer=BurstOffer(
                gpu="NVIDIA A100 80GB",
                region="US",
                hourly_price_usd="1.190000",
                maximum_hourly_price_usd="2.000000",
                maximum_total_usd="4.000000",
                plan_expires_at=1_900_000_000,
                resource_expires_at=1_900_003_600,
                model_id=model_id,
                context_tokens=context_tokens,
                backend="llama_server-cuda",
                skulk_build="2.0.2",
            ),
        )

    async def node_proposals(self, node_id: str, offset: int = 0) -> ProposalPage:
        """Listing is not part of a recommendation."""
        raise AssertionError("not read")

    async def node_proposal(self, reference: ProposalReference) -> ProposalReview:
        """The fresh review supplies the approval fence and plain-text facts."""
        assert reference == self.reference
        return ProposalReview(
            proposal=ProposalSummary(
                reference=reference,
                summary="Rent one A100 for example/large.",
                expires_at=1_900_000_000,
                state=self.review_state,
            ),
            observed_at=1_800_000_000,
            approval_revision="b" * 64 if self.approvable else None,
            fields=(ProposalField(label="GPU", value="NVIDIA A100 80GB"),),
        )


def client(
    *providers: Provider, verdict: Verdict = "external_required"
) -> httpx.AsyncClient:
    app = FastAPI()

    async def assess(model_id: str, context_tokens: int) -> InferenceAssessment:
        assert (model_id, context_tokens) == ("example/large", 8192)
        return assessment(verdict)

    app.include_router(
        create_burst_router(LoadedExtensions(list(providers)), None, assess)
    )
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 1)),
        base_url="http://localhost",
        headers=_HEADERS,
    )


REQUEST = {"modelId": "example/large", "contextTokens": 8192}


async def recommend(http: httpx.AsyncClient, **provider: str) -> BurstRecommendation:
    """Ask for one recommendation and read it back as the typed response."""
    response = await http.post("/v1/burst/recommendations", json=REQUEST | provider)
    assert response.status_code == 200
    return BurstRecommendation.model_validate_json(response.text)


@pytest.mark.parametrize(
    ("verdict", "state"),
    [
        ("local", "not_required"),
        ("insufficient_evidence", "not_required"),
        ("blocked", "blocked"),
    ],
)
async def test_no_provider_is_asked_unless_the_model_needs_remote_capacity(
    verdict: Verdict, state: str
) -> None:
    provider = Provider()
    async with client(provider, verdict=verdict) as http:
        answer = await recommend(http)
    assert answer.state == state and answer.assessment_status == verdict
    assert answer.assessment_reasons == ("No node has enough memory.",)
    assert answer.offer is None and provider.offers == []


async def test_an_offer_comes_back_approvable_with_its_terms_in_dollars() -> None:
    provider = Provider()
    async with client(provider) as http:
        listed = await http.get("/v1/burst/providers")
        response = await http.post("/v1/burst/recommendations", json=REQUEST)
    providers = BurstProviders.model_validate_json(listed.text)
    assert providers.providers[0].state == "available"
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    # The reference crosses the API in the same camelCase the approval takes.
    assert '"proposalDigest":"' + "a" * 64 in response.text
    answer = BurstRecommendation.model_validate_json(response.text)
    assert answer.state == "offer" and answer.reference == provider.reference
    assert answer.approval_revision == "b" * 64
    assert answer.offer is not None and answer.offer.maximum_total_usd == "4.000000"
    assert [(item.label, item.value) for item in answer.review] == [
        ("GPU", "NVIDIA A100 80GB")
    ]
    assert provider.offers == [("gpu", "example/large", 8192)]


async def test_every_missing_provider_case_says_why() -> None:
    async with client() as http:
        none = await recommend(http)
    assert none.state == "no_provider" and "Plugins page" in none.message
    waiting = Provider()
    waiting.state = "configuration_required"
    async with client(waiting) as http:
        unready = await recommend(http)
    assert unready.state == "provider_unavailable"
    assert "Turn on proposals" in unready.message
    assert waiting.offers == []
    first, second = Provider(), Provider("managed.second")
    async with client(first, second) as http:
        ambiguous = await recommend(http)
        named = await recommend(http, pluginId="managed.second", nodeId="gpu")
        missing = await recommend(http, pluginId="managed.absent", nodeId="gpu")
        half = await http.post(
            "/v1/burst/recommendations", json=REQUEST | {"pluginId": "managed.second"}
        )
    assert ambiguous.state == "provider_required"
    assert named.state == "offer" and second.offers and not first.offers
    assert missing.state == "no_provider"
    assert half.status_code == 422


async def test_a_provider_refusal_reveals_nothing_of_its_own() -> None:
    provider = Provider()
    provider.refuse = True
    async with client(provider) as http:
        response = await http.post("/v1/burst/recommendations", json=REQUEST)
    assert response.status_code == 409
    assert "sensitive" not in response.text and "try again" in response.text


def test_burst_routes_take_the_plugin_scopes() -> None:
    assert required_plugin_scope("GET", "/v1/burst/providers") == "plugins:read"
    assert (
        required_plugin_scope("POST", "/v1/burst/recommendations") == "plugins:manage"
    )
    assert required_plugin_scope("POST", "/v1/burst/operations") == "plugins:approve"
    assert (
        required_plugin_scope("DELETE", "/v1/burst/operations/x") == "plugins:approve"
    )


async def test_a_paired_device_needs_manage_scope_for_an_offer(tmp_path: Path) -> None:
    service, exchange = paired_service(tmp_path)
    provider = Provider()
    app = FastAPI()

    async def assess(model_id: str, context_tokens: int) -> InferenceAssessment:
        return assessment("external_required")

    app.include_router(
        create_burst_router(LoadedExtensions([provider]), service, assess)
    )
    service.set_plugin_grant(
        exchange.device_id,
        PluginGrantUpdate(expected_revision=0, scopes=("plugins:read",)),
    )
    headers = {"Authorization": f"Bearer {exchange.access_token}"}
    for application in (app, OperatorGatewayAuthorization(app, service)):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application),
            base_url="https://localhost",
            headers=headers,
        ) as http:
            assert (await http.get("/v1/burst/providers")).status_code == 200
            assert (
                await http.post("/v1/burst/recommendations", json=REQUEST)
            ).status_code == 403
    assert provider.offers == []
    service.set_plugin_grant(
        exchange.device_id,
        PluginGrantUpdate(
            expected_revision=1, scopes=("plugins:read", "plugins:manage")
        ),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=OperatorGatewayAuthorization(app, service)),
        base_url="https://localhost",
        headers=headers,
    ) as http:
        answer = await recommend(http)
    assert answer.state == "offer"


async def test_an_offer_that_cannot_be_approved_is_not_returned_as_one() -> None:
    """An expiry or approval race between offer and review is not an offer."""
    expired = Provider()
    expired.review_state = "expired"
    unapprovable = Provider()
    unapprovable.approvable = False
    for provider in (expired, unapprovable):
        async with client(provider) as http:
            answer = await recommend(http)
        assert answer.state == "provider_unavailable"
        assert "Nothing was rented" in answer.message
        assert answer.reference is None and answer.approval_revision is None
