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
    BurstJob,
    BurstNodeOffer,
    BurstNodeStatus,
    BurstOffer,
    BurstOperation,
    BurstProvider,
    BurstProviders,
    BurstRecommendation,
    BurstRentals,
)
from skulk.extensions.proposal_actions import ProposalApproval, ProposalOperation
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
        self.actors: list[str] = []
        self.released_by: list[str] = []
        self.excluded: list[tuple[str, ...]] = []
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
        self,
        node_id: str,
        model_id: str,
        context_tokens: int,
        excluded_node_ids: tuple[str, ...] = (),
    ) -> BurstNodeOffer:
        """Retain one inert offer, or refuse without saying why to this host."""
        if self.refuse:
            raise ValueError("sensitive provider refusal")
        self.offers.append((node_id, model_id, context_tokens))
        self.excluded.append(excluded_node_ids)
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

    def operation(self, operation_id: str) -> ProposalOperation:
        """The acquisition's durable progress, as the owner journal reports it."""
        return ProposalOperation(
            operation_id=operation_id,
            reference=self.reference,
            phase="accepted",
            updated_at=1_800_000_000,
        )

    async def approve_node_proposal(
        self, mutation: ProposalApproval, operator_id: str
    ) -> ProposalOperation:
        """Record the authenticated actor; the mutation is exactly the offer's."""
        assert mutation.reference == self.reference
        assert mutation.review_revision == "b" * 64
        self.actors.append(operator_id)
        return self.operation(mutation.operation_id)

    async def node_proposal_operation(
        self, node_id: str, operation_id: str
    ) -> ProposalOperation:
        """Not part of the burst routes."""
        raise AssertionError("not read")

    async def resume_node_proposal(
        self, node_id: str, operation_id: str, operator_id: str
    ) -> ProposalOperation:
        """Not part of the burst routes."""
        raise AssertionError("not read")

    def status(self, operation_id: str, *, released: bool) -> BurstNodeStatus:
        return BurstNodeStatus(
            phase="releasing" if released else "ready",
            message="The model is ready on the remote machine.",
            acquisition=self.operation(operation_id).model_copy(
                update={"phase": "acknowledged"}
            ),
            job=BurstJob(
                phase="ready",
                ready=True,
                model_id="example/large",
                instance_id="instance-1",
                resource_expires_at=1_900_003_600,
                observed_at=1_800_000_000,
            ),
            release=self.operation("2" * 32) if released else None,
        )

    async def burst_rentals(self) -> tuple[BurstOperation, ...]:
        """One ready rental, while it is not released."""
        if self.released_by:
            return ()
        return (
            BurstOperation.of(
                self.name, "gpu", "1" * 32, self.status("1" * 32, released=False)
            ),
        )

    async def burst_status(self, node_id: str, operation_id: str) -> BurstNodeStatus:
        """A ready rental."""
        assert node_id == "gpu"
        return self.status(operation_id, released=bool(self.released_by))

    async def burst_release(
        self, node_id: str, operation_id: str, operator_id: str
    ) -> BurstNodeStatus:
        """Release under the authenticated operator only."""
        assert node_id == "gpu"
        self.released_by.append(operator_id)
        return self.status(operation_id, released=True)


def client(
    *providers: Provider, verdict: Verdict = "external_required"
) -> httpx.AsyncClient:
    app = FastAPI()

    async def assess(
        model_id: str, context_tokens: int, excluded: tuple[str, ...]
    ) -> InferenceAssessment:
        assert (model_id, context_tokens) == ("example/large", 8192)
        assessed_exclusions.append(excluded)
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
assessed_exclusions: list[tuple[str, ...]] = []


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

    async def assess(
        model_id: str, context_tokens: int, excluded: tuple[str, ...]
    ) -> InferenceAssessment:
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


OPERATION = "1" * 32
APPROVAL = {
    "operationId": OPERATION,
    "reference": {
        "pluginId": "managed.fixture",
        "nodeId": "gpu",
        "proposalId": "a" * 64,
        "proposalDigest": "a" * 64,
    },
    "approvalRevision": "b" * 64,
}
FOLLOW = f"/v1/burst/operations/{OPERATION}?plugin_id=managed.fixture&node_id=gpu"


async def test_an_offer_is_approved_followed_and_released_through_the_api() -> None:
    provider = Provider()
    async with client(provider) as http:
        approved = await http.post("/v1/burst/operations", json=APPROVAL)
        followed = await http.get(FOLLOW)
        released = await http.delete(FOLLOW)
        missing = await http.get(
            f"/v1/burst/operations/{OPERATION}?plugin_id=managed.absent&node_id=gpu"
        )
    assert approved.status_code == 200
    first = BurstOperation.model_validate_json(approved.text)
    assert first.phase == "approving" and first.operation_id == OPERATION
    assert provider.actors == ["local-owner"]
    ready = BurstOperation.model_validate_json(followed.text)
    assert ready.phase == "ready" and ready.instance_id == "instance-1"
    assert ready.model_id == "example/large"
    gone = BurstOperation.model_validate_json(released.text)
    assert gone.phase == "releasing" and gone.release is not None
    assert provider.released_by == ["local-owner"]
    assert missing.status_code == 404


async def test_approval_and_release_need_the_approve_scope(tmp_path: Path) -> None:
    service, exchange = paired_service(tmp_path)
    provider = Provider()
    app = FastAPI()

    async def assess(
        model_id: str, context_tokens: int, excluded: tuple[str, ...]
    ) -> InferenceAssessment:
        return assessment("external_required")

    app.include_router(
        create_burst_router(LoadedExtensions([provider]), service, assess)
    )
    service.set_plugin_grant(
        exchange.device_id,
        PluginGrantUpdate(
            expected_revision=0, scopes=("plugins:read", "plugins:manage")
        ),
    )
    headers = {"Authorization": f"Bearer {exchange.access_token}"}
    for application in (app, OperatorGatewayAuthorization(app, service)):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application),
            base_url="https://localhost",
            headers=headers,
        ) as http:
            assert (await http.get(FOLLOW)).status_code == 200
            assert (
                await http.post("/v1/burst/operations", json=APPROVAL)
            ).status_code == 403
            assert (await http.delete(FOLLOW)).status_code == 403
    assert provider.actors == [] and provider.released_by == []
    service.set_plugin_grant(
        exchange.device_id,
        PluginGrantUpdate(
            expected_revision=1, scopes=("plugins:read", "plugins:approve")
        ),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=OperatorGatewayAuthorization(app, service)),
        base_url="https://localhost",
        headers=headers,
    ) as http:
        assert (
            await http.post("/v1/burst/operations", json=APPROVAL)
        ).status_code == 200
        assert (
            await http.post(
                "/v1/burst/operations", json=APPROVAL | {"operatorId": "spoofed"}
            )
        ).status_code == 422
        assert (await http.delete(FOLLOW)).status_code == 200
    assert provider.actors == [str(exchange.device_id)]
    assert provider.released_by == [str(exchange.device_id)]


async def test_active_rentals_are_listed_until_released() -> None:
    first, second = Provider(), Provider("managed.second")
    async with client(first, second) as http:
        listed = BurstRentals.model_validate_json(
            (await http.get("/v1/burst/operations")).text
        )
        assert {(item.plugin_id, item.instance_id) for item in listed.rentals} == {
            ("managed.fixture", "instance-1"),
            ("managed.second", "instance-1"),
        }
        assert (await http.delete(FOLLOW)).status_code == 200
        after = BurstRentals.model_validate_json(
            (await http.get("/v1/burst/operations")).text
        )
    assert [item.plugin_id for item in after.rentals] == ["managed.second"]


async def test_excluded_nodes_reach_the_verdict_and_the_provider() -> None:
    """An operator who excluded nodes is assessed and offered without them."""
    provider = Provider()
    assessed_exclusions.clear()
    async with client(provider) as http:
        answer = await recommend(http)
        response = await http.post(
            "/v1/burst/recommendations",
            json=REQUEST | {"excludedNodeIds": ["node-a", "node-b"]},
        )
    assert answer.state == "offer" and response.status_code == 200
    assert assessed_exclusions == [(), ("node-a", "node-b")]
    assert provider.excluded == [(), ("node-a", "node-b")]
