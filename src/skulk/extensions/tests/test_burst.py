"""Burst providers are declared by reserved annotations and answer for themselves."""

from pathlib import Path

import pytest
from pydantic import JsonValue

from skulk.extensions.burst import BurstNodeOffer, burst_memory_pools
from skulk.extensions.capabilities import CapabilityDescriptor
from skulk.extensions.managed import (
    BUDGET_UNAVAILABLE,
    ManagedConnection,
    ManagedNode,
    ManagedOwner,
)
from skulk.extensions.refusals import ManagedRefusalError

PROVIDER = {"skulk.burst.provider": "1", "skulk.burst.memory_pools": "vram"}


def descriptor(annotations: dict[str, str] | None) -> CapabilityDescriptor:
    """One lifecycle contract carrying only the annotations under test."""
    return CapabilityDescriptor(
        id="capacity",
        version="1.0.0",
        title="Capacity",
        description="Remote capacity",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        annotations=annotations,
    )


def test_only_the_understood_contract_declares_a_provider() -> None:
    assert burst_memory_pools(descriptor(PROVIDER)) == ("vram",)
    both = {**PROVIDER, "skulk.burst.memory_pools": "vram,ram,vram"}
    assert burst_memory_pools(descriptor(both)) == ("vram", "ram")
    # A newer contract, a missing or unknown pool, or no annotations at all is
    # not a provider here, so a contract is never half-understood.
    for annotations in (
        {**PROVIDER, "skulk.burst.provider": "2"},
        {"skulk.burst.provider": "1"},
        {**PROVIDER, "skulk.burst.memory_pools": "vram,disk"},
        {"latency": "interactive"},
        None,
    ):
        assert burst_memory_pools(descriptor(annotations)) is None


class _Owner:
    """A managed owner whose control socket is replaced by recorded replies."""

    def __init__(self, tmp_path: Path, *nodes: ManagedNode) -> None:
        self.owner = ManagedOwner(
            ManagedConnection(plugin_id="managed.fixture", state_root=str(tmp_path))
        )
        self.owner.nodes = nodes
        self.owner.available = True
        self.owner.host_callbacks_available = True
        self.sent: list[dict[str, JsonValue]] = []
        self.replies: list[dict[str, JsonValue] | Exception] = []

        async def refresh() -> None:
            return None

        async def request(
            message: dict[str, JsonValue], *, timeout: float = 30
        ) -> dict[str, JsonValue]:
            self.sent.append(message)
            reply = self.replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply

        object.__setattr__(self.owner, "refresh", refresh)
        object.__setattr__(self.owner, "_request", request)


def node(node_id: str, annotations: dict[str, str] | None = PROVIDER) -> ManagedNode:
    return ManagedNode(
        node_id=node_id,
        bundle_id=f"bundle-{node_id}",
        version="0.4.0",
        status="ready",
        configurable=True,
        title="Remote GPUs",
        descriptors=(descriptor(annotations),),
    )


async def test_each_provider_node_answers_its_own_readiness(tmp_path: Path) -> None:
    fixture = _Owner(tmp_path, node("gpu"), node("other", None))
    fixture.replies = [
        {
            "state": "configuration_required",
            "message": "Turn on proposals in this burst provider's settings.",
            "maximum_total_usd": "10.000000",
            "maximum_lifetime_seconds": 3600,
        }
    ]

    (provider,) = await fixture.owner.burst_providers()

    assert fixture.sent == [
        {
            "operation": "burst-readiness",
            "plugin_id": "managed.fixture",
            "node_id": "gpu",
        }
    ]
    assert (provider.plugin_id, provider.node_id, provider.title) == (
        "managed.fixture",
        "gpu",
        "Remote GPUs",
    )
    assert provider.state == "configuration_required"
    assert "Turn on proposals" in provider.message
    assert provider.memory_pools == ("vram",)
    assert provider.maximum_total_usd == "10.000000"


async def test_turned_off_or_unreachable_providers_are_said_so_without_asking(
    tmp_path: Path,
) -> None:
    fixture = _Owner(tmp_path, node("gpu"))
    fixture.owner.manager_enabled = False
    (disabled,) = await fixture.owner.burst_providers()
    assert disabled.state == "disabled" and "turned off" in disabled.message
    fixture.owner.manager_enabled = True
    fixture.owner.host_callbacks_available = False
    (unreachable,) = await fixture.owner.burst_providers()
    assert unreachable.state == "unavailable"
    assert fixture.sent == []
    # A refused or malformed answer is unavailable too, never an error that
    # would hide the other providers.
    fixture.owner.host_callbacks_available = True
    fixture.replies = [ValueError("managed operation refused"), {"state": "maybe"}]
    assert (await fixture.owner.burst_providers())[0].state == "unavailable"
    assert (await fixture.owner.burst_providers())[0].state == "unavailable"


def offer_reply(**reference: str) -> dict[str, JsonValue]:
    return {
        "state": "offer",
        "message": "Remote capacity is available.",
        "reference": {
            "plugin_id": "managed.fixture",
            "node_id": "gpu",
            "proposal_id": "a" * 64,
            "proposal_digest": "a" * 64,
            **reference,
        },
        "offer": {
            "gpu": "NVIDIA A100 80GB",
            "region": "US",
            "hourly_price_usd": "1.190000",
            "maximum_hourly_price_usd": "2.000000",
            "maximum_total_usd": "4.000000",
            "plan_expires_at": 1_900_000_000,
            "resource_expires_at": 1_900_003_600,
            "model_id": "example/large",
            "context_tokens": 8192,
            "backend": "llama_server-cuda",
            "skulk_build": "2.0.2 (dev 02c68630)",
        },
    }


async def test_an_offer_is_forwarded_exactly_and_must_name_this_node(
    tmp_path: Path,
) -> None:
    fixture = _Owner(tmp_path, node("gpu"), node("other", None))
    fixture.replies = [offer_reply()]
    offer = await fixture.owner.burst_offer("gpu", "example/large", 8192)
    assert isinstance(offer, BurstNodeOffer) and offer.state == "offer"
    assert fixture.sent == [
        {
            "operation": "burst-offer",
            "plugin_id": "managed.fixture",
            "node_id": "gpu",
            "model_id": "example/large",
            "context_tokens": 8192,
        }
    ]
    fixture.replies = [offer_reply(node_id="elsewhere")]
    with pytest.raises(ValueError, match="another installation"):
        await fixture.owner.burst_offer("gpu", "example/large", 8192)
    fixture.replies = [offer_reply()]
    with pytest.raises(ValueError, match="another workload"):
        await fixture.owner.burst_offer("gpu", "example/large", 4096)
    with pytest.raises(LookupError):
        await fixture.owner.burst_offer("other", "example/large", 8192)


def status_reply(**acquisition: str) -> dict[str, JsonValue]:
    return {
        "phase": "ready",
        "message": "The model is ready on the remote machine.",
        "acquisition": {
            "operation_id": "1" * 32,
            "reference": {
                "plugin_id": "managed.fixture",
                "node_id": "gpu",
                "proposal_id": "a" * 64,
                "proposal_digest": "a" * 64,
            },
            "phase": "acknowledged",
            "updated_at": 1_800_000_000,
            **acquisition,
        },
        "job": {
            "phase": "ready",
            "ready": True,
            "model_id": "example/large",
            "instance_id": "instance-1",
            "resource_expires_at": 1_800_003_600,
            "observed_at": 1_800_000_000,
        },
        "release": None,
    }


async def test_status_and_release_are_forwarded_for_the_exact_operation(
    tmp_path: Path,
) -> None:
    fixture = _Owner(tmp_path, node("gpu"), node("other", None))
    fixture.replies = [status_reply(), status_reply()]
    status = await fixture.owner.burst_status("gpu", "1" * 32)
    released = await fixture.owner.burst_release("gpu", "1" * 32, "operator-1")
    assert status.job is not None and status.job.instance_id == "instance-1"
    assert released.phase == "ready"
    assert fixture.sent == [
        {
            "operation": "burst-status",
            "plugin_id": "managed.fixture",
            "node_id": "gpu",
            "operation_id": "1" * 32,
        },
        {
            "operation": "burst-release",
            "plugin_id": "managed.fixture",
            "node_id": "gpu",
            "operation_id": "1" * 32,
            "operator_id": "operator-1",
        },
    ]
    fixture.replies = [status_reply(operation_id="2" * 32)]
    with pytest.raises(ValueError, match="another operation"):
        await fixture.owner.burst_status("gpu", "1" * 32)
    with pytest.raises(LookupError):
        await fixture.owner.burst_status("other", "1" * 32)


async def test_rentals_are_listed_per_provider_node_and_fenced(tmp_path: Path) -> None:
    fixture = _Owner(tmp_path, node("gpu"), node("other", None))
    fixture.replies = [{"rentals": [status_reply()]}]
    (rental,) = await fixture.owner.burst_rentals()
    assert (rental.plugin_id, rental.node_id, rental.operation_id) == (
        "managed.fixture",
        "gpu",
        "1" * 32,
    )
    assert rental.instance_id == "instance-1" and rental.phase == "ready"
    assert fixture.sent == [
        {"operation": "burst-list", "plugin_id": "managed.fixture", "node_id": "gpu"}
    ]
    # A node that answers for another installation contributes nothing.
    foreign = status_reply()
    acquisition = foreign["acquisition"]
    assert isinstance(acquisition, dict)
    reference = acquisition["reference"]
    assert isinstance(reference, dict)
    reference["plugin_id"] = "managed.other"
    fixture.replies = [{"rentals": [foreign]}]
    assert await fixture.owner.burst_rentals() == ()


async def test_exclusions_are_sent_only_when_present(tmp_path: Path) -> None:
    """A provider that predates exclusions still gets the ordinary offer."""
    fixture = _Owner(tmp_path, node("gpu"))
    fixture.replies = [offer_reply(), offer_reply()]
    await fixture.owner.burst_offer("gpu", "example/large", 8192)
    await fixture.owner.burst_offer("gpu", "example/large", 8192, ("node-a",))
    assert "excluded_node_ids" not in fixture.sent[0]
    assert fixture.sent[1]["excluded_node_ids"] == ["node-a"]


async def test_the_budget_is_read_from_the_exact_provider_node(tmp_path: Path) -> None:
    fixture = _Owner(tmp_path, node("gpu"), node("other", None))
    fixture.replies = [
        {
            "state": "available",
            "limit_usd": "10.000000",
            "spent_usd": "1.250000",
            "held_usd": "2.250000",
            "left_usd": "6.500000",
            "rentals": 2,
            "counting_since": 1_800_000_000,
            "change_limit_action": "campaign-limits",
            "start_over_action": "start-over",
        }
    ]
    budget = await fixture.owner.burst_budget("gpu")
    assert (budget.limit_usd, budget.spent_usd, budget.left_usd) == (
        "10.000000",
        "1.250000",
        "6.500000",
    )
    assert budget.start_over_action == "start-over"
    assert fixture.sent == [
        {"operation": "burst-budget", "plugin_id": "managed.fixture", "node_id": "gpu"}
    ]
    with pytest.raises(LookupError):
        await fixture.owner.burst_budget("other")
    # An answer outside the contract is refused rather than shown.
    malformed: tuple[dict[str, JsonValue], ...] = (
        {"state": "available", "limit_usd": "ten dollars"},
        # Available without its amounts.
        {"state": "available"},
        # Unavailable without saying why, or with amounts beside it.
        {"state": "unavailable"},
        {"state": "unavailable", "message": "Not now.", "limit_usd": "1.000000"},
    )
    for reply in malformed:
        fixture.replies = [reply]
        with pytest.raises(ValueError):
            await fixture.owner.burst_budget("gpu")
    # A provider released before budget reads refuses the operation.
    fixture.replies = [ManagedRefusalError(None)]
    older = await fixture.owner.burst_budget("gpu")
    assert older.state == "unavailable" and older.message == BUDGET_UNAVAILABLE
    fixture.replies = [
        {"state": "unavailable", "message": "The cleanup service did not answer."}
    ]
    unavailable = await fixture.owner.burst_budget("gpu")
    assert unavailable.limit_usd is None and unavailable.message is not None
