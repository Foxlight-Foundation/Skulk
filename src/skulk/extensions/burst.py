"""Remote capacity from installed burst providers, declared by descriptor annotations.

A capability declares itself a burst provider with reserved descriptor
annotations (``skulk.burst.provider`` and ``skulk.burst.memory_pools``). Core
reads only those keys and never names a provider. Readiness and offers come
from the provider's own owner process; core never sees provider price units,
signing material or credentials. An offer is inert: the provider retains an
unsigned proposal, and approval stays the ordinary owner proposal action under
its own ``plugins:approve`` scope.
"""

from typing import Literal, Protocol, Self, runtime_checkable

from pydantic import Field, model_validator

from skulk.extensions.capabilities import CapabilityDescriptor
from skulk.extensions.configuration import ConfigurationDigest, ConfigurationNodeId
from skulk.extensions.proposal_actions import ProposalOperation, ProposalOperationId
from skulk.extensions.proposal_review import ProposalField, ProposalReference
from skulk.utils.pydantic_ext import FrozenModel

BURST_PROVIDER_ANNOTATION = "skulk.burst.provider"
"""Descriptor annotation naming the burst provider contract version."""

BURST_MEMORY_POOLS_ANNOTATION = "skulk.burst.memory_pools"
"""Descriptor annotation listing the memory pools a provider's machines offer."""

BURST_CONTRACT_VERSION = "1"
"""The only burst provider contract version this core understands."""

type BurstMemoryPool = Literal["vram", "ram"]

_DOLLARS = r"^[0-9]{1,12}\.[0-9]{6}$"

type BurstProviderState = Literal[
    "available", "configuration_required", "unavailable", "disabled"
]


def burst_memory_pools(
    descriptor: CapabilityDescriptor,
) -> tuple[BurstMemoryPool, ...] | None:
    """Return the declared memory pools if this descriptor is a burst provider.

    A descriptor is a provider only when it names exactly the contract version
    this core understands and at least one known memory pool. Anything else is
    not a provider here, so a newer contract is never half-understood.
    """
    annotations = descriptor.annotations or {}
    if annotations.get(BURST_PROVIDER_ANNOTATION) != BURST_CONTRACT_VERSION:
        return None
    pools: list[BurstMemoryPool] = []
    for item in annotations.get(BURST_MEMORY_POOLS_ANNOTATION, "").split(","):
        if item == "vram" or item == "ram":
            if item not in pools:
                pools.append(item)
        else:
            return None
    return tuple(pools) or None


class BurstNodeReadiness(FrozenModel):
    """One provider's own answer to whether it can prepare an offer now."""

    state: Literal["available", "configuration_required", "unavailable"] = Field(
        description="Provider readiness; the message names the first missing step."
    )
    message: str = Field(
        min_length=1, max_length=512, description="Operator-readable next step."
    )
    maximum_total_usd: str | None = Field(
        default=None,
        pattern=_DOLLARS,
        description="Owner ceiling on any one rental's total cost, in US dollars.",
    )
    maximum_lifetime_seconds: int | None = Field(
        default=None,
        gt=0,
        description="Owner ceiling on any one rental's lifetime, in seconds.",
    )


class BurstProvider(FrozenModel):
    """One installed burst provider on this API host and whether it can offer now."""

    plugin_id: str = Field(
        pattern=r"^managed\.[a-z0-9][a-z0-9._-]{0,80}$",
        description="Exact installed managed plugin identity.",
    )
    node_id: ConfigurationNodeId = Field(description="Stable installed node identity.")
    bundle_id: str = Field(description="Bundle the provider node runs.")
    version: str = Field(description="Bundle version the provider node runs.")
    title: str | None = Field(
        default=None, description="Operator title, if the provider declares one."
    )
    memory_pools: tuple[BurstMemoryPool, ...] = Field(
        min_length=1, description="Memory pools the provider's machines offer."
    )
    state: BurstProviderState = Field(
        description=(
            "available, configuration_required, unavailable, or disabled when the "
            "installation is turned off."
        )
    )
    message: str = Field(
        min_length=1, max_length=512, description="Operator-readable next step."
    )
    maximum_total_usd: str | None = Field(
        default=None,
        pattern=_DOLLARS,
        description="Owner ceiling on any one rental's total cost, in US dollars.",
    )
    maximum_lifetime_seconds: int | None = Field(
        default=None,
        gt=0,
        description="Owner ceiling on any one rental's lifetime, in seconds.",
    )


class BurstProviders(FrozenModel):
    """Installed burst providers on this API host, never a cluster-wide absence proof."""

    scope: Literal["local_api_host"] = Field(
        default="local_api_host",
        description="Only providers installed on the API host that answered.",
    )
    observed_at: int = Field(gt=0, description="Read time as UTC Unix seconds.")
    providers: tuple[BurstProvider, ...] = Field(
        max_length=32, description="Every declared provider, ready or not."
    )


class BurstOffer(FrozenModel):
    """What the operator approves, with money as exact US dollar strings."""

    gpu: str = Field(
        min_length=1, max_length=128, description="Provider GPU type for the machine."
    )
    region: str = Field(min_length=1, max_length=128, description="Provider region.")
    hourly_price_usd: str = Field(
        pattern=_DOLLARS,
        description="Quoted hourly price; advisory, never the spending limit.",
    )
    maximum_hourly_price_usd: str = Field(
        pattern=_DOLLARS, description="Owner ceiling on the hourly price."
    )
    maximum_total_usd: str = Field(
        pattern=_DOLLARS,
        description="Worst-case total the approval permits, from owner limits.",
    )
    plan_expires_at: int = Field(
        gt=0,
        description="UTC Unix seconds after which the offer can no longer be approved.",
    )
    resource_expires_at: int = Field(
        gt=0, description="UTC Unix seconds when the rental ends at the latest."
    )
    model_id: str = Field(
        min_length=1, max_length=512, description="Model the rental would serve."
    )
    context_tokens: int = Field(gt=0, description="Context the rental is sized for.")
    backend: str = Field(
        min_length=1, max_length=128, description="Inference backend on the machine."
    )
    skulk_build: str = Field(
        min_length=1, max_length=256, description="Skulk build the machine installs."
    )


type BurstNodeOfferState = Literal[
    "offer",
    "not_required",
    "no_eligible_capacity",
    "skulk_build_unavailable",
    "refused",
    "unavailable",
]


class BurstNodeOffer(FrozenModel):
    """One provider's offer and its retained proposal, or why it has none."""

    state: BurstNodeOfferState = Field(description="Provider outcome.")
    message: str = Field(
        min_length=1, max_length=512, description="Operator-readable outcome."
    )
    reference: ProposalReference | None = Field(
        default=None, description="Retained inert proposal, present only on an offer."
    )
    offer: BurstOffer | None = Field(
        default=None, description="Offer terms, present only on an offer."
    )

    @model_validator(mode="after")
    def _offer_iff_offered(self) -> Self:
        offered = self.state == "offer"
        if (self.offer is not None) != offered or (
            self.reference is not None
        ) != offered:
            raise ValueError("an offer and its reference accompany only an offer")
        return self


@runtime_checkable
class NodeBurstProvider(Protocol):
    """Optional facet of an installation whose nodes provide remote capacity."""

    async def burst_providers(self) -> tuple[BurstProvider, ...]:
        """Read every declared provider node's readiness without preparing anything."""
        ...

    async def burst_offer(
        self, node_id: str, model_id: str, context_tokens: int
    ) -> BurstNodeOffer:
        """Ask one provider node to prepare and retain one inert offer."""
        ...


type BurstPhase = Literal[
    "approving",
    "refused",
    "attention",
    "uncertain",
    "starting",
    "installing",
    "joining",
    "downloading",
    "loading",
    "ready",
    "releasing",
    "released",
]

type BurstJobPhase = Literal[
    "starting",
    "installing",
    "joining",
    "joined",
    "downloading",
    "loading",
    "ready",
    "releasing",
    "absent",
]


class BurstJob(FrozenModel):
    """The provider controller's progress for one rented machine."""

    phase: BurstJobPhase = Field(description="Controller phase for the machine.")
    ready: bool = Field(description="In the controller's fresh ready set.")
    model_id: str | None = Field(
        default=None, max_length=512, description="Model the machine serves."
    )
    instance_id: str | None = Field(
        default=None,
        max_length=128,
        description="Skulk instance serving the model, once observed.",
    )
    resource_expires_at: int = Field(
        gt=0, description="UTC Unix seconds when the rental ends at the latest."
    )
    observed_at: int = Field(gt=0, description="Controller observation time.")


class BurstNodeStatus(FrozenModel):
    """One provider node's view of a rental: approval, machine and release."""

    phase: BurstPhase = Field(description="Summary for the operator, absence first.")
    message: str = Field(
        min_length=1, max_length=512, description="Operator-readable phase."
    )
    acquisition: ProposalOperation = Field(
        description="The approval that rented the machine, with reconciliation."
    )
    job: BurstJob | None = Field(
        default=None, description="Machine progress, once the controller knows it."
    )
    release: ProposalOperation | None = Field(
        default=None, description="The release action, once one was requested."
    )


@runtime_checkable
class NodeBurstOperations(Protocol):
    """Optional facet that follows and releases rentals by their approval ID."""

    async def burst_status(self, node_id: str, operation_id: str) -> BurstNodeStatus:
        """Read approval, machine and release progress without any effect."""
        ...

    async def burst_release(
        self, node_id: str, operation_id: str, operator_id: str
    ) -> BurstNodeStatus:
        """Stop the model, then release the machine under the authenticated operator."""
        ...


@runtime_checkable
class NodeBurstRentals(Protocol):
    """Optional facet that lists an installation's active rentals."""

    async def burst_rentals(self) -> tuple["BurstOperation", ...]:
        """Read rentals not yet confirmed gone, across the installation's nodes."""
        ...


class BurstOperationRequest(FrozenModel):
    """Approve one returned offer, exactly as recommended."""

    operation_id: ProposalOperationId = Field(
        description=(
            "Caller-chosen durable action ID (32 lowercase hex). Save it before "
            "sending: repeating the request with it observes the same action."
        )
    )
    reference: ProposalReference = Field(
        description="The recommendation's reference, unchanged."
    )
    approval_revision: ConfigurationDigest = Field(
        description="The recommendation's approval revision, unchanged."
    )


_ACCEPTED_MESSAGES: dict[str, str] = {
    "approving": "The approval is being signed and sent to the provider.",
    "refused": "The provider refused this approval; nothing was rented.",
    "attention": "The approval was interrupted; resume it from the Plugins page.",
    "uncertain": (
        "The provider may or may not have rented the machine. Its cleanup "
        "follows it; nothing is retried."
    ),
    "starting": "The machine is starting.",
}


class BurstOperation(FrozenModel):
    """One approved rental's progress, from approval to confirmed release."""

    plugin_id: str = Field(description="Provider installation.")
    node_id: ConfigurationNodeId = Field(description="Provider node.")
    operation_id: ProposalOperationId = Field(description="The approval's action ID.")
    phase: BurstPhase = Field(
        description=(
            "approving, refused, attention, uncertain, starting, installing, "
            "joining, downloading, loading, ready, releasing or released."
        )
    )
    message: str = Field(description="Operator-readable phase.")
    instance_id: str | None = Field(
        default=None, description="Skulk instance serving the model, once observed."
    )
    model_id: str | None = Field(default=None, description="Model the machine serves.")
    resource_expires_at: int | None = Field(
        default=None, description="UTC Unix seconds when the rental ends at the latest."
    )
    acquisition: ProposalOperation = Field(
        description="The approval that rented the machine, with reconciliation."
    )
    release: ProposalOperation | None = Field(
        default=None, description="The release action, once one was requested."
    )

    @classmethod
    def accepted(cls, operation: ProposalOperation) -> "BurstOperation":
        """Project a just-submitted approval before the machine exists.

        The provider's status read would compete with the approval it just
        started for the same host channel, so the first answer comes from the
        approval's own durable progress.
        """
        phase: BurstPhase = (
            "refused"
            if operation.phase == "refused"
            else "attention"
            if operation.phase == "approval_interrupted"
            else "uncertain"
            if operation.phase == "uncertain"
            else "starting"
            if operation.phase in ("acknowledged", "succeeded")
            else "approving"
        )
        return cls(
            plugin_id=operation.reference.plugin_id,
            node_id=operation.reference.node_id,
            operation_id=operation.operation_id,
            phase=phase,
            message=_ACCEPTED_MESSAGES[phase],
            acquisition=operation,
        )

    @classmethod
    def of(
        cls, plugin_id: str, node_id: str, operation_id: str, status: BurstNodeStatus
    ) -> "BurstOperation":
        """Project one provider status onto the API's operation view."""
        job = status.job
        return cls(
            plugin_id=plugin_id,
            node_id=node_id,
            operation_id=operation_id,
            phase=status.phase,
            message=status.message,
            instance_id=job.instance_id if job is not None else None,
            model_id=job.model_id if job is not None else None,
            resource_expires_at=job.resource_expires_at if job is not None else None,
            acquisition=status.acquisition,
            release=status.release,
        )


class BurstRentals(FrozenModel):
    """Active rentals known to the burst providers on this API host."""

    scope: Literal["local_api_host"] = Field(
        default="local_api_host",
        description="Only providers installed on the API host that answered.",
    )
    observed_at: int = Field(gt=0, description="Read time as UTC Unix seconds.")
    rentals: tuple[BurstOperation, ...] = Field(
        max_length=64,
        description="Rentals approved and not yet confirmed gone, newest first.",
    )


class BurstRecommendationRequest(FrozenModel):
    """Ask for remote capacity for one model the cluster cannot run."""

    model_id: str = Field(
        min_length=1, max_length=512, description="Exact known model alias."
    )
    context_tokens: int = Field(
        default=8192,
        ge=256,
        le=1_048_576,
        description="Input plus output tokens per request, as the assessment takes.",
    )
    plugin_id: str | None = Field(
        default=None,
        pattern=r"^managed\.[a-z0-9][a-z0-9._-]{0,80}$",
        description="Provider installation; optional when one provider is available.",
    )
    node_id: ConfigurationNodeId | None = Field(
        default=None, description="Provider node; required with plugin_id."
    )

    @model_validator(mode="after")
    def _provider_named_whole(self) -> Self:
        if (self.plugin_id is None) != (self.node_id is None):
            raise ValueError("name both the provider installation and node, or neither")
        return self


type BurstRecommendationState = Literal[
    "offer",
    "not_required",
    "blocked",
    "no_provider",
    "provider_required",
    "provider_unavailable",
    "no_eligible_capacity",
    "skulk_build_unavailable",
    "refused",
]


class BurstRecommendation(FrozenModel):
    """An approvable remote offer, or the reason there is none.

    On an offer, approving ``reference`` with ``approval_revision`` through the
    owner proposal approval rents the machine; nothing is rented before that.
    """

    state: BurstRecommendationState = Field(description="Outcome of the request.")
    message: str = Field(
        min_length=1, max_length=512, description="Operator-readable outcome."
    )
    model_id: str = Field(description="Model the request named.")
    context_tokens: int = Field(description="Context the request named.")
    assessment_status: Literal[
        "local", "external_required", "blocked", "insufficient_evidence"
    ] = Field(description="The cluster's own verdict for this model and context.")
    assessment_reasons: tuple[str, ...] = Field(
        default=(), max_length=16, description="The assessment's stated reasons."
    )
    provider: BurstProvider | None = Field(
        default=None, description="Provider asked, when one was selected."
    )
    reference: ProposalReference | None = Field(
        default=None, description="Retained inert proposal, present only on an offer."
    )
    offer: BurstOffer | None = Field(
        default=None, description="Offer terms, present only on an offer."
    )
    approval_revision: ConfigurationDigest | None = Field(
        default=None,
        description="Approval fence for the reference; null when approval is unavailable.",
    )
    review: tuple[ProposalField, ...] = Field(
        default=(),
        max_length=32,
        description="The provider's plain-text review facts for the offer.",
    )
