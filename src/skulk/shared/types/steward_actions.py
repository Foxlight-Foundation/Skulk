"""Typed, event-sourced actions proposed by the intelligent-fabric steward."""

from datetime import datetime
from typing import Literal, Self, cast

from pydantic import ConfigDict, Field, field_validator, model_validator

from skulk.shared.models.model_cards import ModelCard
from skulk.shared.types.common import CommandId, Id, ModelId, NodeId
from skulk.shared.types.worker.downloads import DownloadAttemptId
from skulk.shared.types.worker.instances import Instance, InstanceId, InstanceMeta
from skulk.shared.types.worker.shards import Sharding
from skulk.utils.pydantic_ext import FrozenModel, TaggedModel


class StewardActionProposalId(Id):
    """Stable identity for one approval-gated steward proposal."""


class StewardPlaceModelAction(TaggedModel):
    """Place one catalog-authorized model through the ordinary planner."""

    model_config = ConfigDict(frozen=True, strict=True)

    model_card: ModelCard = Field(description="Exact authorized model card to place.")
    sharding: Sharding = Field(description="Requested placement sharding strategy.")
    instance_meta: InstanceMeta = Field(description="Requested serving topology.")
    min_nodes: int = Field(ge=1, description="Minimum node count for placement.")
    excluded_nodes: tuple[NodeId, ...] = Field(
        default_factory=tuple,
        description="Nodes excluded from this placement only.",
    )

    @field_validator("excluded_nodes", mode="before")
    @classmethod
    def _coerce_excluded_nodes(cls, value: object) -> object:
        """Restore immutable node exclusions from JSON arrays."""
        if isinstance(value, list):
            return tuple(cast("list[object]", value))
        return value


class StewardStopInstanceAction(TaggedModel):
    """Stop one ordinary operator-managed model instance."""

    model_config = ConfigDict(frozen=True, strict=True)

    instance_id: InstanceId = Field(description="Exact ordinary instance to stop.")
    model_id: ModelId = Field(description="Model identity shown to the approving operator.")


class StewardRestartInstanceAction(TaggedModel):
    """Replace one ordinary instance while preserving its placement intent."""

    model_config = ConfigDict(frozen=True, strict=True)

    instance: Instance = Field(
        description="Exact ordinary instance and placement intent to replace."
    )


class StewardCancelDownloadAction(TaggedModel):
    """Cancel one active model download on one node."""

    model_config = ConfigDict(frozen=True, strict=True)

    node_id: NodeId = Field(description="Exact node whose active download is cancelled.")
    node_name: str = Field(
        min_length=1,
        max_length=128,
        description="Friendly node name shown to the approving operator.",
    )
    model_id: ModelId = Field(description="Model whose active download is cancelled.")
    attempt_id: DownloadAttemptId = Field(
        description="Exact live download attempt reviewed by the operator."
    )


StewardBasicAction = (
    StewardPlaceModelAction
    | StewardStopInstanceAction
    | StewardRestartInstanceAction
    | StewardCancelDownloadAction
)

StewardActionProposalStatus = Literal[
    "pending",
    "approved",
    "dispatched",
    "rejected",
    "expired",
    "failed",
]


class StewardActionProposal(FrozenModel):
    """One bounded proposal and its authoritative approval/execution state."""

    proposal_id: StewardActionProposalId = Field(
        default_factory=StewardActionProposalId,
        description="Stable proposal identity used by approval clients.",
    )
    action: StewardBasicAction = Field(description="Exact typed action awaiting approval.")
    rationale: str = Field(
        min_length=1,
        max_length=1024,
        description="Why the steward recommends the action.",
    )
    evidence: tuple[str, ...] = Field(
        min_length=1,
        max_length=8,
        description="Bounded operator-readable evidence supporting the proposal.",
    )
    expected_effect: str = Field(
        min_length=1,
        max_length=1024,
        description="Expected cluster effect if the proposal is approved.",
    )
    created_at: datetime = Field(description="UTC proposal creation time.")
    expires_at: datetime = Field(description="UTC deadline after which approval is refused.")
    status: StewardActionProposalStatus = Field(
        default="pending",
        description="Authoritative proposal lifecycle state.",
    )
    decided_at: datetime | None = Field(
        default=None,
        description="UTC approval or rejection time.",
    )
    decided_by: str | None = Field(
        default=None,
        max_length=128,
        description="Safe actor class that approved or rejected the proposal.",
    )
    command_id: CommandId | None = Field(
        default=None,
        description="Typed command identity dispatched by an approved proposal.",
    )
    outcome: str | None = Field(
        default=None,
        max_length=1024,
        description="Bounded execution outcome or refusal explanation.",
    )

    @field_validator("created_at", "expires_at", "decided_at", mode="before")
    @classmethod
    def _coerce_datetime(cls, value: object) -> object:
        """Restore strict datetimes from JSON event-log snapshots."""
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        return value

    @field_validator("evidence", mode="before")
    @classmethod
    def _coerce_evidence(cls, value: object) -> object:
        """Restore immutable evidence from JSON arrays."""
        if isinstance(value, list):
            return tuple(cast("list[object]", value))
        return value

    @model_validator(mode="after")
    def _validate_proposal_contract(self) -> Self:
        """Reject ambiguous clocks and unbounded operator-visible evidence."""
        timestamps = (self.created_at, self.expires_at, self.decided_at)
        if any(
            value is not None
            and (value.tzinfo is None or value.utcoffset() is None)
            for value in timestamps
        ):
            raise ValueError("steward proposal timestamps must include a timezone")
        if self.expires_at <= self.created_at:
            raise ValueError("steward proposal expiry must follow creation")
        if any(len(item) > 512 for item in self.evidence):
            raise ValueError("steward proposal evidence entries are limited to 512 chars")
        return self
