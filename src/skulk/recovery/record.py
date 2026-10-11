"""The per-node record that lets a cluster come back after a cold restart.

Replicated ``State`` lives in memory. Failover carries it from a surviving
node, but a single-node reboot, or every node restarting at once, leaves no
survivor: the next master starts empty, so placed models and the steward
proposal history vanish. Each node therefore keeps this small record on its
own disk, derived from its replicated view, and a master that starts cold
reads it back.

The record holds intent, never live wiring. Node IDs are regenerated on every
process start, so a stored ``Instance`` would name runners and nodes that no
longer exist. A placement is kept as what the operator asked for (model,
sharding, width, context, exclusions) plus where it ran, with machines named
by their stable install IDs. Restored steward proposals are only ever
terminal: a restart must never re-arm an action nobody approved after it.
"""

import os
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Literal, cast
from uuid import UUID

from loguru import logger
from pydantic import Field, ValidationError, field_validator

from skulk.master.placement import replacement_command_for_download_failed_instance
from skulk.shared.types.common import ModelId, NodeId, SessionId
from skulk.shared.types.state import State
from skulk.shared.types.steward_actions import (
    StewardActionProposal,
    StewardActionProposalId,
)
from skulk.shared.types.worker.instances import InstanceMeta
from skulk.shared.types.worker.shards import Sharding
from skulk.utils.pydantic_ext import FrozenModel

RECORD_FORMAT_VERSION = 1

RESTARTED_BEFORE_DECISION = "The cluster restarted before a decision."
RESTARTED_BEFORE_FINISH = (
    "The cluster restarted before this finished; nothing further was done."
)


class RecoveryStamp(FrozenModel):
    """When a record was last confirmed current, for choosing the freshest one."""

    session_master_node_id: NodeId = Field(
        description="Master of the session the record was written in (ephemeral)."
    )
    session_election_clock: int = Field(
        ge=0, description="Election clock of that session."
    )
    last_event_applied_idx: int = Field(
        ge=0, description="Last indexed event the writing node had applied."
    )
    written_at: datetime = Field(
        description=(
            "UTC time the content was last confirmed current. Election clocks "
            "and event indices restart with the cluster, so freshness across a "
            "cold restart compares this."
        )
    )

    @field_validator("written_at", mode="before")
    @classmethod
    def _coerce_datetime(cls, value: object) -> object:
        """Restore strict datetimes from the JSON file."""
        return datetime.fromisoformat(value) if isinstance(value, str) else value


class PlacementIntent(FrozenModel):
    """One ordinary placement as the operator asked for it, free of live IDs."""

    model_id: ModelId = Field(description="Model to place again.")
    sharding: Sharding = Field(description="Sharding strategy it was placed with.")
    instance_meta: InstanceMeta = Field(description="Serving topology it used.")
    min_nodes: int = Field(ge=1, description="Width: how many machines it spanned.")
    requested_context_tokens: int | None = Field(
        default=None, description="Context window the operator asked for, if any."
    )
    host_install_ids: tuple[UUID, ...] = Field(
        description="Stable install IDs of the machines it ran on, in rank order."
    )
    excluded_install_ids: tuple[UUID, ...] = Field(
        default=(), description="Machines the operator excluded, by install ID."
    )

    @field_validator("host_install_ids", "excluded_install_ids", mode="before")
    @classmethod
    def _coerce_uuids(cls, value: object) -> object:
        """Restore tuples of UUIDs from JSON arrays of strings."""
        if isinstance(value, list):
            items = cast("list[object]", value)
            return tuple(UUID(item) if isinstance(item, str) else item for item in items)
        return value


class ClusterIntentRecord(FrozenModel):
    """What a cold-started cluster should bring back."""

    format_version: Literal[1] = Field(
        default=RECORD_FORMAT_VERSION, description="Record schema version."
    )
    stamp: RecoveryStamp = Field(description="Freshness of this node's view.")
    placements: tuple[PlacementIntent, ...] = Field(
        default=(), description="Ordinary placements, in the order they were made."
    )
    steward_proposals: tuple[StewardActionProposal, ...] = Field(
        default=(), description="The bounded steward proposal audit."
    )

    @field_validator("placements", "steward_proposals", mode="before")
    @classmethod
    def _coerce_tuples(cls, value: object) -> object:
        """Restore immutable tuples from JSON arrays."""
        return tuple(cast("list[object]", value)) if isinstance(value, list) else value

    def same_content(self, other: "ClusterIntentRecord | None") -> bool:
        """Whether two records would restore the same thing, ignoring freshness."""
        return (
            other is not None
            and self.placements == other.placements
            and self.steward_proposals == other.steward_proposals
        )


def derive_cluster_intent(
    state: State,
    *,
    install_ids: Mapping[NodeId, UUID],
    session_id: SessionId,
    written_at: datetime,
) -> ClusterIntentRecord:
    """Build the record from one node's replicated view.

    System placements (the steward) are left out: the steward's own placement
    invariant brings it back. A node without a known install ID cannot be
    named across a restart, so it is omitted from hosts and exclusions; the
    placement still restores, and the planner chooses in its place.

    Args:
        state: The node's replicated view (it must have applied an event).
        install_ids: Stable install ID per live node, from node telemetry.
        session_id: The session the view belongs to.
        written_at: UTC time to stamp as "current as of".

    Returns:
        The record to persist.
    """
    placements: list[PlacementIntent] = []
    for instance in state.instances.values():
        if instance.system_role is not None:
            continue
        intent = replacement_command_for_download_failed_instance(instance, frozenset())
        hosts = tuple(
            install_ids[node_id]
            for node_id in instance.shard_assignments.node_to_runner
            if node_id in install_ids
        )
        excluded = tuple(
            install_ids[node_id]
            for node_id in sorted(instance.excluded_nodes)
            if node_id in install_ids
        )
        placements.append(
            PlacementIntent(
                model_id=instance.shard_assignments.model_id,
                sharding=intent.sharding,
                instance_meta=intent.instance_meta,
                min_nodes=intent.min_nodes,
                requested_context_tokens=intent.requested_context_tokens,
                host_install_ids=hosts,
                excluded_install_ids=excluded,
            )
        )
    proposals = tuple(
        sorted(
            state.steward_action_proposals.values(),
            key=lambda proposal: proposal.created_at,
        )
    )
    return ClusterIntentRecord(
        stamp=RecoveryStamp(
            session_master_node_id=session_id.master_node_id,
            session_election_clock=session_id.election_clock,
            last_event_applied_idx=max(state.last_event_applied_idx, 0),
            written_at=written_at,
        ),
        placements=tuple(placements),
        steward_proposals=proposals,
    )


def restored_steward_proposals(
    record: ClusterIntentRecord, *, now: datetime
) -> dict[StewardActionProposalId, StewardActionProposal]:
    """Return the proposal history a cold-started master should carry.

    Settled proposals come back unchanged. Anything still open at the restart
    is closed: a pending proposal expires (nobody decided, and its reviewed
    target is gone), and an approved or dispatched one fails, because its
    follow-through ran in the master that is gone. Nothing restored can cause
    an action.
    """
    restored: dict[StewardActionProposalId, StewardActionProposal] = {}
    for proposal in record.steward_proposals:
        if proposal.status == "pending":
            proposal = proposal.model_copy(
                update={
                    "status": "expired",
                    "decided_at": now,
                    "decided_by": "fabric_restart",
                    "outcome": RESTARTED_BEFORE_DECISION,
                }
            )
        elif proposal.status in {"approved", "dispatched"}:
            proposal = proposal.model_copy(
                update={
                    "status": "failed",
                    "decided_at": proposal.decided_at or now,
                    "outcome": RESTARTED_BEFORE_FINISH,
                }
            )
        restored[proposal.proposal_id] = proposal
    return restored


def write_record(path: Path, record: ClusterIntentRecord) -> None:
    """Write the record durably and atomically, readable only by its owner.

    The record exists for abrupt restarts (power loss, a kernel panic), so a
    rename alone is not enough: without a flush the machine can come back with
    the rename applied but the contents missing. The new content goes to a
    sibling file that is owner-only from the moment it exists, is flushed to
    disk, and then replaces the old record in one rename, after which the
    directory entry is flushed too. A crash at any point leaves either the old
    record or the new one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.tmp")
    descriptor = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        # A staging file left by an earlier crash keeps its old mode; narrow it
        # before any record bytes are written.
        os.fchmod(handle.fileno(), 0o600)
        handle.write(record.model_dump_json(by_alias=True).encode())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(staging, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def read_record(path: Path) -> ClusterIntentRecord | None:
    """Return the record on disk, or ``None`` when absent or unreadable.

    An unreadable record is reported and ignored: starting without recovery
    is the behavior before this record existed, never worse.
    """
    try:
        raw = path.read_text()
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as error:
        # Damaged bytes must not stop the node from starting: the record is
        # read while the node is being built.
        logger.warning(f"Restart recovery record unreadable at {path}: {error}")
        return None
    try:
        return ClusterIntentRecord.model_validate_json(raw)
    except ValidationError as error:
        logger.warning(
            f"Restart recovery record at {path} is not valid; ignoring it: "
            f"{error.error_count()} problem(s)"
        )
        return None
