"""Pure helpers for placing a restart recovery record's intents again."""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from skulk.master.placement import (
    PlacementError,
    replacement_command_for_download_failed_instance,
)
from skulk.recovery.record import ClusterIntentRecord, PlacementIntent
from skulk.shared.types.common import NodeId
from skulk.shared.types.events import InstanceFailureRecorded
from skulk.shared.types.restart_recovery import RestartRecoveryAdvert
from skulk.shared.types.telemetry import TelemetryView
from skulk.shared.types.worker.instances import Instance, InstanceFailure, InstanceId

INFO_WAIT_AFTER_SETTLE_SECONDS = 300.0
"""How long restore keeps retrying one placement while the planner still
lacks node information (memory, resources)."""

HOST_WAIT_AFTER_SETTLE_SECONDS = 120.0
"""How long a multi-node cold start waits, after settling, for every peer to
advertise its record and for the chosen record's machines to rejoin."""


@dataclass(frozen=True)
class RecordSource:
    """Where a candidate record lives: this node's own, or a peer's to fetch."""

    written_at: datetime
    node_id: NodeId | None
    """``None`` for this node's own record."""
    advert: RestartRecoveryAdvert | None


def record_sources(
    own: ClusterIntentRecord | None,
    adverts: Mapping[NodeId, RestartRecoveryAdvert],
    *,
    local_node: NodeId,
    connected_node_ids: set[NodeId],
    unfetchable: set[NodeId],
) -> list[RecordSource]:
    """List the records a cold-started master could restore from, newest first.

    A peer's record counts only while it is connected, has a record, runs an
    API to serve it, and has not already failed to serve it. On a tie the
    master's own record wins, since it needs no fetch.

    Args:
        own: This node's record, if it has one.
        adverts: Every node's latest advert, from telemetry.
        local_node: This node, whose own advert (if looped back) is ignored.
        connected_node_ids: Nodes in the current topology.
        unfetchable: Peers whose record could not be fetched this session.

    Returns:
        Candidate sources, freshest first.
    """
    sources: list[RecordSource] = []
    if own is not None:
        sources.append(RecordSource(own.stamp.written_at, None, None))
    for node_id, advert in adverts.items():
        if (
            node_id == local_node
            or node_id not in connected_node_ids
            or node_id in unfetchable
            or advert.written_at is None
            or advert.api_port is None
        ):
            continue
        sources.append(RecordSource(advert.written_at, node_id, advert))
    return sorted(
        sources, key=lambda source: (source.written_at, source.node_id is None), reverse=True
    )


def unrestored_intents(
    intents: Sequence[PlacementIntent], instances: Mapping[InstanceId, Instance]
) -> list[PlacementIntent]:
    """Drop intents an existing ordinary instance already satisfies.

    A master promoted in the middle of a restore inherits the placements its
    predecessor already created; restoring the whole record again would place
    them twice. Each existing ordinary instance of the same model and width
    accounts for one intent.

    Args:
        intents: The record's placements, in order.
        instances: Instances already in replicated state.

    Returns:
        The intents still to request, in their original order.
    """
    available: Counter[tuple[str, int]] = Counter()
    for instance in instances.values():
        if instance.system_role is not None:
            continue
        try:
            width = replacement_command_for_download_failed_instance(
                instance, frozenset()
            ).min_nodes
        except PlacementError:
            continue
        available[(str(instance.shard_assignments.model_id), width)] += 1
    remaining: list[PlacementIntent] = []
    for intent in intents:
        key = (str(intent.model_id), intent.min_nodes)
        if available[key] > 0:
            available[key] -= 1
            continue
        remaining.append(intent)
    return remaining


def live_nodes_by_install_id(
    telemetry_view: TelemetryView, connected_node_ids: set[NodeId]
) -> dict[UUID, NodeId]:
    """Map stable install IDs to the live node IDs of connected machines.

    Args:
        telemetry_view: Node telemetry carrying each node's identity.
        connected_node_ids: Nodes currently in the cluster topology.

    Returns:
        Install ID to node ID, for connected nodes that report one.
    """
    return {
        identity.node_install_id: node_id
        for node_id, identity in telemetry_view.node_identities.items()
        if identity.node_install_id is not None and node_id in connected_node_ids
    }


def intent_nodes(
    intent: PlacementIntent, live: Mapping[UUID, NodeId]
) -> tuple[frozenset[NodeId], frozenset[NodeId]]:
    """Translate an intent's machines into this session's node IDs.

    Args:
        intent: The recorded placement.
        live: Install ID to live node ID, from ``live_nodes_by_install_id``.

    Returns:
        The machines it ran on that are back (preferred), and the machines the
        operator excluded that are back (excluded). Machines that have not
        returned simply drop out of both.
    """
    preferred = frozenset(
        live[install_id] for install_id in intent.host_install_ids if install_id in live
    )
    excluded = frozenset(
        live[install_id]
        for install_id in intent.excluded_install_ids
        if install_id in live
    )
    return preferred - excluded, excluded


def restore_failure_event(
    intent: PlacementIntent, reason: str, *, now: datetime
) -> InstanceFailureRecorded:
    """Record, where the operator looks, that a placement could not come back.

    Args:
        intent: The placement that could not be restored.
        reason: Plain-language cause, completing "Could not bring back X after
            the restart: ...".
        now: UTC time of the decision.

    Returns:
        The failure event; no instance ever existed in this session, so it
        carries a fresh identity.
    """
    return InstanceFailureRecorded(
        failure=InstanceFailure(
            instance_id=InstanceId(),
            model_id=intent.model_id,
            system_role=None,
            error_code="placement_failed",
            error_message=(
                f"Could not bring back {intent.model_id} after the restart: {reason}"
            )[:2048],
            affected_node_ids=(),
            recorded_at=now,
        )
    )
