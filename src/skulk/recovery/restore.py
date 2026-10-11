"""Pure helpers for placing a restart recovery record's intents again."""

from collections.abc import Mapping
from datetime import datetime
from uuid import UUID

from skulk.recovery.record import PlacementIntent
from skulk.shared.types.common import NodeId
from skulk.shared.types.events import InstanceFailureRecorded
from skulk.shared.types.telemetry import TelemetryView
from skulk.shared.types.worker.instances import InstanceFailure, InstanceId

INFO_WAIT_AFTER_SETTLE_SECONDS = 300.0
"""How long restore keeps retrying when the planner still lacks node
information (memory, resources) after the post-election settling window."""


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
