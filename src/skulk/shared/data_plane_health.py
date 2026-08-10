"""Shared positive-evidence health predicates for the inference data plane."""

from collections.abc import Mapping, Set

from skulk.shared.types.common import NodeId
from skulk.shared.types.profiling import NodeResources


def zenoh_isolated_nodes(
    *,
    live_nodes: Set[NodeId],
    node_resources: Mapping[NodeId, NodeResources],
) -> frozenset[NodeId]:
    """Return live nodes whose Zenoh inference data plane is known isolated.

    Isolation is derived only from positive telemetry: at least two live nodes
    must advertise Zenoh, and a returned node must report exactly zero
    connected peers. A missing resource advertisement or ``None`` peer count is
    a startup/telemetry unknown and remains eligible rather than causing a
    false placement failure.

    Args:
        live_nodes: Nodes currently present in cluster topology or membership.
        node_resources: Latest per-node resource and data-plane telemetry.

    Returns:
        The subset of live Zenoh nodes reporting zero connected peers.
    """
    live_zenoh_resources = {
        node_id: resources
        for node_id, resources in node_resources.items()
        if node_id in live_nodes and resources.data_transport == "zenoh"
    }
    if len(live_zenoh_resources) < 2:
        return frozenset()
    return frozenset(
        node_id
        for node_id, resources in live_zenoh_resources.items()
        if resources.zenoh_connected_peers == 0
    )
