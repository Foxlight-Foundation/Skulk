"""Telemetry-plane access for extensions (fabric-citizenship Phase 1).

First-class citizenship on the fabric means plane access. This module carries an
extension's telemetry-plane surface, both halves:

- **Read** (``snapshot_cluster`` / :class:`ClusterNodeView`): an immutable,
  per-node snapshot of what the node currently sees (which peers exist, their
  backends, participation role, accelerator vendor, memory, version, liveness,
  and any capability tags peers advertise). It is how a plugin discovers the
  cluster it is part of, rather than being blind to everything beyond the chat
  request in front of it.
- **Advertise** (``ExtensionContext.advertise_capability``, wired in the API):
  a plugin publishes its own capability tag onto the plane so peers discover it,
  the same way native nodes advertise their backends. The tag rides the ordinary
  telemetry emit path (gossiped last-write-wins), surfacing in every peer's
  ``ClusterNodeView.capabilities``.

The read snapshot is deliberately a **flattened, immutable projection** of the
live ``TelemetryView`` maps, not the mutable view object itself: an extension can
never mutate cluster telemetry directly, and the projection is a stable contract
independent of the internal telemetry types. It is transport-agnostic (a view of
the plane's materialized state, not of libp2p vs Zenoh).
"""

from __future__ import annotations

from datetime import datetime
from typing import final

from pydantic import BaseModel, ConfigDict

from skulk.shared.topology import Topology
from skulk.shared.types.capability_nodes import (
    CAPABILITY_NODES_STALE_AFTER_SECONDS,
    CapabilityNodeSummary,
)
from skulk.shared.types.common import NodeId
from skulk.shared.types.telemetry import TelemetryView
from skulk.shared.types.topology import SocketConnection

# Identity subfields default to this sentinel in ``NodeIdentity`` before their
# reading has actually arrived. The snapshot normalizes it to ``None`` so an
# extension can tell "not known yet" from a real value (the read_cluster
# contract that absent readings are ``None``).
_UNKNOWN = "Unknown"


def _known(value: str | None) -> str | None:
    """Return ``value``, or ``None`` if it is the not-yet-populated sentinel."""
    return None if value is None or value == _UNKNOWN else value


@final
class ClusterNodeView(BaseModel):
    """Immutable snapshot of one node's telemetry, handed to extensions.

    Every field beyond ``node_id`` is optional because telemetry is
    last-write-wins and partial: a node that has gossiped resources but not yet
    identity shows ``friendly_name=None``. A field is ``None`` when the
    corresponding reading has not (yet) been received.

    Attributes:
        node_id: The peer's node identity.
        friendly_name: Human-facing node name (for example ``kite4``), or
            ``None`` if identity telemetry has not arrived.
        backends: The engine/compute backend tags the node advertises (sorted),
            for example ``("llama_cpp", "llama_cpp-vulkan")``; empty if unknown.
        participation: The node's participation role (``full`` / ``management``
            / ...), or ``None`` if resource telemetry has not arrived.
        skulk_version: The node's Skulk version string, or ``None``.
        accelerator_vendor: The accelerator vendor the node reports
            (``apple`` / ``amd`` / ``nvidia`` / ``unknown``), or ``None`` when
            no system telemetry has been received.
        ram_total_bytes: Total RAM in bytes, or ``None``.
        last_telemetry: When this node last received either the peer's explicit
            heartbeat or ordinary fallback telemetry, or ``None``.
        capabilities: Extension-advertised capability tags the node offers
            (sorted), for example ``("memory",)``; empty when the node
            advertises nothing. These are opaque strings set by extensions on
            the advertising node, not interpreted by Skulk core.
    """

    # Strict + frozen: this is an extension-facing contract type, so reject any
    # silent coercion and stay immutable. Not a wire type (never serialized over
    # the network), so no camelCase aliasing.
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    node_id: NodeId
    friendly_name: str | None
    backends: tuple[str, ...]
    participation: str | None
    skulk_version: str | None
    accelerator_vendor: str | None
    ram_total_bytes: int | None
    last_telemetry: datetime | None
    capabilities: tuple[str, ...]


def snapshot_cluster(view: TelemetryView) -> tuple[ClusterNodeView, ...]:
    """Project the live ``TelemetryView`` into an immutable per-node snapshot.

    Pure and side-effect free: it reads the view's maps and returns a new tuple,
    never mutating the view. The node set is the union of every telemetry map, so
    a node visible through any single reading appears. Ordering is stable
    (friendly name, then node id) so callers and tests see deterministic output.

    Args:
        view: The node's live telemetry view.

    Returns:
        One :class:`ClusterNodeView` per node currently visible, sorted stably.
    """
    node_ids: set[NodeId] = set()
    node_ids |= set(view.node_resources)
    node_ids |= set(view.node_identities)
    node_ids |= set(view.node_memory)
    node_ids |= set(view.node_system)
    node_ids |= set(view.node_capabilities)
    node_ids |= set(view.node_last_heartbeat)
    node_ids |= set(view.node_last_telemetry)

    snapshots: list[ClusterNodeView] = []
    for node_id in node_ids:
        resources = view.node_resources.get(node_id)
        identity = view.node_identities.get(node_id)
        memory = view.node_memory.get(node_id)
        system = view.node_system.get(node_id)
        accelerator = system.accelerator if system is not None else None
        snapshots.append(
            ClusterNodeView(
                node_id=node_id,
                # `NodeIdentity` seeds absent subfields with "Unknown" during a
                # partial merge, so normalize those to None to keep the contract.
                friendly_name=_known(identity.friendly_name) if identity is not None else None,
                backends=tuple(sorted(resources.backends)) if resources is not None else (),
                participation=resources.participation if resources is not None else None,
                skulk_version=_known(identity.skulk_version) if identity is not None else None,
                accelerator_vendor=accelerator.vendor if accelerator is not None else None,
                ram_total_bytes=memory.ram_total.in_bytes if memory is not None else None,
                last_telemetry=view.last_liveness_receipt(node_id),
                capabilities=tuple(sorted(view.node_capabilities.get(node_id, frozenset()))),
            )
        )
    snapshots.sort(key=lambda snapshot: (snapshot.friendly_name or "", snapshot.node_id))
    return tuple(snapshots)


MAX_PEER_ADDRESSES = 8
"""Upper bound on the reachable addresses one peer view lists."""


@final
class CapabilityPeerView(BaseModel):
    """One peer host's published capability nodes, as this node sees them.

    How an installed plugin finds the same plugin on other hosts of its
    cluster, and reads the handshakes they address to this host. Only fresh
    readings appear: a host whose last capability-node reading is older than
    the staleness bound is absent, the same rule the dashboard applies.

    Attributes:
        node_id: The peer's node identity.
        friendly_name: Human-facing node name, or ``None`` if identity
            telemetry has not arrived.
        addresses: Addresses this node currently reaches the peer at, from its
            own probed topology edges (sorted, at most eight). Session-only
            edges are left out: their address is the connection's observed
            remote end, which for a NAT'd member is not a dialable listener.
        capability_nodes: The summaries the peer last published.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    node_id: NodeId
    friendly_name: str | None
    addresses: tuple[str, ...]
    capability_nodes: tuple[CapabilityNodeSummary, ...]


def snapshot_capability_peers(
    view: TelemetryView,
    topology: Topology,
    self_node: NodeId,
    now: datetime,
) -> tuple[CapabilityPeerView, ...]:
    """Project fresh peer capability-node readings and reachable addresses.

    Pure and side-effect free. This host itself is excluded; its own
    summaries are what it publishes, not what it receives.

    Args:
        view: The node's live telemetry view.
        topology: The node's current topology.
        self_node: This node's identity.
        now: The current time, compared with each reading's receipt time.

    Returns:
        One view per peer with a fresh reading, sorted by name then identity.
    """
    reachable: dict[NodeId, set[str]] = {}
    for connection in topology.out_edges(self_node):
        edge = connection.edge
        if isinstance(edge, SocketConnection) and not edge.session:
            reachable.setdefault(connection.sink, set()).add(
                edge.sink_multiaddr.ip_address
            )
    peers: list[CapabilityPeerView] = []
    for node_id, summaries in view.node_capability_nodes.items():
        received = view.node_capability_nodes_received_at.get(node_id)
        if (
            node_id == self_node
            or received is None
            or (now - received).total_seconds() > CAPABILITY_NODES_STALE_AFTER_SECONDS
        ):
            continue
        identity = view.node_identities.get(node_id)
        peers.append(
            CapabilityPeerView(
                node_id=node_id,
                friendly_name=_known(identity.friendly_name)
                if identity is not None
                else None,
                addresses=tuple(sorted(reachable.get(node_id, set())))[
                    :MAX_PEER_ADDRESSES
                ],
                capability_nodes=summaries,
            )
        )
    peers.sort(key=lambda peer: (peer.friendly_name or "", peer.node_id))
    return tuple(peers)
