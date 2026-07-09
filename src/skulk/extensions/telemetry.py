"""Read access to the telemetry plane for extensions (fabric-citizenship Phase 1).

First-class citizenship on the fabric means plane access. This module gives an
extension the *read* half of the telemetry plane: an immutable, per-node snapshot
of what the node currently sees (which peers exist, their backends, participation
role, accelerator vendor, memory, version, and liveness). It is how a plugin
discovers the cluster it is part of, rather than being blind to everything beyond
the chat request in front of it.

The snapshot is deliberately a **flattened, immutable projection** of the live
``TelemetryView`` maps, not the mutable view object itself: an extension can never
mutate cluster telemetry, and the projection is a stable contract independent of
the internal telemetry types. It is transport-agnostic (a view of the plane's
materialized state, not of libp2p vs Zenoh). The *advertise* half (a plugin
publishing its own capability onto the plane) is a separate, later slice because
it is a wire-schema change.
"""

from __future__ import annotations

from datetime import datetime
from typing import final

from pydantic import BaseModel

from skulk.shared.types.common import NodeId
from skulk.shared.types.telemetry import TelemetryView


@final
class ClusterNodeView(BaseModel, frozen=True):
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
        last_telemetry: When this node last received any telemetry from the
            peer (its liveness signal), or ``None``.
    """

    node_id: NodeId
    friendly_name: str | None
    backends: tuple[str, ...]
    participation: str | None
    skulk_version: str | None
    accelerator_vendor: str | None
    ram_total_bytes: int | None
    last_telemetry: datetime | None


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
                friendly_name=identity.friendly_name if identity is not None else None,
                backends=tuple(sorted(resources.backends)) if resources is not None else (),
                participation=resources.participation if resources is not None else None,
                skulk_version=identity.skulk_version if identity is not None else None,
                accelerator_vendor=accelerator.vendor if accelerator is not None else None,
                ram_total_bytes=memory.ram_total.in_bytes if memory is not None else None,
                last_telemetry=view.node_last_telemetry.get(node_id),
            )
        )
    snapshots.sort(key=lambda snapshot: (snapshot.friendly_name or "", snapshot.node_id))
    return tuple(snapshots)
