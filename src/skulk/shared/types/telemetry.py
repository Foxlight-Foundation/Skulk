"""The telemetry plane (#279): per-node live readings, off the event log.

Telemetry is gossiped last-write-wins on its own topic and kept in a
``TelemetryView`` that lives outside the event-sourced ``State``. It is never
indexed by the master and never persisted: a stale reading coalesces under the
next one instead of inflating the log. This is the plane a management/edge node
subscribes to so it can render the dashboard and route placement without
joining the inference data plane.

Phase 1 slice 1 carries only ``NodeResources`` here (single reader: the
planner); memory and the rest of the ``node_*`` maps migrate in later slices.
"""

from __future__ import annotations

from skulk.shared.types.common import NodeId
from skulk.shared.types.profiling import NodeResources
from skulk.utils.info_gatherer.info_gatherer import GatheredInfo
from skulk.utils.pydantic_ext import CamelCaseModel


class NodeTelemetry(CamelCaseModel):
    """A single node's latest telemetry reading, published on the TELEMETRY
    topic. Mirrors ``NodeGatheredInfo`` but is a plain message, not a log
    event: it is gossiped node -> all and never enters the event log."""

    node_id: NodeId
    info: GatheredInfo


class TelemetryView:
    """Latest-per-node telemetry, maintained outside event-sourced ``State``.

    Owned by the ``Node`` so it survives master re-election (a freshly elected
    master keeps the cluster's current readings instead of starting blind and
    risking a placement on a management node during the repopulation window).
    Updated only by the telemetry subscriber; readers (planner, API previews)
    hold a reference and read the maps directly.
    """

    def __init__(self) -> None:
        self.node_resources: dict[NodeId, NodeResources] = {}

    def apply(self, message: NodeTelemetry) -> None:
        """Coalesce one telemetry message into the latest-value view."""
        info = message.info
        if isinstance(info, NodeResources):
            self.node_resources[message.node_id] = info
        # Other GatheredInfo variants still travel the event log in slice 1;
        # they migrate onto this plane (and this dispatch) in later slices.
