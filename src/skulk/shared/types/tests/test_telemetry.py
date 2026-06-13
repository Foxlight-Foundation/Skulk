"""Telemetry plane coverage (#279): wire round-trip + view coalescing.

The NodeTelemetry message must survive the gossip path (model_dump_json ->
bytes -> model_validate_json under the topic codec) or node telemetry never
reaches the view and the planner reads nothing. Mirrors the #287 lesson where
an in-process-only test missed a strict round-trip failure.
"""

from skulk.routing.topics import TELEMETRY
from skulk.shared.types.common import NodeId
from skulk.shared.types.profiling import NodeResources
from skulk.shared.types.telemetry import NodeTelemetry, TelemetryView


def test_node_telemetry_survives_topic_codec_round_trip() -> None:
    msg = NodeTelemetry(
        node_id=NodeId("node-a"),
        info=NodeResources(backends=frozenset({"mlx"}), participation="management"),
    )
    restored = TELEMETRY.deserialize(TELEMETRY.serialize(msg))
    assert restored == msg
    assert isinstance(restored.info, NodeResources)
    assert restored.info.participation == "management"
    assert restored.info.backends == frozenset({"mlx"})


def test_view_keeps_latest_per_node() -> None:
    view = TelemetryView()
    node = NodeId("node-a")
    view.apply(NodeTelemetry(node_id=node, info=NodeResources(participation="full")))
    assert view.node_resources[node].participation == "full"
    # last write wins — a later reading replaces the earlier one
    view.apply(
        NodeTelemetry(node_id=node, info=NodeResources(participation="management"))
    )
    assert view.node_resources[node].participation == "management"
    assert len(view.node_resources) == 1
