"""Telemetry plane coverage (#279): wire round-trip + view coalescing.

The NodeTelemetry message must survive the gossip path (model_dump_json ->
bytes -> model_validate_json under the topic codec) or node telemetry never
reaches the view and the planner reads nothing. Mirrors the #287 lesson where
an in-process-only test missed a strict round-trip failure.
"""

from skulk.routing.topics import TELEMETRY
from skulk.shared.types.common import NodeId
from skulk.shared.types.profiling import (
    MemoryUsage,
    NodeResources,
    SystemPerformanceProfile,
)
from skulk.shared.types.telemetry import NodeTelemetry, TelemetryView
from skulk.utils.info_gatherer.info_gatherer import MactopMetrics


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


def test_view_coalesces_memory_telemetry() -> None:
    # node_memory moved onto the telemetry plane in slice 2: a MemoryUsage
    # reading must land in view.node_memory, last-write-wins per node.
    view = TelemetryView()
    node = NodeId("node-a")
    view.apply(
        NodeTelemetry(
            node_id=node,
            info=MemoryUsage.from_bytes(
                ram_total=16 * 2**30,
                ram_available=8 * 2**30,
                swap_total=0,
                swap_available=0,
            ),
        )
    )
    assert view.node_memory[node].ram_available.in_bytes == 8 * 2**30
    view.apply(
        NodeTelemetry(
            node_id=node,
            info=MemoryUsage.from_bytes(
                ram_total=16 * 2**30,
                ram_available=4 * 2**30,
                swap_total=0,
                swap_available=0,
            ),
        )
    )
    assert view.node_memory[node].ram_available.in_bytes == 4 * 2**30
    assert len(view.node_memory) == 1


def test_view_coalesces_mactop_memory_and_system() -> None:
    # MactopMetrics carries BOTH memory and the system profile; one reading
    # must populate node_memory AND node_system (the combined dispatch path).
    view = TelemetryView()
    node = NodeId("node-a")
    view.apply(
        NodeTelemetry(
            node_id=node,
            info=MactopMetrics(
                system_profile=SystemPerformanceProfile(sys_power=42.0),
                memory=MemoryUsage.from_bytes(
                    ram_total=32 * 2**30,
                    ram_available=20 * 2**30,
                    swap_total=0,
                    swap_available=0,
                ),
            ),
        )
    )
    assert view.node_memory[node].ram_available.in_bytes == 20 * 2**30
    assert view.node_system[node].sys_power == 42.0
