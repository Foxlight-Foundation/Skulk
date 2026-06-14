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


def test_prune_drops_all_telemetry_for_a_node() -> None:
    # On NodeTimedOut the view must drop the node entirely (it has no natural
    # expiry); otherwise a dead node lingers as a ghost in /state and skews
    # capacity/energy aggregates (#279 slice 2).
    gib = 1024 * 1024 * 1024
    reading = MemoryUsage.from_bytes(
        ram_total=16 * gib, ram_available=8 * gib, swap_total=0, swap_available=0
    )
    profile = SystemPerformanceProfile(sys_power=10.0)
    view = TelemetryView()
    a, b = NodeId("node-a"), NodeId("node-b")
    for node in (a, b):
        view.apply(NodeTelemetry(node_id=node, info=NodeResources(participation="full")))
        view.apply(
            NodeTelemetry(
                node_id=node,
                info=MactopMetrics(system_profile=profile, memory=reading),
            )
        )
    view.prune(a)
    for m in (view.node_resources, view.node_memory, view.node_system):
        assert a not in m
        assert b in m  # only the pruned node is dropped
