"""Telemetry plane coverage (#279): wire round-trip + view coalescing.

The NodeTelemetry message must survive the gossip path (model_dump_json ->
bytes -> model_validate_json under the topic codec) or node telemetry never
reaches the view and the planner reads nothing. Mirrors the #287 lesson where
an in-process-only test missed a strict round-trip failure.
"""

from skulk.routing.topics import TELEMETRY
from skulk.shared.types.common import NodeId
from skulk.shared.types.memory import Memory
from skulk.shared.types.profiling import (
    DiskUsage,
    MemoryUsage,
    NodeResources,
    SystemPerformanceProfile,
)
from skulk.shared.types.telemetry import NodeTelemetry, TelemetryView
from skulk.utils.info_gatherer.info_gatherer import (
    MactopMetrics,
    MiscData,
    NodeDiskUsage,
    RdmaCtlStatus,
    StaticNodeInformation,
)


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


def test_view_coalesces_disk_and_rdma_telemetry() -> None:
    # Slice 3: disk + rdma-ctl readings land in their own view maps.
    view = TelemetryView()
    node = NodeId("node-a")
    view.apply(
        NodeTelemetry(
            node_id=node,
            info=NodeDiskUsage(
                disk_usage=DiskUsage(
                    total=Memory.from_bytes(500 * 2**30),
                    available=Memory.from_bytes(200 * 2**30),
                )
            ),
        )
    )
    assert view.node_disk[node].available.in_bytes == 200 * 2**30
    view.apply(
        NodeTelemetry(
            node_id=node,
            info=RdmaCtlStatus(enabled=True, interfaces_present=True),
        )
    )
    assert view.node_rdma_ctl[node].enabled is True


def test_view_merges_identity_from_two_readings() -> None:
    # Slice 3: identity is assembled from MiscData (friendly name) and
    # StaticNodeInformation (static fields); the two readings must MERGE into one
    # NodeIdentity rather than overwrite each other (mirrors the event applier's
    # accumulation), regardless of arrival order.
    view = TelemetryView()
    node = NodeId("node-a")
    view.apply(NodeTelemetry(node_id=node, info=MiscData(friendly_name="Kite 2")))
    view.apply(
        NodeTelemetry(
            node_id=node,
            info=StaticNodeInformation(
                model="Mac mini",
                chip="M4",
                os_version="26.4",
                os_build_version="X",
                skulk_version="1.2.0",
                skulk_commit="abc123",
            ),
        )
    )
    identity = view.node_identities[node]
    assert identity.friendly_name == "Kite 2"  # preserved across the static merge
    assert identity.chip_id == "M4"
    assert identity.skulk_version == "1.2.0"
    # a later friendly-name update keeps the static fields
    view.apply(
        NodeTelemetry(node_id=node, info=MiscData(friendly_name="Kite 2 (renamed)"))
    )
    identity = view.node_identities[node]
    assert identity.friendly_name == "Kite 2 (renamed)"
    assert identity.chip_id == "M4"


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
        view.apply(
            NodeTelemetry(node_id=node, info=NodeResources(participation="full"))
        )
        view.apply(
            NodeTelemetry(
                node_id=node,
                info=MactopMetrics(system_profile=profile, memory=reading),
            )
        )
        # slice-3 maps too
        view.apply(NodeTelemetry(node_id=node, info=MiscData(friendly_name="n")))
        view.apply(
            NodeTelemetry(
                node_id=node,
                info=NodeDiskUsage(
                    disk_usage=DiskUsage(
                        total=Memory.from_bytes(500 * gib),
                        available=Memory.from_bytes(200 * gib),
                    )
                ),
            )
        )
        view.apply(
            NodeTelemetry(
                node_id=node,
                info=RdmaCtlStatus(enabled=False, interfaces_present=False),
            )
        )
    view.prune(a)
    for m in (
        view.node_resources,
        view.node_memory,
        view.node_system,
        view.node_identities,
        view.node_disk,
        view.node_rdma_ctl,
    ):
        assert a not in m
        assert b in m  # only the pruned node is dropped
