"""Tests for the extension telemetry-plane read surface (fabric-citizenship Phase 1)."""

from datetime import datetime, timezone

from skulk.extensions.telemetry import ClusterNodeView, snapshot_cluster
from skulk.shared.types.common import NodeId
from skulk.shared.types.profiling import (
    AcceleratorMetrics,
    MemoryUsage,
    NodeIdentity,
    NodeResources,
    SystemPerformanceProfile,
)
from skulk.shared.types.telemetry import TelemetryView


def _full_node(view: TelemetryView, node_id: NodeId, name: str, vendor: str) -> None:
    view.node_resources[node_id] = NodeResources(
        backends=frozenset({"mlx", "mlx-metal"}), participation="full"
    )
    view.node_identities[node_id] = NodeIdentity(friendly_name=name, skulk_version="1.4.2")
    view.node_memory[node_id] = MemoryUsage.from_bytes(
        ram_total=16_000_000_000, ram_available=8_000_000_000, swap_total=0, swap_available=0
    )
    view.node_system[node_id] = SystemPerformanceProfile(
        accelerator=AcceleratorMetrics(vendor=vendor)  # type: ignore[arg-type]
    )
    view.node_last_telemetry[node_id] = datetime(2026, 7, 9, 6, 0, tzinfo=timezone.utc)


def test_snapshot_projects_full_node() -> None:
    """A node with complete telemetry snapshots into a fully-populated view."""
    view = TelemetryView()
    _full_node(view, NodeId("n-kite1"), "kite1", "apple")

    snapshot = snapshot_cluster(view)
    assert len(snapshot) == 1
    node = snapshot[0]
    assert isinstance(node, ClusterNodeView)
    assert node.node_id == NodeId("n-kite1")
    assert node.friendly_name == "kite1"
    assert node.backends == ("mlx", "mlx-metal")  # sorted
    assert node.participation == "full"
    assert node.skulk_version == "1.4.2"
    assert node.accelerator_vendor == "apple"
    assert node.ram_total_bytes == 16_000_000_000
    assert node.last_telemetry == datetime(2026, 7, 9, 6, 0, tzinfo=timezone.utc)


def test_snapshot_is_stably_sorted_and_heterogeneous() -> None:
    """Multiple nodes sort by friendly name; vendors are reported per node."""
    view = TelemetryView()
    _full_node(view, NodeId("n-kite4"), "kite4", "amd")
    _full_node(view, NodeId("n-kite1"), "kite1", "apple")

    snapshot = snapshot_cluster(view)
    assert [n.friendly_name for n in snapshot] == ["kite1", "kite4"]
    assert [n.accelerator_vendor for n in snapshot] == ["apple", "amd"]


def test_snapshot_tolerates_partial_telemetry() -> None:
    """A node seen through one reading only shows None for the absent fields."""
    view = TelemetryView()
    # Resources only: no identity, memory, system, or liveness yet.
    view.node_resources[NodeId("n-partial")] = NodeResources(
        backends=frozenset({"llama_cpp"}), participation="management"
    )

    snapshot = snapshot_cluster(view)
    assert len(snapshot) == 1
    node = snapshot[0]
    assert node.node_id == NodeId("n-partial")
    assert node.backends == ("llama_cpp",)
    assert node.participation == "management"
    assert node.friendly_name is None
    assert node.skulk_version is None
    assert node.accelerator_vendor is None
    assert node.ram_total_bytes is None
    assert node.last_telemetry is None


def test_snapshot_empty_view() -> None:
    """An empty view snapshots to an empty tuple, not an error."""
    assert snapshot_cluster(TelemetryView()) == ()


def test_snapshot_does_not_mutate_the_view() -> None:
    """Reading is pure: the underlying view is unchanged."""
    view = TelemetryView()
    _full_node(view, NodeId("n-kite1"), "kite1", "apple")
    before = dict(view.node_resources)
    snapshot_cluster(view)
    assert view.node_resources == before
    assert set(view.node_identities) == {NodeId("n-kite1")}
