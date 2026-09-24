# pyright: reportPrivateUsage=false
"""Ordered preparation completion closes the telemetry race before placement."""

from skulk.master.main import Master
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.events import AudioCppPreparationCompleted, IndexedEvent
from skulk.shared.types.profiling import NodeResources
from skulk.shared.types.state import State
from skulk.shared.types.telemetry import TelemetryView


def test_preparation_completion_makes_build_ready_on_master() -> None:
    """Placement sees the worker-verified build when completion is indexed."""
    node = NodeId("music-node")
    master = object.__new__(Master)
    master.state = State()
    master.state.topology.add_node(node)
    master._telemetry_view = TelemetryView()
    master._runner_loaded_at = {}
    resources = NodeResources(
        backends=frozenset({"audio_cpp", "audio_cpp-cpu"}),
        engine_builds={"audio_cpp-cpu": "verified-build"},
    )
    completion = AudioCppPreparationCompleted(
        request_id=CommandId("prepare-music"),
        target_node=node,
        owner_node=NodeId("api-node"),
        success=True,
        resources=resources,
    )

    master._apply_indexed_event(IndexedEvent(event=completion, idx=0))

    assert master._telemetry_view.node_resources[node] == resources


def test_preparation_completion_does_not_restore_timed_out_node() -> None:
    """A late completion cannot resurrect a node absent from topology."""
    node = NodeId("departed-node")
    master = object.__new__(Master)
    master.state = State()
    master._telemetry_view = TelemetryView()
    master._runner_loaded_at = {}
    completion = AudioCppPreparationCompleted(
        request_id=CommandId("prepare-music"),
        target_node=node,
        owner_node=NodeId("api-node"),
        success=True,
        resources=NodeResources(
            backends=frozenset({"audio_cpp-cpu"}),
            engine_builds={"audio_cpp-cpu": "verified-build"},
        ),
    )

    master._apply_indexed_event(IndexedEvent(event=completion, idx=0))

    assert node not in master._telemetry_view.node_resources
