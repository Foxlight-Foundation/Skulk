"""Coverage for resource telemetry from nodes started without a worker."""

import anyio

from skulk.main import (
    _publish_management_node_resources,  # pyright: ignore[reportPrivateUsage]
)
from skulk.shared.types.common import NodeId
from skulk.shared.types.profiling import NodeResources
from skulk.shared.types.telemetry import NodeTelemetry
from skulk.utils.channels import channel


async def test_management_node_advertises_transport_without_placement() -> None:
    """A no-worker node remains visible to transport checks but not placement."""
    telemetry_send, telemetry_recv = channel[NodeTelemetry]()

    with anyio.fail_after(30):
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(
                _publish_management_node_resources,
                NodeId("api-only-node"),
                True,
                "zenoh",
                telemetry_send,
                None,
                0.01,
            )
            telemetry = await telemetry_recv.receive()
            assert telemetry.node_id == NodeId("api-only-node")
            assert isinstance(telemetry.info, NodeResources)
            assert telemetry.info.data_transport == "zenoh"
            assert telemetry.info.api_available is True
            assert telemetry.info.participation == "management"
            assert telemetry.info.backends == frozenset()
            task_group.cancel_scope.cancel()


async def test_management_node_publishes_capability_changes_and_empty_withdrawal() -> (
    None
):
    """Management-only peers advertise plugins without ever advertising compute."""
    from skulk.utils.info_gatherer.info_gatherer import NodeCapabilities

    tags = {"managed.echo"}
    sender, receiver = channel[NodeTelemetry]()
    with anyio.fail_after(5):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(
                _publish_management_node_resources,
                NodeId("management"),
                True,
                "zenoh",
                sender,
                None,
                0.01,
                lambda: frozenset(tags),
            )
            first = await receiver.receive()
            assert isinstance(first.info, NodeResources)
            assert first.info.backends == frozenset()
            reading = await receiver.receive()
            assert isinstance(reading.info, NodeCapabilities)
            assert reading.info.capabilities == frozenset(tags)
            tags.clear()
            while True:
                reading = await receiver.receive()
                if isinstance(reading.info, NodeCapabilities):
                    assert reading.info.capabilities == frozenset()
                    break
            tasks.cancel_scope.cancel()


async def test_management_node_publishes_capability_nodes_and_empty_withdrawal() -> (
    None
):
    """Capability-node summaries follow the same publish-and-clear discipline as tags."""
    from skulk.shared.types.capability_nodes import CapabilityNodeSummary
    from skulk.utils.info_gatherer.info_gatherer import NodeCapabilityNodes

    summary = CapabilityNodeSummary.model_validate(
        {
            "plugin_id": "foxlight.video-studio",
            "node_id": "studio",
            "bundle_id": "foxlight.video-studio",
            "version": "1.0.0",
            "status": "ready",
            "owner_available": True,
        }
    )
    published: list[CapabilityNodeSummary] = [summary]
    sender, receiver = channel[NodeTelemetry]()
    with anyio.fail_after(5):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(
                _publish_management_node_resources,
                NodeId("management"),
                True,
                "zenoh",
                sender,
                None,
                0.01,
                None,
                lambda: tuple(published),
            )
            while True:
                reading = await receiver.receive()
                if isinstance(reading.info, NodeCapabilityNodes):
                    assert reading.info.nodes == (summary,)
                    break
            published.clear()
            while True:
                reading = await receiver.receive()
                if isinstance(reading.info, NodeCapabilityNodes):
                    assert reading.info.nodes == ()
                    break
            # After the clearing reading, no further empty readings follow.
            seen_empty_again = False
            with anyio.move_on_after(0.1):
                while True:
                    reading = await receiver.receive()
                    if isinstance(reading.info, NodeCapabilityNodes):
                        seen_empty_again = True
            assert not seen_empty_again
            tasks.cancel_scope.cancel()
