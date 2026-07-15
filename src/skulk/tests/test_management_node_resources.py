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
                "zenoh",
                telemetry_send,
                0.01,
            )
            telemetry = await telemetry_recv.receive()
            assert telemetry.node_id == NodeId("api-only-node")
            assert isinstance(telemetry.info, NodeResources)
            assert telemetry.info.data_transport == "zenoh"
            assert telemetry.info.participation == "management"
            assert telemetry.info.backends == frozenset()
            task_group.cancel_scope.cancel()
