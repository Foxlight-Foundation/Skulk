"""Coverage for resolved node resource telemetry publication."""

import anyio

from skulk.shared.types.profiling import NodeResources
from skulk.utils.channels import channel
from skulk.utils.info_gatherer.info_gatherer import GatheredInfo, InfoGatherer


async def test_monitor_publishes_resolved_data_transport() -> None:
    """The gatherer emits the exact DATA transport resolved during startup."""
    info_send, info_recv = channel[GatheredInfo]()
    gatherer = InfoGatherer(
        info_sender=info_send,
        data_transport="zenoh",
        node_resources_poll_interval=0.01,
    )

    with anyio.fail_after(30):
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(
                gatherer._monitor_node_resources  # pyright: ignore[reportPrivateUsage]
            )
            resources = await info_recv.receive()
            assert isinstance(resources, NodeResources)
            assert resources.data_transport == "zenoh"
            task_group.cancel_scope.cancel()
