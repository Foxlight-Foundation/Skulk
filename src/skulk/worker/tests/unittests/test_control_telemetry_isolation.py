"""Worker liveness must survive a saturated real ordered-control outbox."""

import anyio
import pytest

from skulk.routing.tests.test_local_control_delivery import (
    _components,  # pyright: ignore[reportPrivateUsage]
    _echo,  # pyright: ignore[reportPrivateUsage]
)
from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.events import IndexedEvent, NodeGatheredInfo, TestEvent
from skulk.shared.types.profiling import NetworkInterfaceInfo, NodeResources
from skulk.shared.types.telemetry import NodeTelemetry
from skulk.utils.channels import channel
from skulk.utils.info_gatherer.info_gatherer import (
    GatheredInfo,
    NodeHeartbeat,
    NodeNetworkInterfaces,
)
from skulk.worker.main import Worker


@pytest.mark.parametrize("release", [False, True])
async def test_full_outbox_preserves_telemetry_and_coalesces_control(
    release: bool,
) -> None:
    """Real delivery backpressure cannot suppress heartbeats or grow a FIFO."""
    router, outbound, incoming = _components()
    router.pending_event_limit = 1
    blocking = TestEvent()
    producer = router.sender()
    controls = router.sender()
    telemetry, observed_telemetry = channel[NodeTelemetry](1)
    info, readings = channel[GatheredInfo](1)
    _, indexed = channel[IndexedEvent]()
    worker = Worker(
        node_id=router.node_id,
        event_receiver=indexed,
        event_sender=controls,
        command_sender=channel[ForwarderCommand]()[0],
        download_command_sender=channel[ForwarderDownloadCommand]()[0],
        telemetry_sender=telemetry,
    )
    finished = anyio.Event()

    async def forwarding() -> None:
        await worker._forward_info(readings)  # pyright: ignore[reportPrivateUsage]
        finished.set()

    final = NodeNetworkInterfaces(
        ifaces=[NetworkInterfaceInfo(name="example0", ip_address="192.0.2.100")]
    )
    with anyio.fail_after(8):
        async with router._tg as group:  # pyright: ignore[reportPrivateUsage]
            group.start_soon(router._run_ext_in)  # pyright: ignore[reportPrivateUsage]
            await producer.send(blocking)
            first = await outbound.receive()
            assert first.event == blocking
            group.start_soon(forwarding)
            for batch in range(8):
                for number in range(10):
                    await info.send(
                        NodeNetworkInterfaces(
                            ifaces=[
                                NetworkInterfaceInfo(
                                    name="example0",
                                    ip_address=f"192.0.2.{batch * 10 + number + 1}",
                                )
                            ]
                        )
                    )
                await info.send(NodeHeartbeat())
                heartbeat = await observed_telemetry.receive()
                assert heartbeat.node_id == router.node_id
                assert isinstance(heartbeat.info, NodeHeartbeat)
            await info.send(final)
            expected_resources = NodeResources(
                backends=frozenset(), architecture="test"
            )
            await info.send(expected_resources)
            resource = await observed_telemetry.receive()
            assert resource.info == expected_resources
            # The ordinary source is still unacknowledged throughout all nine
            # telemetry deliveries; no test directly bypasses reliable admission.
            assert set(router.out_for_delivery) == {blocking.event_id}
            assert len(worker._queued_forwarded_info) <= 1  # pyright: ignore[reportPrivateUsage]
            info.close()
            if release:
                await _echo(router, incoming, blocking, 0)
                delivered: list[NodeGatheredInfo] = []
                while not delivered or delivered[-1].info != final:
                    frame = await outbound.receive()
                    assert isinstance(frame.event, NodeGatheredInfo)
                    delivered.append(frame.event)
                    await _echo(router, incoming, frame.event, len(delivered))
                await finished.wait()
                # One admission wait, one producer slot, one blocked publisher
                # and one latest value are finite even after 80 changed polls.
                assert len(delivered) <= 4
                assert router.out_for_delivery == {}
                assert worker._queued_forwarded_info == {}  # pyright: ignore[reportPrivateUsage]
            group.cancel_scope.cancel()
