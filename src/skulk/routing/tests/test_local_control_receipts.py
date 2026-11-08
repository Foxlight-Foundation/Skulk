"""Cumulative delivery receipts free skipped work without corrupting replay."""

import anyio
import pytest

from skulk.master.main import Master
from skulk.routing.tests.test_local_control_delivery import (
    _components,  # pyright: ignore[reportPrivateUsage]
)
from skulk.shared.types.commands import (
    ForwarderCommand,
    ForwarderDownloadCommand,
    RequestEventLog,
)
from skulk.shared.types.common import NodeId, SessionId, SystemId
from skulk.shared.types.events import (
    Event,
    LocalForwarderEvent,
    TaskDeleted,
    TracingStateChanged,
)
from skulk.shared.types.state_sync import LocalEventReceipt, StateSyncMessage
from skulk.shared.types.tasks import TaskId
from skulk.utils.channels import channel


async def test_skipped_tasks_release_bounded_outbox_and_allow_following_work() -> None:
    """Actual master consumption frees no-op tasks that have no indexed echo."""
    router, outbound, inbound = _components()
    router.pending_event_limit = 1
    router.retry_base_seconds = 0.01
    router.retry_cap_seconds = 0.01
    requests, request_wire = channel[LocalEventReceipt]()
    responses, response_wire = channel[LocalEventReceipt]()
    master_input, master_receipts = channel[tuple[str | None, LocalEventReceipt]]()
    worker_input, worker_receipts = channel[tuple[str | None, LocalEventReceipt]]()
    router.receipt_sender = requests
    master = Master(
        router.session_id.master_node_id,
        router.session_id,
        command_receiver=channel[ForwarderCommand]()[1],
        event_sender=channel[Event]()[0],
        local_event_receiver=outbound,
        global_event_sender=inbound,
        state_sync_receiver=channel[StateSyncMessage]()[1],
        state_sync_sender=channel[StateSyncMessage]()[0],
        download_command_sender=channel[ForwarderDownloadCommand]()[0],
    )

    async def request_transport() -> None:
        async for receipt in request_wire:
            await master_input.send((str(router.node_id), receipt))

    async def response_transport() -> None:
        async for receipt in response_wire:
            await worker_input.send((str(master.node_id), receipt))

    producer = router.sender()
    finished = anyio.Event()

    async def produce() -> None:
        await producer.send(TaskDeleted(task_id=TaskId()))
        await producer.send(TaskDeleted(task_id=TaskId()))
        await producer.send(TracingStateChanged(enabled=True))
        finished.set()

    try:
        with anyio.fail_after(12):
            async with router._tg as group:  # pyright: ignore[reportPrivateUsage]
                group.start_soon(router._run_ext_in)  # pyright: ignore[reportPrivateUsage]
                group.start_soon(router._simple_retry)  # pyright: ignore[reportPrivateUsage]
                group.start_soon(router._receive_delivery_receipts, worker_receipts)  # pyright: ignore[reportPrivateUsage]
                group.start_soon(master._event_processor)  # pyright: ignore[reportPrivateUsage]
                group.start_soon(
                    master._local_receipt_processor,  # pyright: ignore[reportPrivateUsage]
                    responses,
                    master_receipts,
                )
                group.start_soon(request_transport)
                group.start_soon(response_transport)
                group.start_soon(produce)
                while (
                    not finished.is_set()
                    or not master.state.tracing_enabled
                    or router.out_for_delivery
                ):
                    await anyio.sleep(0.01)
                # Stale lifecycle events were consumed, never persisted as fake
                # events just to obtain acknowledgements.
                assert len(master._event_log) == 1  # pyright: ignore[reportPrivateUsage]
                assert master._consumed_local_indices == {  # pyright: ignore[reportPrivateUsage]
                    source: 2
                    for source in master._multi_buffer.stores  # pyright: ignore[reportPrivateUsage]
                }
                assert router._pending_bytes == 0  # pyright: ignore[reportPrivateUsage]
                group.cancel_scope.cancel()
    finally:
        master._event_log.close()  # pyright: ignore[reportPrivateUsage]


async def test_untrusted_or_unmatched_receipts_cannot_release_delivery() -> None:
    """Only this session's master can retire this node's matching source prefix."""
    router, outbound, _ = _components()
    incoming, receiver = channel[tuple[str | None, LocalEventReceipt]]()
    producer = router.sender()
    async with router._tg as group:  # pyright: ignore[reportPrivateUsage]
        group.start_soon(router._receive_delivery_receipts, receiver)  # pyright: ignore[reportPrivateUsage]
        await producer.send(TaskDeleted(task_id=TaskId()))
        frame: LocalForwarderEvent = await outbound.receive()
        valid = LocalEventReceipt(
            kind="response",
            requester=router.node_id,
            source=frame.origin,
            session_id=router.session_id,
            through_idx=frame.origin_idx,
            next_global_idx=3,
        )
        for origin, receipt in [
            ("impostor", valid),
            (None, valid),
            ("master", valid.model_copy(update={"requester": NodeId("other")})),
            (
                "master",
                valid.model_copy(
                    update={
                        "session_id": SessionId(
                            master_node_id=NodeId("master"), election_clock=2
                        )
                    }
                ),
            ),
            ("master", valid.model_copy(update={"through_idx": -1})),
        ]:
            await incoming.send((origin, receipt))
        await anyio.sleep(0.01)
        assert frame.event.event_id in router.out_for_delivery
        await incoming.send(("master", valid))
        with anyio.fail_after(2):
            while router.out_for_delivery:
                await anyio.sleep(0)
        assert router._pending_bytes == 0  # pyright: ignore[reportPrivateUsage]
        command = await router.command_sender.clone_receiver().receive()
        assert isinstance(command.command, RequestEventLog)
        assert command.command.since_idx == 0
        group.cancel_scope.cancel()


async def test_master_consumption_receipt_waits_for_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An advanced ordering buffer alone cannot acknowledge unapplied work."""
    router, outbound, inbound = _components()
    requests, request_receiver = channel[tuple[str | None, LocalEventReceipt]]()
    responses, response_receiver = channel[LocalEventReceipt]()
    local_consumption: list[tuple[SystemId, int, int]] = []

    async def consumed(source: SystemId, through_idx: int, next_global_idx: int) -> None:
        local_consumption.append((source, through_idx, next_global_idx))

    master = Master(
        router.session_id.master_node_id,
        router.session_id,
        command_receiver=channel[ForwarderCommand]()[1],
        event_sender=channel[Event]()[0],
        local_event_receiver=outbound,
        global_event_sender=inbound,
        state_sync_receiver=channel[StateSyncMessage]()[1],
        state_sync_sender=channel[StateSyncMessage]()[0],
        download_command_sender=channel[ForwarderDownloadCommand]()[0],
        local_event_consumed=consumed,
    )
    entered, release = anyio.Event(), anyio.Event()

    async def persist(*, force: bool = False) -> None:
        entered.set()
        await release.wait()

    monkeypatch.setattr(master, "_persist_snapshot", persist)
    producer = router.sender()
    try:
        with anyio.fail_after(3):
            async with router._tg as group:  # pyright: ignore[reportPrivateUsage]
                group.start_soon(master._event_processor)  # pyright: ignore[reportPrivateUsage]
                group.start_soon(
                    master._local_receipt_processor,  # pyright: ignore[reportPrivateUsage]
                    responses,
                    request_receiver,
                )
                await producer.send(TracingStateChanged(enabled=True))
                await entered.wait()
                assert not master._consumed_local_indices  # pyright: ignore[reportPrivateUsage]
                assert local_consumption == []
                source = next(iter(master._multi_buffer.stores))  # pyright: ignore[reportPrivateUsage]
                request = LocalEventReceipt(
                    kind="request",
                    requester=router.node_id,
                    source=source,
                    session_id=router.session_id,
                )
                await requests.send((str(router.node_id), request))
                await anyio.sleep(0.01)
                assert response_receiver.collect() == []
                release.set()
                while source not in master._consumed_local_indices:  # pyright: ignore[reportPrivateUsage]
                    await anyio.sleep(0)
                await requests.send((str(router.node_id), request))
                receipt = await response_receiver.receive()
                assert receipt.through_idx == 0 and receipt.next_global_idx == 1
                assert local_consumption == [(source, 0, 1)]
                group.cancel_scope.cancel()
    finally:
        master._event_log.close()  # pyright: ignore[reportPrivateUsage]
