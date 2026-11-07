"""A consumed local prefix must not retire an unreceived global tail."""

import anyio

from skulk.routing.tests.test_local_control_delivery import (
    _components,  # pyright: ignore[reportPrivateUsage]
)
from skulk.shared.types.commands import RequestEventLog
from skulk.shared.types.common import NodeId, SessionId, SystemId
from skulk.shared.types.events import (
    GlobalForwarderEvent,
    TaskDeleted,
    TracingStateChanged,
)
from skulk.shared.types.state_sync import LocalEventReceipt
from skulk.shared.types.tasks import TaskId
from skulk.utils.channels import channel


async def test_untrusted_receipts_cannot_start_background_replay() -> None:
    """A forged head cannot drive replay after the local outbox is empty."""
    router, _, _ = _components()
    receipts, receipt_input = channel[tuple[str | None, LocalEventReceipt]]()
    commands = router.command_sender.clone_receiver()
    receipt = LocalEventReceipt(
        kind="response",
        requester=router.node_id,
        source=SystemId(),
        session_id=router.session_id,
        through_idx=0,
        next_global_idx=100,
    )
    async with router._tg as group:  # pyright: ignore[reportPrivateUsage]
        group.start_soon(router._receive_delivery_receipts, receipt_input)  # pyright: ignore[reportPrivateUsage]
        group.start_soon(router._retry_receipt_replay)  # pyright: ignore[reportPrivateUsage]
        for origin, untrusted in [
            ("impostor", receipt),
            (None, receipt),
            ("master", receipt.model_copy(update={"requester": NodeId("other")})),
            (
                "master",
                receipt.model_copy(
                    update={
                        "session_id": SessionId(
                            master_node_id=NodeId("master"),
                            election_clock=2,
                        )
                    }
                ),
            ),
        ]:
            await receipts.send((origin, untrusted))
        await anyio.sleep(0.6)
        assert commands.collect() == []
        group.cancel_scope.cancel()


async def test_lost_replay_request_retries_after_last_local_frame_is_retired() -> None:
    """An empty reliable outbox cannot hide a missing replay request."""
    router, outbound, incoming = _components()
    router.nack_base_seconds = router.nack_cap_seconds = 0.01
    receipts, receipt_input = channel[tuple[str | None, LocalEventReceipt]]()
    commands = router.command_sender.clone_receiver()
    indexed = router.receiver()
    producer = router.sender()
    with anyio.fail_after(4):
        async with router._tg as group:  # pyright: ignore[reportPrivateUsage]
            group.start_soon(router._run_ext_in)  # pyright: ignore[reportPrivateUsage]
            group.start_soon(router._receive_delivery_receipts, receipt_input)  # pyright: ignore[reportPrivateUsage]
            group.start_soon(router._retry_receipt_replay)  # pyright: ignore[reportPrivateUsage]
            await producer.send(TaskDeleted(task_id=TaskId()))
            local = await outbound.receive()
            await receipts.send(
                (
                    "master",
                    LocalEventReceipt(
                        kind="response",
                        requester=router.node_id,
                        source=local.origin,
                        session_id=router.session_id,
                        through_idx=0,
                        next_global_idx=1,
                    ),
                )
            )
            lost = await commands.receive()
            assert isinstance(lost.command, RequestEventLog)
            assert lost.command.since_idx == 0
            assert not router.out_for_delivery
            # Drop the first request. No local event remains to trigger probes.
            retry = await commands.receive()
            assert isinstance(retry.command, RequestEventLog)
            assert retry.command.since_idx == 0
            event = TracingStateChanged(enabled=True)
            await incoming.send(
                GlobalForwarderEvent(
                    origin=router.session_id.master_node_id,
                    origin_idx=0,
                    session=router.session_id,
                    event=event,
                )
            )
            assert (await indexed.receive()).event == event
            await anyio.sleep(0.6)
            assert commands.collect() == []
            group.cancel_scope.cancel()


async def test_lost_final_replay_frame_and_already_retired_receipt_retry_current_cursor() -> (
    None
):
    """Partial replay and a newer trusted head survive without matching locals."""
    router, outbound, incoming = _components()
    router.nack_base_seconds = router.nack_cap_seconds = 0.01
    receipts, receipt_input = channel[tuple[str | None, LocalEventReceipt]]()
    commands = router.command_sender.clone_receiver()
    indexed = router.receiver()
    producer = router.sender()
    with anyio.fail_after(5):
        async with router._tg as group:  # pyright: ignore[reportPrivateUsage]
            group.start_soon(router._run_ext_in)  # pyright: ignore[reportPrivateUsage]
            group.start_soon(router._receive_delivery_receipts, receipt_input)  # pyright: ignore[reportPrivateUsage]
            group.start_soon(router._retry_receipt_replay)  # pyright: ignore[reportPrivateUsage]
            await producer.send(TaskDeleted(task_id=TaskId()))
            local = await outbound.receive()
            receipt = LocalEventReceipt(
                kind="response",
                requester=router.node_id,
                source=local.origin,
                session_id=router.session_id,
                through_idx=0,
                next_global_idx=2,
            )
            await receipts.send(("master", receipt))
            initial = await commands.receive()
            assert isinstance(initial.command, RequestEventLog)
            assert initial.command.since_idx == 0
            assert not router.out_for_delivery

            async def deliver(index: int) -> None:
                event = TracingStateChanged(enabled=index % 2 == 0)
                await incoming.send(
                    GlobalForwarderEvent(
                        origin=router.session_id.master_node_id,
                        origin_idx=index,
                        session=router.session_id,
                        event=event,
                    )
                )
                assert (await indexed.receive()).event == event

            # Only the first frame arrives. A duplicate receipt has no retained
            # source to match, but its authenticated newer head still matters.
            await deliver(0)
            await receipts.send(
                ("master", receipt.model_copy(update={"next_global_idx": 3}))
            )
            retry = await commands.receive()
            assert isinstance(retry.command, RequestEventLog)
            assert retry.command.since_idx == 1
            await deliver(1)
            # Drop the final replay frame again: no out-of-order tail exists to
            # start the ordinary gap NACK, so receipt recovery must retry it.
            final_retry = await commands.receive()
            assert isinstance(final_retry.command, RequestEventLog)
            assert final_retry.command.since_idx == 2
            await deliver(2)
            await anyio.sleep(0.6)
            assert commands.collect() == []
            group.cancel_scope.cancel()
