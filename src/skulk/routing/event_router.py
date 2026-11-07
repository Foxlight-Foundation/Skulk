from dataclasses import dataclass, field
from math import isfinite
from random import random

import anyio
from anyio import BrokenResourceError, ClosedResourceError
from anyio.abc import CancelScope
from loguru import logger

from skulk.shared.types.commands import ForwarderCommand, RequestEventLog
from skulk.shared.types.common import NodeId, SessionId, SystemId
from skulk.shared.types.events import (
    Event,
    EventId,
    GlobalForwarderEvent,
    IndexedEvent,
    LocalForwarderEvent,
    NodeGatheredInfo,
    StateSnapshotHydrated,
    is_persistable_control_event,
)
from skulk.shared.types.state_sync import (
    LocalEventReceipt,
    StateSnapshot,
    StateSyncMessage,
)
from skulk.utils.channels import Receiver, Sender, channel
from skulk.utils.event_buffer import ConflictingDuplicateIndexError, OrderedBuffer
from skulk.utils.task_group import TaskGroup


@dataclass
class EventRouter:
    """Bound local control delivery to one master session and release indexed events.

    Unacknowledged local events retain their original source sequence until the
    master echoes them. Admission and retry batches are bounded so a partition
    applies backpressure without discarding an ordering dependency.
    """

    node_id: NodeId
    session_id: SessionId
    command_sender: Sender[ForwarderCommand]
    state_sync_sender: Sender[StateSyncMessage]
    state_sync_receiver: Receiver[tuple[str | None, StateSyncMessage]]
    external_inbound: Receiver[GlobalForwarderEvent]
    external_outbound: Sender[LocalForwarderEvent]
    receipt_sender: Sender[LocalEventReceipt] | None = None
    receipt_receiver: Receiver[tuple[str | None, LocalEventReceipt]] | None = None
    _system_id: SystemId = field(init=False, default_factory=SystemId)
    internal_outbound: list[Sender[IndexedEvent]] = field(
        init=False, default_factory=list
    )
    event_buffer: OrderedBuffer[Event] = field(
        init=False, default_factory=OrderedBuffer
    )
    # event_id -> (next retry deadline, attempts so far, event). Entries are
    # popped when the event round-trips back from the master as an indexed
    # broadcast. Forgetting one event creates a permanent hole in the master's
    # per-source ordering buffer, so later publications cannot repair membership.
    # Storage, producer queues and retry batches must bound pressure instead.
    out_for_delivery: dict[EventId, tuple[float, int, LocalForwarderEvent]] = field(
        init=False, default_factory=dict
    )
    retry_base_seconds: float = 5.0
    retry_cap_seconds: float = 30.0
    retry_batch_size: int = 16
    pending_event_limit: int = 256
    pending_byte_limit: int = 16 * 1024 * 1024
    _pending_event_bytes: dict[EventId, int] = field(init=False, default_factory=dict)
    _pending_bytes: int = field(init=False, default=0)
    _warned_delivery_ids: set[EventId] = field(init=False, default_factory=set)
    _rejected_event_types: set[tuple[type[Event], type[object] | None]] = field(
        init=False, default_factory=set
    )
    _delivery_capacity: anyio.Condition = field(
        init=False, default_factory=anyio.Condition
    )
    _tg: TaskGroup = field(init=False, default_factory=TaskGroup)

    _nack_cancel_scope: CancelScope | None = field(init=False, default=None)
    _nack_attempts: int = field(init=False, default=0)
    nack_base_seconds: float = 0.5
    nack_cap_seconds: float = 10.0
    snapshot_request_timeout_seconds: float = 1.0
    snapshot_request_attempts: int = 3
    _bootstrap_complete: anyio.Event = field(init=False, default_factory=anyio.Event)
    _release_lock: anyio.Lock = field(init=False, default_factory=anyio.Lock)

    def __post_init__(self) -> None:
        """Reject invalid admission limits or retry delays before starting tasks."""
        if (
            min(
                self.pending_event_limit, self.pending_byte_limit, self.retry_batch_size
            )
            <= 0
        ):
            raise ValueError(
                "local control admission and retry limits must be positive"
            )
        if (
            not isfinite(self.retry_base_seconds)
            or not isfinite(self.retry_cap_seconds)
            or min(self.retry_base_seconds, self.retry_cap_seconds) <= 0
        ):
            raise ValueError("local control retry delays must be finite and positive")

    async def run(self):
        try:
            async with self._tg as tg:
                tg.start_soon(self._bootstrap)
                tg.start_soon(self._run_ext_in)
                tg.start_soon(self._simple_retry)
                if self.receipt_receiver is not None:
                    tg.start_soon(self._receive_delivery_receipts, self.receipt_receiver)
        finally:
            self.external_outbound.close()
            self.state_sync_receiver.close()
            if self.receipt_sender is not None:
                self.receipt_sender.close()
            if self.receipt_receiver is not None:
                self.receipt_receiver.close()
            for send in self.internal_outbound:
                send.close()

    async def _simple_retry(self):
        while True:
            await anyio.sleep(1 + random())
            sent = 0
            retried_sources: set[SystemId] = set()
            # Snapshot entries because indexed echoes can remove them while a
            # network send yields. Oldest due entries repair sequence gaps first.
            for e_id, (deadline, attempts, event) in list(
                self.out_for_delivery.items()
            ):
                if anyio.current_time() <= deadline:
                    continue
                if self.out_for_delivery.get(e_id) != (deadline, attempts, event):
                    continue
                if attempts >= 6 and e_id not in self._warned_delivery_ids:
                    self._warned_delivery_ids.add(e_id)
                    logger.warning(
                        "Local control acknowledgement is still pending; "
                        "retaining the ordered event for bounded retry: "
                        f"{type(event.event).__name__} ({e_id})"
                    )
                # Saturating the exponent avoids overflow in a long partition.
                # A capped delay preserves recovery without fixed-rate storms.
                backoff = min(
                    self.retry_cap_seconds,
                    self.retry_base_seconds * (2.0 ** min(attempts, 6)),
                )
                self.out_for_delivery[e_id] = (
                    anyio.current_time() + backoff,
                    min(attempts + 1, 6),
                    event,
                )
                await self.external_outbound.send(event)
                retried_sources.add(event.origin)
                sent += 1
                if sent >= self.retry_batch_size:
                    break
            # A skipped stale task has no indexed echo. Ask once per retried
            # source for the master's fully consumed prefix, never per poll.
            if self.receipt_sender is not None:
                for source in retried_sources:
                    await self.receipt_sender.send(LocalEventReceipt(
                        kind="request", requester=self.node_id, source=source,
                        session_id=self.session_id,
                    ))

    async def _receive_delivery_receipts(
        self, receiver: Receiver[tuple[str | None, LocalEventReceipt]],
    ) -> None:
        """Release consumed source prefixes only on an authenticated master reply."""
        with receiver as receipts:
            async for origin, receipt in receipts:
                if (
                    receipt.kind != "response"
                    or receipt.requester != self.node_id
                    or receipt.session_id != self.session_id
                    or not self._is_trusted_snapshot_origin(origin)
                ):
                    continue
                matched = [
                    identifier for identifier, (_, _, frame) in self.out_for_delivery.items()
                    if frame.origin == receipt.source and frame.origin_idx <= receipt.through_idx
                ]
                if not matched:
                    continue
                for identifier in matched:
                    await self._acknowledge_local_event(identifier)
                # A consumption receipt is not replicated-state delivery. If
                # indexed echoes were lost, replay still has to repair State.
                if receipt.next_global_idx > self.event_buffer.next_idx_to_release:
                    await self.command_sender.send(ForwarderCommand(
                        origin=self._system_id,
                        command=RequestEventLog(since_idx=self.event_buffer.next_idx_to_release),
                    ))

    def sender(self) -> Sender[Event]:
        """Create a backpressured control producer with one queued event.

        The small queue permits initialization before ``run`` starts while
        keeping an admitted-outbox wait from moving unbounded storage upstream.
        """
        send, recv = channel[Event](1)
        if self._tg.is_running():
            self._tg.start_soon(self._ingest, SystemId(), recv)
        else:
            self._tg.queue(self._ingest, SystemId(), recv)
        return send

    def receiver(self) -> Receiver[IndexedEvent]:
        send, recv = channel[IndexedEvent]()
        self.internal_outbound.append(send)
        return recv

    def shutdown(self) -> None:
        self._tg.cancel_tasks()

    async def _ingest(self, system_id: SystemId, recv: Receiver[Event]):
        idx = 0
        with recv as events:
            async for event in events:
                if not is_persistable_control_event(event):
                    # The master never echoes rejected data/telemetry events.
                    # They cannot enter a reliable control outbox or consume a
                    # sequence that valid events would then wait behind.
                    payload_type = (
                        type(event.info)
                        if isinstance(event, NodeGatheredInfo)
                        else None
                    )
                    rejected_type = (type(event), payload_type)
                    if rejected_type not in self._rejected_event_types:
                        self._rejected_event_types.add(rejected_type)
                        logger.warning(
                            "Rejected non-control event before local sequencing: "
                            f"{type(event).__name__}"
                            + (
                                f" carrying {payload_type.__name__}"
                                if payload_type
                                else ""
                            )
                        )
                    continue
                f_ev = LocalForwarderEvent(
                    origin_idx=idx,
                    origin=system_id,
                    session=self.session_id,
                    event=event,
                )
                event_bytes = len(f_ev.model_dump_json().encode("utf-8"))
                async with self._delivery_capacity:
                    duplicate = False
                    while True:
                        if event.event_id in self.out_for_delivery:
                            duplicate = True
                            break
                        if not self.out_for_delivery or (
                            len(self.out_for_delivery) < self.pending_event_limit
                            and self._pending_bytes + event_bytes
                            <= self.pending_byte_limit
                        ):
                            break
                        await self._delivery_capacity.wait()
                    if duplicate:
                        # Re-submitting the same in-flight event must not mint
                        # a second local index with one shared acknowledgement.
                        continue
                    # An individually larger legal message is admitted alone;
                    # the existing transport limit still bounds that message.
                    self._pending_event_bytes[event.event_id] = event_bytes
                    self._pending_bytes += event_bytes
                    self.out_for_delivery[event.event_id] = (
                        anyio.current_time() + self.retry_base_seconds,
                        0,
                        f_ev,
                    )
                idx += 1
                # Retain before publication: a local-master echo may arrive
                # before this await returns and must release actual capacity.
                await self.external_outbound.send(f_ev)

    async def _acknowledge_local_event(self, event_id: EventId) -> None:
        """Release retained payload accounting after an authenticated master echo."""
        async with self._delivery_capacity:
            if event_id not in self.out_for_delivery:
                return
            self.out_for_delivery.pop(event_id)
            self._warned_delivery_ids.discard(event_id)
            self._pending_bytes -= self._pending_event_bytes.pop(event_id, 0)
            self._delivery_capacity.notify_all()

    async def _run_ext_in(self):
        with self.external_inbound as events:
            async for event in events:
                if event.session != self.session_id:
                    continue
                if event.origin != self.session_id.master_node_id:
                    continue

                try:
                    self.event_buffer.ingest(event.origin_idx, event.event)
                except ConflictingDuplicateIndexError as exc:
                    logger.warning(
                        "Conflicting duplicate global event index detected; "
                        "dropping buffered tail and requesting replay "
                        f"(idx={exc.idx}, session_id={self.session_id}, "
                        f"origin={event.origin}, existing_event_id="
                        f"{getattr(exc.existing, 'event_id', 'unknown')}, "
                        f"incoming_event_id={getattr(exc.incoming, 'event_id', 'unknown')})"
                    )
                    self._recover_from_conflicting_event(exc.idx)
                    continue
                event_id = event.event.event_id
                await self._acknowledge_local_event(event_id)

                if not self._bootstrap_complete.is_set():
                    continue

                await self._release_ready_events()

    def _recover_from_conflicting_event(self, idx: int) -> None:
        """Discard the conflicting buffered tail and request authoritative replay."""
        self.event_buffer.truncate_from(idx)
        self._nack_attempts = 0
        if self._nack_cancel_scope is not None:
            self._nack_cancel_scope.cancel()

        if self._bootstrap_complete.is_set():
            self._tg.start_soon(self._nack_request, idx)

    async def _bootstrap(self) -> None:
        replay_start_idx = 0

        snapshot = await self._await_snapshot()
        if snapshot is not None:
            await self._hydrate_from_snapshot(snapshot)
            replay_start_idx = snapshot.last_event_applied_idx + 1

        self._bootstrap_complete.set()
        await self.command_sender.send(
            ForwarderCommand(
                origin=self._system_id,
                command=RequestEventLog(since_idx=replay_start_idx),
            )
        )
        await self._release_ready_events()

    async def _await_snapshot(self) -> StateSnapshot | None:
        with self.state_sync_receiver as messages:
            for attempt in range(self.snapshot_request_attempts):
                await self.state_sync_sender.send(
                    StateSyncMessage(
                        kind="request",
                        requester=self._system_id,
                        session_id=self.session_id,
                    )
                )
                with anyio.move_on_after(self.snapshot_request_timeout_seconds):
                    async for origin, message in messages:
                        if message.kind != "response":
                            continue
                        if message.requester != self._system_id:
                            continue
                        if message.session_id != self.session_id:
                            logger.warning(
                                "Ignoring state snapshot response for mismatched session"
                            )
                            continue
                        if not self._is_trusted_snapshot_origin(origin):
                            logger.warning(
                                "Ignoring state snapshot response from non-master origin"
                            )
                            continue
                        assert message.snapshot is not None
                        if message.snapshot.session_id != self.session_id:
                            logger.warning(
                                "Ignoring state snapshot with mismatched embedded session"
                            )
                            continue
                        return message.snapshot
                if attempt < self.snapshot_request_attempts - 1:
                    logger.debug(
                        "Retrying state snapshot bootstrap request "
                        f"(attempt={attempt + 2}/{self.snapshot_request_attempts})"
                    )
        return None

    def _is_trusted_snapshot_origin(self, origin: str | None) -> bool:
        """Accept authenticated master responses, including local self-master replies."""
        if origin == str(self.session_id.master_node_id):
            return True
        return origin is None and self.node_id == self.session_id.master_node_id

    async def _hydrate_from_snapshot(self, snapshot: StateSnapshot) -> None:
        self.event_buffer.next_idx_to_release = snapshot.last_event_applied_idx + 1
        self.event_buffer.store = {
            idx: event
            for idx, event in self.event_buffer.store.items()
            if idx >= self.event_buffer.next_idx_to_release
        }
        if snapshot.last_event_applied_idx < 0:
            return
        await self._send_internal_event(
            IndexedEvent(
                idx=snapshot.last_event_applied_idx,
                event=StateSnapshotHydrated(state=snapshot.state),
            )
        )

    async def _release_ready_events(self) -> None:
        # Bootstrap replay and live event ingestion can both try to drain the ordered
        # buffer. Serialize release so `next_idx_to_release` cannot advance past
        # events that are still waiting to be delivered to consumers.
        async with self._release_lock:
            drained = self.event_buffer.drain_indexed()
            if drained:
                self._nack_attempts = 0
                if self._nack_cancel_scope:
                    self._nack_cancel_scope.cancel()

            if not drained and (
                self._nack_cancel_scope is None or self._nack_cancel_scope.cancel_called
            ):
                self._tg.start_soon(
                    self._nack_request, self.event_buffer.next_idx_to_release
                )
                return

            for idx, event in drained:
                await self._send_internal_event(IndexedEvent(idx=idx, event=event))

    async def _send_internal_event(self, indexed_event: IndexedEvent) -> None:
        to_clear = set[int]()
        for i, sender in enumerate(self.internal_outbound):
            try:
                await sender.send(indexed_event)
            except (ClosedResourceError, BrokenResourceError):
                to_clear.add(i)
        for i in sorted(to_clear, reverse=True):
            self.internal_outbound.pop(i)

    async def _nack_request(self, since_idx: int) -> None:
        # We request all events after (and including) the missing index.
        # This function is started whenever we receive an event that is out of sequence.
        # It is cancelled as soon as we receiver an event that is in sequence.

        if since_idx < 0:
            logger.warning(f"Negative value encountered for nack request {since_idx=}")
            since_idx = 0

        with CancelScope() as scope:
            self._nack_cancel_scope = scope
            delay: float = self.nack_base_seconds * (2.0**self._nack_attempts)
            delay = min(self.nack_cap_seconds, delay)
            self._nack_attempts += 1
            try:
                await anyio.sleep(delay)
                logger.info(
                    f"Nack attempt {self._nack_attempts}: Requesting Event Log from {since_idx}"
                )
                await self.command_sender.send(
                    ForwarderCommand(
                        origin=self._system_id,
                        command=RequestEventLog(since_idx=since_idx),
                    )
                )
            finally:
                if self._nack_cancel_scope is scope:
                    self._nack_cancel_scope = None
