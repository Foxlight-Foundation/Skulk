"""Authoritative source handshake over the existing addressed binary media plane.

The selected worker requests input after seeing its own admitted task, avoiding
cross-plane races where source bytes outrun TaskCreated. Packet metadata cannot
admit a task or choose a destination/model/path. The source store remains an
injected file owner; this module starts no process and registers no receiver.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, final

import anyio
from anyio import to_thread

from skulk.routing.vision_media import AudioSourcePayload, VisionMediaPacket
from skulk.shared.models.model_cards import ModelId
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.state import State
from skulk.shared.types.tasks import AudioSeparation, MusicEditing, TaskStatus
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.worker.audio_source_types import AudioSourceTask


class SourceCopies[SourceTask: AudioSourceTask = AudioSeparation](Protocol):
    """Private bounded file effects; the worker supplies its exclusive source store."""

    def admit(self, task: SourceTask) -> None:
        """Reserve one original for this authoritative nonterminal task."""
        ...

    def append(self, command_id: CommandId, sequence: int, data: bytes) -> None:
        """Write a bounded raw frame without changing the admitted source identity."""
        ...

    def complete(self, command_id: CommandId) -> Path | None:
        """Return a private path only after complete independent WAV verification."""
        ...

    def delete(self, command_id: CommandId) -> None:
        """Delete only the owned copy; failed cleanup retains its reservation."""
        ...


async def separation_file_effect[Result](effect: Callable[[], Result]) -> Result:
    """Join owned file work before allowing coroutine cancellation to escape."""
    task = asyncio.create_task(to_thread.run_sync(effect))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # A cancelled asyncio task can otherwise abandon to_thread's file
        # effect. Keep ownership until it settles, including repeated cancels,
        # so a late write cannot recreate a source after deletion/close.
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        with contextlib.suppress(Exception):
            task.result()
        raise


def local_separation_input_task(
    state: State, node_id: NodeId, command_id: CommandId
) -> AudioSeparation | None:
    """Resolve one live complete local task without trusting any packet's routing.

    Source ownership follows the master's actual task and full rank-zero single
    host assignment. Missing, terminal, ambiguous, changed or nonlocal placements
    are not input capacity. Actual runtime/build policy remains enforced at load.
    """
    matches = (
        task
        for task in state.tasks.values()
        if isinstance(task, AudioSeparation) and task.command_id == command_id
    )
    task = next(matches, None)
    if task is None or next(matches, None) is not None:
        return None
    if task.task_status not in (TaskStatus.Pending, TaskStatus.Running):
        return None
    instance = state.instances.get(task.instance_id)
    if instance is None:
        return None
    assignments = instance.shard_assignments
    runner = assignments.node_to_runner.get(node_id)
    if (
        runner is None
        or len(assignments.node_to_runner) != 1
        or len(assignments.runner_to_shard) != 1
    ):
        return None
    shard = assignments.runner_to_shard.get(runner)
    if (
        not isinstance(shard, PipelineShardMetadata)
        or shard.model_card.separation is None
        or shard.model_card.model_id != task.task_params.model
        or shard.device_rank != 0
        or shard.world_size != 1
        or shard.start_layer != 0
        or shard.end_layer != shard.model_card.n_layers
        or shard.n_layers != shard.model_card.n_layers
        or shard.resolved_backend
        not in {
            "demucs-cpu",
            "demucs-metal",
            "demucs-cuda",
            "demucs-rocm",
        }
        or shard.resolved_engine_build is None
        or shard.separation_policy is None
    ):
        return None
    return task


@dataclass
class _Input[SourceTask: AudioSourceTask]:
    task: SourceTask
    deadline: float
    requested_at: float = float("-inf")
    opened: bool = False
    completed: bool = False
    accepted: bool = False
    failed: bool = False


class AudioInputReceiver[SourceTask: AudioSourceTask]:
    """Request, verify and acknowledge four bounded originals before execution.

    The owner calls begin after authoritative task creation, receive for directed
    frames, tick for finite transfer deadlines/retries, and ready during planning.
    All methods run on one async loop. File work runs off-loop. Source release
    and close require real resident readers to have drained first; terminal State
    alone is not proof that a cancelling native process has stopped reading.
    """

    def __init__(
        self,
        node_id: NodeId,
        state: Callable[[], State],
        copies: SourceCopies[SourceTask],
        send: Callable[[VisionMediaPacket], Awaitable[None]],
        fail: Callable[[SourceTask, str], Awaitable[None]],
        *,
        local_task: Callable[[State, NodeId, CommandId], SourceTask | None],
        payload: AudioSourcePayload,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Bind existing owners and effects without files, downloads or execution."""
        self.node_id, self.state, self.copies = node_id, state, copies
        self.send, self.fail, self.clock = send, fail, clock
        self.local_task = local_task
        self.payload: AudioSourcePayload = payload
        self._entries: dict[CommandId, _Input[SourceTask]] = {}
        self._recent: OrderedDict[CommandId, None] = OrderedDict()
        self._lock = anyio.Lock()
        self._closed = False

    def _live(self, entry: _Input[SourceTask]) -> bool:
        task = self.local_task(self.state(), self.node_id, entry.task.command_id)
        return task is not None and (
            task.task_id == entry.task.task_id
            and task.instance_id == entry.task.instance_id
            and task.owner_node == entry.task.owner_node
            and task.task_params == entry.task.task_params
        )

    async def _failed(self, entry: _Input[SourceTask], detail: str) -> None:
        if entry.failed:
            return
        entry.failed = True
        entry.accepted = False
        if self._live(entry):
            with anyio.fail_after(5):
                await self.fail(entry.task, detail)

    async def _send(self, entry: _Input[SourceTask], packet: VisionMediaPacket) -> bool:
        try:
            with anyio.fail_after(5):
                await self.send(packet)
        except (anyio.ClosedResourceError, anyio.BrokenResourceError, TimeoutError):
            await self._failed(entry, "separation source transport is unavailable")
            return False
        return True

    async def _request(self, entry: _Input[SourceTask]) -> None:
        if self._closed:
            return
        entry.requested_at = self.clock()
        await self._send(
            entry,
            VisionMediaPacket(
                source_node=self.node_id,
                target_node=entry.task.owner_node,
                command_id=entry.task.command_id,
                model=ModelId(entry.task.task_params.model),
                payload=self.payload,
                kind="requested",
                sequence=0,
                image_count=1,
                total_chunks=entry.task.task_params.total_input_chunks,
            ),
        )

    async def begin(self, command_id: CommandId) -> bool:
        """Admit only an actual pending local task, then ask its owner for bytes."""
        async with self._lock:
            if self._closed or command_id in self._recent:
                return False
            current = self._entries.get(command_id)
            if current is not None:
                return not current.failed and self._live(current)
            task = self.local_task(self.state(), self.node_id, command_id)
            if task is None or task.task_status is not TaskStatus.Pending:
                return False
            if len(self._entries) >= 4:
                with anyio.fail_after(5):
                    await self.fail(task, "separation input capacity is full")
                return False
            entry = _Input(task=task, deadline=self.clock() + 180)
            self._entries[command_id] = entry
            try:
                await separation_file_effect(lambda: self.copies.admit(task))
            except (OSError, ValueError):
                await self._failed(entry, "separation source storage admission failed")
                return False
            if not self._live(entry):
                return False
            await self._request(entry)
            return not self._closed and not entry.failed

    async def receive(self, packet: VisionMediaPacket) -> bool:
        """Acknowledge only this owner's verified source for a still-live task.

        Unknown/spoofed/late frames cannot allocate storage or cancel another
        owner. Valid opened/chunk/completed packets may reorder. Once accepted,
        duplicate frames cannot delete a source a resident might be reading.
        """
        if packet.payload != self.payload or packet.target_node != self.node_id:
            return False
        async with self._lock:
            entry = self._entries.get(packet.command_id)
            if self._closed or entry is None or entry.failed or not self._live(entry):
                return False
            task = entry.task
            if (
                packet.source_node != task.owner_node
                or packet.model != task.task_params.model
            ):
                return False
            if packet.kind == "requested":
                return False
            if entry.accepted:
                if packet.kind in ("opened", "completed"):
                    await self._acknowledge(entry)
                return entry.accepted
            if self.clock() >= entry.deadline:
                await self._failed(entry, "separation source transfer deadline expired")
                return False
            if packet.kind in ("cancelled", "transport_failed"):
                await self._failed(
                    entry, "separation source transfer stopped before verification"
                )
                return False
            chunks = task.task_params.total_input_chunks
            if packet.kind not in ("opened", "chunk", "completed"):
                return False
            if (
                packet.total_chunks != chunks
                or packet.kind == "chunk"
                and packet.image_index != 0
                or packet.kind in ("opened", "completed")
                and packet.image_count != 1
                or packet.kind == "completed"
                and packet.sha256 != task.task_params.source.sha256
            ):
                await self._failed(
                    entry, "separation source declaration differs from admitted input"
                )
                return False
            try:
                if packet.kind == "opened":
                    entry.opened = True
                elif packet.kind == "completed":
                    entry.completed = True
                else:
                    await separation_file_effect(
                        lambda: self.copies.append(
                            task.command_id, packet.sequence, packet.data
                        )
                    )
                if entry.opened and entry.completed:
                    path = await separation_file_effect(
                        lambda: self.copies.complete(task.command_id)
                    )
                    if path is not None and self._live(entry):
                        await self._acknowledge(entry)
            except (OSError, ValueError):
                await self._failed(
                    entry, "separation source failed independent WAV verification"
                )
            return entry.accepted

    async def _acknowledge(self, entry: _Input[SourceTask]) -> None:
        if self._closed or not self._live(entry):
            return
        if not entry.accepted and self.clock() >= entry.deadline:
            await self._failed(entry, "separation source transfer deadline expired")
            return
        accepted = VisionMediaPacket(
            source_node=self.node_id,
            target_node=entry.task.owner_node,
            command_id=entry.task.command_id,
            model=ModelId(entry.task.task_params.model),
            payload=self.payload,
            kind="accepted",
            sequence=entry.task.task_params.total_input_chunks + 1,
        )
        if await self._send(entry, accepted) and not self._closed and self._live(entry):
            entry.accepted = True

    def ready(self, command_id: CommandId) -> bool:
        """Observe acknowledged input for planning without file I/O or native work."""
        entry = self._entries.get(command_id)
        return bool(
            not self._closed
            and entry is not None
            and entry.accepted
            and not entry.failed
            and self._live(entry)
        )

    def tracked_tasks(self) -> tuple[SourceTask, ...]:
        """Return bounded admitted identities, including terminal undrained copies."""
        return tuple(entry.task for entry in self._entries.values())

    def ready_commands(self) -> frozenset[CommandId]:
        """Snapshot only acknowledged live inputs for ordinary worker planning."""
        return frozenset(command for command in self._entries if self.ready(command))

    async def tick(self) -> None:
        """Retry bounded requests until open; fail finite unfinished transfers."""
        async with self._lock:
            if self._closed:
                return
            for entry in tuple(self._entries.values()):
                if entry.accepted or entry.failed or not self._live(entry):
                    continue
                now = self.clock()
                if now >= entry.deadline:
                    await self._failed(
                        entry, "separation source transfer deadline expired"
                    )
                elif not entry.opened and now - entry.requested_at >= 2:
                    await self._request(entry)

    async def release_after_drain(self, command_id: CommandId) -> None:
        """Release an owned copy only after the supervisor has drained its reader."""
        async with self._lock:
            if command_id not in self._entries:
                return
            await separation_file_effect(lambda: self.copies.delete(command_id))
            self._entries.pop(command_id)
            self._recent[command_id] = None
            self._recent.move_to_end(command_id)
            while len(self._recent) > 64:
                self._recent.popitem(last=False)

    async def close_after_drain(self) -> None:
        """Stop new admission and retain failed deletion for a later close retry."""
        self._closed = True
        for command_id in tuple(self._entries):
            await self.release_after_drain(command_id)



def local_music_edit_input_task(
    state: State, node_id: NodeId, command_id: CommandId,
) -> MusicEditing | None:
    """Resolve one admitted native edit from complete authoritative local State."""
    matches = (
        task for task in state.tasks.values()
        if isinstance(task, MusicEditing) and task.command_id == command_id
    )
    task = next(matches, None)
    if task is None or next(matches, None) is not None or task.task_status not in {TaskStatus.Pending, TaskStatus.Running}:
        return None
    instance = state.instances.get(task.instance_id)
    if instance is None:
        return None
    assignments = instance.shard_assignments
    runner = assignments.node_to_runner.get(node_id)
    if runner is None or len(assignments.node_to_runner) != 1 or len(assignments.runner_to_shard) != 1:
        return None
    shard = assignments.runner_to_shard.get(runner)
    if (
        shard is None or shard.model_card.model_id != task.task_params.model
        or shard.model_card.music is None or shard.model_card.music.native is None
        or shard.world_size != 1 or shard.device_rank != 0 or shard.start_layer != 0
        or shard.end_layer != shard.model_card.n_layers or shard.n_layers != shard.model_card.n_layers
        or shard.resolved_backend not in {"ace_step-cpu", "ace_step-metal", "ace_step-cuda", "ace_step-rocm"}
        or shard.resolved_engine_build is None
    ):
        return None
    return task


@final
class SeparationInputReceiver(AudioInputReceiver[AudioSeparation]):
    """Existing separation policy over the shared bounded original handshake."""

    def __init__(
        self, node_id: NodeId, state: Callable[[], State], copies: SourceCopies[AudioSeparation],
        send: Callable[[VisionMediaPacket], Awaitable[None]],
        fail: Callable[[AudioSeparation, str], Awaitable[None]], *, clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Retain separation's exact source/rank/policy admission and payload."""
        super().__init__(node_id, state, copies, send, fail,
            local_task=local_separation_input_task, payload="separation_source", clock=clock)


@final
class MusicEditInputReceiver(AudioInputReceiver[MusicEditing]):
    """Native edit policy over the shared bounded original handshake."""

    def __init__(
        self, node_id: NodeId, state: Callable[[], State], copies: SourceCopies[MusicEditing],
        send: Callable[[VisionMediaPacket], Awaitable[None]],
        fail: Callable[[MusicEditing, str], Awaitable[None]], *, clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Bind native edit identity and an isolated source payload to real custody."""
        super().__init__(node_id, state, copies, send, fail,
            local_task=local_music_edit_input_task, payload="music_edit_source", clock=clock)
