"""Actual Worker ownership of native originals and generated/edited WAV results.

Authoritative TaskCreated starts custody before dispatch. One journal covers
source-conditioned jobs and generations with no uploaded original, so process
restart cannot orphan unbounded results or replay an uncertain native edit.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import final

import anyio

from skulk.routing.output_media import OutputMediaPacket
from skulk.routing.vision_media import VisionMediaPacket
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.music import MusicOutputManifest
from skulk.shared.types.state import State
from skulk.shared.types.tasks import MusicEditing, MusicGeneration, TaskStatus
from skulk.shared.types.worker.runners import RunnerId
from skulk.worker.native_music_output import NativeMusicOutputProducer, NativeMusicTask
from skulk.worker.runner.ace_step.result_custody import reclaim_music_result
from skulk.worker.runner.demucs.source_fence import (
    SourceReadFence,
    acquire_source_read_fence,
)
from skulk.worker.separation_custody import (
    SeparationSourceCustody,
    separation_reader_key,
)
from skulk.worker.separation_input import MusicEditInputReceiver, separation_file_effect
from skulk.worker.separation_sources import SeparationSourceStore


def local_native_music_task(
    state: State, node: NodeId, command: CommandId
) -> NativeMusicTask | None:
    """Require a unique complete local task with the stamped native model policy."""
    matches = (
        task
        for task in state.tasks.values()
        if isinstance(task, (MusicGeneration, MusicEditing))
        and task.command_id == command
    )
    task = next(matches, None)
    if (
        task is None
        or next(matches, None) is not None
        or task.task_status not in (TaskStatus.Pending, TaskStatus.Running)
    ):
        return None
    instance = state.instances.get(task.instance_id)
    if instance is None:
        return None
    assignments = instance.shard_assignments
    runner = assignments.node_to_runner.get(node)
    if (
        runner is None
        or len(assignments.node_to_runner) != 1
        or len(assignments.runner_to_shard) != 1
    ):
        return None
    shard = assignments.runner_to_shard.get(runner)
    if (
        shard is None
        or shard.model_card.model_id != task.task_params.model
        or shard.model_card.music is None
        or shard.model_card.music.native is None
        or shard.world_size != 1
        or shard.device_rank != 0
        or shard.start_layer != 0
        or shard.end_layer != shard.model_card.n_layers
        or shard.n_layers != shard.model_card.n_layers
        or shard.native_music_policy is None
        or shard.resolved_backend != shard.native_music_policy.backend
        or shard.resolved_engine_build
        != "ace_step@sha256:" + shard.native_music_policy.payload_sha256
    ):
        return None
    return task


@dataclass(frozen=True)
class _Reader:
    task: NativeMusicTask
    runner: RunnerId
    root: Path


@final
class NativeMusicMediaOwner:
    """Connect the bounded source receiver, durable writer intent and WAV sender."""

    def __init__(
        self,
        node: NodeId,
        state: Callable[[], State],
        source_root: Path,
        readers_root: Path,
        send_input: Callable[[VisionMediaPacket], Awaitable[None]],
        send_output: Callable[[OutputMediaPacket], Awaitable[None]],
        fail: Callable[[NativeMusicTask, str], Awaitable[None]],
        drained: Callable[[NativeMusicTask, RunnerId, Path], bool],
    ) -> None:
        """Bind Worker effects; no files, input, package or process starts here."""
        self.node, self.state = node, state
        self.source_root, self.readers_root = source_root, readers_root
        self.send_input, self.fail, self.drained = send_input, fail, drained
        self.sources: SeparationSourceStore[MusicEditing] | None = None
        self.inputs: MusicEditInputReceiver | None = None
        self.outputs = NativeMusicOutputProducer(node, send_output, fail)
        self._readers: dict[CommandId, _Reader] = {}
        self._rendered: set[CommandId] = set()
        self._ready_generations: set[CommandId] = set()
        self._initialization = anyio.Lock()

    def _reader_key(self, task: MusicEditing) -> str:
        reader = self._readers.get(task.command_id)
        if reader is None or reader.task != task:
            raise ValueError("native music source has no captured reader")
        return reader.root.name

    def _open_sources(self) -> None:
        if self.sources is None:
            custody = SeparationSourceCustody[MusicEditing](
                self.source_root,
                self.readers_root,
                self._reader_key,
                reclaim_result=reclaim_music_result,
            )
            self.sources = SeparationSourceStore(self.source_root, custody=custody)

    async def _initialize(self) -> None:
        async with self._initialization:
            if self.sources is None:
                await separation_file_effect(self._open_sources)
            if self.inputs is None:
                assert self.sources is not None
                self.inputs = MusicEditInputReceiver(
                    self.node,
                    self.state,
                    self.sources,
                    self.send_input,
                    self.fail,
                )

    async def prepare_reader(self, root: Path) -> SourceReadFence:
        """Reconcile drained prior jobs before the new runner inherits its fence."""
        deadline = time.monotonic() + 30
        await self._initialize()
        assert self.sources is not None
        # The root-service fence excludes replacement Workers while this
        # acquire/release proves old readers drained and journal recovery runs.
        observed = await acquire_source_read_fence(root)
        observed.close()
        await separation_file_effect(self.sources.recover_stale)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                "native source reconciliation exceeded its complete budget"
            )
        return await acquire_source_read_fence(root, timeout_seconds=remaining)

    async def begin(self, command: CommandId) -> None:
        """Reserve actual local work before either generation or edit can dispatch."""
        task = local_native_music_task(self.state(), self.node, command)
        if task is None or task.task_status != TaskStatus.Pending:
            return
        await self._initialize()
        assert self.sources is not None and self.inputs is not None
        if local_native_music_task(self.state(), self.node, command) != task:
            return
        previous = self._readers.get(command)
        if previous is not None:
            if previous.task != task:
                raise ValueError("native music command changed its source or owner")
            return
        if len(self._readers) >= 4:
            raise ValueError("native music source/result admission is full")
        snapshot = self.state()
        instance = snapshot.instances.get(task.instance_id)
        if instance is None:
            return
        runner = instance.shard_assignments.node_to_runner.get(self.node)
        if runner is None:
            return
        root = self.readers_root / separation_reader_key(task.instance_id, runner)
        self._readers[command] = _Reader(task, runner, root)
        # Keep the captured reader even on an uncertain file-effect failure.
        # Terminal reconciliation then proves drain before dropping its intent.
        if isinstance(task, MusicEditing):
            await self.inputs.begin(command)
        else:
            await separation_file_effect(
                lambda: self.sources_reserve(command, root.name)
            )
            self._ready_generations.add(command)

    def sources_reserve(self, command: CommandId, reader_key: str) -> None:
        """Serialize generation's durable reservation through the shared file owner."""
        assert self.sources is not None
        self.sources.reserve_result(command, reader_key)

    def ready_commands(self) -> frozenset[CommandId]:
        """Gate dispatch on source verification or generation's durable reservation."""
        edits: frozenset[CommandId] = (
            self.inputs.ready_commands() if self.inputs is not None else frozenset()
        )
        return edits | frozenset(self._ready_generations)

    async def receive_input(self, packet: VisionMediaPacket) -> None:
        """Consume only an already admitted native source handshake."""
        if self.inputs is not None:
            await self.inputs.receive(packet)

    def register_output(
        self,
        command: CommandId,
        owner: NodeId | None,
        model: str,
        manifest: MusicOutputManifest,
    ) -> bool:
        """Bind completion to retained task ownership before addressed delivery."""
        reader = self._readers.get(command)
        if (
            reader is None
            or owner != reader.task.owner_node
            or model != reader.task.task_params.model
        ):
            raise ValueError("native completion has no matching admitted media owner")
        registered = self.outputs.register(reader.task, reader.root, manifest)
        self._rendered.add(command)
        return registered

    def owns(self, command: CommandId) -> bool:
        """Distinguish native generation callbacks from the existing audio.cpp path."""
        return command in self._readers

    async def reject_output(self, command: CommandId) -> None:
        """Fail an invalid local completion without losing its original file owner."""
        reader = self._readers.get(command)
        if reader is not None:
            self.outputs.cancel(command)
            await self.fail(
                reader.task,
                "Native music completion did not match its admitted media owner; retry the operation.",
            )

    async def reconcile(self) -> None:
        """Retain originals/results until both native readers and transfer settle."""
        if self.inputs is not None:
            await self.inputs.tick()
        await self.outputs.tick()
        for command, reader in tuple(self._readers.items()):
            live = self.state().tasks.get(reader.task.task_id)
            terminal = live is None or live.task_status not in (
                TaskStatus.Pending,
                TaskStatus.Running,
            )
            # Terminal DATA lets the API retire execution before the WAV sender
            # finishes. A registered result has independent bounded ACK custody;
            # ordinary task removal must not cancel that valid transfer.
            if terminal and (
                (live is None and command not in self._rendered)
                or (live is not None and live.task_status != TaskStatus.Complete)
            ):
                self.outputs.cancel(command)
            if not terminal or self.outputs.pending(command):
                continue
            drained = command in self._rendered or await separation_file_effect(
                lambda reader=reader: self.drained(
                    reader.task, reader.runner, reader.root
                )
            )
            if drained:
                if isinstance(reader.task, MusicEditing):
                    assert self.inputs is not None
                    await self.inputs.release_after_drain(command)
                else:
                    assert self.sources is not None
                    await separation_file_effect(
                        lambda command=command: self.sources_release(command)
                    )
                self._readers.pop(command, None)
                self._rendered.discard(command)
                self._ready_generations.discard(command)
        if self.sources is not None:
            await separation_file_effect(self.sources.recover_stale)

    def sources_release(self, command: CommandId) -> None:
        """Retain failed generation cleanup under the original serialized owner."""
        assert self.sources is not None
        self.sources.release_result(command)

    async def fail_recovered(self) -> None:
        """Fail this Worker's uncertain snapshot jobs without replaying inference."""
        for task in self.state().tasks.values():
            if (
                isinstance(task, (MusicGeneration, MusicEditing))
                and task.command_id not in self._readers
                and local_native_music_task(self.state(), self.node, task.command_id)
                == task
            ):
                await self.fail(
                    task,
                    "Native music was interrupted before this worker recovered; retry the operation.",
                )

    async def detach(self) -> None:
        """Release source service ownership after ingress/file tasks have joined."""
        if self.sources is not None:
            await separation_file_effect(self.sources.detach)
