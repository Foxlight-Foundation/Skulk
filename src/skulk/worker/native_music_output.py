"""Deliver native music from its journaled workspace through verified reverse ACKs.

Generation and editing share one bounded WAV owner. The Worker supplies the
authoritative task and captured reader root; no packet can select a local path.
The owning API independently verifies bytes before acknowledging completion.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import stat
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypedDict, final

import anyio
from loguru import logger

from skulk.routing.output_media import OutputMediaPacket
from skulk.shared.constants import SKULK_MAX_CHUNK_SIZE
from skulk.shared.models.model_cards import ModelId
from skulk.shared.types.audio_separation import (
    MAX_SEPARATION_SOURCE_BYTES,
    pcm24_from_wav,
)
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.music import MusicOutputManifest
from skulk.shared.types.native_music import (
    NATIVE_MUSIC_MAX_FRAMES,
    NATIVE_MUSIC_SAMPLE_RATE,
)
from skulk.shared.types.tasks import MusicEditing, MusicGeneration
from skulk.utils.runtime_files import read_private
from skulk.worker.runner.ace_step.result_custody import (
    music_result_directory,
    reclaim_result_directory,
)
from skulk.worker.separation_input import separation_file_effect

NativeMusicTask = MusicGeneration | MusicEditing


class _PacketAddress(TypedDict):
    source_node: NodeId
    target_node: NodeId
    command_id: CommandId
    model: ModelId
    purpose: Literal["music", "music_edit"]


@dataclass
class _Delivery:
    task: NativeMusicTask
    directory: Path
    manifest: MusicOutputManifest
    purpose: Literal["music", "music_edit"]
    terminal: int | None = None
    accepted: bool = False
    aborted: bool = False
    sending: bool = False
    finished: bool = False
    changed: anyio.Event = field(default_factory=anyio.Event)


@final
class NativeMusicOutputProducer:
    """Retain four native results until their sender drains and cleanup succeeds."""

    def __init__(
        self,
        node_id: NodeId,
        send: Callable[[OutputMediaPacket], Awaitable[None]],
        fail: Callable[[NativeMusicTask, str], Awaitable[None]],
        *,
        timeout_seconds: float = 300,
    ) -> None:
        """Bind addressed effects without opening files or admitting inference."""
        if not 0 < timeout_seconds <= 300:
            raise ValueError("native music delivery needs a bounded five-minute budget")
        self.node_id, self.send, self.fail = node_id, send, fail
        self.timeout_seconds = timeout_seconds
        self._entries: dict[CommandId, _Delivery] = {}

    def register(
        self, task: NativeMusicTask, root: Path, manifest: MusicOutputManifest
    ) -> bool:
        """Bind one native completion to its actual command and captured reader root."""
        directory = music_result_directory(root, task.command_id)
        previous = self._entries.get(task.command_id)
        if previous is not None:
            if (previous.task, previous.directory, previous.manifest) != (
                task,
                directory,
                manifest,
            ):
                raise ValueError(
                    "native music completion changed its admitted identity"
                )
            return False
        if len(self._entries) >= 4:
            raise ValueError("native music output custody is full")
        if (
            manifest.size_bytes > MAX_SEPARATION_SOURCE_BYTES
            or manifest.sample_rate != 48000
            or manifest.channels != 2
        ):
            raise ValueError("native music output exceeds its bounded WAV contract")
        self._entries[task.command_id] = _Delivery(
            task,
            directory,
            manifest,
            "music_edit" if isinstance(task, MusicEditing) else "music",
        )
        return True

    def pending(self, command: CommandId) -> bool:
        """Include waiting ACKs and failed cleanup in the result reservation."""
        return command in self._entries

    def receive(self, packet: OutputMediaPacket) -> bool:
        """Accept only the exact owner/model/role and a terminal already emitted."""
        entry = self._entries.get(packet.command_id)
        if entry is None or (
            packet.target_node != self.node_id
            or packet.source_node != entry.task.owner_node
            or packet.model != entry.task.task_params.model
            or packet.purpose != entry.purpose
        ):
            return False
        if packet.kind in ("cancelled", "transport_failed") and packet.sequence == 0:
            entry.aborted = True
        elif entry.terminal is None or packet.sequence != entry.terminal:
            return False
        elif packet.kind == "accepted":
            entry.accepted = True
        elif packet.kind in ("cancelled", "transport_failed"):
            entry.aborted = True
        else:
            return False
        entry.changed.set()
        return True

    def cancel(self, command: CommandId) -> None:
        """Stop this sender without releasing files still held by its coroutine."""
        entry = self._entries.get(command)
        if entry is not None:
            entry.aborted = True
            entry.changed.set()

    async def _send(self, entry: _Delivery, packet: OutputMediaPacket) -> None:
        if entry.aborted:
            raise ValueError("native music output delivery was cancelled")
        with anyio.fail_after(5):
            await self.send(packet)

    async def _stream(self, entry: _Delivery) -> None:
        path, manifest = entry.directory / "result.wav", entry.manifest

        def verify() -> tuple[int, int, int]:
            directory = entry.directory.lstat()
            if (
                not stat.S_ISDIR(directory.st_mode)
                or directory.st_uid != os.getuid()
                or stat.S_IMODE(directory.st_mode) != 0o700
            ):
                raise ValueError("native music result directory is not private")
            facts, _ = pcm24_from_wav(read_private(path, MAX_SEPARATION_SOURCE_BYTES))
            if (
                facts.size_bytes != manifest.size_bytes
                or facts.sha256 != manifest.sha256
                or facts.sample_rate != manifest.sample_rate
                or facts.channels != manifest.channels
                or not NATIVE_MUSIC_SAMPLE_RATE
                <= facts.frames
                <= NATIVE_MUSIC_MAX_FRAMES
                or abs(facts.frames / facts.sample_rate - manifest.duration_seconds)
                > 0.5 / facts.sample_rate
            ):
                raise ValueError(
                    "native music WAV differs from its measured completion"
                )
            info = path.lstat()
            if info.st_nlink != 1:
                raise ValueError("native music result is not exclusively owned")
            return info.st_dev, info.st_ino, info.st_mtime_ns

        identity = await separation_file_effect(verify)
        chunks = (
            manifest.size_bytes + SKULK_MAX_CHUNK_SIZE - 1
        ) // SKULK_MAX_CHUNK_SIZE
        common = _PacketAddress(
            source_node=self.node_id,
            target_node=entry.task.owner_node,
            command_id=entry.task.command_id,
            model=ModelId(entry.task.task_params.model),
            purpose=entry.purpose,
        )
        await self._send(
            entry,
            OutputMediaPacket(
                **common,
                kind="opened",
                sequence=0,
                total_chunks=chunks,
                total_bytes=manifest.size_bytes,
                content_type="audio/wav",
            ),
        )
        digest = hashlib.sha256()
        for sequence in range(1, chunks + 1):
            offset = (sequence - 1) * SKULK_MAX_CHUNK_SIZE
            length = min(SKULK_MAX_CHUNK_SIZE, manifest.size_bytes - offset)

            def read_frame(offset: int = offset, length: int = length) -> bytes:
                descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                try:
                    info = os.fstat(descriptor)
                    if (
                        (info.st_dev, info.st_ino, info.st_mtime_ns) != identity
                        or info.st_size != manifest.size_bytes
                        or not stat.S_ISREG(info.st_mode)
                        or info.st_uid != os.getuid()
                        or info.st_mode & 0o077
                        or info.st_nlink != 1
                    ):
                        raise ValueError("native music result changed while streaming")
                    frame = os.pread(descriptor, length, offset)
                    if len(frame) != length:
                        raise ValueError("native music result is incomplete")
                    return frame
                finally:
                    os.close(descriptor)

            data = await separation_file_effect(read_frame)
            digest.update(data)
            await self._send(
                entry,
                OutputMediaPacket(
                    **common,
                    kind="chunk",
                    sequence=sequence,
                    total_chunks=chunks,
                    data=data,
                ),
            )
        if digest.hexdigest() != manifest.sha256:
            raise ValueError("native music result changed during delivery")
        # A same-node API can acknowledge inside send. Establish correlation
        # first, but retain the directory until that entire send has returned.
        entry.terminal = chunks + 1
        await self._send(
            entry,
            OutputMediaPacket(
                **common,
                kind="completed",
                sequence=chunks + 1,
                total_chunks=chunks,
                total_bytes=manifest.size_bytes,
                sha256=manifest.sha256,
            ),
        )

    async def deliver(self, command: CommandId) -> None:
        """Verify the WAV, stream it, await custody ACK and join before deletion."""
        entry = self._entries[command]
        if entry.sending or entry.finished:
            return
        entry.sending = True
        try:
            with anyio.fail_after(self.timeout_seconds):
                await self._stream(entry)
                while not entry.accepted:
                    if entry.aborted:
                        raise ValueError("native music output delivery was cancelled")
                    entry.changed = anyio.Event()
                    await entry.changed.wait()
        except Exception as error:  # noqa: BLE001 - settle this bounded media owner
            logger.opt(exception=error).warning("Native music result delivery failed")
            with contextlib.suppress(
                anyio.BrokenResourceError, anyio.ClosedResourceError, TimeoutError
            ):
                with anyio.fail_after(5):
                    await self.fail(
                        entry.task,
                        "Native music result delivery failed; retry the operation.",
                    )
        finally:
            # File effects join before cancellation reaches here; no abandoned
            # thread may read a directory while cleanup removes its last copy.
            entry.sending = False
            entry.finished = True
            with anyio.CancelScope(shield=True):
                await self._cleanup(command, entry)

    async def _cleanup(self, command: CommandId, entry: _Delivery) -> None:
        try:
            reclaimed = await separation_file_effect(
                lambda: reclaim_result_directory(entry.directory)
            )
            if not reclaimed:
                return
        except (OSError, ValueError):
            logger.warning(
                "Native music result cleanup retained its bounded reservation"
            )
            return
        self._entries.pop(command, None)

    async def tick(self) -> None:
        """Retry failed cleanup without admitting another result into its slot."""
        for command, entry in tuple(self._entries.items()):
            if entry.finished and not entry.sending:
                await self._cleanup(command, entry)
