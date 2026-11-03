"""Deliver four private stems with bounded ownership through reverse ACKs.

The worker reserves source capacity until this owner settles. Native completion
provides the fixed result root and metadata; peers never supply local paths.
All file effects are joined before deletion, including coroutine cancellation.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import stat
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import final

import anyio
from loguru import logger

from skulk.routing.output_media import OutputMediaPacket, OutputMediaPurpose
from skulk.shared.constants import SKULK_MAX_CHUNK_SIZE
from skulk.shared.models.model_cards import ModelId
from skulk.shared.types.audio_separation import (
    MAX_SEPARATION_STEM_BYTES,
    SEPARATION_STEMS,
    AudioSeparationOutput,
    SeparatedStem,
    stem_from_wav,
)
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.tasks import AudioSeparation
from skulk.utils.runtime_files import read_private
from skulk.worker.runner.demucs.model import separation_result_directory
from skulk.worker.runner.demucs.result_custody import reclaim_result_directory
from skulk.worker.separation_input import separation_file_effect


def separation_purpose(stem: str) -> OutputMediaPurpose:
    """Resolve only the four explicit stem roles, never an arbitrary media tag."""
    match stem:
        case "drums":
            return "separation_drums"
        case "bass":
            return "separation_bass"
        case "other":
            return "separation_other"
        case "vocals":
            return "separation_vocals"
        case _:
            raise ValueError("unknown separation stem")


def _owned_directory(directory: Path) -> None:
    info = directory.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_mode & 0o077
    ):
        raise ValueError("separation result directory is not private")


def remove_separation_result(directory: Path) -> None:
    """Remove four known WAVs and their bounded native completion record.

    Unexpected content or symlinks remain untouched and keep the reservation.
    There is no recursive deletion or cleanup based on packet-provided paths.
    """
    if not reclaim_result_directory(directory):
        raise ValueError("separation result ownership is unresolved; retain its admission slot")


@dataclass
class _Delivery:
    task: AudioSeparation
    directory: Path
    output: AudioSeparationOutput
    completed: dict[OutputMediaPurpose, int] = field(default_factory=dict)
    accepted: set[OutputMediaPurpose] = field(default_factory=set)
    changed: anyio.Event = field(default_factory=anyio.Event)
    aborted: bool = False
    sending: bool = False
    finished: bool = False


@final
class SeparationOutputProducer:
    """Own at most four result sets until senders drain and cleanup succeeds."""

    def __init__(
        self,
        node_id: NodeId,
        send: Callable[[OutputMediaPacket], Awaitable[None]],
        fail: Callable[[AudioSeparation, str], Awaitable[None]],
        *,
        timeout_seconds: float = 300,
    ) -> None:
        """Bind effects without reading files or admitting native work."""
        if not 0 < timeout_seconds <= 300:
            raise ValueError(
                "separation delivery requires a bounded five-minute budget"
            )
        self.node_id, self.send, self.fail = node_id, send, fail
        self.timeout_seconds = timeout_seconds
        self._entries: dict[CommandId, _Delivery] = {}

    def register(
        self, task: AudioSeparation, root: Path, output: AudioSeparationOutput
    ) -> bool:
        """Reserve a completed result only for its actual task/source identity."""
        previous = self._entries.get(task.command_id)
        if previous is not None:
            if (previous.task, previous.output, previous.directory) != (
                task,
                output,
                separation_result_directory(root, task.command_id),
            ):
                raise ValueError("separation output changed its admitted identity")
            return False
        if len(self._entries) >= 4 or output.source != task.task_params.source:
            raise ValueError(
                "separation output has no capacity or differs from admitted source"
            )
        self._entries[task.command_id] = _Delivery(
            task, separation_result_directory(root, task.command_id), output
        )
        return True

    def pending(self, command_id: CommandId) -> bool:
        """Include sending, waiting for ACK and failed cleanup in admission."""
        return command_id in self._entries

    def receive(self, packet: OutputMediaPacket) -> bool:
        """Accept only correlated reverse terminals for already emitted roles."""
        entry = self._entries.get(packet.command_id)
        if entry is None or (
            packet.target_node != self.node_id
            or packet.source_node != entry.task.owner_node
            or packet.model != entry.task.task_params.model
        ):
            return False
        if packet.kind in ("cancelled", "transport_failed") and packet.sequence == 0:
            # The owning API may cancel after DATA has finished but before this
            # role opened. Match owner/model/known role, without requiring an ACK
            # terminal that this unfinished transfer has not emitted yet.
            if packet.purpose not in {separation_purpose(name) for name in SEPARATION_STEMS}:
                return False
            entry.aborted = True
            entry.changed.set()
            return True
        expected = entry.completed.get(packet.purpose)
        if expected is None or packet.sequence != expected:
            return False
        if packet.kind == "accepted":
            entry.accepted.add(packet.purpose)
        elif packet.kind in ("cancelled", "transport_failed"):
            entry.aborted = True
        else:
            return False
        entry.changed.set()
        return True

    def cancel(self, command_id: CommandId) -> None:
        """Stop owned transfer work; its coroutine retains files until it drains."""
        entry = self._entries.get(command_id)
        if entry is not None:
            entry.aborted = True
            entry.changed.set()

    async def _send(self, entry: _Delivery, packet: OutputMediaPacket) -> None:
        if entry.aborted:
            raise ValueError("separation output delivery was cancelled")
        with anyio.fail_after(5):
            await self.send(packet)

    async def _stem(self, entry: _Delivery, stem: SeparatedStem) -> None:
        path = entry.directory / f"{stem.name}.wav"

        def verify() -> tuple[int, int, int]:
            _owned_directory(entry.directory)
            data = read_private(path, MAX_SEPARATION_STEM_BYTES)
            if stem_from_wav(stem.name, data) != stem:
                raise ValueError("separation output differs from native manifest")
            info = path.lstat()
            if info.st_nlink != 1:
                raise ValueError("separation output is not exclusively owned")
            return info.st_dev, info.st_ino, info.st_mtime_ns

        identity = await separation_file_effect(verify)
        purpose = separation_purpose(stem.name)
        chunks = (stem.size_bytes + SKULK_MAX_CHUNK_SIZE - 1) // SKULK_MAX_CHUNK_SIZE
        await self._send(
            entry,
            OutputMediaPacket(
                source_node=self.node_id,
                target_node=entry.task.owner_node,
                command_id=entry.task.command_id,
                model=ModelId(entry.task.task_params.model),
                purpose=purpose,
                kind="opened",
                sequence=0,
                total_chunks=chunks,
                total_bytes=stem.size_bytes,
                content_type="audio/wav",
            ),
        )
        digest = hashlib.sha256()
        for sequence in range(1, chunks + 1):
            offset = (sequence - 1) * SKULK_MAX_CHUNK_SIZE
            expected_bytes = min(SKULK_MAX_CHUNK_SIZE, stem.size_bytes - offset)

            def read_frame(
                offset: int = offset, expected_bytes: int = expected_bytes
            ) -> bytes:
                descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                try:
                    info = os.fstat(descriptor)
                    if (
                        (info.st_dev, info.st_ino, info.st_mtime_ns) != identity
                        or info.st_size != stem.size_bytes
                        or not stat.S_ISREG(info.st_mode)
                        or info.st_uid != os.getuid()
                        or info.st_mode & 0o077
                        or info.st_nlink != 1
                    ):
                        raise ValueError("separation output changed while streaming")
                    data = os.pread(descriptor, expected_bytes, offset)
                    if len(data) != expected_bytes:
                        raise ValueError("separation output is incomplete")
                    return data
                finally:
                    os.close(descriptor)

            data = await separation_file_effect(read_frame)
            digest.update(data)
            await self._send(
                entry,
                OutputMediaPacket(
                    source_node=self.node_id,
                    target_node=entry.task.owner_node,
                    command_id=entry.task.command_id,
                    model=ModelId(entry.task.task_params.model),
                    purpose=purpose,
                    kind="chunk",
                    sequence=sequence,
                    total_chunks=chunks,
                    data=data,
                ),
            )
        if digest.hexdigest() != stem.sha256:
            raise ValueError("separation output changed during delivery")
        # A same-node ACK can arrive during send, so establish its exact expected
        # terminal before publishing. Cleanup still waits for this whole sender.
        entry.completed[purpose] = chunks + 1
        await self._send(
            entry,
            OutputMediaPacket(
                source_node=self.node_id,
                target_node=entry.task.owner_node,
                command_id=entry.task.command_id,
                model=ModelId(entry.task.task_params.model),
                purpose=purpose,
                kind="completed",
                sequence=chunks + 1,
                total_chunks=chunks,
                total_bytes=stem.size_bytes,
                sha256=stem.sha256,
            ),
        )

    async def deliver(self, command_id: CommandId) -> None:
        """Verify/send all four roles, await ACKs and then release owned files."""
        entry = self._entries[command_id]
        if entry.sending or entry.finished:
            return
        entry.sending = True
        try:
            with anyio.fail_after(self.timeout_seconds):
                for stem in entry.output.stems:
                    await self._stem(entry, stem)
                while len(entry.accepted) != len(SEPARATION_STEMS):
                    if entry.aborted:
                        raise ValueError("separation output delivery was cancelled")
                    entry.changed = anyio.Event()
                    await entry.changed.wait()
        except Exception as error:  # noqa: BLE001 - settle this bounded media owner
            logger.opt(exception=error).warning("Separation result delivery failed")
            with contextlib.suppress(
                anyio.BrokenResourceError, anyio.ClosedResourceError, TimeoutError
            ):
                with anyio.fail_after(5):
                    await self.fail(
                        entry.task,
                        "Separation result delivery failed; retry the operation.",
                    )
        finally:
            # separation_file_effect joins outstanding reads before it lets cancellation
            # reach this boundary. No sender can still be using these paths.
            entry.sending = False
            entry.finished = True
            with anyio.CancelScope(shield=True):
                await self._cleanup(command_id, entry)

    async def _cleanup(self, command_id: CommandId, entry: _Delivery) -> None:
        try:
            await separation_file_effect(
                lambda: remove_separation_result(entry.directory)
            )
        except (OSError, ValueError):
            logger.warning("Separation result cleanup retained its bounded reservation")
            return
        self._entries.pop(command_id, None)

    async def tick(self) -> None:
        """Retry failed deletion without abandoning or expanding admission."""
        for command_id, entry in tuple(self._entries.items()):
            if entry.finished and not entry.sending:
                await self._cleanup(command_id, entry)
