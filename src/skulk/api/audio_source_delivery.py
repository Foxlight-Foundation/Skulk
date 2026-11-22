"""Addressed original-WAV delivery shared by native audio API owners."""

import hashlib
import os
from collections.abc import Awaitable, Callable
from pathlib import Path

import anyio

from skulk.routing.vision_media import AudioSourcePayload, VisionMediaPacket
from skulk.shared.constants import SKULK_MAX_CHUNK_SIZE
from skulk.shared.models.model_cards import ModelId
from skulk.shared.types.common import NodeId
from skulk.worker.audio_source_types import AudioSourceTask
from skulk.worker.separation_input import separation_file_effect


async def send_original_wav(
    path: Path,
    task: AudioSourceTask,
    target: NodeId,
    payload: AudioSourcePayload,
    send: Callable[[VisionMediaPacket], Awaitable[None]],
    current: Callable[[], bool],
) -> None:
    """Send a retained private original under total/per-frame transfer deadlines.

    The service owns admission, source custody, authoritative target binding and
    cancellation. Every joined read checks the exact encoded length and the full
    transfer rehashes against the accepted original. No caller path is accepted.
    """
    source, chunks = task.task_params.source, task.task_params.total_input_chunks

    async def emit(packet: VisionMediaPacket) -> None:
        with anyio.fail_after(5):
            await send(packet)

    with anyio.fail_after(180):
        if not current():
            return
        await emit(
            VisionMediaPacket(
                source_node=task.owner_node,
                target_node=target,
                command_id=task.command_id,
                model=ModelId(task.task_params.model),
                payload=payload,
                kind="opened",
                sequence=0,
                image_count=1,
                total_chunks=chunks,
            )
        )
        digest = hashlib.sha256()
        for sequence in range(1, chunks + 1):
            if not current():
                return
            offset = (sequence - 1) * SKULK_MAX_CHUNK_SIZE
            size = min(SKULK_MAX_CHUNK_SIZE, source.size_bytes - offset)

            def read_frame(offset: int = offset, size: int = size) -> bytes:
                descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                try:
                    return os.pread(descriptor, size, offset)
                finally:
                    os.close(descriptor)

            data = await separation_file_effect(read_frame)
            if len(data) != size or not data:
                raise ValueError("Retained original is incomplete")
            digest.update(data)
            if not current():
                return
            await emit(
                VisionMediaPacket(
                    source_node=task.owner_node,
                    target_node=target,
                    command_id=task.command_id,
                    model=ModelId(task.task_params.model),
                    payload=payload,
                    kind="chunk",
                    sequence=sequence,
                    total_chunks=chunks,
                    image_index=0,
                    data=data,
                )
            )
        if digest.hexdigest() != source.sha256:
            raise ValueError("Retained original changed")
        if current():
            await emit(
                VisionMediaPacket(
                    source_node=task.owner_node,
                    target_node=target,
                    command_id=task.command_id,
                    model=ModelId(task.task_params.model),
                    payload=payload,
                    kind="completed",
                    sequence=chunks + 1,
                    total_chunks=chunks,
                    image_count=1,
                    sha256=source.sha256,
                )
            )
