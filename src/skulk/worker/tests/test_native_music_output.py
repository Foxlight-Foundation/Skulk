"""Actual native WAV streaming, API custody, reverse ACK races and joined cleanup."""

import asyncio
from pathlib import Path

import anyio
import pytest

from skulk.api.music_edit_media import MusicEditMediaStore
from skulk.api.video_store import VideoStore
from skulk.routing.output_media import (
    OutputMediaPacket,
    output_media_for_api,
    output_media_for_worker,
)
from skulk.routing.router import OutboundPacket, TopicRouter
from skulk.routing.topics import OUTPUT_MEDIA
from skulk.shared.tests.music_edit_fixtures import music_edit_fixture
from skulk.shared.types.common import NodeId
from skulk.shared.types.music import MusicGenerationTaskParams, MusicOutputManifest
from skulk.shared.types.tasks import MusicEditing, MusicGeneration
from skulk.shared.types.worker.instances import InstanceId
from skulk.utils.channels import channel
from skulk.utils.runtime_files import write_private
from skulk.worker.native_music_output import NativeMusicOutputProducer, NativeMusicTask
from skulk.worker.runner.ace_step.result_custody import (
    music_result_directory,
    prepare_result_directory,
)
from skulk.worker.runner.ace_step.tests.test_result_custody import parents


def completion(
    root: Path,
    *,
    editing: bool = True,
) -> tuple[NativeMusicTask, MusicOutputManifest, bytes]:
    """Write one real journaled native result; this proves files, never inference."""
    job, output, _, data = music_edit_fixture()
    task: NativeMusicTask
    if editing:
        task = MusicEditing(
            command_id=job.id,
            owner_node=NodeId("api"),
            instance_id=InstanceId("native"),
            task_params=job.params,
        )
    else:
        task = MusicGeneration(
            command_id=job.id,
            owner_node=NodeId("api"),
            instance_id=InstanceId("native"),
            task_params=MusicGenerationTaskParams(
                model=job.params.model,
                prompt=job.params.prompt,
                seconds=1,
                seed=0,
            ),
        )
    directory = music_result_directory(root, task.command_id)
    prepare_result_directory(directory, parents(root))
    write_private(directory / "result.wav", data)
    write_private(directory / "result.json", output.result.model_dump_json().encode())
    facts = output.result.output
    return (
        task,
        MusicOutputManifest(
            size_bytes=facts.size_bytes,
            sha256=facts.sha256,
            sample_rate=facts.sample_rate,
            channels=facts.channels,
            duration_seconds=facts.frames / facts.sample_rate,
        ),
        data,
    )


@pytest.mark.parametrize("editing", [False, True])
async def test_real_api_custody_and_synchronous_ack_retain_sender(
    tmp_path: Path,
    *,
    editing: bool,
) -> None:
    """A reverse ACK inside completed send cannot delete a sender's live directory."""
    root = tmp_path / "native"
    task, manifest, data = completion(root, editing=editing)
    directory = music_result_directory(root, task.command_id)
    failure: list[str] = []
    store = (
        MusicEditMediaStore(tmp_path / "api")
        if editing
        else VideoStore(tmp_path / "api")
    )
    retained: list[bytes] = []

    async def fail(_task: NativeMusicTask, message: str) -> None:
        failure.append(message)

    async def send(packet: OutputMediaPacket) -> None:
        assert directory.exists() and producer.pending(task.command_id)
        if packet.kind == "opened":
            assert packet.total_bytes is not None and packet.total_chunks is not None
            if isinstance(store, MusicEditMediaStore):
                store.open_assembly(
                    packet.command_id,
                    total_bytes=packet.total_bytes,
                    total_chunks=packet.total_chunks,
                )
            else:
                store.open_assembly(
                    packet.command_id,
                    "music",
                    content_type="audio/wav",
                    total_bytes=packet.total_bytes,
                    total_chunks=packet.total_chunks,
                )
        elif packet.kind == "chunk":
            if isinstance(store, MusicEditMediaStore):
                store.append(packet.command_id, packet.sequence, packet.data)
            else:
                store.append(packet.command_id, "music", packet.sequence, packet.data)
        elif packet.kind == "completed":
            assert packet.total_chunks is not None and packet.sha256 is not None
            if isinstance(store, MusicEditMediaStore):
                _, output, _, _ = music_edit_fixture()
                stored = store.commit(
                    packet.command_id,
                    output.result.output,
                    total_chunks=packet.total_chunks,
                )
            else:
                stored = store.commit(
                    packet.command_id,
                    "music",
                    sha256=packet.sha256,
                    total_chunks=packet.total_chunks,
                )
            retained.append(stored.file_path.read_bytes())
            assert producer.receive(packet.accepted())
            await asyncio.sleep(0)
            assert directory.exists()

    producer = NativeMusicOutputProducer(NodeId("worker"), send, fail)
    try:
        assert producer.register(task, root, manifest)
        await producer.deliver(task.command_id)
        assert retained == [data] and not failure
        assert not directory.exists() and not producer.pending(task.command_id)
    finally:
        if isinstance(store, MusicEditMediaStore):
            store.close()


async def test_early_or_foreign_ack_cannot_release_native_output(
    tmp_path: Path,
) -> None:
    """Only the emitted terminal from the exact addressed owner settles custody."""
    task, manifest, _ = completion(tmp_path)
    failed: list[str] = []
    completed = asyncio.Event()
    terminal: list[OutputMediaPacket] = []

    async def fail(_task: NativeMusicTask, message: str) -> None:
        failed.append(message)

    async def send(packet: OutputMediaPacket) -> None:
        if packet.kind == "opened":
            assert not producer.receive(
                OutputMediaPacket(
                    source_node=task.owner_node,
                    target_node=NodeId("worker"),
                    command_id=task.command_id,
                    model=packet.model,
                    purpose="music_edit",
                    kind="accepted",
                    sequence=99,
                )
            )
        if packet.kind == "completed":
            terminal.append(packet)
            completed.set()

    producer = NativeMusicOutputProducer(NodeId("worker"), send, fail)
    assert producer.register(task, tmp_path, manifest)
    delivery = asyncio.create_task(producer.deliver(task.command_id))
    await asyncio.wait_for(completed.wait(), 3)
    ack = terminal[0].accepted()
    assert not producer.receive(
        ack.model_copy(update={"source_node": NodeId("foreign")})
    )
    assert producer.pending(task.command_id) and not delivery.done()
    assert producer.receive(ack)
    await delivery
    assert not failed and not producer.pending(task.command_id)


@pytest.mark.parametrize("editing", [False, True])
async def test_actual_same_node_rendezvous_delivers_native_result_and_reverse_ack(
    tmp_path: Path,
    *,
    editing: bool,
) -> None:
    """Use actual zero-buffer topic fan-out, preserving legacy music admission."""
    task, manifest, data = completion(tmp_path / "native", editing=editing)
    task = task.model_copy(update={"owner_node": NodeId("same-node")})
    network_send, network_receive = channel[OutboundPacket](1)
    router = TopicRouter(OUTPUT_MEDIA, network_send, local_routing_key="same-node")
    api_receive = router.new_receiver(accept=output_media_for_api)
    worker_receive = router.new_receiver(accept=output_media_for_worker)
    sender = router.new_sender()
    store = VideoStore(tmp_path / "api")
    retained: list[bytes] = []
    failures: list[str] = []

    async def fail(_task: NativeMusicTask, message: str) -> None:
        failures.append(message)

    producer = NativeMusicOutputProducer(NodeId("same-node"), sender.send, fail)

    async def api_ingress() -> None:
        with api_receive:
            async for packet in api_receive:
                if packet.kind == "opened":
                    assert (
                        packet.total_bytes is not None
                        and packet.total_chunks is not None
                    )
                    store.open_assembly(
                        packet.command_id,
                        "music",
                        content_type="audio/wav",
                        total_bytes=packet.total_bytes,
                        total_chunks=packet.total_chunks,
                    )
                elif packet.kind == "chunk":
                    store.append(
                        packet.command_id, "music", packet.sequence, packet.data
                    )
                elif packet.kind == "completed":
                    assert packet.sha256 is not None and packet.total_chunks is not None
                    stored = store.commit(
                        packet.command_id,
                        "music",
                        sha256=packet.sha256,
                        total_chunks=packet.total_chunks,
                    )
                    retained.append(stored.file_path.read_bytes())
                    await sender.send(packet.accepted())

    async def worker_ingress() -> None:
        with worker_receive:
            async for packet in worker_receive:
                producer.receive(packet)

    try:
        async with anyio.create_task_group() as group:
            group.start_soon(router.run)
            group.start_soon(api_ingress)
            group.start_soon(worker_ingress)
            try:
                assert producer.register(task, tmp_path / "native", manifest)
                with anyio.fail_after(5):
                    await producer.deliver(task.command_id)
                assert retained == [data] and not failures
                assert not producer.pending(task.command_id)
            finally:
                group.cancel_scope.cancel()
    finally:
        network_send.close()
        network_receive.close()


@pytest.mark.parametrize("fault", ["modified", "symlink", "timeout", "cancel"])
async def test_failed_transfer_reclaims_only_after_owned_sender_drains(
    tmp_path: Path,
    fault: str,
) -> None:
    """Corrupt or abandoned native results fail rather than expose partial content."""
    task, manifest, _ = completion(tmp_path)
    directory = music_result_directory(tmp_path, task.command_id)
    path = directory / "result.wav"
    failed: list[str] = []

    async def fail(_task: NativeMusicTask, message: str) -> None:
        failed.append(message)

    async def send(packet: OutputMediaPacket) -> None:
        if fault == "cancel" and packet.kind == "chunk":
            producer.cancel(task.command_id)

    if fault == "modified":
        path.write_bytes(b"not a WAV")
    elif fault == "symlink":
        path.unlink()
        path.symlink_to(tmp_path / "unowned")
    producer = NativeMusicOutputProducer(
        NodeId("worker"), send, fail, timeout_seconds=0.1
    )
    assert producer.register(task, tmp_path, manifest)
    await producer.deliver(task.command_id)
    assert failed
    if fault == "symlink":
        assert producer.pending(task.command_id) and path.is_symlink()
    else:
        assert not producer.pending(task.command_id) and not directory.exists()
