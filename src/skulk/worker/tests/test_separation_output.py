"""Four actual WAV roles, reverse ACK races and retained file ownership."""

import asyncio
from pathlib import Path

import pytest

from skulk.api.separation_media import SeparationMediaStore
from skulk.api.tests.test_separation_media import wav
from skulk.routing.output_media import OutputMediaPacket
from skulk.shared.types.audio_separation import (
    SEPARATION_STEMS,
    AudioSeparationOutput,
    NativeSeparationResult,
    SeparationStemName,
    stem_from_wav,
)
from skulk.shared.types.common import NodeId
from skulk.shared.types.tasks import AudioSeparation
from skulk.worker.runner.demucs.model import separation_result_directory
from skulk.worker.runner.demucs.result_custody import remember_result_directory
from skulk.worker.separation_output import SeparationOutputProducer
from skulk.worker.tests.test_separation_input import fixture


def result(root: Path, task: AudioSeparation) -> AudioSeparationOutput:
    """Write four real private PCM24 WAVs with independent metadata."""
    directory = separation_result_directory(root, task.command_id)
    directory.mkdir(mode=0o700, parents=True)
    directory.parent.chmod(0o700)
    remember_result_directory(directory)
    data = wav(rate=44100)
    for name in SEPARATION_STEMS:
        path = directory / f"{name}.wav"
        path.write_bytes(data)
        path.chmod(0o600)
    facts = tuple(stem_from_wav(name, data) for name in SEPARATION_STEMS)
    output = AudioSeparationOutput(
        source=task.task_params.source,
        stems=facts,
        common_gain=1.0,
        pre_gain_peak=facts[0].peak,
    )
    record = directory / "result.json"
    record.write_text(
        NativeSeparationResult(
            output=output,
            device="cpu",
            load_seconds=1.0,
            separation_seconds=1.0,
            total_seconds=2.0,
            process_peak_rss_bytes=1024,
        ).model_dump_json()
    )
    record.chmod(0o600)
    return output


def role(packet: OutputMediaPacket) -> SeparationStemName:
    """Require the four known packet roles in the actual accepting file owner."""
    match packet.purpose:
        case "separation_drums":
            return "drums"
        case "separation_bass":
            return "bass"
        case "separation_other":
            return "other"
        case "separation_vocals":
            return "vocals"
        case _:
            raise ValueError("not a stem packet")


async def test_actual_four_role_delivery_to_accepting_store_and_synchronous_ack(
    tmp_path: Path,
) -> None:
    """Same-loop ACKs cannot race cleanup ahead of another role's sender."""
    current, _ = fixture()
    root = tmp_path / "native"
    output = result(root, current.task)
    directory = separation_result_directory(root, current.task.command_id)
    store = SeparationMediaStore(tmp_path / "api")
    completed: list[SeparationStemName] = []
    failures: list[str] = []

    async def fail(_task: AudioSeparation, message: str) -> None:
        failures.append(message)

    async def send(packet: OutputMediaPacket) -> None:
        name = role(packet)
        assert directory.is_dir() and producer.pending(packet.command_id)
        if packet.kind == "opened":
            assert packet.total_bytes is not None and packet.total_chunks is not None
            store.open_assembly(
                packet.command_id,
                name,
                total_bytes=packet.total_bytes,
                total_chunks=packet.total_chunks,
            )
        elif packet.kind == "chunk":
            store.append(packet.command_id, name, packet.sequence, packet.data)
        elif packet.kind == "completed":
            assert packet.sha256 is not None and packet.total_chunks is not None
            retained = store.commit(
                packet.command_id,
                name,
                sha256=packet.sha256,
                total_chunks=packet.total_chunks,
            )
            assert retained.facts == next(
                stem for stem in output.stems if stem.name == name
            )
            completed.append(name)
            assert producer.receive(packet.accepted())
            # Even the fourth ACK cannot unlink while send itself is running.
            assert directory.is_dir()

    producer = SeparationOutputProducer(NodeId("node-0"), send, fail)
    try:
        assert producer.register(current.task, root, output)
        await producer.deliver(current.task.command_id)
        assert completed == list(SEPARATION_STEMS) and not failures
        assert not directory.exists() and not producer.pending(current.task.command_id)
    finally:
        store.close()


async def test_missing_ack_expires_and_drops_only_drained_private_results(
    tmp_path: Path,
) -> None:
    """A disappeared accepting API cannot retain unbounded native result sets."""
    current, _ = fixture()
    root = tmp_path / "native"
    output = result(root, current.task)
    sent: list[OutputMediaPacket] = []
    failed: list[AudioSeparation] = []

    async def send(packet: OutputMediaPacket) -> None:
        sent.append(packet)

    async def fail(task: AudioSeparation, _message: str) -> None:
        failed.append(task)

    producer = SeparationOutputProducer(
        NodeId("node-0"), send, fail, timeout_seconds=0.5
    )
    producer.register(current.task, root, output)
    await producer.deliver(current.task.command_id)
    assert sum(packet.kind == "completed" for packet in sent) == 4
    assert failed == [current.task] and not producer.pending(current.task.command_id)
    assert not separation_result_directory(root, current.task.command_id).exists()


async def test_peer_failure_cannot_delete_files_while_sender_is_active(
    tmp_path: Path,
) -> None:
    """Reverse failure requests a stop; the in-flight sender still owns paths."""
    current, _ = fixture()
    root = tmp_path / "native"
    output = result(root, current.task)
    directory = separation_result_directory(root, current.task.command_id)
    started, release = asyncio.Event(), asyncio.Event()
    failed: list[str] = []

    async def send(packet: OutputMediaPacket) -> None:
        if packet.kind == "completed":
            assert not producer.receive(
                packet.accepted().model_copy(update={"source_node": NodeId("foreign")})
            )
            assert not producer.receive(
                packet.accepted().model_copy(update={"model": "foreign/model"})
            )
            assert not producer.receive(
                packet.accepted().model_copy(update={"sequence": 0})
            )
            assert producer.receive(
                packet.accepted().model_copy(update={"kind": "cancelled"})
            )
            started.set()
            await release.wait()
            assert directory.exists()

    async def fail(_task: AudioSeparation, message: str) -> None:
        failed.append(message)

    producer = SeparationOutputProducer(NodeId("node-0"), send, fail)
    producer.register(current.task, root, output)
    task = asyncio.create_task(producer.deliver(current.task.command_id))
    try:
        await asyncio.wait_for(started.wait(), 3)
        await producer.tick()
        assert directory.exists() and producer.pending(current.task.command_id)
        release.set()
        await task
        assert len(failed) == 1 and not directory.exists()
    finally:
        release.set()
        await task


async def test_foreign_cleanup_content_retains_capacity_until_reconciled(
    tmp_path: Path,
) -> None:
    """No recursive deletion hides an unresolved file or admits a fifth result."""
    current, _ = fixture()
    root = tmp_path / "native"
    output = result(root, current.task)
    directory = separation_result_directory(root, current.task.command_id)
    foreign = directory / "unrelated"
    foreign.write_text("preserve")

    async def send(packet: OutputMediaPacket) -> None:
        if packet.kind == "completed":
            assert producer.receive(packet.accepted())

    async def fail(_task: AudioSeparation, _message: str) -> None:
        raise AssertionError("delivery should succeed")

    producer = SeparationOutputProducer(NodeId("node-0"), send, fail)
    producer.register(current.task, root, output)
    await producer.deliver(current.task.command_id)
    assert foreign.read_text() == "preserve" and producer.pending(
        current.task.command_id
    )
    assert all((directory / f"{name}.wav").exists() for name in SEPARATION_STEMS)
    for number in range(3):
        other = current.task.model_copy(
            update={"command_id": type(current.task.command_id)(f"another-{number}")}
        )
        producer.register(other, root, output)
    with pytest.raises(ValueError, match="capacity"):
        producer.register(
            current.task.model_copy(
                update={"command_id": type(current.task.command_id)("fifth")}
            ),
            root,
            output,
        )
    foreign.unlink()
    await producer.tick()
    assert not producer.pending(current.task.command_id) and not directory.exists()


async def test_changed_stem_refuses_delivery_and_preserves_source_identity(
    tmp_path: Path,
) -> None:
    """A native metadata claim is checked again against real WAV facts."""
    current, _ = fixture()
    root = tmp_path / "native"
    output = result(root, current.task)
    directory = separation_result_directory(root, current.task.command_id)
    (directory / "drums.wav").write_bytes(wav(value=2000, rate=8000))
    sent: list[OutputMediaPacket] = []
    failed: list[str] = []

    async def send(packet: OutputMediaPacket) -> None:
        sent.append(packet)

    async def fail(_task: AudioSeparation, message: str) -> None:
        failed.append(message)

    producer = SeparationOutputProducer(NodeId("node-0"), send, fail)
    producer.register(current.task, root, output)
    await producer.deliver(current.task.command_id)
    assert not sent and len(failed) == 1 and not directory.exists()
