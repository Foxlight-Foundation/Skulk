"""Authoritative source handshake and real async races over injected file effects."""

import asyncio
import io
import threading
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import anyio
import pytest

from skulk.master.main import separation_generation_events
from skulk.master.tests.test_separation_admission import command, state
from skulk.routing.vision_media import (
    VisionMediaPacket,
    decode_vision_media_packet,
    encode_vision_media_packet,
)
from skulk.shared.models.model_cards import ModelId
from skulk.shared.types.audio_separation import source_from_wav
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.events import TaskCreated
from skulk.shared.types.state import State
from skulk.shared.types.tasks import AudioSeparation, TaskId, TaskStatus
from skulk.shared.types.worker.instances import InstanceId
from skulk.shared.types.worker.runners import RunnerId, RunnerReady
from skulk.worker.separation_input import (
    SeparationInputReceiver,
    local_separation_input_task,
)
from skulk.worker.separation_sources import SeparationSourceStore


def original() -> bytes:
    """Return a real one-second mono PCM16 original accepted by the source contract."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setparams((1, 2, 8000, 8000, "NONE", "not compressed"))
        audio.writeframes(bytes(16000))
    return buffer.getvalue()


class Copies:
    """Model file effects only; real private-file custody has its own native proof."""

    def __init__(self) -> None:
        self.tasks: dict[CommandId, AudioSeparation] = {}
        self.chunks: dict[CommandId, dict[int, bytes]] = {}
        self.deleted: list[CommandId] = []
        self.on_complete: Callable[[], None] = lambda: None
        self.fail_delete = False

    def admit(self, task: AudioSeparation) -> None:
        self.tasks[task.command_id] = task
        self.chunks[task.command_id] = {}

    def append(self, command_id: CommandId, sequence: int, data: bytes) -> None:
        self.chunks[command_id].setdefault(sequence, data)

    def complete(self, command_id: CommandId) -> Path | None:
        task = self.tasks[command_id]
        chunks = self.chunks[command_id]
        if len(chunks) != task.task_params.total_input_chunks:
            return None
        content = b"".join(chunks[index] for index in range(1, len(chunks) + 1))
        if source_from_wav(content) != task.task_params.source:
            raise ValueError("fixture independent source verification failed")
        self.on_complete()
        return Path("private-owned-source")

    def delete(self, command_id: CommandId) -> None:
        if self.fail_delete:
            self.fail_delete = False
            raise OSError("fixture retained cleanup")
        self.deleted.append(command_id)
        self.tasks.pop(command_id, None)
        self.chunks.pop(command_id, None)


@dataclass
class Fixture:
    snapshot: State
    task: AudioSeparation
    copies: Copies
    sent: list[VisionMediaPacket]
    failures: list[tuple[TaskId, str]]
    now: float = 0.0
    unavailable: bool = False


def fixture() -> tuple[Fixture, SeparationInputReceiver]:
    """Use actual master task creation and placement facts, with bounded file effects."""
    snapshot = state(RunnerReady())
    request = command()
    request = request.model_copy(
        update={
            "task_params": request.task_params.model_copy(
                update={
                    "source": source_from_wav(original()),
                    "total_input_chunks": 2,
                }
            )
        }
    )
    events, _ = separation_generation_events(snapshot, request)
    assert isinstance(events[0], TaskCreated) and isinstance(
        events[0].task, AudioSeparation
    )
    task = events[0].task
    current = Fixture(
        snapshot.model_copy(update={"tasks": {task.task_id: task}}),
        task,
        Copies(),
        [],
        [],
    )

    async def send(packet: VisionMediaPacket) -> None:
        if current.unavailable:
            raise anyio.ClosedResourceError
        current.sent.append(packet)

    async def fail(task: AudioSeparation, detail: str) -> None:
        current.failures.append((task.task_id, detail))

    return current, SeparationInputReceiver(
        NodeId("node-0"),
        lambda: current.snapshot,
        current.copies,
        send,
        fail,
        clock=lambda: current.now,
    )


def packets(task: AudioSeparation) -> tuple[VisionMediaPacket, ...]:
    """Create the exact raw source response to the worker's request."""
    common = dict(
        source_node=task.owner_node,
        target_node=NodeId("node-0"),
        command_id=task.command_id,
        model=ModelId(task.task_params.model),
        payload="separation_source",
        total_chunks=2,
    )
    return (
        VisionMediaPacket.model_validate(
            common | {"kind": "opened", "sequence": 0, "image_count": 1}
        ),
        VisionMediaPacket.model_validate(
            common
            | {
                "kind": "chunk",
                "sequence": 1,
                "image_index": 0,
                "data": original()[:8000],
            }
        ),
        VisionMediaPacket.model_validate(
            common
            | {
                "kind": "chunk",
                "sequence": 2,
                "image_index": 0,
                "data": original()[8000:],
            }
        ),
        VisionMediaPacket.model_validate(
            common
            | {
                "kind": "completed",
                "sequence": 3,
                "image_count": 1,
                "sha256": task.task_params.source.sha256,
            }
        ),
    )


async def transfer(receiver: SeparationInputReceiver, task: AudioSeparation) -> None:
    """Deliver valid bytes with completion and later chunk reordered ahead of open."""
    opened, first, second, completed = packets(task)
    for packet in (completed, second, opened, opened, first):
        await receiver.receive(packet)


async def test_actual_private_source_owner_composes_with_reordered_handshake(
    tmp_path: Path,
) -> None:
    """The real private WAV owner, not a fake path, unlocks acknowledged input."""
    current, _ = fixture()
    copies = SeparationSourceStore(tmp_path / "inputs")

    async def send(packet: VisionMediaPacket) -> None:
        current.sent.append(packet)

    async def fail(task: AudioSeparation, detail: str) -> None:
        current.failures.append((task.task_id, detail))

    receiver = SeparationInputReceiver(
        NodeId("node-0"), lambda: current.snapshot, copies, send, fail
    )
    try:
        assert await receiver.begin(current.task.command_id)
        await transfer(receiver, current.task)
        retained = copies.ready_path(current.task.command_id)
        assert retained is not None and retained.name == "original.wav"
        assert retained.read_bytes() == original()
        assert retained.stat().st_mode & 0o777 == 0o600
        assert (
            receiver.ready(current.task.command_id)
            and current.sent[-1].kind == "accepted"
        )
        assert not current.failures
        terminal = current.task.model_copy(update={"task_status": TaskStatus.Cancelled})
        current.snapshot = current.snapshot.model_copy(
            update={"tasks": {terminal.task_id: terminal}}
        )
        assert not receiver.ready(current.task.command_id)
        assert retained.is_file()  # Terminal State does not prove a reader drained.
        await receiver.release_after_drain(current.task.command_id)
        assert not retained.exists()
    finally:
        await receiver.close_after_drain()
        copies.close()


async def test_worker_requests_only_its_actual_task_then_acknowledges_verified_source() -> (
    None
):
    """No packet can create admission; planning opens only after complete verification."""
    current, receiver = fixture()
    assert not receiver.ready(current.task.command_id)
    assert not await receiver.receive(packets(current.task)[0])
    assert current.copies.tasks == {} and current.sent == []
    assert await receiver.begin(current.task.command_id)
    assert current.sent[0].kind == "requested" and current.sent[0].is_terminal
    assert current.sent[0].source_node == NodeId("node-0")
    assert current.sent[0].target_node == current.task.owner_node
    assert (
        decode_vision_media_packet(encode_vision_media_packet(current.sent[0]))
        == current.sent[0]
    )
    assert await receiver.begin(current.task.command_id)
    assert len(current.sent) == 1
    await transfer(receiver, current.task)
    assert receiver.ready(current.task.command_id)
    assert current.sent[-1].kind == "accepted" and not current.failures
    assert set(current.copies.tasks) == {current.task.command_id}
    # Late cancellation must follow core task cancellation/drain; a raw frame
    # cannot remove the original while the resident might already be reading.
    cancelled = current.sent[-1].model_copy(
        update={
            "source_node": current.task.owner_node,
            "target_node": NodeId("node-0"),
            "kind": "cancelled",
        }
    )
    assert await receiver.receive(cancelled)
    assert not current.copies.deleted
    await receiver.release_after_drain(current.task.command_id)
    assert not receiver.ready(current.task.command_id)
    assert current.copies.deleted == [current.task.command_id]
    assert not await receiver.begin(current.task.command_id)


@pytest.mark.parametrize("change", ["source", "target", "model", "command"])
async def test_spoofed_source_cannot_write_acknowledge_or_cancel_another_task(
    change: str,
) -> None:
    """Caller-owned routing metadata never replaces authoritative task ownership."""
    current, receiver = fixture()
    await receiver.begin(current.task.command_id)
    packet = packets(current.task)[1]
    choices: dict[str, dict[str, object]] = {
        "source": {"source_node": NodeId("other")},
        "target": {"target_node": NodeId("other")},
        "model": {"model": ModelId("other/model")},
        "command": {"command_id": CommandId("other-command")},
    }
    updates = choices[change]
    assert not await receiver.receive(packet.model_copy(update=updates))
    assert current.copies.chunks[current.task.command_id] == {}
    assert not current.failures and not receiver.ready(current.task.command_id)
    await receiver.close_after_drain()


async def test_requests_retry_until_open_and_unfinished_transfer_has_finite_deadline() -> (
    None
):
    """Missing task/source cross-plane delivery cannot leave a pending job forever."""
    current, receiver = fixture()
    await receiver.begin(current.task.command_id)
    current.now = 1
    await receiver.tick()
    assert len(current.sent) == 1
    current.now = 2
    await receiver.tick()
    assert len(current.sent) == 2
    await receiver.receive(packets(current.task)[0])
    current.now = 4
    await receiver.tick()
    assert len(current.sent) == 2
    current.now = 180
    await receiver.tick()
    await receiver.tick()
    assert len(current.failures) == 1 and "deadline" in current.failures[0][1]
    assert not receiver.ready(current.task.command_id)
    await receiver.close_after_drain()


async def test_unavailable_transport_cannot_publish_input_readiness() -> None:
    """A missing data plane fails actionably while private cleanup remains owned."""
    current, receiver = fixture()
    current.unavailable = True
    assert not await receiver.begin(current.task.command_id)
    assert len(current.failures) == 1 and "transport" in current.failures[0][1]
    assert not receiver.ready(current.task.command_id)
    await receiver.close_after_drain()


async def test_failed_ack_keeps_verified_source_out_of_execution() -> None:
    """Verified bytes alone cannot open the worker's acknowledged-input gate."""
    current, receiver = fixture()
    await receiver.begin(current.task.command_id)
    frames = packets(current.task)
    for packet in frames[:-1]:
        await receiver.receive(packet)
    current.unavailable = True
    assert not await receiver.receive(frames[-1])
    assert len(current.failures) == 1 and not receiver.ready(current.task.command_id)
    assert not any(packet.kind == "accepted" for packet in current.sent)
    await receiver.close_after_drain()


async def test_four_copy_bound_refuses_more_authoritative_transfers() -> None:
    """Even an over-admitted task snapshot cannot grow native input without bound."""
    current, receiver = fixture()
    tasks = [
        current.task.model_copy(
            update={
                "task_id": TaskId(f"task-{number}"),
                "command_id": CommandId(f"source-{number}"),
            }
        )
        for number in range(5)
    ]
    current.snapshot = current.snapshot.model_copy(
        update={"tasks": {task.task_id: task for task in tasks}}
    )
    for task in tasks[:4]:
        assert await receiver.begin(task.command_id)
    assert not await receiver.begin(tasks[4].command_id)
    assert len(current.copies.tasks) == 4 and len(current.failures) == 1
    await receiver.close_after_drain()


@pytest.mark.parametrize("fault", ["digest", "slot", "chunks", "wav"])
async def test_declared_or_measured_source_mismatch_fails_without_ack(
    fault: str,
) -> None:
    """Both public metadata and independently measured WAV bytes bind the source."""
    current, receiver = fixture()
    await receiver.begin(current.task.command_id)
    frames = list(packets(current.task))
    if fault == "digest":
        frames[-1] = frames[-1].model_copy(update={"sha256": "a" * 64})
    elif fault == "slot":
        frames[1] = frames[1].model_copy(update={"image_index": 1})
    elif fault == "chunks":
        frames[0] = frames[0].model_copy(update={"total_chunks": 3})
    else:
        frames[2] = frames[2].model_copy(update={"data": original()[8000:-1] + b"x"})
    for packet in frames:
        await receiver.receive(packet)
    assert not receiver.ready(current.task.command_id)
    assert len(current.failures) == 1
    assert not any(packet.kind == "accepted" for packet in current.sent)
    await receiver.close_after_drain()


@pytest.mark.parametrize("race", ["cancel", "deadline"])
async def test_state_or_deadline_change_during_verification_prevents_ack(
    race: str,
) -> None:
    """An off-loop verification result cannot override current task cancellation."""
    current, receiver = fixture()
    await receiver.begin(current.task.command_id)
    entered, released = threading.Event(), threading.Event()

    def pause() -> None:
        entered.set()
        assert released.wait(2)

    current.copies.on_complete = pause
    frames = packets(current.task)
    for packet in frames[:-1]:
        await receiver.receive(packet)
    pending = asyncio.create_task(receiver.receive(frames[-1]))
    async with asyncio.timeout(2):
        while not entered.is_set():
            await asyncio.sleep(0.005)
    if race == "cancel":
        cancelled = current.task.model_copy(
            update={"task_status": TaskStatus.Cancelled}
        )
        current.snapshot = current.snapshot.model_copy(
            update={"tasks": {cancelled.task_id: cancelled}}
        )
    else:
        current.now = 181
    released.set()
    assert not await pending
    assert not any(packet.kind == "accepted" for packet in current.sent)
    await receiver.close_after_drain()


async def test_failed_close_retains_copy_ownership_and_can_retry() -> None:
    """No new transfer or request can appear while failed deletion still drains."""
    current, receiver = fixture()
    await receiver.begin(current.task.command_id)
    current.copies.fail_delete = True
    with pytest.raises(OSError, match="retained cleanup"):
        await receiver.close_after_drain()
    await receiver.tick()
    assert len(current.sent) == 1 and not await receiver.begin(current.task.command_id)
    assert current.task.command_id in current.copies.tasks
    await receiver.close_after_drain()
    assert current.copies.deleted == [current.task.command_id]


async def test_cancelled_receiver_joins_file_verification_before_cleanup() -> None:
    """No abandoned verification thread can recreate input after the owner closes."""
    current, receiver = fixture()
    await receiver.begin(current.task.command_id)
    entered, released = threading.Event(), threading.Event()

    def pause() -> None:
        entered.set()
        assert released.wait(2)

    current.copies.on_complete = pause
    frames = packets(current.task)
    for packet in frames[:-1]:
        await receiver.receive(packet)
    pending = asyncio.create_task(receiver.receive(frames[-1]))
    async with asyncio.timeout(2):
        while not entered.is_set():
            await asyncio.sleep(0.005)
    pending.cancel()
    await asyncio.sleep(0)
    pending.cancel()
    await asyncio.sleep(0)
    assert not pending.done() and not current.copies.deleted
    released.set()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert not receiver.ready(current.task.command_id)
    await receiver.close_after_drain()
    assert current.copies.deleted == [current.task.command_id]


@pytest.mark.parametrize(
    "fault", ["terminal", "remote", "partial", "rank", "duplicate"]
)
def test_input_task_lookup_requires_actual_complete_local_nonterminal_assignment(
    fault: str,
) -> None:
    """Neither a model alias nor a forged route grants access to a source copy."""
    current, _ = fixture()
    snapshot = current.snapshot
    if fault == "terminal":
        task = current.task.model_copy(update={"task_status": TaskStatus.Complete})
        snapshot = snapshot.model_copy(update={"tasks": {task.task_id: task}})
    elif fault == "remote":
        assert (
            local_separation_input_task(
                snapshot, NodeId("other"), current.task.command_id
            )
            is None
        )
        return
    elif fault == "duplicate":
        duplicate = current.task.model_copy(update={"task_id": TaskId("other")})
        snapshot = snapshot.model_copy(
            update={"tasks": {**snapshot.tasks, duplicate.task_id: duplicate}}
        )
    else:
        instance = snapshot.instances[InstanceId("instance-0")]
        assignments = instance.shard_assignments
        shard = assignments.runner_to_shard[RunnerId("runner-0")]
        changed = {"device_rank": 1} if fault == "rank" else {"end_layer": 4}
        shard = shard.model_copy(update=changed)
        instance = instance.model_copy(
            update={
                "shard_assignments": assignments.model_copy(
                    update={"runner_to_shard": {RunnerId("runner-0"): shard}}
                )
            }
        )
        snapshot = snapshot.model_copy(
            update={"instances": {instance.instance_id: instance}}
        )
    assert (
        local_separation_input_task(snapshot, NodeId("node-0"), current.task.command_id)
        is None
    )


@pytest.mark.parametrize("payload", ["base64_image", "reference_media"])
def test_existing_vision_payloads_cannot_acquire_new_source_request_semantics(
    payload: str,
) -> None:
    """The added request kind is scoped to the explicit native original role."""
    current, _ = fixture()
    with pytest.raises(ValueError, match="only separation source"):
        VisionMediaPacket.model_validate(
            {
                "source_node": "node-0",
                "target_node": str(current.task.owner_node),
                "model": current.task.task_params.model,
                "command_id": str(current.task.command_id),
                "payload": payload,
                "kind": "requested",
                "sequence": 0,
                "image_count": 1,
                "total_chunks": 2,
            }
        )
