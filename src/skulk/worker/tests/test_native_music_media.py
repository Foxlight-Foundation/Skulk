"""Worker-native task custody with actual private files and addressed input frames.

Synthetic WAVs and controlled completion replace inference only. These cases
exercise actual source storage, command journals, planner readiness and recovery.
"""

import asyncio
import multiprocessing
import threading
from pathlib import Path
from typing import Literal

import pytest

from skulk.api.music_edit_media import MusicEditMediaStore
from skulk.api.video_store import VideoStore
from skulk.master.tests.test_music_edit_admission import state
from skulk.routing.output_media import OutputMediaPacket
from skulk.routing.vision_media import VisionMediaPacket
from skulk.shared.models.native_music_runtime import resolve_native_music_policy
from skulk.shared.models.tests.test_native_music_runtime import probe
from skulk.shared.tests.music_edit_fixtures import music_edit_fixture
from skulk.shared.types.common import CommandId, ModelId, NodeId
from skulk.shared.types.diagnostics import RunnerDiagnosticUpdate
from skulk.shared.types.events import Event
from skulk.shared.types.music import MusicGenerationTaskParams, MusicOutputManifest
from skulk.shared.types.state import State
from skulk.shared.types.tasks import (
    MusicEditing,
    MusicGeneration,
    Task,
    TaskId,
    TaskStatus,
)
from skulk.shared.types.worker.instances import BoundInstance
from skulk.shared.types.worker.runners import RunnerReady
from skulk.utils.channels import channel, mp_channel
from skulk.utils.runtime_files import write_private
from skulk.worker.native_music_media import (
    NativeMusicMediaOwner,
    local_native_music_task,
)
from skulk.worker.native_music_output import NativeMusicTask
from skulk.worker.plan import plan
from skulk.worker.runner.ace_step.result_custody import (
    music_result_directory,
    prepare_result_directory,
)
from skulk.worker.runner.ace_step.tests.test_result_custody import parents
from skulk.worker.runner.runner_supervisor import RunnerSupervisor
from skulk.worker.separation_custody import separation_reader_key
from skulk.worker.separation_sources import SeparationSourceStore


def fixture(*, editing: bool) -> tuple[State, NativeMusicTask, bytes]:
    """Stamp a complete pure native policy on the ordinary model/task State."""
    snapshot, _ = state()
    instance = next(iter(snapshot.instances.values()))
    assignments = instance.shard_assignments
    runner = next(iter(assignments.runner_to_shard))
    shard = assignments.runner_to_shard[runner]
    policy = resolve_native_music_policy(
        shard.model_card,
        "ace_step-cpu",
        probe().model_copy(update={"payload_sha256": "d" * 64}),
    )
    shard = shard.model_copy(update={"native_music_policy": policy})
    instance = instance.model_copy(
        update={
            "shard_assignments": assignments.model_copy(
                update={"runner_to_shard": {runner: shard}}
            )
        }
    )
    job, _, source, _ = music_edit_fixture()
    task: NativeMusicTask
    if editing:
        task = MusicEditing(
            command_id=job.id,
            owner_node=NodeId("api"),
            instance_id=instance.instance_id,
            task_params=job.params,
        )
    else:
        task = MusicGeneration(
            command_id=CommandId("generate"),
            owner_node=NodeId("api"),
            instance_id=instance.instance_id,
            task_params=MusicGenerationTaskParams(
                model=job.params.model, prompt="A warm folk song", seconds=1, seed=0
            ),
        )
    return (
        snapshot.model_copy(
            update={
                "instances": {instance.instance_id: instance},
                "tasks": {task.task_id: task},
            }
        ),
        task,
        source,
    )


async def test_generation_is_not_ready_until_durable_intent_finishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A planning tick during joined disk work cannot outrun writer admission."""
    snapshot, task, _ = fixture(editing=False)
    entered, release = threading.Event(), threading.Event()
    original = SeparationSourceStore[MusicEditing].reserve_result

    def reserve(
        store: SeparationSourceStore[MusicEditing], command: CommandId, key: str
    ) -> None:
        entered.set()
        assert release.wait(5)
        original(store, command, key)

    monkeypatch.setattr(SeparationSourceStore, "reserve_result", reserve)

    async def source(_packet: VisionMediaPacket) -> None:
        raise AssertionError("generation must not fabricate an uploaded original")

    async def output(_packet: OutputMediaPacket) -> None:
        raise AssertionError("no native completion has been supplied")

    async def fail(_task: NativeMusicTask, _message: str) -> None:
        raise AssertionError("valid controlled custody must succeed")

    owner = NativeMusicMediaOwner(
        NodeId("node-0"),
        lambda: snapshot,
        tmp_path / "source",
        tmp_path / "readers",
        source,
        output,
        fail,
        lambda _task, _runner, _root: True,
    )
    work = asyncio.create_task(owner.begin(task.command_id))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        assert task.command_id not in owner.ready_commands()
        release.set()
        await work
        assert task.command_id in owner.ready_commands()
        assert not (tmp_path / "source" / str(task.command_id)).exists()
        snapshot = snapshot.model_copy(
            update={
                "tasks": {
                    task.task_id: task.model_copy(
                        update={"task_status": TaskStatus.Cancelled}
                    )
                }
            }
        )
        await owner.reconcile()
        assert not owner.owns(task.command_id)
    finally:
        release.set()
        await work
        await owner.detach()


async def test_edit_requires_verified_original_and_releases_after_native_drain(
    tmp_path: Path,
) -> None:
    """Actual requested/open/chunk/completed frames gate the native edit planner."""
    snapshot, task, original = fixture(editing=True)
    requested: list[VisionMediaPacket] = []
    failures: list[str] = []
    drained = False

    async def source(packet: VisionMediaPacket) -> None:
        requested.append(packet)

    async def output(_packet: OutputMediaPacket) -> None:
        raise AssertionError("no output has been supplied")

    async def fail(_task: NativeMusicTask, message: str) -> None:
        failures.append(message)

    owner = NativeMusicMediaOwner(
        NodeId("node-0"),
        lambda: snapshot,
        tmp_path / "source",
        tmp_path / "readers",
        source,
        output,
        fail,
        lambda _task, _runner, _root: drained,
    )
    try:
        await owner.begin(task.command_id)
        assert (
            requested[0].kind == "requested"
            and requested[0].payload == "music_edit_source"
        )
        assert task.command_id not in owner.ready_commands()
        assert isinstance(task, MusicEditing)
        for packet in (
            VisionMediaPacket(
                source_node=task.owner_node,
                target_node=NodeId("node-0"),
                command_id=task.command_id,
                model=ModelId(task.task_params.model),
                payload="music_edit_source",
                kind="opened",
                sequence=0,
                total_chunks=1,
                image_count=1,
            ),
            VisionMediaPacket(
                source_node=task.owner_node,
                target_node=NodeId("node-0"),
                command_id=task.command_id,
                model=ModelId(task.task_params.model),
                payload="music_edit_source",
                kind="chunk",
                image_index=0,
                sequence=1,
                total_chunks=1,
                data=original,
            ),
            VisionMediaPacket(
                source_node=task.owner_node,
                target_node=NodeId("node-0"),
                command_id=task.command_id,
                model=ModelId(task.task_params.model),
                payload="music_edit_source",
                kind="completed",
                sequence=2,
                total_chunks=1,
                image_count=1,
                sha256=task.task_params.source.sha256,
            ),
        ):
            await owner.receive_input(packet)
        assert not failures and task.command_id in owner.ready_commands()
        copied = tmp_path / "source" / str(task.command_id) / "original.wav"
        assert copied.read_bytes() == original
        snapshot = snapshot.model_copy(
            update={
                "tasks": {
                    task.task_id: task.model_copy(
                        update={"task_status": TaskStatus.Cancelled}
                    )
                }
            }
        )
        await owner.reconcile()
        assert copied.exists() and owner.owns(task.command_id)
        drained = True
        await owner.reconcile()
        assert not copied.exists() and not owner.owns(task.command_id)
    finally:
        await owner.detach()


async def test_replacement_worker_retains_generation_result_until_inherited_fence_drains(
    tmp_path: Path,
) -> None:
    """No-original generations still have a durable writer intent after a crash."""
    snapshot, task, _ = fixture(editing=False)
    failures: list[str] = []

    async def source(_packet: VisionMediaPacket) -> None:
        raise AssertionError("generation has no original")

    async def output(_packet: OutputMediaPacket) -> None:
        raise AssertionError("no completion was supplied")

    async def fail(_task: NativeMusicTask, message: str) -> None:
        failures.append(message)

    def create() -> NativeMusicMediaOwner:
        return NativeMusicMediaOwner(
            NodeId("node-0"),
            lambda: snapshot,
            tmp_path / "source",
            tmp_path / "readers",
            source,
            output,
            fail,
            lambda _task, _runner, _root: False,
        )

    owner = create()
    instance = next(iter(snapshot.instances.values()))
    runner = next(iter(instance.shard_assignments.runner_to_shard))
    root = tmp_path / "readers" / separation_reader_key(task.instance_id, runner)
    fence = await owner.prepare_reader(root)
    await owner.begin(task.command_id)
    directory = music_result_directory(root, task.command_id)
    prepare_result_directory(directory, parents(root))
    write_private(directory / "result.wav", b"partial result")
    await owner.detach()
    replacement = create()
    try:
        await replacement.fail_recovered()
        assert failures and task.command_id not in replacement.ready_commands()
        other = await replacement.prepare_reader(tmp_path / "readers" / ("f" * 64))
        other.close()
        assert directory.exists()
        fence.close()
        assert replacement.sources is not None
        assert await asyncio.to_thread(replacement.sources.recover_stale) == 1
        assert not directory.exists()
    finally:
        fence.close()
        await replacement.detach()


def test_foreign_or_unstamped_native_assignment_cannot_admit_media() -> None:
    """The original transport cannot turn incomplete placement into a writer."""
    snapshot, task, _ = fixture(editing=True)
    assert local_native_music_task(snapshot, NodeId("foreign"), task.command_id) is None
    instance = next(iter(snapshot.instances.values()))
    runner = next(iter(instance.shard_assignments.runner_to_shard))
    shard = instance.shard_assignments.runner_to_shard[runner].model_copy(
        update={"native_music_policy": None}
    )
    changed = instance.model_copy(
        update={
            "shard_assignments": instance.shard_assignments.model_copy(
                update={"runner_to_shard": {runner: shard}}
            )
        }
    )
    snapshot = snapshot.model_copy(
        update={"instances": {instance.instance_id: changed}}
    )
    assert local_native_music_task(snapshot, NodeId("node-0"), task.command_id) is None


@pytest.mark.parametrize("editing", [False, True])
async def test_actual_planner_waits_for_native_media_admission(
    *, editing: bool
) -> None:
    """Ready native models still cannot dispatch an unjournaled generation or edit."""
    snapshot, task, _ = fixture(editing=editing)
    instance = snapshot.instances[task.instance_id]
    runner_id = instance.shard_assignments.node_to_runner[NodeId("node-0")]
    bound = BoundInstance(
        instance=instance, bound_node_id=NodeId("node-0"), bound_runner_id=runner_id
    )
    process = multiprocessing.Process()
    event_send, event_recv = channel[Event]()
    child_events, events = mp_channel[Event]()
    child_diagnostics, diagnostics = mp_channel[RunnerDiagnosticUpdate]()
    tasks, child_tasks = mp_channel[Task]()
    cancellations, child_cancellations = mp_channel[TaskId]()
    try:
        supervisor = RunnerSupervisor(
            shard_metadata=bound.bound_shard,
            bound_instance=bound,
            runner_process=process,
            initialize_timeout=10,
            _ev_recv=events,
            _diag_recv=diagnostics,
            _task_sender=tasks,
            _event_sender=event_send,
            _cancel_sender=cancellations,
        )
        supervisor.status = RunnerReady()
        assert (
            plan(
                NodeId("node-0"),
                {runner_id: supervisor},
                {},
                snapshot.instances,
                snapshot.runners,
                snapshot.tasks,
            )
            is None
        )
        assert (
            plan(
                NodeId("node-0"),
                {runner_id: supervisor},
                {},
                snapshot.instances,
                snapshot.runners,
                snapshot.tasks,
                native_music_media_ready=frozenset({task.command_id}),
            )
            == task
        )
    finally:
        for endpoint in (
            event_send,
            event_recv,
            child_events,
            events,
            child_diagnostics,
            diagnostics,
            tasks,
            child_tasks,
            cancellations,
            child_cancellations,
        ):
            endpoint.close()
        process.close()


@pytest.mark.parametrize("editing", [False, True])
@pytest.mark.parametrize("terminal", ["removed", "complete", "cancelled", "failed"])
async def test_terminal_handoff_retains_verified_media_until_its_own_ack(
    tmp_path: Path,
    *,
    editing: bool,
    terminal: Literal["removed", "complete", "cancelled", "failed"],
) -> None:
    """Real files/producer/API custody survive normal task removal, not cancellation."""
    snapshot, task, original = fixture(editing=editing)
    _, output, _, data = music_edit_fixture(str(task.command_id))
    facts = output.result.output
    manifest = MusicOutputManifest(
        size_bytes=facts.size_bytes,
        sha256=facts.sha256,
        sample_rate=facts.sample_rate,
        channels=facts.channels,
        duration_seconds=facts.frames / facts.sample_rate,
    )
    sent: list[OutputMediaPacket] = []
    failure: list[str] = []
    retained: list[bytes] = []
    store = (
        MusicEditMediaStore(tmp_path / "api")
        if editing
        else VideoStore(tmp_path / "api")
    )

    async def source(_packet: VisionMediaPacket) -> None:
        pass

    async def fail(_task: NativeMusicTask, message: str) -> None:
        failure.append(message)

    async def send(packet: OutputMediaPacket) -> None:
        sent.append(packet)
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
                stored = store.commit(
                    packet.command_id, facts, total_chunks=packet.total_chunks
                )
            else:
                stored = store.commit(
                    packet.command_id,
                    "music",
                    sha256=packet.sha256,
                    total_chunks=packet.total_chunks,
                )
            retained.append(stored.file_path.read_bytes())
            assert owner.outputs.receive(packet.accepted())

    owner = NativeMusicMediaOwner(
        NodeId("node-0"),
        lambda: snapshot,
        tmp_path / "source",
        tmp_path / "readers",
        source,
        send,
        fail,
        lambda _task, _runner, _root: True,
    )
    instance = snapshot.instances[task.instance_id]
    runner = instance.shard_assignments.node_to_runner[NodeId("node-0")]
    root = tmp_path / "readers" / separation_reader_key(task.instance_id, runner)
    fence = await owner.prepare_reader(root)
    try:
        await owner.begin(task.command_id)
        if isinstance(task, MusicEditing):
            for packet in (
                VisionMediaPacket(
                    source_node=task.owner_node,
                    target_node=NodeId("node-0"),
                    command_id=task.command_id,
                    model=ModelId(task.task_params.model),
                    payload="music_edit_source",
                    kind="opened",
                    sequence=0,
                    total_chunks=1,
                    image_count=1,
                ),
                VisionMediaPacket(
                    source_node=task.owner_node,
                    target_node=NodeId("node-0"),
                    command_id=task.command_id,
                    model=ModelId(task.task_params.model),
                    payload="music_edit_source",
                    kind="chunk",
                    image_index=0,
                    sequence=1,
                    total_chunks=1,
                    data=original,
                ),
                VisionMediaPacket(
                    source_node=task.owner_node,
                    target_node=NodeId("node-0"),
                    command_id=task.command_id,
                    model=ModelId(task.task_params.model),
                    payload="music_edit_source",
                    kind="completed",
                    sequence=2,
                    total_chunks=1,
                    image_count=1,
                    sha256=task.task_params.source.sha256,
                ),
            ):
                await owner.receive_input(packet)
        assert task.command_id in owner.ready_commands()
        directory = music_result_directory(root, task.command_id)
        prepare_result_directory(directory, parents(root))
        write_private(directory / "result.wav", data)
        assert owner.register_output(
            task.command_id, task.owner_node, str(task.task_params.model), manifest
        )
        if terminal == "removed":
            snapshot = snapshot.model_copy(update={"tasks": {}})
        else:
            snapshot = snapshot.model_copy(
                update={
                    "tasks": {
                        task.task_id: task.model_copy(
                            update={
                                "task_status": {
                                    "complete": TaskStatus.Complete,
                                    "cancelled": TaskStatus.Cancelled,
                                    "failed": TaskStatus.Failed,
                                }[terminal]
                            }
                        )
                    }
                }
            )
        # Terminal DATA cleanup can remove execution before the independent sender's
        # first scheduling/verification step. No timing race or native model is needed.
        await owner.reconcile()
        assert directory.exists() and owner.owns(task.command_id)
        await owner.outputs.deliver(task.command_id)
        if terminal in ("removed", "complete"):
            assert retained == [data] and not failure
        else:
            assert failure and not retained and not sent
        assert not owner.outputs.pending(task.command_id) and not directory.exists()
        await owner.reconcile()
        assert not owner.owns(task.command_id)
    finally:
        fence.close()
        await owner.detach()
        if isinstance(store, MusicEditMediaStore):
            store.close()
