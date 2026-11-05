# pyright: reportPrivateUsage=false
"""Worker-owned native observations and preparation; library effects are injected."""

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock

import anyio
import pytest

from skulk.provisioning.demucs_inventory import DEMUCS_PACKAGES
from skulk.routing.output_media import OutputMediaPacket
from skulk.routing.vision_media import VisionMediaPacket
from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.events import (
    DemucsPreparationCompleted,
    DemucsPreparationRequested,
    Event,
    IndexedEvent,
    StateSnapshotHydrated,
    TaskFailed,
)
from skulk.shared.types.profiling import NodeResources
from skulk.shared.types.tasks import TaskStatus
from skulk.shared.types.telemetry import NodeTelemetry
from skulk.shared.types.worker.instances import BoundInstance
from skulk.shared.types.worker.runners import RunnerReady
from skulk.utils.channels import Receiver, channel
from skulk.utils.info_gatherer.info_gatherer import GatheredInfo, InfoGatherer
from skulk.utils.runtime_files import RuntimeLock, write_private
from skulk.worker import main as worker_module
from skulk.worker import plan as worker_plan
from skulk.worker.demucs_preparation import DemucsPreparationController
from skulk.worker.main import Worker
from skulk.worker.runner.demucs.model import separation_result_directory
from skulk.worker.runner.demucs.result_custody import remember_result_directory
from skulk.worker.runner.runner_supervisor import RunnerSupervisor
from skulk.worker.tests.test_demucs_preparation import Manager
from skulk.worker.tests.test_separation_input import fixture as input_fixture
from skulk.worker.tests.test_separation_input import packets


async def fixture(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    offline: bool = False,
) -> tuple[Worker, Manager, Receiver[Event], Receiver[NodeTelemetry]]:
    """Construct the actual worker without launching its network/planning loops."""
    _, indexed = channel[IndexedEvent]()
    events, output = channel[Event]()
    commands, _ = channel[ForwarderCommand]()
    downloads, _ = channel[ForwarderDownloadCommand]()
    telemetry, observations = channel[NodeTelemetry]()
    worker = Worker(
        NodeId("worker"),
        event_receiver=indexed,
        event_sender=events,
        command_sender=commands,
        download_command_sender=downloads,
        telemetry_sender=telemetry,
        offline=offline,
    )
    worker._demucs_platforms = frozenset({"macosx_14_0_arm64"})
    # This fixture isolates separator inventory; native ACE has its own actual
    # Worker/cache/resource bridge tests and no prepared library in this one.
    worker._ace_step_platforms = frozenset()
    manager = Manager(root)
    manager.release.set()
    worker._demucs_preparation = DemucsPreparationController(manager, offline=offline)
    resources = NodeResources(
        backends=frozenset({"audio_cpp-metal", "demucs-cuda"}),
        on_demand_backends=frozenset({"demucs-cuda"}),
        engine_builds={
            "audio_cpp-metal": "existing-music-build",
            "demucs-cuda": "unverified",
        },
        installable_engine_builds={"demucs-cuda": "unimplemented-profile"},
    )
    monkeypatch.setattr(NodeResources, "gather", AsyncMock(return_value=resources))
    return worker, manager, output, observations


async def test_inventory_read_performs_no_native_effect_and_never_advertises_candidate_as_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same owner preserves other engines and publishes only actual verified lanes."""
    worker, manager, _, observations = await fixture(tmp_path, monkeypatch)
    try:
        resources = await worker._gather_node_resources()
        assert resources.backends == frozenset({"audio_cpp-metal"})
        assert resources.engine_builds == {"audio_cpp-metal": "existing-music-build"}
        assert not resources.on_demand_backends
        assert resources.installable_engine_builds == {
            "demucs-cpu": DEMUCS_PACKAGES[0].build_identity,
            "demucs-metal": DEMUCS_PACKAGES[0].build_identity,
        }
        assert not manager.calls and manager.closed == 0
        manager.builds = {"demucs-cpu": DEMUCS_PACKAGES[0].build_identity}
        observed = await worker._gather_node_resources()
        assert observed.backends == frozenset({"audio_cpp-metal", "demucs-cpu"})
        assert observed.engine_builds["demucs-cpu"] == DEMUCS_PACKAGES[0].build_identity
        await worker._publish_fresh_node_resources()
        assert (await observations.receive()).info == observed
        manager.builds.clear()
        assert (await worker._gather_node_resources()).backends == resources.backends
    finally:
        await worker._demucs_preparation.close()


@pytest.mark.parametrize("offline", [False, True])
async def test_explicit_preparation_orders_verified_snapshot_and_preserves_offline_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, offline: bool
) -> None:
    """The worker cannot signal success using package availability or stale facts."""
    worker, manager, output, observations = await fixture(
        tmp_path, monkeypatch, offline=offline
    )
    try:
        request = DemucsPreparationRequested(
            request_id=CommandId("prepare"),
            target_node=worker.node_id,
            owner_node=NodeId("api"),
            package_sha256=DEMUCS_PACKAGES[0].sha256,
            device="cpu",
            expires_at=time.time() + 30,
        )
        await worker._prepare_demucs_engine(request)
        completed = await output.receive()
        assert isinstance(completed, DemucsPreparationCompleted) and completed.success
        assert (
            completed.resources is not None
            and completed.resources.engine_builds["demucs-cpu"]
            == DEMUCS_PACKAGES[0].build_identity
        )
        assert (await observations.receive()).info == completed.resources
        assert manager.calls[0][0:2] == ("cpu", offline)
        assert 0 < manager.calls[0][2] <= 30
    finally:
        await worker._demucs_preparation.close()


async def test_unknown_package_has_no_install_effect_or_success_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A preparation request cannot substitute another artifact or harm other engines."""
    worker, manager, output, _ = await fixture(tmp_path, monkeypatch)
    try:
        request = DemucsPreparationRequested(
            request_id=CommandId("wrong-package"),
            target_node=worker.node_id,
            owner_node=NodeId("api"),
            package_sha256="0" * 64,
            device="cpu",
            expires_at=time.time() + 30,
        )
        await worker._prepare_demucs_engine(request)
        completed = await output.receive()
        assert (
            isinstance(completed, DemucsPreparationCompleted) and not completed.success
        )
        assert (
            completed.resources is None
            and completed.error is not None
            and not manager.calls
        )
        assert (await worker._gather_node_resources()).engine_builds == {
            "audio_cpp-metal": "existing-music-build"
        }
    finally:
        await worker._demucs_preparation.close()


async def test_periodic_resources_use_same_worker_owner_for_readiness_and_withdrawal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A periodic gather cannot overwrite freshly verified native readiness with old facts."""
    worker, manager, _, _ = await fixture(tmp_path, monkeypatch)
    send, receive = channel[GatheredInfo]()
    gatherer = InfoGatherer(
        send,
        node_resources_provider=worker._gather_node_resources,
        node_resources_poll_interval=0.01,
    )
    try:
        with anyio.fail_after(3):
            async with anyio.create_task_group() as group:
                group.start_soon(gatherer._monitor_node_resources)
                first = await receive.receive()
                assert (
                    isinstance(first, NodeResources)
                    and "demucs-cpu" not in first.backends
                )
                manager.builds = {"demucs-cpu": DEMUCS_PACKAGES[0].build_identity}
                second = await receive.receive()
                assert (
                    isinstance(second, NodeResources)
                    and second.engine_builds["demucs-cpu"]
                    == DEMUCS_PACKAGES[0].build_identity
                )
                manager.builds.clear()
                third = await receive.receive()
                assert (
                    isinstance(third, NodeResources)
                    and "demucs-cpu" not in third.backends
                )
                group.cancel_scope.cancel()
    finally:
        await worker._demucs_preparation.close()


@pytest.mark.parametrize("foreign_result", [False, True])
async def test_actual_worker_source_handshake_gates_planning_and_retains_live_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    foreign_result: bool,
) -> None:
    """Real Worker ingress/custody cannot execute unverified or reclaim live input."""
    worker, _, _, _ = await fixture(tmp_path, monkeypatch)
    current, _ = input_fixture()
    worker.node_id = NodeId("node-0")
    worker.state = current.snapshot
    source_send, source_receive = channel[VisionMediaPacket]()
    into_worker, worker_packets = channel[VisionMediaPacket]()
    output_send, _ = channel[OutputMediaPacket]()
    worker._vision_media_packet_sender = source_send
    worker._vision_media_packet_receiver = worker_packets
    worker._output_media_packet_sender = output_send
    monkeypatch.setattr(
        worker_module, "SKULK_SEPARATION_INPUT_DIR", tmp_path / "inputs"
    )
    monkeypatch.setattr(
        worker_module, "SKULK_SEPARATION_OUTPUT_DIR", tmp_path / "readers"
    )
    instance = worker.state.instances[current.task.instance_id]
    runner_id = instance.shard_assignments.node_to_runner[worker.node_id]
    supervisor = RunnerSupervisor.create(
        bound_instance=BoundInstance(
            instance=instance, bound_node_id=worker.node_id, bound_runner_id=runner_id
        ),
        event_sender=worker.event_sender.clone(),
    )
    supervisor.status = RunnerReady()
    worker.runners[runner_id] = supervisor
    reader: RuntimeLock | None = None
    try:
        await worker._begin_separation_input(current.task.command_id)
        assert (await source_receive.receive()).kind == "requested"
        receiver = worker._separation_inputs
        assert receiver is not None and receiver.ready_commands() == frozenset()
        assert (
            worker_plan._pending_tasks(
                worker.runners,
                worker.state.tasks,
                worker.state.runners,
                None,
                frozenset(),
                frozenset(),
                receiver.ready_commands(),
            )
            is None
        )
        with anyio.fail_after(3):
            async with anyio.create_task_group() as group:
                group.start_soon(worker._vision_media_packet_ingress)
                for packet in reversed(packets(current.task)):
                    await into_worker.send(packet)
                assert (await source_receive.receive()).kind == "accepted"
                group.cancel_scope.cancel()
        assert (
            worker_plan._pending_tasks(
                worker.runners,
                worker.state.tasks,
                worker.state.runners,
                None,
                frozenset(),
                frozenset(),
                receiver.ready_commands(),
            )
            == current.task
        )
        sources = worker._separation_sources
        assert sources is not None
        original = sources.ready_path(current.task.command_id)
        assert original is not None and original.exists()
        # Use a real kernel reader fence after losing the local parent mapping;
        # terminal replicated State still does not authorize deleting its WAV.
        root = worker._separation_readers[current.task.command_id][1]
        reader = RuntimeLock(root, name="attachment.lock")
        # Simulate a native crash after output creation but before any completion
        # reaches the producer. The durable source intent is the only owner left.
        result = separation_result_directory(root, current.task.command_id)
        result.parent.mkdir(mode=0o700, exist_ok=True)
        result.mkdir(mode=0o700)
        remember_result_directory(result)
        write_private(result / "drums.wav", b"partial native output")
        if foreign_result:
            write_private(result / "keep", b"unknown file")
        worker.runners.clear()
        cancelled = current.task.model_copy(
            update={"task_status": TaskStatus.Cancelled}
        )
        worker.state = worker.state.model_copy(
            update={"tasks": {cancelled.task_id: cancelled}}
        )
        await worker._reconcile_separation_media()
        assert original.exists() and result.exists() and len(receiver.tracked_tasks()) == 1
        reader.close()
        if foreign_result:
            with pytest.raises(OSError, match="owned cleanup"):
                await worker._reconcile_separation_media()
            assert original.exists() and result.exists()
            assert len(receiver.tracked_tasks()) == 1
            (result / "keep").unlink()
        await worker._reconcile_separation_media()
        assert not original.exists() and not result.exists()
        assert not result.with_name("." + result.name + ".owner.json").exists()
        assert receiver.tracked_tasks() == ()
        assert not worker._separation_readers
    finally:
        if reader is not None:
            reader.close()
        if worker._separation_sources is not None:
            worker._separation_sources.detach()
        supervisor._ev_recv.close()
        supervisor._diag_recv.close()
        supervisor._task_sender.close()
        supervisor._cancel_sender.close()
        assert supervisor._realtime_audio_sender is not None
        supervisor._realtime_audio_sender.close()
        supervisor.runner_process.close()
        await worker._demucs_preparation.close()


@pytest.mark.parametrize(("local", "known", "fails"), [
    (True, False, True), (False, False, False), (True, True, False),
])
async def test_actual_snapshot_bootstrap_fails_only_unrecovered_local_separations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    local: bool, known: bool, fails: bool,
) -> None:
    """Follower bootstrap cannot cancel a healthy remote task or replay a live owner."""
    worker, _, outgoing, _ = await fixture(tmp_path, monkeypatch)
    current, receiver = input_fixture()
    worker.node_id = NodeId("node-0" if local else "unrelated-follower")
    indexed, events = channel[IndexedEvent]()
    worker.event_receiver = events
    def preserve_config(*_arguments: object) -> None:
        """This event test never mutates the operator's model-trust configuration."""
    monkeypatch.setattr(worker_module, "persist_model_trust_config", preserve_config)
    if known:
        assert await receiver.begin(current.task.command_id)
        worker._separation_inputs = receiver
    try:
        async with worker._tg as group:
            group.start_soon(worker._event_applier)
            snapshot = current.snapshot.model_copy(update={"last_event_applied_idx": 0})
            await indexed.send(IndexedEvent(
                idx=0, event=StateSnapshotHydrated(state=snapshot)
            ))
            with anyio.fail_after(2):
                while current.task.task_id not in worker.state.tasks:
                    await anyio.sleep(0)
            if fails:
                with anyio.fail_after(2):
                    failure = await outgoing.receive()
                assert isinstance(failure, TaskFailed) and failure.task_id == current.task.task_id
            else:
                with anyio.move_on_after(0.1) as observation:
                    await outgoing.receive()
                assert observation.cancel_called
            group.cancel_scope.cancel()
    finally:
        indexed.close()
        if known:
            await receiver.close_after_drain()
        await worker._demucs_preparation.close()


async def test_restored_reader_reconciles_old_results_before_holding_new_lifetime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A replacement model cannot hold the very fence its stale cleanup needs."""
    from skulk.worker.runner.demucs.tests.test_result_custody import owned_result
    from skulk.worker.tests.test_separation_custody import KEY, store
    from skulk.worker.tests.test_separation_sources import retain, task

    previous = store(tmp_path)
    retain(previous, task("prior-native"))
    root = tmp_path / "readers" / KEY
    old_reader = RuntimeLock(root, name="attachment.lock")
    output = owned_result(tmp_path, "prior-native")
    (output / "drums.wav").write_bytes(b"partial native result")
    (output / "drums.wav").chmod(0o600)
    original = tmp_path / "inputs/prior-native/original.wav"
    previous.detach()
    worker, _, _, _ = await fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        worker_module, "SKULK_SEPARATION_INPUT_DIR", tmp_path / "inputs"
    )
    monkeypatch.setattr(
        worker_module, "SKULK_SEPARATION_OUTPUT_DIR", tmp_path / "readers"
    )
    preparing = asyncio.create_task(worker._prepare_separation_reader(root))
    try:
        await asyncio.sleep(0.1)
        assert not preparing.done() and original.exists() and output.exists()
        old_reader.close()
        new_reader = await asyncio.wait_for(preparing, 5)
        try:
            assert not original.exists() and not output.exists()
            assert worker._separation_sources is not None
            assert worker._separation_sources.unresolved_source_count() == 0
            with pytest.raises(BlockingIOError):
                RuntimeLock(root, name="attachment.lock")
        finally:
            new_reader.close()
    finally:
        old_reader.close()
        if not preparing.done():
            preparing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await preparing
        if worker._separation_sources is not None:
            worker._separation_sources.detach()
        await worker._demucs_preparation.close()
