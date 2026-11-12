# pyright: reportPrivateUsage=false
"""Election replacement must join a Worker stopped before task-group startup."""

import threading
from collections.abc import Callable

import anyio
import pytest
from anyio import to_thread
from anyio.lowlevel import checkpoint

from skulk import facts
from skulk.facts.derive import BackendDerivation
from skulk.routing.router import OutboundPacket, TopicRouter
from skulk.routing.topics import VISION_MEDIA
from skulk.routing.vision_media import AudioSourcePayload, VisionMediaPacket
from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.common import CommandId, ModelId, NodeId
from skulk.shared.types.events import Event, IndexedEvent
from skulk.shared.types.telemetry import NodeTelemetry
from skulk.utils.channels import channel
from skulk.worker import main as worker_module
from skulk.worker.main import Worker


def worker_fixture(monkeypatch: pytest.MonkeyPatch) -> tuple[Worker, list[str]]:
    """Keep real ownership/cleanup while replacing hardware startup effects."""
    _, inbound = channel[IndexedEvent]()
    outbound, _ = channel[Event]()
    commands, _ = channel[ForwarderCommand]()
    downloads, _ = channel[ForwarderDownloadCommand]()
    telemetry, _ = channel[NodeTelemetry]()
    worker = Worker(
        NodeId("startup-worker"),
        event_receiver=inbound,
        event_sender=outbound,
        command_sender=commands,
        download_command_sender=downloads,
        telemetry_sender=telemetry,
    )
    effects: list[str] = []

    def effect(name: str) -> Callable[[], None]:
        """Record a startup boundary without touching native processes/files."""
        return lambda: effects.append(name)

    monkeypatch.setattr(worker, "_reconcile_staging_on_startup", effect("staging"))
    monkeypatch.setattr(worker_module, "sweep_orphaned_vllm_engines", effect("vllm"))
    monkeypatch.setattr(worker_module, "sweep_orphaned_comfy_servers", effect("comfy"))
    monkeypatch.setattr(
        worker_module, "_purge_stale_video_directories", effect("video")
    )
    monkeypatch.setattr(
        facts,
        "current_backend_derivation",
        lambda: BackendDerivation(backends=frozenset()),
    )
    return worker, effects


async def test_shutdown_before_scheduled_run_joins_cleanup_without_starting_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An election can retire a queued Worker before its coroutine first runs."""
    worker, effects = worker_fixture(monkeypatch)
    with anyio.fail_after(5):
        async with anyio.create_task_group() as group:
            group.start_soon(worker.shutdown)
            await checkpoint()
            group.start_soon(worker.run)
    assert worker._stopped.is_set()
    assert effects == []
    await worker.shutdown()


async def test_shutdown_during_offloop_bootstrap_joins_before_returning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retirement drains a started facts read and prevents late startup effects."""
    worker, effects = worker_fixture(monkeypatch)
    entered = threading.Event()
    release = threading.Event()
    returned = threading.Event()

    def blocked_derivation() -> BackendDerivation:
        """Hold the actual run's off-loop startup boundary across shutdown."""
        entered.set()
        assert release.wait(5)
        returned.set()
        return BackendDerivation(backends=frozenset())

    monkeypatch.setattr(facts, "current_backend_derivation", blocked_derivation)
    with anyio.fail_after(5):
        async with anyio.create_task_group() as group:
            group.start_soon(worker.run)
            assert await to_thread.run_sync(lambda: entered.wait(5))
            try:
                group.start_soon(worker.shutdown)
                await checkpoint()
                assert not worker._stopped.is_set()
            finally:
                release.set()
    assert returned.is_set()
    assert worker._stopped.is_set()
    assert effects == ["staging", "vllm", "comfy"]


async def test_active_shutdown_joins_children_and_native_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Returning from shutdown requires both child drain and native owner cleanup."""
    worker, _ = worker_fixture(monkeypatch)
    active = anyio.Event()
    child_drained = anyio.Event()
    cleanup_entered = anyio.Event()
    release_cleanup = anyio.Event()

    async def child() -> None:
        """Expose a real child lifetime inside the worker task group."""
        try:
            active.set()
            await anyio.sleep_forever()
        finally:
            child_drained.set()

    async def tasks() -> None:
        """Replace hardware admission with a supervised active task."""
        async with worker._tg as group:
            group.start_soon(child)
            await anyio.sleep_forever()

    async def cleanup() -> None:
        """Hold the real native-owner cleanup boundary across retirement."""
        cleanup_entered.set()
        await release_cleanup.wait()

    monkeypatch.setattr(worker, "_run_tasks", tasks)
    monkeypatch.setattr(worker._ace_step_preparation, "close", cleanup)
    with anyio.fail_after(5):
        async with anyio.create_task_group() as group:
            group.start_soon(worker.run)
            await active.wait()
            group.start_soon(worker.shutdown)
            await cleanup_entered.wait()
            assert child_drained.is_set()
            assert not worker._stopped.is_set()
            release_cleanup.set()
    assert worker._stopped.is_set()


@pytest.mark.parametrize("during_startup", [False, True])
@pytest.mark.parametrize("payload", ["music_edit_source", "separation_source"])
async def test_retired_startup_worker_cannot_block_directed_source_request(
    monkeypatch: pytest.MonkeyPatch,
    during_startup: bool,
    payload: AudioSourcePayload,
) -> None:
    """An unused worker subscription must not strand a live API's source handshake."""
    worker, _ = worker_fixture(monkeypatch)
    network, _ = channel[OutboundPacket]()
    topic = TopicRouter(VISION_MEDIA, network)
    worker._vision_media_packet_receiver = topic.new_receiver()
    api = topic.new_receiver()
    entered = threading.Event()
    release = threading.Event()

    def blocked_derivation() -> BackendDerivation:
        """Retire a real Worker before its media receive task can start."""
        entered.set()
        assert release.wait(5)
        return BackendDerivation(backends=frozenset())

    with anyio.fail_after(5):
        async with anyio.create_task_group() as group:
            if during_startup:
                monkeypatch.setattr(facts, "current_backend_derivation", blocked_derivation)
                group.start_soon(worker.run)
                assert await to_thread.run_sync(lambda: entered.wait(5))
                try:
                    group.start_soon(worker.shutdown)
                    await checkpoint()
                finally:
                    release.set()
            else:
                group.start_soon(worker.shutdown)
                await checkpoint()
                group.start_soon(worker.run)
    assert worker._stopped.is_set()
    packet = VisionMediaPacket(
        source_node=NodeId("source-worker"),
        target_node=NodeId("owner-api"),
        command_id=CommandId(),
        model=ModelId("native-audio-model"),
        sequence=0,
        kind="requested",
        total_chunks=66,
        image_count=1,
        payload=payload,
    )

    async def consume() -> None:
        """Exercise the real live receiver while the old receiver is idle."""
        assert await api.receive() == packet

    with anyio.fail_after(2):
        async with anyio.create_task_group() as group:
            group.start_soon(consume)
            await topic.publish(packet)
    assert len(topic.senders) == 1
    api.close()
