# pyright: reportPrivateUsage=false
"""Worker tests for exact-transfer staging capacity admission."""

import threading
from pathlib import Path

import anyio
import pytest

from skulk.routing.router import get_node_id_keypair
from skulk.shared.models.model_cards import (
    ModelCard,
    ModelId,
    ModelTask,
    RuntimeCapabilityCardConfig,
)
from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.common import NodeId
from skulk.shared.types.events import Event, IndexedEvent, NodeDownloadProgress
from skulk.shared.types.memory import Memory
from skulk.shared.types.state import State
from skulk.shared.types.tasks import CreateRunner
from skulk.shared.types.worker.downloads import DownloadCompleted, DownloadPending
from skulk.shared.types.worker.instances import InstanceId
from skulk.shared.types.worker.runners import RunnerId
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.store.config import StagingNodeConfig
from skulk.store.model_store_client import ModelStoreClient
from skulk.store.staging_eviction import (
    LAST_USED_MARKER_FILENAME,
    StagingCapacityError,
    StagingEvictionReport,
)
from skulk.utils.channels import Receiver, channel
from skulk.worker.main import Worker, _staging_model_ids
from skulk.worker.tests.unittests.conftest import get_mlx_ring_instance


def _shard(model_id: str, storage_bytes: int) -> PipelineShardMetadata:
    card = ModelCard(
        model_id=ModelId(model_id),
        storage_size=Memory.from_bytes(storage_bytes),
        n_layers=4,
        hidden_size=64,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )
    return PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=4,
        n_layers=4,
    )


def _stage(root: Path, model_id: str, size_bytes: int) -> Path:
    directory = root / model_id.replace("/", "--")
    directory.mkdir(parents=True)
    (directory / "model.safetensors").write_bytes(b"\0" * size_bytes)
    (directory / LAST_USED_MARKER_FILENAME).touch()
    return directory


def _worker(
    staging_root: Path,
) -> tuple[Worker, Receiver[Event]]:
    _, indexed_receiver = channel[IndexedEvent]()
    event_sender, event_receiver = channel[Event]()
    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    worker = Worker(
        node_id=NodeId(get_node_id_keypair().to_node_id()),
        event_receiver=indexed_receiver,
        event_sender=event_sender,
        command_sender=command_sender,
        download_command_sender=download_sender,
        store_client=ModelStoreClient(store_host="store.local"),
        staging_config=StagingNodeConfig(node_cache_path=str(staging_root)),
    )
    return worker, event_receiver


def test_staging_protection_includes_separate_served_draft_repo() -> None:
    card = _shard("org/base", storage_bytes=100).model_card.model_copy(
        update={
            "runtime": RuntimeCapabilityCardConfig(
                served_spec_type="draft_simple",
                served_spec_draft_repo="org/draft",
                served_spec_draft_file="draft.gguf",
            )
        }
    )

    assert _staging_model_ids(card) == {"org/base", "org/draft"}


@pytest.mark.asyncio
async def test_preflight_protects_partial_incoming_and_resets_evicted_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    incoming = _shard("org/incoming", storage_bytes=100)
    idle = _shard("org/idle", storage_bytes=120)
    incoming_directory = _stage(tmp_path, "org/incoming", size_bytes=40)
    idle_directory = _stage(tmp_path, "org/idle", size_bytes=120)
    worker, event_receiver = _worker(tmp_path)
    worker.state = State(
        downloads={
            worker.node_id: [
                DownloadCompleted(
                    node_id=worker.node_id,
                    shard_metadata=idle,
                    total=idle.model_card.storage_size,
                    model_directory=str(idle_directory),
                )
            ]
        }
    )
    monkeypatch.setattr(
        "skulk.worker.main.MINIMUM_STAGING_FREE_DISK_BYTES",
        100,
    )

    def _free_bytes(_path: Path) -> int:
        return 50 + (120 if not idle_directory.exists() else 0)

    monkeypatch.setattr(
        "skulk.store.staging_eviction._filesystem_free_bytes", _free_bytes
    )

    await worker.prepare_staging_transfer(
        incoming,
        frozenset({"org/incoming"}),
        additional_bytes=60,
    )

    assert incoming_directory.exists()
    assert not idle_directory.exists()
    reset_events = [
        event
        for event in event_receiver.collect()
        if isinstance(event, NodeDownloadProgress)
    ]
    assert len(reset_events) == 1
    assert isinstance(reset_events[0].download_progress, DownloadPending)
    assert reset_events[0].download_progress.shard_metadata == idle


@pytest.mark.asyncio
async def test_runtime_eviction_counters_delayed_download_completion(
    tmp_path: Path,
) -> None:
    """Remember an eviction until a just-completed transfer becomes visible."""
    completed = _shard("org/completed", storage_bytes=120)
    incoming = _shard("org/incoming", storage_bytes=100)
    worker, event_receiver = _worker(tmp_path)
    completed_directory = tmp_path / "org--completed"

    await worker._reset_download_state_for_evicted(
        StagingEvictionReport(evicted_model_ids=["org/completed"]),
        incoming,
    )

    assert "org--completed" in worker._stale_downloads_pending_reset
    assert event_receiver.collect() == []

    worker.state = State(
        downloads={
            worker.node_id: [
                DownloadCompleted(
                    node_id=worker.node_id,
                    shard_metadata=completed,
                    total=completed.model_card.storage_size,
                    model_directory=str(completed_directory),
                )
            ]
        }
    )
    await worker._reset_stale_downloads_from_state()

    reset_events = [
        event
        for event in event_receiver.collect()
        if isinstance(event, NodeDownloadProgress)
    ]
    assert len(reset_events) == 1
    assert isinstance(reset_events[0].download_progress, DownloadPending)
    assert reset_events[0].download_progress.shard_metadata == completed


@pytest.mark.asyncio
async def test_preflight_fails_cleanly_when_only_protected_data_remains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    incoming = _shard("org/incoming", storage_bytes=100)
    incoming_directory = _stage(tmp_path, "org/incoming", size_bytes=40)
    worker, _event_receiver = _worker(tmp_path)
    monkeypatch.setattr(
        "skulk.worker.main.MINIMUM_STAGING_FREE_DISK_BYTES",
        100,
    )

    def _free_bytes(_path: Path) -> int:
        return 50

    monkeypatch.setattr(
        "skulk.store.staging_eviction._filesystem_free_bytes",
        _free_bytes,
    )

    with pytest.raises(StagingCapacityError) as raised:
        await worker.prepare_staging_transfer(
            incoming,
            frozenset({"org/incoming"}),
            additional_bytes=60,
        )

    assert "need 0.0 GiB free" in str(raised.value)
    assert "only 0.0 GiB is available" in str(raised.value)
    assert incoming_directory.exists()


@pytest.mark.asyncio
async def test_preflight_fails_closed_when_disk_capacity_cannot_be_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    incoming = _shard("org/incoming", storage_bytes=100)
    _stage(tmp_path, "org/incoming", size_bytes=40)
    worker, _event_receiver = _worker(tmp_path)
    monkeypatch.setattr(
        "skulk.worker.main.MINIMUM_STAGING_FREE_DISK_BYTES",
        100,
    )

    def _unreadable_disk(_path: Path) -> int:
        raise OSError("disk metadata unavailable")

    monkeypatch.setattr(
        "skulk.store.staging_eviction._filesystem_free_bytes",
        _unreadable_disk,
    )

    with pytest.raises(StagingCapacityError) as raised:
        await worker.prepare_staging_transfer(
            incoming,
            frozenset({"org/incoming"}),
            additional_bytes=60,
        )

    assert "Could not verify staging disk capacity" in str(raised.value)
    assert "download was not started" in str(raised.value)


@pytest.mark.asyncio
async def test_zero_allocation_preflight_skips_reserve_and_eviction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hardlink staging remains available below the free-space reserve."""
    incoming = _shard("org/incoming", storage_bytes=100)
    worker, _event_receiver = _worker(tmp_path)

    def _unexpected_capacity_pass(
        _models_in_use: frozenset[str],
        _required_free_bytes: int,
        _enforce_recent_budget: bool,
        _fail_on_error: bool,
    ) -> None:
        raise AssertionError("zero-allocation staging must not enforce reserve")

    monkeypatch.setattr(
        worker,
        "_enforce_staging_budget",
        _unexpected_capacity_pass,
    )

    await worker.prepare_staging_transfer(
        incoming,
        frozenset({"org/incoming"}),
        additional_bytes=0,
    )


@pytest.mark.asyncio
async def test_runner_creation_waits_for_capacity_eviction_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    incoming = _shard("org/incoming", storage_bytes=100)
    worker, _event_receiver = _worker(tmp_path)
    runner_id = RunnerId()
    instance_id = InstanceId()
    instance = get_mlx_ring_instance(
        instance_id=instance_id,
        model_id=incoming.model_card.model_id,
        node_to_runner={worker.node_id: runner_id},
        runner_to_shard={runner_id: incoming},
    )
    worker.state = State(instances={instance_id: instance})

    eviction_started = threading.Event()
    release_eviction = threading.Event()

    def _blocking_capacity_pass(
        _models_in_use: frozenset[str],
        _required_free_bytes: int,
        _enforce_recent_budget: bool,
        _fail_on_error: bool,
    ) -> None:
        eviction_started.set()
        assert release_eviction.wait(timeout=2)

    monkeypatch.setattr(worker, "_enforce_staging_budget", _blocking_capacity_pass)
    runner_creation_started = anyio.Event()

    async def _record_runner_creation(
        _worker: Worker, _task: CreateRunner
    ) -> None:
        runner_creation_started.set()

    monkeypatch.setattr(Worker, "_execute_create_runner", _record_runner_creation)

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(
            worker.prepare_staging_transfer,
            incoming,
            frozenset({"org/incoming"}),
            1,
        )
        with anyio.fail_after(2):
            while not eviction_started.is_set():
                await anyio.sleep(0.01)

        task_group.start_soon(worker._plan_next_task_with_staging_guard)
        await anyio.sleep(0.05)
        assert not runner_creation_started.is_set()

        release_eviction.set()
        with anyio.fail_after(2):
            await runner_creation_started.wait()
