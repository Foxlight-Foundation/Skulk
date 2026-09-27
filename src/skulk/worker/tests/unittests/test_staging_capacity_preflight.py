# pyright: reportPrivateUsage=false
"""Worker tests for exact-transfer staging capacity admission."""

import os
import threading
import time
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
from skulk.shared.types.worker.shards import (
    PipelineShardMetadata,
    RpcDonorShardMetadata,
)
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
    *,
    keep_recent_gb: float = 40.0,
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
        staging_config=StagingNodeConfig(
            node_cache_path=str(staging_root),
            staging_keep_recent_gb=keep_recent_gb,
        ),
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


def _age_marker(directory: Path, seconds: float) -> None:
    stamp = time.time() - seconds
    os.utime(directory / LAST_USED_MARKER_FILENAME, (stamp, stamp))


def test_startup_reconciliation_keeps_what_was_serving(tmp_path: Path) -> None:
    """A restart or a recreated worker keeps the models that were serving.

    Nothing is in use at startup and the node id is new, so the recency
    marker decides: a model used within the window survives even though it
    alone exceeds the grace budget, here zero; an old idle copy does not.
    """
    serving = _stage(tmp_path, "org/serving", size_bytes=40)
    stale = _stage(tmp_path, "org/stale", size_bytes=40)
    _age_marker(serving, 120)
    _age_marker(stale, 2 * 3600)
    worker, _event_receiver = _worker(tmp_path, keep_recent_gb=0)

    worker._reconcile_staging_on_startup()

    assert serving.exists()
    assert not stale.exists()
    assert worker._stale_downloads_pending_reset == {"org--stale"}


def test_placed_models_are_in_use_and_donor_shards_are_not(tmp_path: Path) -> None:
    """A placement on this node protects its model before its runner exists.

    Another node's placement protects nothing here, and neither does an RPC
    donor shard: a donor lends memory and never reads the model.
    """
    placed = _shard("org/placed", storage_bytes=40)
    elsewhere = _shard("org/elsewhere", storage_bytes=40)
    donated = _shard("org/donated", storage_bytes=40)
    donor = RpcDonorShardMetadata(
        model_card=donated.model_card,
        device_rank=1,
        world_size=2,
        start_layer=0,
        end_layer=0,
        n_layers=donated.n_layers,
    )
    worker, _event_receiver = _worker(tmp_path)
    here, there, lent = RunnerId(), RunnerId(), RunnerId()
    placed_instance, elsewhere_instance, donor_instance = (
        InstanceId(),
        InstanceId(),
        InstanceId(),
    )
    worker.state = State(
        instances={
            placed_instance: get_mlx_ring_instance(
                instance_id=placed_instance,
                model_id=placed.model_card.model_id,
                node_to_runner={worker.node_id: here},
                runner_to_shard={here: placed},
            ),
            elsewhere_instance: get_mlx_ring_instance(
                instance_id=elsewhere_instance,
                model_id=elsewhere.model_card.model_id,
                node_to_runner={NodeId("other-node"): there},
                runner_to_shard={there: elsewhere},
            ),
            donor_instance: get_mlx_ring_instance(
                instance_id=donor_instance,
                model_id=donated.model_card.model_id,
                node_to_runner={worker.node_id: lent},
                runner_to_shard={lent: donor},
            ),
        }
    )

    assert worker._models_in_use() == {"org/placed"}


def test_in_use_markers_are_refreshed_once_a_minute(tmp_path: Path) -> None:
    """Serving models keep a fresh marker for the next startup to read."""
    placed = _shard("org/placed", storage_bytes=40)
    placed_directory = _stage(tmp_path, "org/placed", size_bytes=40)
    idle_directory = _stage(tmp_path, "org/idle", size_bytes=40)
    _age_marker(placed_directory, 3600)
    _age_marker(idle_directory, 3600)
    worker, _event_receiver = _worker(tmp_path)
    runner_id, instance_id = RunnerId(), InstanceId()
    worker.state = State(
        instances={
            instance_id: get_mlx_ring_instance(
                instance_id=instance_id,
                model_id=placed.model_card.model_id,
                node_to_runner={worker.node_id: runner_id},
                runner_to_shard={runner_id: placed},
            )
        }
    )
    marker = placed_directory / LAST_USED_MARKER_FILENAME
    idle_marker = idle_directory / LAST_USED_MARKER_FILENAME

    worker._refresh_in_use_markers()
    assert time.time() - marker.stat().st_mtime < 60
    assert time.time() - idle_marker.stat().st_mtime > 3000

    # Within the minute nothing is touched again.
    _age_marker(placed_directory, 3600)
    worker._refresh_in_use_markers()
    assert time.time() - marker.stat().st_mtime > 3000

    assert worker._in_use_markers_refreshed_at is not None
    worker._in_use_markers_refreshed_at -= 61
    worker._refresh_in_use_markers()
    assert time.time() - marker.stat().st_mtime < 60
