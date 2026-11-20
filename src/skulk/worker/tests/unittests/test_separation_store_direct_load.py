# pyright: reportPrivateUsage=false
"""Worker and deferred separator loading share a canonical Store artifact path."""

import shutil
from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path
from typing import Literal, cast

import pytest

import skulk.download.coordinator as coordinator_module
import skulk.download.download_utils as download_utils
import skulk.shared.constants as constants
from skulk.download.coordinator import DownloadCoordinator
from skulk.download.download_utils import RepoDownloadProgress
from skulk.download.impl_shard_downloader import ResumableShardDownloader
from skulk.shared.models.model_cards import ModelCard
from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.events import Event, IndexedEvent, RunnerStatusUpdated
from skulk.shared.types.memory import Memory
from skulk.shared.types.state import State
from skulk.shared.types.tasks import LoadModel, Task
from skulk.shared.types.telemetry import NodeTelemetry
from skulk.shared.types.worker.downloads import DownloadCompleted
from skulk.shared.types.worker.instances import BoundInstance
from skulk.shared.types.worker.runners import RunnerFailed, RunnerLoading, RunnerStatus
from skulk.store.config import StagingNodeConfig
from skulk.store.installed_cards import (
    build_installed_card_record,
    write_installed_card,
)
from skulk.store.model_store import StoreModelEntry, StoreRegistryIndex
from skulk.store.model_store_client import ModelStoreClient, ModelStoreDownloader
from skulk.utils.channels import channel
from skulk.utils.runtime_files import RuntimeLock
from skulk.worker.main import Worker
from skulk.worker.runner.demucs.context import SeparationRunnerContext
from skulk.worker.runner.demucs.model import SeparationModel
from skulk.worker.runner.demucs.source_fence import SourceReadFence
from skulk.worker.runner.demucs.tests.test_model import fixture
from skulk.worker.runner.runner_supervisor import RunnerSupervisor


@pytest.mark.parametrize("store_is_local", [False, True])
@pytest.mark.parametrize("staging_enabled", [False, True])
@pytest.mark.parametrize("offline", [False, True])
@pytest.mark.parametrize("startup_candidate", [False, True])
async def test_worker_dispatches_store_direct_separator_through_deferred_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, staging_enabled: bool, offline: bool,
    startup_candidate: bool, store_is_local: bool,
) -> None:
    """Configured discovery reaches staged or Store-direct deferred loading.

    Local or shared Store custody, path resolution, Worker dispatch and deferred
    model binding are real. Staging transfer and native effects are injected; this is not inference.
    """
    actual, factory, _, _ = fixture(tmp_path)
    bound = actual.bound
    card = bound.bound_shard.model_card
    bundle = card.artifact_bundle
    assert bundle is not None and card.source_revision is not None
    revision = card.source_revision
    store = tmp_path / "store"
    store.mkdir()
    canonical = store / (
        f"{card.model_id.normalize()}--revision-{card.source_revision}"
        f"--bundle-{bundle.bundle_id.removeprefix('bundle_')[:16]}"
    )
    (tmp_path / "model").rename(canonical)
    (canonical / ".skulk-source-revision").write_text(card.source_revision + "\n")
    record = build_installed_card_record(canonical, card)
    write_installed_card(canonical, record)
    entry = StoreModelEntry(
        model_id=str(card.model_id),
        store_path=canonical.name,
        source_repository=str(card.artifact_repository),
        source_revision=card.source_revision,
        files=[item.path for item in record.files],
        downloaded_at="2026-11-06T00:00:00+00:00",
        total_bytes=sum(item.size_bytes for item in record.files),
    )
    (store / "registry.json").write_text(
        StoreRegistryIndex(entries={str(card.model_id): entry}).model_dump_json()
    )
    staging = tmp_path / "staging"
    monkeypatch.setattr(constants, "SKULK_MODELS_PATH", (staging, store))
    monkeypatch.setattr(download_utils, "SKULK_MODELS_DIR", tmp_path / "unused")
    monkeypatch.setattr(coordinator_module, "SKULK_MODELS_DIR", tmp_path / "unused")
    staging_configuration = StagingNodeConfig(
        enabled=staging_enabled, node_cache_path=str(staging)
    )
    store_client = ModelStoreClient(store_host="store.invalid", local_store_path=store if store_is_local else None)
    inner = ResumableShardDownloader(offline=offline)
    downloader = ModelStoreDownloader(inner, store_client, staging_configuration)
    _, download_receiver = channel[ForwarderDownloadCommand]()
    discovery_events, _ = channel[Event]()
    telemetry, _ = channel[NodeTelemetry]()
    coordinator = DownloadCoordinator(
        node_id=bound.bound_node_id,
        shard_downloader=downloader,
        download_command_receiver=download_receiver,
        event_sender=discovery_events,
        telemetry_sender=telemetry,
        offline=offline,
        staging_cache_path=staging,
    )
    assert await coordinator._resolve_or_refresh_complete_artifact(
        bound.bound_shard
    ) == (None if staging_enabled else canonical)

    async def empty_status() -> AsyncIterator[tuple[Path, RepoDownloadProgress]]:
        if startup_candidate:
            yield canonical, RepoDownloadProgress(
                repo_id=str(card.model_id), repo_revision=revision,
                shard=bound.bound_shard, completed_files=1, total_files=1,
                downloaded=card.storage_size, downloaded_this_session=Memory.from_bytes(0),
                total=card.storage_size, overall_speed=0, overall_eta=timedelta(0),
                status="complete",
            )

    async def catalog() -> list[ModelCard]:
        return [card]

    class ScanComplete(BaseException):
        """Stop after one actual periodic discovery cycle without a timed race."""

    async def stop_scan(_seconds: float) -> None:
        raise ScanComplete

    with monkeypatch.context() as scan_patch:
        scan_patch.setattr(inner, "get_shard_download_status", empty_status)
        scan_patch.setattr(coordinator_module, "get_model_cards", catalog)
        scan_patch.setattr(coordinator_module.anyio, "sleep", stop_scan)
        with pytest.raises(ScanComplete):
            await coordinator._emit_existing_download_progress()
    assert isinstance(
        coordinator.download_status.get(card.model_id), DownloadCompleted
    ) == (not staging_enabled)
    discovered = coordinator.download_status.get(card.model_id)
    if isinstance(discovered, DownloadCompleted):
        assert discovered.read_only, "Canonical Store custody is never a deletable cache"
    expected_directory = canonical
    if staging_enabled:
        # The transfer effect is injected; real discovery must demand this
        # staged alias before Worker and deferred loading can share it.
        expected_directory = staging / card.model_id.normalize()
        shutil.copytree(canonical, expected_directory)
        write_installed_card(
            expected_directory, build_installed_card_record(expected_directory, card)
        )
        assert await coordinator._resolve_or_refresh_complete_artifact(
            bound.bound_shard
        ) == expected_directory
    reader = RuntimeLock(tmp_path / "reader", name="attachment.lock")
    fence = SourceReadFence(reader.descriptor)
    reader.close()
    context = SeparationRunnerContext(
        actual.runtime, actual.root, actual.source_root, fence
    )

    def bind(
        owner: SeparationRunnerContext, placement: BoundInstance, directory: Path
    ) -> SeparationModel:
        assert owner is context and placement == bound and directory == expected_directory
        return SeparationModel(
            placement,
            actual.runtime,
            directory,
            actual.root,
            actual.source_root,
            create_resident=factory,
        )

    monkeypatch.setattr(SeparationRunnerContext, "bind", bind)
    model = context.model(bound)
    dispatched: list[Task] = []

    class RecordingRunner:
        """Exercise the deferred native binding at the supervisor effect boundary."""

        status: RunnerStatus = RunnerLoading()

        async def start_task(self, task: Task) -> None:
            """Record dispatch and run the actual deferred load with fake native effects."""
            assert isinstance(task, LoadModel)
            dispatched.append(task)
            model.start(is_cancelled=lambda: False)

    def fit_accepted(*_args: object, **_kwargs: object) -> str | None:
        return None

    monkeypatch.setattr(Worker, "_local_shard_fit_error", fit_accepted)
    _, indexed_receiver = channel[IndexedEvent]()
    events, event_receiver = channel[Event]()
    commands, _ = channel[ForwarderCommand]()
    downloads, _ = channel[ForwarderDownloadCommand]()
    worker = Worker(
        node_id=bound.bound_node_id,
        event_receiver=indexed_receiver,
        event_sender=events,
        command_sender=commands,
        download_command_sender=downloads,
        store_client=store_client,
        staging_config=staging_configuration,
    )
    worker.state = State(instances={bound.instance.instance_id: bound.instance})
    runner = RecordingRunner()
    worker.runners[bound.bound_runner_id] = cast(RunnerSupervisor, cast(object, runner))
    task = LoadModel(instance_id=bound.instance.instance_id)
    try:
        await worker._start_runner_task(task)
        assert dispatched == [task] and model.is_ready()
        assert len(factory.calls) == 1
        assert factory.calls[0][2].weights_path == str(
            expected_directory / "955717e8.safetensors"
        )
        assert not (store / card.model_id.normalize()).exists()
        assert not any(
            isinstance(event, RunnerStatusUpdated)
            and isinstance(event.runner_status, RunnerFailed)
            for event in event_receiver.collect()
        )
    finally:
        model.close()
        fence.close()


@pytest.mark.parametrize("store_is_local", [False, True])
@pytest.mark.parametrize("staging_enabled", [False, True])
@pytest.mark.parametrize("location", ["canonical", "staged", "default"])
async def test_download_delete_preserves_store_custody_and_removes_only_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, staging_enabled: bool,
    location: Literal["canonical", "staged", "default"], store_is_local: bool,
) -> None:
    """A stale writable status cannot delete canonical weights or their index.

    The filesystem, Store wrapper and coordinator deletion path are real. Both
    staging modes retain local/shared custody while ordinary cache deletion works.
    """
    actual, _, _, _ = fixture(tmp_path)
    bound = actual.bound
    card = bound.bound_shard.model_card
    store = tmp_path / "store"
    store.mkdir()
    canonical = store / "registered-canonical-bundle"
    (tmp_path / "model").rename(canonical)
    record = build_installed_card_record(canonical, card)
    write_installed_card(canonical, record)
    entry = StoreModelEntry(
        model_id=str(card.model_id), store_path=canonical.name,
        source_repository=str(card.artifact_repository),
        source_revision=card.source_revision,
        files=[item.path for item in record.files],
        downloaded_at="2026-11-06T00:00:00+00:00",
        total_bytes=sum(item.size_bytes for item in record.files),
    )
    index_path = store / "registry.json"
    index_path.write_text(
        StoreRegistryIndex(entries={str(card.model_id): entry}).model_dump_json()
    )
    index_before = index_path.read_bytes()
    default_root = tmp_path / "downloads"
    default = default_root / card.model_id.normalize()
    staged_root = tmp_path / "staging"
    staged = staged_root / card.model_id.normalize()
    shutil.copytree(canonical, default)
    shutil.copytree(canonical, staged)
    monkeypatch.setattr(download_utils, "SKULK_MODELS_DIR", default_root)
    monkeypatch.setattr(coordinator_module, "SKULK_MODELS_DIR", default_root)
    unregistered: list[object] = []
    monkeypatch.setattr(
        coordinator_module, "unregister_installed_card_record", unregistered.append
    )
    configuration = StagingNodeConfig(
        enabled=staging_enabled, node_cache_path=str(staged_root)
    )
    downloader = ModelStoreDownloader(
        ResumableShardDownloader(offline=True),
        ModelStoreClient(store_host="store.invalid", local_store_path=store if store_is_local else None),
        configuration,
    )
    _, commands = channel[ForwarderDownloadCommand]()
    events, _ = channel[Event]()
    telemetry, _ = channel[NodeTelemetry]()
    coordinator = DownloadCoordinator(
        node_id=bound.bound_node_id, shard_downloader=downloader,
        download_command_receiver=commands, event_sender=events,
        telemetry_sender=telemetry, staging_cache_path=staged_root,
    )
    directory = {"canonical": canonical, "staged": staged, "default": default}[location]
    coordinator.download_status[card.model_id] = DownloadCompleted(
        node_id=bound.bound_node_id, shard_metadata=bound.bound_shard,
        total=card.storage_size, model_directory=str(directory), read_only=False,
    )
    await coordinator._delete_download(card.model_id)
    assert canonical.is_dir() and (canonical / "955717e8.safetensors").is_file()
    assert index_path.read_bytes() == index_before
    if location == "canonical":
        assert default.is_dir() and staged.is_dir()
        assert card.model_id in coordinator.download_status
        assert not unregistered
    else:
        assert not directory.exists() and not default.exists()
        assert card.model_id not in coordinator.download_status
        assert unregistered == [card.model_id]
