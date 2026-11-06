# pyright: reportPrivateUsage=false
"""Worker and deferred separator loading share a canonical Store artifact path."""

from pathlib import Path
from typing import cast

import pytest

import skulk.download.download_utils as download_utils
import skulk.shared.constants as constants
from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.events import Event, IndexedEvent, RunnerStatusUpdated
from skulk.shared.types.state import State
from skulk.shared.types.tasks import LoadModel, Task
from skulk.shared.types.worker.instances import BoundInstance
from skulk.shared.types.worker.runners import RunnerFailed, RunnerLoading, RunnerStatus
from skulk.store.config import StagingNodeConfig
from skulk.store.installed_cards import (
    build_installed_card_record,
    write_installed_card,
)
from skulk.store.model_store import StoreModelEntry, StoreRegistryIndex
from skulk.store.model_store_client import ModelStoreClient
from skulk.utils.channels import channel
from skulk.utils.runtime_files import RuntimeLock
from skulk.worker.main import Worker
from skulk.worker.runner.demucs.context import SeparationRunnerContext
from skulk.worker.runner.demucs.model import SeparationModel
from skulk.worker.runner.demucs.source_fence import SourceReadFence
from skulk.worker.runner.demucs.tests.test_model import fixture
from skulk.worker.runner.runner_supervisor import RunnerSupervisor


async def test_worker_dispatches_store_direct_separator_through_deferred_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completed canonical download reaches the runner without a staged alias.

    File custody, path resolution, Worker dispatch and deferred model binding are
    real. Native model effects are injected; this is not an inference test.
    """
    actual, factory, _, _ = fixture(tmp_path)
    bound = actual.bound
    card = bound.bound_shard.model_card
    bundle = card.artifact_bundle
    assert bundle is not None and card.source_revision is not None
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
    monkeypatch.setattr(constants, "SKULK_MODELS_PATH", (store,))
    monkeypatch.setattr(download_utils, "SKULK_MODELS_DIR", tmp_path / "unused")
    reader = RuntimeLock(tmp_path / "reader", name="attachment.lock")
    fence = SourceReadFence(reader.descriptor)
    reader.close()
    context = SeparationRunnerContext(
        actual.runtime, actual.root, actual.source_root, fence
    )

    def bind(
        owner: SeparationRunnerContext, placement: BoundInstance, directory: Path
    ) -> SeparationModel:
        assert owner is context and placement == bound and directory == canonical
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
        store_client=ModelStoreClient(store_host="store.invalid"),
        staging_config=StagingNodeConfig(
            enabled=False, node_cache_path=str(tmp_path / "staging")
        ),
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
            canonical / "955717e8.safetensors"
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
