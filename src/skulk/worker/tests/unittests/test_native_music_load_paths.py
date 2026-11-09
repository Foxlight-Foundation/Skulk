# pyright: reportPrivateUsage=false
"""Worker preflight and native binding select the same recorded primary generation."""

from collections.abc import Mapping
from pathlib import Path
from typing import Literal, cast

import pytest

import skulk.download.download_utils as download_utils
import skulk.shared.constants as constants
from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.events import Event, IndexedEvent, RunnerStatusUpdated
from skulk.shared.types.state import State
from skulk.shared.types.tasks import LoadModel, Task
from skulk.shared.types.worker.instances import BoundInstance
from skulk.shared.types.worker.runners import RunnerFailed, RunnerLoading, RunnerStatus
from skulk.store.installed_cards import (
    build_installed_card_record,
    write_installed_card,
)
from skulk.store.model_store import StoreModelEntry, StoreRegistryIndex
from skulk.utils.channels import channel
from skulk.utils.runtime_files import RuntimeLock
from skulk.worker.main import Worker
from skulk.worker.runner.ace_step.context import NativeMusicRunnerContext
from skulk.worker.runner.ace_step.model import NativeMusicModel
from skulk.worker.runner.ace_step.tests.test_model import fixture
from skulk.worker.runner.demucs.source_fence import SourceReadFence
from skulk.worker.runner.runner_supervisor import RunnerSupervisor


@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("invalid", [None, "missing", "stale"])
async def test_worker_native_primary_matches_deferred_source_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    indexed: bool,
    invalid: Literal["missing", "stale"] | None,
) -> None:
    """Recorded generations load without aliases; absent or stale primaries fail.

    Real selected files, installed records, path resolution, Worker dispatch and
    deferred native binding are exercised. Only native execution and live resource
    admission are injected; this is not hardware or music qualification.
    """
    actual, factory, _ = fixture(tmp_path)
    bound = actual.bound
    card = bound.bound_shard.model_card
    store = tmp_path / "store"
    store.mkdir()
    sources: dict[tuple[str, str], Path] = {}
    entries: dict[str, StoreModelEntry] = {}
    for selected in card.native_music_artifact_sources():
        identity = selected.repository, selected.revision
        primary = selected.repository == str(card.artifact_repository)
        alias, _ = download_utils.companion_artifact_location(card, *identity)
        directory = store / (
            alias.normalize() + "--installed-generation"
            if indexed
            else alias.normalize()
        )
        actual.sources[identity].rename(directory)
        sources[identity] = directory
        download_utils.write_source_revision_marker(directory, selected.revision)
        recorded_card = (
            card.model_copy(update={"base_model": "different custom policy"})
            if primary and invalid == "stale"
            else card
        )
        installed = build_installed_card_record(
            directory,
            recorded_card,
            artifact_role="base" if primary else "music_component",
            artifact_model_id=str(alias),
            artifact_repository=selected.repository,
            artifact_revision=selected.revision,
            owner_model_id=None if primary else str(card.model_id),
            owner_card_id=None if primary else card.registry_card_id,
        )
        write_installed_card(directory, installed)
        entries[str(alias)] = StoreModelEntry(
            model_id=str(alias),
            store_path=directory.name,
            source_repository=selected.repository,
            source_revision=selected.revision,
            files=[item.path for item in installed.files],
            downloaded_at="2026-11-06T00:00:00+00:00",
            total_bytes=sum(item.size_bytes for item in installed.files),
        )
    (store / "registry.json").write_text(
        StoreRegistryIndex(entries=entries).model_dump_json()
    )
    primary = sources[str(card.artifact_repository), str(card.source_revision)]
    if invalid == "missing":
        (primary / "config.json").unlink()
    monkeypatch.setattr(constants, "SKULK_MODELS_PATH", (store,))
    monkeypatch.setattr(download_utils, "SKULK_MODELS_DIR", tmp_path / "absent")
    reader = RuntimeLock(tmp_path / "reader", name="attachment.lock")
    fence = SourceReadFence(reader.descriptor)
    reader.close()
    context = NativeMusicRunnerContext(
        actual.runtime, actual.root, actual.source_root, fence, lambda _process: None
    )

    def bind(
        owner: NativeMusicRunnerContext,
        placement: BoundInstance,
        selected: Mapping[tuple[str, str], Path],
    ) -> NativeMusicModel:
        assert owner is context and placement == bound and selected == sources
        return NativeMusicModel(
            placement,
            actual.runtime,
            selected,
            actual.root,
            actual.source_root,
            create_resident=factory,
        )

    monkeypatch.setattr(NativeMusicRunnerContext, "bind", bind)
    model = context.model(bound)
    dispatched: list[Task] = []

    class RecordingRunner:
        """Dispatch to the actual deferred binding while injecting native effects."""

        status: RunnerStatus = RunnerLoading()

        async def start_task(self, task: Task) -> None:
            """Run deferred load only after the real Worker accepts this primary."""
            assert isinstance(task, LoadModel)
            dispatched.append(task)
            model.start(is_cancelled=lambda: False)

    def fit_accepted(*_arguments: object, **_keywords: object) -> str | None:
        return None

    monkeypatch.setattr(Worker, "_local_shard_fit_error", fit_accepted)
    _, indexed_receiver = channel[IndexedEvent]()
    events, received = channel[Event]()
    commands, _ = channel[ForwarderCommand]()
    downloads, _ = channel[ForwarderDownloadCommand]()
    worker = Worker(
        node_id=bound.bound_node_id,
        event_receiver=indexed_receiver,
        event_sender=events,
        command_sender=commands,
        download_command_sender=downloads,
    )
    worker.state = State(instances={bound.instance.instance_id: bound.instance})
    runner = RecordingRunner()
    worker.runners[bound.bound_runner_id] = cast(RunnerSupervisor, cast(object, runner))
    task = LoadModel(instance_id=bound.instance.instance_id)
    try:
        await worker._start_runner_task(task)
        failures = [
            event
            for event in received.collect()
            if isinstance(event, RunnerStatusUpdated)
            and isinstance(event.runner_status, RunnerFailed)
        ]
        if invalid is None:
            assert dispatched == [task] and model.is_ready() and factory.calls == 1
            assert not failures
        else:
            assert not dispatched and not model.is_ready() and factory.calls == 0
            assert len(failures) == 1
    finally:
        model.close()
        fence.close()
