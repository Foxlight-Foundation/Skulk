"""Native ACE downloads retain the complete pinned three-component data closure."""

import asyncio
import contextlib
import tomllib
from collections.abc import Callable
from pathlib import Path

import anyio
import pytest

from skulk.download import coordinator as coordinator_module
from skulk.download import download_utils
from skulk.download.coordinator import DownloadCoordinator
from skulk.download.impl_shard_downloader import ResumableShardDownloader
from skulk.shared import constants
from skulk.shared.models.model_cards import ModelCard, ModelTask
from skulk.shared.tests.native_music_fixtures import native_music_fixture
from skulk.shared.types.commands import ForwarderDownloadCommand, StartDownload
from skulk.shared.types.common import ModelId, NodeId, SystemId
from skulk.shared.types.events import Event, NodeDownloadProgress
from skulk.shared.types.memory import Memory
from skulk.shared.types.telemetry import NodeTelemetry
from skulk.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    FileListEntry,
)
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.store.installed_cards import (
    build_installed_card_record,
    read_installed_card,
    write_installed_card,
)
from skulk.store.preservation_closure import companion_selections
from skulk.utils.channels import channel


def _card() -> ModelCard:
    source = Path(__file__).parents[2] / "shared/models/tests/ace-native-card.toml"
    return ModelCard.model_validate(tomllib.loads(source.read_text()))


def _shard(card: ModelCard) -> PipelineShardMetadata:
    return PipelineShardMetadata(model_card=card, device_rank=0, world_size=1,
        start_layer=0, end_layer=card.n_layers, n_layers=card.n_layers)


def test_native_companion_selection_retains_encoder_and_vae_once() -> None:
    """Shared-source roles must be one required exact eleven-file transaction."""
    card = _card()
    specs = download_utils.companion_download_specs(card)
    assert len(specs) == 1
    shard, paths, required = specs[0]
    assert required and len(paths) == 11
    assert shard.model_card.artifact_bundle is not None
    assert shard.model_card.artifact_bundle.root is None
    assert set(paths) == {item.path for item in shard.model_card.artifact_bundle.files}
    assert companion_selections(card) == (("music_component", shard.model_card),)


@pytest.mark.parametrize("required_only", [False, True])
def test_native_profile_cannot_be_inferred_download_complete(required_only: bool) -> None:
    """Cached and offline shortcuts cannot mistake primary bytes for full closure."""
    assert not download_utils.model_companions_present_on_disk(_card(), required_only=required_only)


@pytest.mark.parametrize("shared_primary", [False, True])
async def test_native_direct_download_and_verified_offline_reopen(
    shared_primary: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ordinary downloader retains all fourteen pins and reopens without any network."""
    card, objects = native_music_fixture(shared_primary=shared_primary)
    fetched: list[tuple[str, str, str]] = []
    original_listing = download_utils.fetch_file_list_with_cache
    original_download = download_utils.download_file_with_retry

    async def listing(repository: ModelId, revision: str, **_kwargs: object) -> list[FileListEntry]:
        return [FileListEntry(type="file", path=path, size=len(data))
            for (repo, rev, path), data in objects.items() if (repo, rev) == (str(repository), revision)] + [FileListEntry(type="file", path="unselected.py", size=3)]

    async def download(repository: ModelId, revision: str, path: str, target_dir: Path,
        on_progress: Callable[[int, int, bool], None], expected_size: int | None = None,
        expected_object_id: str | None = None, **_kwargs: object) -> Path:
        key = (str(repository), revision, path)
        data = objects[key]
        assert expected_size == len(data) and expected_object_id is not None
        fetched.append(key)
        target = target_dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        on_progress(len(data), len(data), True)
        return target

    monkeypatch.setattr(download_utils, "SKULK_MODELS_DIR", tmp_path)
    monkeypatch.setattr("skulk.download.impl_shard_downloader.SKULK_MODELS_DIR", tmp_path)
    monkeypatch.setattr(constants, "SKULK_MODELS_PATH", (tmp_path,))
    monkeypatch.setattr(download_utils, "fetch_file_list_with_cache", listing)
    monkeypatch.setattr(download_utils, "download_file_with_retry", download)
    assert len(await download_utils.resolve_allow_patterns(_shard(card))) == (14 if shared_primary else 3)
    primary = await ResumableShardDownloader().ensure_shard(_shard(card))
    assert primary == tmp_path / card.model_id.normalize()
    assert set(fetched) == set(objects) and len(fetched) == 14
    assert download_utils.model_companions_present_on_disk(card, required_only=True)
    record = read_installed_card(primary)
    assert record and len(record.files) == (14 if shared_primary else 3)
    assert not (primary / "unselected.py").exists()

    async def no_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("offline reopening attempted network access")

    monkeypatch.setattr(download_utils, "fetch_file_list_with_cache", original_listing)
    monkeypatch.setattr(download_utils, "download_file_with_retry", original_download)
    monkeypatch.setattr(download_utils, "file_meta", no_network)
    assert await ResumableShardDownloader(offline=True).ensure_shard(_shard(card)) == primary
    missing = card.native_music_artifact_sources()[-1].files[-1]
    source = card.native_music_artifact_sources()[-1]
    alias, _ = download_utils.companion_artifact_location(card, source.repository, source.revision)
    (tmp_path / alias.normalize() / missing.path).unlink()
    assert not download_utils.model_companions_present_on_disk(card)
    with pytest.raises(FileNotFoundError, match="Offline native music"):
        await ResumableShardDownloader(offline=True).ensure_shard(_shard(card))


@pytest.mark.parametrize("skip_download", [False, True])
async def test_explicit_partial_patterns_do_not_certify_native_download(
    skip_download: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raw partial selections fail rather than claiming a complete native base."""
    card, objects = native_music_fixture()
    async def listing(*_args: object, **_kwargs: object) -> list[FileListEntry]:
        return [FileListEntry(type="file", path=path, size=len(data))
            for (repo, _rev, path), data in objects.items() if repo == str(card.artifact_repository)]
    monkeypatch.setattr(download_utils, "SKULK_MODELS_DIR", tmp_path)
    monkeypatch.setattr(download_utils, "fetch_file_list_with_cache", listing)
    downloader = ResumableShardDownloader()
    with pytest.raises(FileNotFoundError, match="file is absent"):
        await download_utils.download_shard(_shard(card), downloader.on_progress_wrapper,
            skip_download=skip_download, allow_patterns=["config.json"], skip_internet=True)
    assert not (tmp_path / card.model_id.normalize() / "model.safetensors").exists()
    assert read_installed_card(tmp_path / card.model_id.normalize()) is None


def test_native_installed_record_rejects_missing_primary_files(tmp_path: Path) -> None:
    """An installed record must bind actual complete bytes, not just valid metadata."""
    card, _objects = native_music_fixture()
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="complete exact source selection"):
        build_installed_card_record(tmp_path, card)


@pytest.mark.parametrize("offline", [False, True])
@pytest.mark.parametrize("failure_kind", ["missing_repository", "missing_primary", "wrong_size"])
async def test_native_download_failure_keeps_coordinator_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, offline: bool, failure_kind: str
) -> None:
    """Real queued admission must fail only native work, then complete a cached model."""
    native = _card()
    assert native.artifact_bundle is not None
    selected = native.artifact_bundle.files
    async def missing_source(*_args: object, **_kwargs: object) -> list[FileListEntry]:
        if failure_kind == "missing_repository":
            raise FileNotFoundError("Required native source is absent")
        return [FileListEntry(type="file", path=item.path,
            size=item.size_bytes + (1 if failure_kind == "wrong_size" and item.path == "model.safetensors" else 0))
            for item in selected if failure_kind != "missing_primary" or item.path != "model.safetensors"]

    monkeypatch.setattr(download_utils, "fetch_file_list_with_cache", missing_source)
    node = NodeId("native-download-test")
    ordinary = ModelCard(
        model_id=ModelId("test/cached-gguf"), storage_size=Memory.from_bytes(7),
        n_layers=1, hidden_size=1, supports_tensor=False,
        tasks=[ModelTask.TextGeneration], gguf_file="model.gguf",
        trust_remote_code=False,
    )
    cached = tmp_path / ordinary.model_id.normalize()
    cached.mkdir()
    (cached / "config.json").write_text("{}")
    (cached / "model.gguf").write_bytes(b"weights")
    write_installed_card(cached, build_installed_card_record(cached, ordinary))
    monkeypatch.setattr(constants, "SKULK_MODELS_PATH", (tmp_path,))
    monkeypatch.setattr(download_utils, "SKULK_MODELS_DIR", tmp_path)
    monkeypatch.setattr(coordinator_module, "SKULK_MODELS_DIR", tmp_path)

    async def no_background_catalog() -> list[ModelCard]:
        return []

    monkeypatch.setattr(coordinator_module, "get_model_cards", no_background_catalog)
    monkeypatch.setattr("skulk.download.impl_shard_downloader.get_model_cards", no_background_catalog)
    commands, incoming = channel[ForwarderDownloadCommand]()
    events, received = channel[Event]()
    telemetry, _ = channel[NodeTelemetry]()
    coordinator = DownloadCoordinator(
        node_id=node, shard_downloader=ResumableShardDownloader(offline=offline),
        download_command_receiver=incoming, event_sender=events,
        telemetry_sender=telemetry, offline=offline,
    )

    async def observe(model: ModelId) -> DownloadFailed | DownloadCompleted:
        with anyio.fail_after(5):
            while True:
                event = await received.receive()
                if isinstance(event, NodeDownloadProgress):
                    progress = event.download_progress
                    if progress.shard_metadata.model_card.model_id == model and isinstance(progress, (DownloadFailed, DownloadCompleted)):
                        return progress

    task = asyncio.create_task(coordinator.run())
    try:
        await commands.send(ForwarderDownloadCommand(
            origin=SystemId("native-test"),
            command=StartDownload(target_node_id=node, shard_metadata=_shard(native)),
        ))
        failed = await observe(native.model_id)
        assert isinstance(failed, DownloadFailed)
        if failure_kind == "missing_repository":
            assert "native" in failed.error_message.lower() or "Required companion" in failed.error_message
        else:
            assert ("file is absent" if failure_kind == "missing_primary" else "size mismatch") in failed.error_message
        assert not (tmp_path / native.model_id.normalize()).exists()
        assert not coordinator.active_downloads
        await commands.send(ForwarderDownloadCommand(
            origin=SystemId("native-test"),
            command=StartDownload(target_node_id=node, shard_metadata=_shard(ordinary)),
        ))
        completed = await observe(ordinary.model_id)
        assert isinstance(completed, DownloadCompleted)
        assert completed.model_directory == str(cached)
        assert not task.done()
    finally:
        await coordinator.shutdown()
        with contextlib.suppress(asyncio.CancelledError):
            await task
