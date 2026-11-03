"""Native reader metadata must not authorize incomplete artifact transactions."""

import asyncio
import contextlib
import tomllib
from pathlib import Path

import anyio
import pytest
from foxlight_model_preservation.models import PreservationError

from skulk.download import coordinator as coordinator_module
from skulk.download import download_utils
from skulk.download.coordinator import DownloadCoordinator
from skulk.download.impl_shard_downloader import ResumableShardDownloader
from skulk.shared import constants
from skulk.shared.models.model_cards import ModelCard, ModelTask
from skulk.shared.types.commands import ForwarderDownloadCommand, StartDownload
from skulk.shared.types.common import ModelId, NodeId, SystemId
from skulk.shared.types.events import Event, NodeDownloadProgress
from skulk.shared.types.memory import Memory
from skulk.shared.types.telemetry import NodeTelemetry
from skulk.shared.types.worker.downloads import DownloadCompleted, DownloadFailed
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.store.installed_cards import (
    build_installed_card_record,
    write_installed_card,
)
from skulk.store.model_store import ModelStore
from skulk.store.preservation_closure import companion_selections
from skulk.utils.channels import channel


def _card() -> ModelCard:
    source = Path(__file__).parents[2] / "shared/models/tests/ace-native-card.toml"
    return ModelCard.model_validate(tomllib.loads(source.read_text()))


def _shard(card: ModelCard) -> PipelineShardMetadata:
    return PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=card.n_layers,
        n_layers=card.n_layers,
    )


def test_native_companion_selection_cannot_silently_omit_both_components() -> None:
    """Refuse an incomplete transaction even though reading the card is valid."""
    with pytest.raises(ValueError, match="native music.*artifact"):
        download_utils.companion_download_specs(_card())


@pytest.mark.parametrize("required_only", [False, True])
def test_native_profile_cannot_be_inferred_download_complete(required_only: bool) -> None:
    """Cached and offline shortcuts cannot mistake primary bytes for full closure."""
    assert not download_utils.model_companions_present_on_disk(
        _card(), required_only=required_only
    )


async def test_native_primary_selection_is_not_a_complete_download() -> None:
    """Direct callers cannot resolve only three files and label the model complete."""
    with pytest.raises(ValueError, match="native music.*artifact"):
        await download_utils.resolve_allow_patterns(_shard(_card()))


@pytest.mark.parametrize("offline", [False, True])
@pytest.mark.parametrize("config_only", [False, True])
async def test_native_downloader_refuses_before_cache_or_network_effects(
    offline: bool, config_only: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Configuration-only and offline paths must also honor the unsupported layout."""
    cache = tmp_path / "not-created"
    monkeypatch.setattr(download_utils, "SKULK_MODELS_DIR", cache)
    downloader = ResumableShardDownloader(offline=offline)
    with pytest.raises(ValueError, match="native music.*artifact"):
        await downloader.ensure_shard(_shard(_card()), config_only=config_only)
    assert not cache.exists()


@pytest.mark.parametrize("skip_download", [False, True])
async def test_explicit_patterns_cannot_bypass_native_download_refusal(
    skip_download: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raw shard request fails before creating directories, even with supplied paths."""
    cache = tmp_path / "not-created"
    monkeypatch.setattr(download_utils, "SKULK_MODELS_DIR", cache)
    downloader = ResumableShardDownloader()
    with pytest.raises(ValueError, match="native music.*artifact"):
        await download_utils.download_shard(
            _shard(_card()),
            downloader.on_progress_wrapper,
            skip_download=skip_download,
            allow_patterns=["config.json"],
            skip_internet=True,
        )
    assert not cache.exists()


def test_native_preservation_cannot_select_primary_only_closure() -> None:
    """Both export and recovery share this refusal before selecting any byte work."""
    with pytest.raises(PreservationError, match="native music.*artifact"):
        companion_selections(_card())


def test_native_installed_record_cannot_certify_primary_only_bytes(tmp_path: Path) -> None:
    """Record creation refuses before inspecting an uncreated artifact directory."""
    with pytest.raises(ValueError, match="native music.*artifact"):
        build_installed_card_record(tmp_path / "missing", _card())


async def test_native_store_download_is_rejected_before_admission(tmp_path: Path) -> None:
    """Direct store callers cannot bypass the shard downloader's layout refusal."""
    root = tmp_path / "not-created"
    store = ModelStore(root, recover_installed_cards=False)
    card = _card()
    with pytest.raises(ValueError, match="native music.*artifact"):
        await store.request_download(str(card.model_id), model_card=card)
    assert not store.list_active_downloads()
    assert not root.exists()


@pytest.mark.parametrize("offline", [False, True])
async def test_native_download_failure_keeps_coordinator_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, offline: bool
) -> None:
    """Real queued admission must fail only native work, then complete a cached model."""
    node = NodeId("native-download-test")
    native = _card()
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
        assert "native music artifact" in failed.error_message
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
