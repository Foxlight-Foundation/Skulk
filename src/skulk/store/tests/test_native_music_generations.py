# pyright: reportPrivateUsage=false
"""Native source generations preserve independent owners and exact selections."""

import asyncio
import hashlib
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import anyio
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from foxlight_model_preservation.signing import create_key, load_key

from skulk.download import download_utils
from skulk.shared import constants
from skulk.shared.models.model_cards import ModelCard, NativeMusicArtifactSource
from skulk.shared.tests.native_music_fixtures import native_music_fixture
from skulk.shared.types.common import ModelId
from skulk.store import model_store_server as server_module
from skulk.store.installed_cards import (
    manifest_sha256,
    read_installed_card,
    verify_installed_card,
    write_installed_card,
)
from skulk.store.model_store import ModelStore
from skulk.store.model_store_server import ModelStoreServer
from skulk.store.preservation import recover_home
from skulk.store.preservation_export import plan_preservation, preserve_installed
from skulk.store.tests.test_native_music_store import (
    _finished,
    _install,
    _network_fixture,
)


def _other_selection(
    card: ModelCard,
    objects: dict[tuple[str, str, str], bytes],
    *,
    prefix: str = "replacement",
    other_model: bool = False,
) -> ModelCard:
    """Create another valid exact encoder selection at the same pinned commit."""
    native = card.music.native if card.music is not None else None
    assert native is not None
    encoder = native.text_encoder
    bundle = encoder.artifact_bundle
    assert bundle.root is not None
    root = f"{prefix}/{bundle.root}"
    files = tuple(
        item.model_copy(update={"path": f"{prefix}/{item.path}"})
        for item in bundle.files
    )
    for old, new in zip(bundle.files, files, strict=True):
        objects[encoder.repo, encoder.revision, new.path] = objects[
            encoder.repo, encoder.revision, old.path
        ]
    grouped = NativeMusicArtifactSource(
        repository=encoder.repo, revision=encoder.revision, files=files
    )
    replacement = encoder.model_copy(
        update={
            "artifact_bundle": grouped.download_bundle().model_copy(
                update={"root": root}
            )
        }
    )
    assert card.music is not None
    result = card.model_copy(
        update={
            "music": card.music.model_copy(
                update={
                    "native": native.model_copy(update={"text_encoder": replacement})
                }
            ),
            "model_id": ModelId("test/native-other") if other_model else card.model_id,
        }
    )
    return ModelCard.model_validate(result.model_dump())


async def _request(store: ModelStore, card: ModelCard) -> Path:
    status = await store.request_download(
        str(card.model_id),
        model_card=card,
        source_repository=str(card.artifact_repository),
        source_revision=card.source_revision,
    )
    await _finished(status)
    assert status.status == "complete", status.error
    path = store.get_store_path(str(card.model_id))
    assert path is not None
    return path


@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
async def test_shared_native_data_retains_both_owner_closures_without_refetch(
    signed: bool,
    reverse: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two custom or signed cards preserve/reopen shared bytes in either order."""
    first, objects = native_music_fixture()
    second = first.model_copy(update={"model_id": ModelId("test/native-other")})
    if signed:
        first = first.model_copy(
            update={"is_custom": False, "registry_card_id": "card_" + "a" * 52}
        )
        second = second.model_copy(
            update={"is_custom": False, "registry_card_id": "card_" + "b" * 52}
        )
    cards = (second, first) if reverse else (first, second)
    fetched = _network_fixture(monkeypatch, objects)
    store = ModelStore(tmp_path, recover_installed_cards=False)
    paths = [await _request(store, card) for card in cards]
    initial_fetches = len(fetched)
    assert (
        sum(repository == "test/native-components" for repository, _, _ in fetched)
        == 11
    )
    source = cards[0].native_music_artifact_sources()[-1]
    alias, _ = download_utils.companion_artifact_location(
        cards[0], source.repository, source.revision
    )
    component = store.get_store_path(str(alias))
    assert component is not None
    original_record = read_installed_card(component)
    assert original_record is not None and original_record.model_card == cards[0]
    reopened = ModelStore(tmp_path)
    monkeypatch.setattr(constants, "SKULK_MODELS_PATH", (tmp_path,))
    for card, primary in zip(cards, paths, strict=True):
        assert reopened.entry_matches_artifact(
            str(card.model_id),
            card.source_revision,
            str(card.artifact_repository),
            card,
        )
        assert download_utils.model_companions_present_on_disk(card)
        assert plan_preservation(primary).source_bytes > 0
        await _request(reopened, card)
    assert len(fetched) == initial_fetches
    assert read_installed_card(component) == original_record


@pytest.mark.parametrize("shared_primary", [False, True])
async def test_native_changed_selection_round_trip_retains_exact_generations(
    shared_primary: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing native roots at one commit cannot contaminate either generation."""
    first, objects = native_music_fixture(shared_primary=shared_primary)
    second = _other_selection(first, objects)
    fetched = _network_fixture(monkeypatch, objects)
    store = ModelStore(tmp_path, recover_installed_cards=False)
    first_path = await _request(store, first)
    first_record = read_installed_card(first_path)
    assert first_record is not None
    assert verify_installed_card(first_path, first_record, verify_hashes=True)
    second_path = await _request(store, second)
    second_record = read_installed_card(second_path)
    assert second_record is not None
    if shared_primary:
        assert second_path != first_path
    assert verify_installed_card(second_path, second_record, verify_hashes=True)
    monkeypatch.setattr(constants, "SKULK_MODELS_PATH", (tmp_path,))
    assert download_utils.model_companions_present_on_disk(second)
    assert plan_preservation(second_path).source_bytes > 0
    reopened = ModelStore(tmp_path)
    restored = await _request(reopened, first)
    assert reopened.entry_matches_artifact(
        str(first.model_id),
        first.source_revision,
        str(first.artifact_repository),
        first,
    )
    assert download_utils.model_companions_present_on_disk(first)
    assert plan_preservation(restored).source_bytes > 0
    assert fetched and all(path != "modeling_unselected.py" for _, _, path in fetched)


@pytest.mark.parametrize("different_selection", [False, True])
async def test_concurrent_native_owners_share_only_identical_source_tasks(
    different_selection: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shared exact source downloads deduplicate; different sources stay isolated."""
    first, objects = native_music_fixture()
    second = (
        _other_selection(first, objects, other_model=True)
        if different_selection
        else first.model_copy(update={"model_id": ModelId("test/native-other")})
    )
    fetched = _network_fixture(monkeypatch, objects)
    original = download_utils.download_file_with_retry
    entered, release = asyncio.Event(), asyncio.Event()

    async def held_download(
        model_id: ModelId,
        revision: str,
        path: str,
        target_dir: Path,
        on_progress: Callable[[int, int, bool], None] = lambda _a, _b, _c: None,
        on_connection_lost: Callable[[], None] = lambda: None,
        skip_internet: bool = False,
        expected_size: int | None = None,
        expected_object_id: str | None = None,
    ) -> Path:
        entered.set()
        await release.wait()
        return await original(
            model_id,
            revision,
            path,
            target_dir,
            on_progress=on_progress,
            on_connection_lost=on_connection_lost,
            skip_internet=skip_internet,
            expected_size=expected_size,
            expected_object_id=expected_object_id,
        )

    monkeypatch.setattr(download_utils, "download_file_with_retry", held_download)
    store = ModelStore(tmp_path, recover_installed_cards=False)
    pending = [asyncio.create_task(_request(store, card)) for card in (first, second)]
    try:
        with anyio.fail_after(5):
            await entered.wait()
            expected_sources = 2 if different_selection else 1
            while (
                sum(
                    status.artifact_role == "music_component"
                    for status in store._active_downloads.values()
                )
                < expected_sources
            ):
                await asyncio.sleep(0)
        release.set()
        await asyncio.gather(*pending)
    finally:
        release.set()
        await asyncio.gather(*pending, return_exceptions=True)
    assert set(fetched) == set(objects) and len(fetched) == len(objects)
    for card in (first, second):
        assert store.entry_matches_artifact(
            str(card.model_id),
            card.source_revision,
            str(card.artifact_repository),
            card,
        )
        primary = store.get_store_path(str(card.model_id))
        assert primary is not None and plan_preservation(primary).source_bytes > 0


@pytest.mark.parametrize("prefix", ["original", "metal"])
async def test_exact_native_companion_selection_survives_legacy_exclusions(
    prefix: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Actual direct fallback honors explicitly selected native repository paths."""
    from skulk.download.impl_shard_downloader import ResumableShardDownloader

    card, objects = native_music_fixture()
    card = _other_selection(card, objects, prefix=prefix)
    fetched = _network_fixture(monkeypatch, objects)
    monkeypatch.setattr(download_utils, "SKULK_MODELS_DIR", tmp_path)
    monkeypatch.setattr(constants, "SKULK_MODELS_PATH", (tmp_path,))
    shard, patterns, required = download_utils.companion_download_specs(card)[0]
    assert required
    await ResumableShardDownloader().ensure_shard(shard)
    assert set(fetched) == {
        (
            str(shard.model_card.artifact_repository),
            shard.model_card.source_revision,
            path,
        )
        for path in patterns
    }
    assert all(path != "modeling_unselected.py" for _, _, path in fetched)


async def test_both_native_owners_preserve_and_recover_shared_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each owner's real signed capture recovers its own complete offline closure."""
    first, objects = native_music_fixture()
    second = first.model_copy(update={"model_id": ModelId("test/native-other")})
    _network_fixture(monkeypatch, objects)
    store = ModelStore(tmp_path / "installed", recover_installed_cards=False)
    paths = [await _request(store, card) for card in (first, second)]
    key = tmp_path / "owner.pem"
    trust = create_key(key, None)
    for index, (card, primary) in enumerate(zip((first, second), paths, strict=True)):
        archive = tmp_path / f"archive-{index}"
        archive.mkdir()
        manifest = await asyncio.to_thread(
            preserve_installed, primary, archive, (load_key(key, None),)
        )
        home = tmp_path / f"recovered-{index}"
        await asyncio.to_thread(
            recover_home, archive, manifest, trust, home, accept_owner_capture=True
        )
        recovered = ModelStore(home / "model-store")
        entry = recovered.get_entry(str(card.model_id))
        assert entry is not None and entry.installed_card is not None
        effective = entry.installed_card.model_card
        assert effective.music == card.music
        monkeypatch.setattr(constants, "SKULK_MODELS_PATH", (home / "model-store",))
        assert download_utils.model_companions_present_on_disk(effective)
        for item in recovered.list_models():
            assert item.installed_card is not None
            assert verify_installed_card(
                home / "model-store" / item.store_path,
                item.installed_card,
                verify_hashes=True,
            )


@pytest.mark.parametrize("shared_primary", [False, True])
@pytest.mark.parametrize("failure", ["missing", "wrong_object", "cancel"])
async def test_failed_native_selection_keeps_previous_registered_generation(
    shared_primary: bool,
    failure: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed/cancelled replacement never damages the selected working model."""
    first, objects = native_music_fixture(shared_primary=shared_primary)
    _network_fixture(monkeypatch, objects)
    store = ModelStore(tmp_path, recover_installed_cards=False)
    primary = await _request(store, first)
    previous = store.get_entry(str(first.model_id))
    assert previous is not None and previous.installed_card is not None
    second = _other_selection(first, objects)
    native = second.music.native if second.music is not None else None
    assert native is not None
    encoder = native.text_encoder
    target = next(
        item.path
        for item in encoder.artifact_bundle.files
        if item.path.endswith("config.json")
    )
    if failure == "wrong_object":
        key = encoder.repo, encoder.revision, target
        objects[key] = b"!" * len(objects[key])
    _network_fixture(
        monkeypatch, objects, missing=target if failure == "missing" else None
    )
    original = download_utils.download_file_with_retry
    entered, release = asyncio.Event(), asyncio.Event()

    async def hold(
        repository: ModelId,
        revision: str,
        path: str,
        target_dir: Path,
        on_progress: Callable[[int, int, bool], None],
        expected_size: int | None = None,
        expected_object_id: str | None = None,
        **_kwargs: object,
    ) -> Path:
        if path.startswith("replacement/"):
            entered.set()
            await release.wait()
        return await original(
            repository,
            revision,
            path,
            target_dir,
            on_progress,
            expected_size=expected_size,
            expected_object_id=expected_object_id,
        )

    if failure == "cancel":
        monkeypatch.setattr(download_utils, "download_file_with_retry", hold)
    status = await store.request_download(
        str(second.model_id),
        model_card=second,
        source_repository=str(second.artifact_repository),
        source_revision=second.source_revision,
    )
    try:
        if failure == "cancel":
            with anyio.fail_after(5):
                await entered.wait()
            await store.cancel_download(str(second.model_id))
            assert status.status == "cancelled"
        else:
            await _finished(status)
            assert status.status == "failed" and status.error
        assert store.get_entry(str(first.model_id)) == previous
        assert verify_installed_card(
            primary, previous.installed_card, verify_hashes=True
        )
        monkeypatch.setattr(constants, "SKULK_MODELS_PATH", (tmp_path,))
        assert download_utils.model_companions_present_on_disk(first)
    finally:
        release.set()
        await asyncio.gather(
            *tuple(store._download_tasks_by_model.values()), return_exceptions=True
        )


async def test_legacy_native_source_can_be_adopted_without_rewriting_its_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """New aliases use independent verified links; deleting one cannot revive it."""
    card, objects = native_music_fixture()
    _install(tmp_path, card, objects)
    legacy = tmp_path / "test--native-components"
    old_record = read_installed_card(legacy)
    assert old_record is not None
    fetched = _network_fixture(monkeypatch, objects)
    store = ModelStore(tmp_path)
    source = next(
        source
        for source in card.native_music_artifact_sources()
        if source.repository != str(card.artifact_repository)
    )
    alias = str(source.model_id())
    status = await store.request_download(
        alias,
        source_repository=source.repository,
        source_revision=source.revision,
        model_card=card,
        artifact_role="music_component",
        owner_model_id=str(card.model_id),
        owner_card_id=card.registry_card_id,
    )
    await _finished(status)
    assert status.status == "complete" and not fetched
    adopted = store.get_store_path(alias)
    assert adopted is not None and adopted != legacy
    assert read_installed_card(legacy) == old_record
    assert store.delete_model(alias)
    assert verify_installed_card(legacy, old_record, verify_hashes=True)
    monkeypatch.setattr(constants, "SKULK_MODELS_PATH", (tmp_path,))
    monkeypatch.setattr(download_utils, "SKULK_MODELS_DIR", tmp_path / "empty")
    assert (
        download_utils.resolve_native_music_source_directory(
            card, source.repository, source.revision
        )
        is None
    )
    assert not store.entry_matches_artifact(
        str(card.model_id), card.source_revision, str(card.artifact_repository), card
    )


@pytest.mark.parametrize("shared_primary", [False, True])
async def test_completed_native_download_revalidates_lost_required_object(
    shared_primary: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated offline StartDownload refuses stale success, then survives restart."""
    from skulk.download import coordinator as coordinator_module
    from skulk.download.coordinator import DownloadCoordinator
    from skulk.download.impl_shard_downloader import ResumableShardDownloader
    from skulk.shared.types.commands import ForwarderDownloadCommand
    from skulk.shared.types.common import NodeId
    from skulk.shared.types.events import Event
    from skulk.shared.types.telemetry import NodeTelemetry
    from skulk.shared.types.worker.downloads import DownloadCompleted, DownloadFailed
    from skulk.shared.types.worker.shards import PipelineShardMetadata
    from skulk.utils.channels import channel

    card, objects = native_music_fixture(shared_primary=shared_primary)
    _network_fixture(monkeypatch, objects)
    store = ModelStore(tmp_path / "store", recover_installed_cards=False)
    await _request(store, card)
    monkeypatch.setattr(constants, "SKULK_MODELS_PATH", (tmp_path / "store",))
    monkeypatch.setattr(download_utils, "SKULK_MODELS_DIR", tmp_path / "empty")
    monkeypatch.setattr(coordinator_module, "SKULK_MODELS_DIR", tmp_path / "empty")
    _commands, incoming = channel[ForwarderDownloadCommand]()
    events, _received = channel[Event]()
    telemetry, _ = channel[NodeTelemetry]()
    coordinator = DownloadCoordinator(
        NodeId("native-repeat"),
        ResumableShardDownloader(offline=True),
        incoming,
        events,
        telemetry,
        offline=True,
    )
    shard = PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=card.n_layers,
        n_layers=card.n_layers,
    )
    await coordinator._start_download(shard)
    assert isinstance(coordinator.download_status[card.model_id], DownloadCompleted)
    source = card.native_music_artifact_sources()[-1]
    alias, _ = download_utils.companion_artifact_location(
        card, source.repository, source.revision
    )
    directory = store.get_store_path(str(alias))
    assert directory is not None
    item = source.files[-1]
    removed = directory / item.path
    removed.unlink()
    await coordinator._start_download(shard)
    assert isinstance(coordinator.download_status[card.model_id], DownloadFailed)
    assert not coordinator.active_downloads
    removed.write_bytes(objects[source.repository, source.revision, item.path])
    restarted = DownloadCoordinator(
        NodeId("native-repeat"),
        ResumableShardDownloader(offline=True),
        incoming,
        events,
        telemetry,
        offline=True,
    )
    await restarted._start_download(shard)
    assert isinstance(restarted.download_status[card.model_id], DownloadCompleted)


@pytest.mark.parametrize("forged_alias", [False, True])
async def test_native_component_http_binds_derived_alias_to_its_selection(
    forged_alias: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP source aliases cannot target unrelated canonical store entries."""
    from urllib.parse import quote

    card, objects = native_music_fixture()
    fetched = _network_fixture(monkeypatch, objects)
    source = next(
        source
        for source in card.native_music_artifact_sources()
        if source.repository != str(card.artifact_repository)
    )
    alias = "test/unrelated" if forged_alias else str(source.model_id())

    async def cards() -> list[ModelCard]:
        return [card]

    monkeypatch.setattr(server_module, "get_all_model_cards", cards)
    store = ModelStore(tmp_path, recover_installed_cards=False)
    server = ModelStoreServer(store, host="127.0.0.1", port=0)
    async with TestClient(TestServer(server._app)) as client:
        response = await client.post(
            f"/models/{quote(alias, safe='')}/download",
            json={
                "source_repository": source.repository,
                "source_revision": source.revision,
                "owner_model_id": str(card.model_id),
                "artifact_role": "music_component",
            },
        )
        assert response.status == (409 if forged_alias else 200), await response.text()
        if forged_alias:
            assert not fetched and store.get_entry(alias) is None
        else:
            status = store.get_download_status(alias)
            assert status is not None
            await _finished(status)
            assert status.status == "complete" and len(fetched) == 11


async def test_native_staging_keeps_shared_owners_independent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Actual store HTTP staging preserves both owner cards and one shared source."""
    from skulk.download.impl_shard_downloader import ResumableShardDownloader
    from skulk.shared.types.worker.shards import PipelineShardMetadata
    from skulk.store.config import StagingNodeConfig
    from skulk.store.model_store_client import ModelStoreClient, ModelStoreDownloader

    first, objects = native_music_fixture()
    second = first.model_copy(update={"model_id": ModelId("test/native-other")})
    fetched = _network_fixture(monkeypatch, objects)
    store = ModelStore(tmp_path / "store", recover_installed_cards=False)
    for card in (first, second):
        await _request(store, card)
    fetch_count = len(fetched)

    async def cards() -> list[ModelCard]:
        return [first, second]

    monkeypatch.setattr(server_module, "get_all_model_cards", cards)
    server = ModelStoreServer(store, host="127.0.0.1", port=0)
    staging = tmp_path / "staging"
    async with TestServer(server._app) as running:
        port = running.make_url("/").port
        assert isinstance(port, int)
        client = ModelStoreClient("127.0.0.1", port)
        downloader = ModelStoreDownloader(
            ResumableShardDownloader(),
            client,
            StagingNodeConfig(node_cache_path=str(staging)),
            allow_hf_fallback=False,
        )
        paths: list[Path] = []
        for card in (first, second):
            shard = PipelineShardMetadata(
                model_card=card,
                device_rank=0,
                world_size=1,
                start_layer=0,
                end_layer=card.n_layers,
                n_layers=card.n_layers,
            )
            paths.append(await downloader.ensure_shard(shard))
        for card, primary in zip((first, second), paths, strict=True):
            installed = read_installed_card(primary)
            assert installed is not None and installed.model_card == card
            assert verify_installed_card(primary, installed, verify_hashes=True)
            assert plan_preservation(primary).source_bytes > 0
        assert len(fetched) == fetch_count


@pytest.mark.parametrize("cached", [False, True])
async def test_native_metadata_lookup_keeps_unrelated_http_responsive(
    cached: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Held native cache metadata reads cannot block the coordinator event loop."""
    from threading import Event as ThreadEvent

    from skulk.download import coordinator as coordinator_module
    from skulk.download.coordinator import DownloadCoordinator
    from skulk.download.impl_shard_downloader import ResumableShardDownloader
    from skulk.shared.types.commands import ForwarderDownloadCommand
    from skulk.shared.types.common import NodeId
    from skulk.shared.types.events import Event
    from skulk.shared.types.telemetry import NodeTelemetry
    from skulk.shared.types.worker.shards import PipelineShardMetadata
    from skulk.utils.channels import channel

    card, objects = native_music_fixture()
    primary = _install(tmp_path, card, objects)
    monkeypatch.setattr(constants, "SKULK_MODELS_PATH", (tmp_path,))
    monkeypatch.setattr(download_utils, "SKULK_MODELS_DIR", tmp_path / "empty")
    _commands, incoming = channel[ForwarderDownloadCommand]()
    events, _received = channel[Event]()
    telemetry, _ = channel[NodeTelemetry]()
    coordinator = DownloadCoordinator(
        NodeId("native-held"),
        ResumableShardDownloader(offline=True),
        incoming,
        events,
        telemetry,
        offline=True,
    )
    shard = PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=card.n_layers,
        n_layers=card.n_layers,
    )
    if cached:
        await coordinator._start_download(shard)
    entered, release = ThreadEvent(), ThreadEvent()
    original = coordinator_module.resolve_native_music_source_directory

    def held(
        model_card: ModelCard,
        repository: str,
        revision: str,
        *,
        require_card_identity: bool = False,
    ) -> Path | None:
        entered.set()
        assert release.wait(5), "native metadata read was not released"
        return original(
            model_card,
            repository,
            revision,
            require_card_identity=require_card_identity,
        )

    monkeypatch.setattr(
        coordinator_module, "resolve_native_music_source_directory", held
    )
    server = ModelStoreServer(ModelStore(tmp_path), host="127.0.0.1", port=0)
    async with TestClient(TestServer(server._app)) as client:
        pending = asyncio.create_task(coordinator._start_download(shard))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            response = await client.get("/health")
            assert response.status == 200 and not pending.done()
        finally:
            release.set()
            await pending
    assert coordinator.download_status[card.model_id].model_directory == str(primary)


async def test_native_direct_reuse_preserves_shared_component_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct fallback retains the exact first source record for another owner."""
    from skulk.download.impl_shard_downloader import ResumableShardDownloader
    from skulk.shared.types.worker.shards import PipelineShardMetadata

    first, objects = native_music_fixture()
    second = first.model_copy(update={"model_id": ModelId("test/native-other")})
    _network_fixture(monkeypatch, objects)
    monkeypatch.setattr(constants, "SKULK_MODELS_PATH", (tmp_path,))
    monkeypatch.setattr(download_utils, "SKULK_MODELS_DIR", tmp_path)
    component = download_utils.companion_download_specs(first)[0][0].model_card.model_id
    old = None
    for card in (first, second):
        shard = PipelineShardMetadata(
            model_card=card,
            device_rank=0,
            world_size=1,
            start_layer=0,
            end_layer=card.n_layers,
            n_layers=card.n_layers,
        )
        await ResumableShardDownloader().ensure_shard(shard)
        observed = read_installed_card(tmp_path / component.normalize())
        assert observed is not None
        if old is None:
            old = observed
        else:
            assert observed == old
    for card in (first, second):
        assert plan_preservation(tmp_path / card.model_id.normalize()).source_bytes > 0


@pytest.mark.parametrize("shared_primary", [False, True])
@pytest.mark.parametrize("reopen", [False, True])
@pytest.mark.parametrize("corruption", ["git_blob", "forged_git_manifest", "tensor"])
async def test_native_cache_admission_repairs_same_size_corrupted_objects(
    shared_primary: bool,
    reopen: bool,
    corruption: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Neither completed-status nor reopened-index reuse can certify wrong bytes."""
    card, objects = native_music_fixture(shared_primary=shared_primary)
    card = card.model_copy(
        update={"is_custom": False, "registry_card_id": "card_" + "a" * 52}
    )
    _network_fixture(monkeypatch, objects)
    store = ModelStore(tmp_path, recover_installed_cards=False)
    await _request(store, card)
    source = card.native_music_artifact_sources()[-1]
    alias, _ = download_utils.companion_artifact_location(
        card, source.repository, source.revision
    )
    directory = store.get_store_path(str(alias))
    entry = store.get_entry(str(alias))
    assert (
        directory is not None and entry is not None and entry.installed_card is not None
    )
    expected = next(
        item
        for item in source.files
        if (
            item.path.endswith(".safetensors")
            if corruption == "tensor"
            else item.path.endswith("config.json")
        )
    )
    original = objects[source.repository, source.revision, expected.path]
    changed = (
        b"!" * len(original)
        if corruption == "tensor"
        else original.replace(b"true", b"null")
    )
    assert changed != original and len(changed) == len(original)
    target = directory / expected.path
    target.write_bytes(changed)
    if corruption == "forged_git_manifest":
        files = tuple(
            item.model_copy(update={"sha256": hashlib.sha256(changed).hexdigest()})
            if item.path == expected.path
            else item
            for item in entry.installed_card.files
        )
        forged = entry.installed_card.model_copy(
            update={"files": files, "manifest_sha256": manifest_sha256(files)}
        )
        write_installed_card(directory, forged)
        store.register_model(
            str(alias),
            directory,
            entry.files,
            entry.total_bytes,
            source_repository=source.repository,
            source_revision=source.revision,
            installed_card=forged,
        )
    if reopen:
        store = ModelStore(tmp_path)
    await _request(store, card)
    assert target.read_bytes() == original
    repaired = read_installed_card(directory)
    assert repaired is not None and verify_installed_card(
        directory, repaired, verify_hashes=True
    )


@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("lose_index", [False, True])
async def test_removing_historical_native_owner_preserves_surviving_model(
    signed: bool,
    lose_index: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deleting the first owner does not revoke independent shared data custody."""
    first, objects = native_music_fixture()
    second = first.model_copy(update={"model_id": ModelId("test/native-other")})
    if signed:
        first = first.model_copy(
            update={"is_custom": False, "registry_card_id": "card_" + "a" * 52}
        )
        second = second.model_copy(
            update={"is_custom": False, "registry_card_id": "card_" + "b" * 52}
        )
    fetched = _network_fixture(monkeypatch, objects)
    store = ModelStore(tmp_path, recover_installed_cards=False)
    await _request(store, first)
    survivor = await _request(store, second)
    before = len(fetched)
    source = next(
        item
        for item in first.native_music_artifact_sources()
        if item.repository != str(first.artifact_repository)
    )
    alias = str(source.model_id())
    shared = store.get_store_path(alias)
    assert shared is not None
    original_record = read_installed_card(shared)
    assert original_record is not None and original_record.owner_model_id == str(
        first.model_id
    )
    assert await store.delete_model_serialized(str(first.model_id))
    if lose_index:
        (tmp_path / "registry.json").unlink()
    reopened = ModelStore(tmp_path)
    assert reopened.get_entry(str(first.model_id)) is None
    assert reopened.get_entry(alias) is not None
    monkeypatch.setattr(constants, "SKULK_MODELS_PATH", (tmp_path,))
    monkeypatch.setattr(download_utils, "SKULK_MODELS_DIR", tmp_path / "empty")
    assert download_utils.model_companions_present_on_disk(second)
    assert plan_preservation(survivor).source_bytes > 0
    await _request(reopened, second)
    assert len(fetched) == before
    assert read_installed_card(shared) == original_record


@pytest.mark.parametrize("reopen", [False, True])
async def test_native_cached_integrity_reads_do_not_block_http_or_publish_early(
    reopen: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both cache-admission paths await real integrity reads outside the loop."""
    from threading import Event

    from skulk.store import model_store as store_module
    from skulk.store.installed_cards import InstalledCardRecord

    card, objects = native_music_fixture()
    _network_fixture(monkeypatch, objects)
    store = ModelStore(tmp_path, recover_installed_cards=False)
    await _request(store, card)
    if reopen:
        store = ModelStore(tmp_path)
    entered, release = Event(), Event()
    original = store_module.verify_installed_card

    def held(
        directory: Path, record: InstalledCardRecord, *, verify_hashes: bool = False
    ) -> bool:
        if verify_hashes:
            entered.set()
            assert release.wait(5), "native source integrity read was not released"
        return original(directory, record, verify_hashes=verify_hashes)

    monkeypatch.setattr(store_module, "verify_installed_card", held)
    server = ModelStoreServer(store, host="127.0.0.1", port=0)
    async with TestClient(TestServer(server._app)) as client:
        pending = asyncio.create_task(_request(store, card))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            response = await client.get("/health")
            assert response.status == 200 and not pending.done()
        finally:
            release.set()
            await pending


@pytest.mark.parametrize("reopen", [False, True])
async def test_native_cache_wait_does_not_block_transfer_cancel_or_fresh_admission(
    reopen: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real transfer owner stays cancellable while another cache request waits."""
    card, objects = native_music_fixture(shared_primary=True)
    transfer = _other_selection(card, objects, prefix="held", other_model=True)
    fresh = _other_selection(card, objects, prefix="fresh", other_model=True).model_copy(
        update={"model_id": ModelId("test/native-fresh")}
    )
    _network_fixture(monkeypatch, objects)
    store = ModelStore(tmp_path, recover_installed_cards=False)
    await _request(store, card)
    if reopen:
        store = ModelStore(tmp_path)
    original = download_utils.download_file_with_retry
    entered, release = asyncio.Event(), asyncio.Event()
    waiting = asyncio.Event()

    class ObservedTransferLock(asyncio.Lock):
        """Observe real contention without reading asyncio's private waiter list."""

        async def acquire(self) -> Literal[True]:
            if self.locked():
                waiting.set()
            return await super().acquire()

    store._download_transfer_lock = ObservedTransferLock()

    async def held(
        model_id: ModelId, revision: str, path: str, target_dir: Path,
        on_progress: Callable[[int, int, bool], None],
        *, expected_size: int | None = None, expected_object_id: str | None = None,
    ) -> Path:
        if path.startswith("held/"):
            entered.set()
            await release.wait()
        return await original(
            model_id, revision, path, target_dir, on_progress,
            expected_size=expected_size, expected_object_id=expected_object_id,
        )

    monkeypatch.setattr(download_utils, "download_file_with_retry", held)
    status = await store.request_download(
        str(transfer.model_id), model_card=transfer,
        source_repository=str(transfer.artifact_repository),
        source_revision=transfer.source_revision,
    )
    cached: asyncio.Task[Path] | None = None
    try:
        with anyio.fail_after(5):
            await entered.wait()
        cached = asyncio.create_task(_request(store, card))
        # Observe actual transfer-fence contention rather than a timed sleep.
        with anyio.fail_after(5):
            await waiting.wait()
        with anyio.fail_after(1):
            queued = await store.request_download(
                str(fresh.model_id), model_card=fresh,
                source_repository=str(fresh.artifact_repository),
                source_revision=fresh.source_revision,
            )
            assert queued.status == "pending"
            cancelled = await store.cancel_download(str(transfer.model_id))
        assert cancelled is status and status.status == "cancelled"
        await cached
        assert await store.cancel_download(str(fresh.model_id)) is not None or queued.status == "complete"
    finally:
        release.set()
        for pending in tuple(store._download_tasks):
            pending.cancel()
        await asyncio.gather(*store._download_tasks, return_exceptions=True)
        if cached is not None:
            if not cached.done():
                cached.cancel()
            await asyncio.gather(cached, return_exceptions=True)


@pytest.mark.parametrize("shared_primary", [False, True])
@pytest.mark.parametrize("reopen", [False, True])
@pytest.mark.parametrize("tensor", [False, True])
async def test_native_corrupt_cache_recovers_through_real_file_download(
    shared_primary: bool, reopen: bool, tensor: bool,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real HEAD/GET/resume/hash logic must replace a same-size damaged object."""
    card, objects = native_music_fixture(shared_primary=shared_primary)
    real_download = download_utils.download_file_with_retry
    _network_fixture(monkeypatch, objects)
    monkeypatch.setattr(download_utils, "download_file_with_retry", real_download)
    gets: list[tuple[str, str, str]] = []

    async def headers() -> dict[str, str]:
        return {}

    async def object_response(request: web.Request) -> web.Response:
        repository = request.match_info["owner"] + "/" + request.match_info["name"]
        key = repository, request.match_info["revision"], request.match_info["path"]
        data = objects.get(key)
        if data is None:
            return web.Response(status=404)
        digest = (
            hashlib.sha256(data).hexdigest()
            if key[2].endswith((".safetensors", ".pt"))
            else hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        )
        if request.method == "GET":
            gets.append(key)
        return web.Response(body=data, headers={"ETag": digest})

    application = web.Application()
    application.router.add_route(
        "*", "/{owner}/{name}/resolve/{revision}/{path:.*}", object_response
    )
    monkeypatch.setattr(download_utils, "get_download_headers", headers)
    async with TestServer(application) as hub:
        endpoint = str(hub.make_url("/")).rstrip("/")
        monkeypatch.setattr(download_utils, "get_hf_endpoint", lambda: endpoint)
        store = ModelStore(tmp_path, recover_installed_cards=False)
        await _request(store, card)
        assert len(gets) == len(objects)
        source = card.native_music_artifact_sources()[-1]
        item = next(
            file for file in source.files
            if (file.path.endswith(".safetensors") if tensor else file.path.endswith("config.json"))
        )
        alias, _ = download_utils.companion_artifact_location(
            card, source.repository, source.revision
        )
        directory = store.get_store_path(str(alias))
        assert directory is not None
        target = directory / item.path
        original = objects[source.repository, source.revision, item.path]
        target.write_bytes(bytes((original[0] ^ 1,)) + original[1:])
        gets.clear()
        if reopen:
            store = ModelStore(tmp_path)
        await _request(store, card)
        assert target.read_bytes() == original
        assert gets == [(source.repository, source.revision, item.path)]
        record = read_installed_card(directory)
        assert record is not None and verify_installed_card(
            directory, record, verify_hashes=True
        )


@pytest.mark.parametrize("offline", [False, True])
@pytest.mark.parametrize("tensor", [False, True])
@pytest.mark.parametrize("damage", [None, "same_size", "short"])
async def test_pinned_file_offline_reuse_keeps_original_identity_authoritative(
    offline: bool, tensor: bool, damage: str | None,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit offline and network-fallback paths cannot bless corrupted pins."""
    original = b"original pinned object"
    target = tmp_path / "object"
    data = (
        original if damage is None else original[:-1] if damage == "short"
        else bytes((original[0] ^ 1,)) + original[1:]
    )
    target.write_bytes(data)
    digest = (
        "sha256:" + hashlib.sha256(original).hexdigest() if tensor else
        "git-sha1:" + hashlib.sha1(f"blob {len(original)}\0".encode() + original).hexdigest()
    )

    async def missing_metadata(
        _model: ModelId, _revision: str, _path: str
    ) -> tuple[int, str]:
        raise ConnectionError("controlled unavailable Hub")

    monkeypatch.setattr(download_utils, "file_meta", missing_metadata)
    if damage is None:
        assert await download_utils._download_file(
            ModelId("test/pinned"), "a" * 40, "object", tmp_path,
            skip_internet=offline, expected_size=len(original), expected_object_id=digest,
        ) == target
    else:
        with pytest.raises(ValueError, match="object mismatch"):
            await download_utils._download_file(
                ModelId("test/pinned"), "a" * 40, "object", tmp_path,
                skip_internet=offline, expected_size=len(original), expected_object_id=digest,
            )
    assert target.read_bytes() == data


@pytest.mark.parametrize("reopen", [False, True])
async def test_cancelled_native_cache_check_keeps_its_fence_until_thread_drain(
    reopen: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repeated cancellation cannot release a live sidecar reader/writer's fence."""
    from threading import Event

    from skulk.store import model_store as store_module
    from skulk.store.installed_cards import InstalledCardRecord

    card, objects = native_music_fixture(shared_primary=True)
    fresh = _other_selection(card, objects, other_model=True)
    _network_fixture(monkeypatch, objects)
    store = ModelStore(tmp_path, recover_installed_cards=False)
    await _request(store, card)
    if reopen:
        store = ModelStore(tmp_path)
    entered, release = Event(), Event()
    original = store_module.verify_installed_card

    def held(
        directory: Path, record: InstalledCardRecord, *, verify_hashes: bool = False
    ) -> bool:
        if verify_hashes:
            entered.set()
            assert release.wait(5), "native cache reader was not released"
        return original(directory, record, verify_hashes=verify_hashes)

    monkeypatch.setattr(store_module, "verify_installed_card", held)
    pending = asyncio.create_task(_request(store, card))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        assert pending.cancel()
        await asyncio.sleep(0)
        assert not pending.done() and store._download_transfer_lock.locked()
        assert pending.cancel()
        await asyncio.sleep(0)
        assert not pending.done() and store._download_transfer_lock.locked()
        with anyio.fail_after(1):
            queued = await store.request_download(
                str(fresh.model_id), model_card=fresh,
                source_repository=str(fresh.artifact_repository),
                source_revision=fresh.source_revision,
            )
            assert queued.status == "pending"
            assert await store.cancel_download(str(fresh.model_id)) is queued
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert not store._download_transfer_lock.locked()
        assert store.entry_matches_artifact(
            str(card.model_id), card.source_revision, str(card.artifact_repository),
            card, verify_hashes=True,
        )
    finally:
        release.set()
        if not pending.done():
            pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)


async def test_native_peer_recovery_keeps_data_but_honors_actual_alias_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A surviving owner can recover bytes without restoring a deleted model."""
    card, objects = native_music_fixture()
    other = card.model_copy(update={"model_id": ModelId("test/native-other")})
    _network_fixture(monkeypatch, objects)
    source_store = ModelStore(tmp_path / "source", recover_installed_cards=False)
    source_primary = await _request(source_store, card)
    source = next(
        item
        for item in card.native_music_artifact_sources()
        if item.repository != str(card.artifact_repository)
    )
    alias = str(source.model_id())
    directory = source_store.get_store_path(alias)
    assert directory is not None
    record = read_installed_card(directory)
    base = read_installed_card(source_primary)
    assert record is not None and base is not None
    store = ModelStore(tmp_path / "target", recover_installed_cards=False)
    for owner in (card, other):
        await _request(store, owner)
    assert await store.delete_model_serialized(str(card.model_id))
    missing = store.get_store_path(alias)
    assert missing is not None
    shutil.rmtree(missing)

    async def export(request: web.Request) -> web.Response:
        assert request.headers["X-Skulk-Store-Node"] == "native-data-target"
        return web.Response(body=(directory / request.match_info["path"]).read_bytes())

    app = web.Application()
    app.router.add_get("/store/internal/exports/{token}/{path:.*}", export)
    async with TestServer(app) as server:
        options = dict(
            source_base_url=str(server.make_url("")),
            capability_token="native-data",
            target_node_id="native-data-target",
        )
        await store.import_peer_artifact(record, **options)
        recovered = ModelStore(tmp_path / "target")
        assert recovered.get_entry(str(card.model_id)) is None
        assert recovered.entry_matches_artifact(
            str(other.model_id),
            other.source_revision,
            str(other.artifact_repository),
            other,
        )
        with pytest.raises(ValueError, match="suppressed"):
            await store.import_peer_artifact(base, **options)
        assert await store.delete_model_serialized(alias)
        with pytest.raises(ValueError, match="suppressed"):
            await store.import_peer_artifact(record, **options)
