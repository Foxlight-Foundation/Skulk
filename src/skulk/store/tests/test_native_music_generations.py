# pyright: reportPrivateUsage=false
"""Native source generations preserve independent owners and exact selections."""

import asyncio
from collections.abc import Callable
from pathlib import Path

import anyio
import pytest
from aiohttp.test_utils import TestClient, TestServer
from foxlight_model_preservation.signing import create_key, load_key

from skulk.download import download_utils
from skulk.shared import constants
from skulk.shared.models.model_cards import ModelCard, NativeMusicArtifactSource
from skulk.shared.tests.native_music_fixtures import native_music_fixture
from skulk.shared.types.common import ModelId
from skulk.store import model_store_server as server_module
from skulk.store.installed_cards import read_installed_card, verify_installed_card
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
