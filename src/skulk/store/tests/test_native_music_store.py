# pyright: reportPrivateUsage=false
"""Native music byte custody through real store admission and offline preservation."""

import asyncio
from collections.abc import Callable
from pathlib import Path

import anyio
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from foxlight_model_preservation.models import PreservationError
from foxlight_model_preservation.signing import create_key, load_key

from skulk.download import download_utils
from skulk.shared import constants
from skulk.shared.models.model_cards import ModelCard
from skulk.shared.tests.native_music_fixtures import native_music_fixture
from skulk.shared.types.common import ModelId
from skulk.shared.types.worker.downloads import FileListEntry
from skulk.store import model_store_server as server_module
from skulk.store.installed_cards import (
    build_installed_card_record,
    read_installed_card,
    verify_installed_card,
    write_installed_card,
)
from skulk.store.model_store import ModelStore, StoreDownloadStatus, StoreRegistryIndex
from skulk.store.model_store_client import (
    _declared_companion_files,
    _staged_directory_looks_complete,
)
from skulk.store.model_store_server import ModelStoreServer
from skulk.store.preservation import recover_home
from skulk.store.preservation_export import plan_preservation, preserve_installed


async def _finished(status: StoreDownloadStatus) -> None:
    with anyio.fail_after(5):
        while status.status in {"pending", "downloading"}:
            await asyncio.sleep(0.01)


def _network_fixture(
    monkeypatch: pytest.MonkeyPatch, objects: dict[tuple[str, str, str], bytes],
    *, missing: str | None = None,
) -> list[tuple[str, str, str]]:
    fetched: list[tuple[str, str, str]] = []
    async def listing(repository: ModelId, revision: str, **_kwargs: object) -> list[FileListEntry]:
        return [FileListEntry(type="file", path=path, size=len(data))
            for (repo, rev, path), data in objects.items()
            if (repo, rev) == (str(repository), revision) and path != missing
        ] + [FileListEntry(type="file", path="modeling_unselected.py", size=10)]

    async def download(repository: ModelId, revision: str, path: str, target_dir: Path,
        on_progress: Callable[[int, int, bool], None], expected_size: int | None = None,
        expected_object_id: str | None = None, **_kwargs: object) -> Path:
        key = str(repository), revision, path
        data = objects[key]
        assert expected_size == len(data) and expected_object_id is not None
        fetched.append(key)
        target = target_dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        on_progress(len(data), len(data), True)
        return target

    monkeypatch.setattr(download_utils, "fetch_file_list_with_cache", listing)
    monkeypatch.setattr(download_utils, "download_file_with_retry", download)
    return fetched


def _install(root: Path, card: ModelCard, objects: dict[tuple[str, str, str], bytes]) -> Path:
    for source in card.native_music_artifact_sources():
        primary = source.repository == str(card.artifact_repository)
        alias = card.model_id if primary else ModelId(source.repository)
        directory = root / alias.normalize()
        for item in source.files:
            target = directory / item.path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(objects[source.repository, source.revision, item.path])
        download_utils.write_source_revision_marker(directory, source.revision)
        record = build_installed_card_record(directory, card,
            artifact_role="base" if primary else "music_component",
            artifact_model_id=str(alias), artifact_repository=source.repository,
            artifact_revision=source.revision,
            owner_model_id=None if primary else str(card.model_id),
            owner_card_id=None if primary else card.registry_card_id,
        )
        write_installed_card(directory, record)
    return root / card.model_id.normalize()


@pytest.mark.parametrize("shared_primary", [False, True])
async def test_store_download_completes_only_after_all_native_files_and_reopens(
    shared_primary: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One base Download request acquires every source and repairs a lost component."""
    card, objects = native_music_fixture(shared_primary=shared_primary)
    fetched = _network_fixture(monkeypatch, objects)
    store = ModelStore(tmp_path, recover_installed_cards=False)

    async def request() -> None:
        status = await store.request_download(str(card.model_id), model_card=card,
            source_repository=str(card.artifact_repository), source_revision=card.source_revision)
        await _finished(status)
        assert status.status == "complete", status.error

    await request()
    assert set(fetched) == set(objects) and len(fetched) == 14
    await request()
    assert len(fetched) == 14
    store = ModelStore(tmp_path)
    assert store.entry_matches_artifact(str(card.model_id), card.source_revision,
        str(card.artifact_repository), card)
    monkeypatch.setattr(constants, "SKULK_MODELS_PATH", (tmp_path,))
    assert download_utils.model_companions_present_on_disk(card)
    source = card.native_music_artifact_sources()[-1]
    alias = str(card.model_id) if shared_primary else source.repository
    directory = store.get_store_path(alias)
    assert directory is not None
    removed = directory / source.files[-1].path
    removed.unlink()
    assert not store.entry_matches_artifact(str(card.model_id), card.source_revision,
        str(card.artifact_repository), card)
    await request()
    assert removed.is_file()
    assert store.entry_matches_artifact(str(card.model_id), card.source_revision,
        str(card.artifact_repository), card)


@pytest.mark.parametrize("shared_primary", [False, True])
@pytest.mark.parametrize("signed,refresh", [(False, False), (True, False), (True, True)])
async def test_offline_download_reopens_indexed_native_store_without_direct_cache(
    shared_primary: bool, signed: bool, refresh: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ordinary offline Download must reuse the exact complete reopened store."""
    from skulk.download import coordinator as coordinator_module
    from skulk.download.coordinator import DownloadCoordinator
    from skulk.download.impl_shard_downloader import ResumableShardDownloader
    from skulk.shared.types.commands import ForwarderDownloadCommand
    from skulk.shared.types.common import NodeId
    from skulk.shared.types.events import Event
    from skulk.shared.types.telemetry import NodeTelemetry
    from skulk.shared.types.worker.downloads import DownloadCompleted
    from skulk.shared.types.worker.shards import PipelineShardMetadata
    from skulk.utils.channels import channel

    card, objects = native_music_fixture(shared_primary=shared_primary)
    if signed:
        card = card.model_copy(update={"is_custom": False, "registry_card_id": "card_" + "a" * 52})
    _network_fixture(monkeypatch, objects)
    root = tmp_path / "store"
    store = ModelStore(root, recover_installed_cards=False)
    status = await store.request_download(str(card.model_id), model_card=card,
        source_repository=str(card.artifact_repository), source_revision=card.source_revision)
    await _finished(status)
    assert status.status == "complete", status.error
    reopened = ModelStore(root)
    primary = reopened.get_store_path(str(card.model_id))
    assert primary is not None and primary.name != card.model_id.normalize()
    if refresh:
        card = card.model_copy(update={"registry_card_id": "card_" + "b" * 52})
    direct = tmp_path / "direct"
    monkeypatch.setattr(constants, "SKULK_MODELS_PATH", (root,))
    monkeypatch.setattr(download_utils, "SKULK_MODELS_DIR", direct)
    monkeypatch.setattr(coordinator_module, "SKULK_MODELS_DIR", direct)

    async def no_network(*_args: object, **_kwargs: object) -> list[FileListEntry]:
        raise AssertionError("a complete offline store must not probe the upstream repository")

    monkeypatch.setattr(download_utils, "fetch_file_list_with_cache", no_network)
    _commands, incoming = channel[ForwarderDownloadCommand]()
    events, _received = channel[Event]()
    telemetry, _ = channel[NodeTelemetry]()
    coordinator = DownloadCoordinator(NodeId("native-store-reopen"),
        ResumableShardDownloader(offline=True), incoming, events, telemetry, offline=True)
    shard = PipelineShardMetadata(model_card=card, device_rank=0, world_size=1,
        start_layer=0, end_layer=card.n_layers, n_layers=card.n_layers)
    await coordinator._start_download(shard)
    completed = coordinator.download_status[card.model_id]
    assert isinstance(completed, DownloadCompleted)
    assert completed.model_directory == str(primary) and completed.read_only
    assert not coordinator.active_downloads and not direct.exists()
    record = read_installed_card(primary)
    assert record is not None and record.model_card.registry_card_id == card.registry_card_id


@pytest.mark.parametrize("signed", [False, True])
def test_native_primary_lookup_skips_data_equivalent_stale_card_policy(
    signed: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exact primary admission searches past stale policy without borrowing its trust."""
    card, objects = native_music_fixture()
    if signed:
        card = card.model_copy(update={"is_custom": False, "registry_card_id": "card_" + "a" * 52})
        stale = card.model_copy(update={"registry_card_id": "card_" + "b" * 52})
    else:
        assert card.music is not None
        stale = card.model_copy(update={"music": card.music.model_copy(
            update={"max_seconds": card.music.max_seconds - 1})})
    _install(tmp_path / "first", stale, objects)
    complete = _install(tmp_path / "second", card, objects)
    monkeypatch.setattr(constants, "SKULK_MODELS_PATH", (tmp_path / "first", tmp_path / "second"))
    monkeypatch.setattr(download_utils, "SKULK_MODELS_DIR", tmp_path / "unused")
    assert card.source_revision is not None
    assert download_utils.resolve_native_music_source_directory(
        card, str(card.artifact_repository), card.source_revision, require_card_identity=True,
    ) == complete
    (complete / ".skulk/installed-card.json").unlink()
    assert download_utils.resolve_native_music_source_directory(
        card, str(card.artifact_repository), card.source_revision, require_card_identity=True,
    ) is None


async def test_missing_native_component_fails_only_its_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing VAE data cannot register a base or leave its request falsely complete."""
    card, objects = native_music_fixture()
    fetched = _network_fixture(monkeypatch, objects, missing="vae/diffusion_pytorch_model.safetensors")
    store = ModelStore(tmp_path, recover_installed_cards=False)
    status = await store.request_download(str(card.model_id), model_card=card,
        source_repository=str(card.artifact_repository), source_revision=card.source_revision)
    await _finished(status)
    assert status.status == "failed" and status.error and "Required native music source" in status.error
    assert not fetched and not store.is_in_store(str(card.model_id))


async def test_cancelled_native_base_does_not_complete_or_cancel_a_shared_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancellation retires the parent request and leaves ordinary child custody reusable."""
    card, objects = native_music_fixture()
    fetched = _network_fixture(monkeypatch, objects)
    original_download = download_utils.download_file_with_retry
    entered = asyncio.Event()
    release = asyncio.Event()

    async def wait_for_release(repository: ModelId, revision: str, path: str,
        target_dir: Path, on_progress: Callable[[int, int, bool], None],
        expected_size: int | None = None, expected_object_id: str | None = None,
        **_kwargs: object) -> Path:
        entered.set()
        await release.wait()
        return await original_download(repository, revision, path, target_dir,
            on_progress, expected_size=expected_size, expected_object_id=expected_object_id)

    monkeypatch.setattr(download_utils, "download_file_with_retry", wait_for_release)
    store = ModelStore(tmp_path, recover_installed_cards=False)
    status = await store.request_download(str(card.model_id), model_card=card,
        source_repository=str(card.artifact_repository), source_revision=card.source_revision)
    try:
        with anyio.fail_after(5):
            await entered.wait()
        assert status.status != "complete" and not store.is_in_store(str(card.model_id))
        assert await store.cancel_download(str(card.model_id)) is status
        assert status.status == "cancelled" and not fetched
    finally:
        release.set()
        child = store.get_download_status("test/native-components")
        if child is not None:
            await _finished(child)
    assert child is not None and child.status == "complete"
    assert len(fetched) == 11 and not store.is_in_store(str(card.model_id))
    retry = await store.request_download(str(card.model_id), model_card=card,
        source_repository=str(card.artifact_repository), source_revision=card.source_revision)
    await _finished(retry)
    assert retry.status == "complete" and len(fetched) == 14


@pytest.mark.parametrize("revision,role,expected_status", [
    ("b" * 40, "music_component", 200),
    ("c" * 40, "music_component", 409),
    ("b" * 40, "image_component", 409),
])
async def test_native_component_http_request_binds_owner_role_and_revision(
    revision: str, role: str, expected_status: int,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Actual wire admission verifies a grouped music source against its trusted owner."""
    card, objects = native_music_fixture()
    fetched = _network_fixture(monkeypatch, objects)
    async def cards() -> list[ModelCard]:
        return [card]
    monkeypatch.setattr(server_module, "get_all_model_cards", cards)
    store = ModelStore(tmp_path, recover_installed_cards=False)
    server = ModelStoreServer(store, host="127.0.0.1", port=0)
    async with TestClient(TestServer(server._app)) as client:
        response = await client.post("/models/test%2Fnative-components/download", json={
            "source_repository": "test/native-components", "source_revision": revision,
            "owner_model_id": str(card.model_id), "artifact_role": role,
        })
        assert response.status == expected_status, await response.text()
        if expected_status != 200:
            assert not fetched and store.get_download_status("test/native-components") is None
            return
        status = store.get_download_status("test/native-components")
        assert status is not None
        await _finished(status)
        assert status.status == "complete", status.error
    assert len(fetched) == 11


async def test_native_base_http_download_acquires_the_complete_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ordinary public store Download route includes required native companions."""
    card, objects = native_music_fixture()
    fetched = _network_fixture(monkeypatch, objects)
    async def cards() -> list[ModelCard]:
        return [card]
    monkeypatch.setattr(server_module, "get_all_model_cards", cards)
    store = ModelStore(tmp_path, recover_installed_cards=False)
    server = ModelStoreServer(store, host="127.0.0.1", port=0)
    async with TestClient(TestServer(server._app)) as client:
        response = await client.post("/models/test%2Fnative-music/download", json={
            "source_repository": str(card.artifact_repository), "source_revision": card.source_revision,
            "artifact_bundle_id": card.artifact_bundle.bundle_id if card.artifact_bundle else None,
        })
        assert response.status == 200, await response.text()
        status = store.get_download_status(str(card.model_id))
        assert status is not None
        await _finished(status)
        assert status.status == "complete", status.error
    assert len(fetched) == 14 and set(fetched) == set(objects)


async def test_native_peer_record_verifies_its_grouped_source_against_signed_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A peer cannot keep a signed owner id while altering its role or source revision."""
    card, objects = native_music_fixture()
    card = card.model_copy(update={"is_custom": False, "registry_card_id": "card_" + "a" * 52})
    primary = _install(tmp_path, card, objects)
    record = read_installed_card(primary.parent / "test--native-components")
    assert record is not None and record.verification == "registry_verified"
    async def lookup(_card_id: str, **_kwargs: object) -> None:
        return None
    monkeypatch.setattr(server_module, "get_registry_card_by_id", lookup)
    monkeypatch.setattr(server_module, "get_current_registry_cards", lambda: [card])
    await server_module._require_trusted_registry_peer_record(record)
    for update in ({"artifact_revision": "c" * 40}, {"artifact_role": "image_component"},
        {"owner_card_id": "card_" + "b" * 52}):
        with pytest.raises(web.HTTPConflict):
            await server_module._require_trusted_registry_peer_record(record.model_copy(update=update))


@pytest.mark.parametrize("shared_primary", [False, True])
def test_native_preservation_round_trip_reopens_complete_closure_offline(
    shared_primary: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real signed export/recovery retains all roles and their exact loader roots."""
    card, objects = native_music_fixture(shared_primary=shared_primary)
    primary = _install(tmp_path / "installed", card, objects)
    archive = tmp_path / "archive"
    archive.mkdir()
    key = tmp_path / "owner.pem"
    trust = create_key(key, None)
    plan = plan_preservation(primary)
    assert len(plan.artifacts) == (1 if shared_primary else 2)
    manifest = preserve_installed(primary, archive, (load_key(key, None),))
    home = tmp_path / "recovered"
    recover_home(archive, manifest, trust, home, accept_owner_capture=True)
    store = ModelStore(home / "model-store")
    for entry in store.list_models():
        assert entry.installed_card is not None
        assert verify_installed_card(home / "model-store" / entry.store_path,
            entry.installed_card, verify_hashes=True)
    monkeypatch.setattr(constants, "SKULK_MODELS_PATH", (home / "model-store",))
    recovered = store.get_entry(str(card.model_id))
    assert recovered and recovered.installed_card
    recovered_card = recovered.installed_card.model_card
    assert recovered_card.music == card.music
    assert download_utils.model_companions_present_on_disk(recovered_card)
    assert store.entry_matches_artifact(str(card.model_id), card.source_revision,
        str(card.artifact_repository), recovered_card)


def test_native_preservation_refuses_an_omitted_component(tmp_path: Path) -> None:
    """A valid old sidecar cannot authorize export after its VAE bytes disappeared."""
    card, objects = native_music_fixture()
    primary = _install(tmp_path / "installed", card, objects)
    (primary.parent / "test--native-components/vae/config.json").unlink()
    archive = tmp_path / "archive"
    archive.mkdir()
    key = tmp_path / "owner.pem"
    create_key(key, None)
    with pytest.raises((PreservationError, FileNotFoundError)):
        preserve_installed(primary, archive, (load_key(key, None),))


def test_native_staging_and_installed_manifest_require_every_group_file(tmp_path: Path) -> None:
    """Staging refuses a lone weight or a record edited down to primary-only evidence."""
    card, objects = native_music_fixture()
    shard, _patterns, _required = download_utils.companion_download_specs(card)[0]
    expected = _declared_companion_files(shard, card, "music_component")
    assert len(expected) == 11
    primary = _install(tmp_path / "installed", card, objects)
    companion = primary.parent / "test--native-components"
    assert _staged_directory_looks_complete(companion, shard.model_card, expected)
    record = read_installed_card(companion)
    assert record is not None
    incomplete = record.model_copy(update={"files": record.files[:-1]})
    assert not verify_installed_card(companion, incomplete)
    (companion / "vae/config.json").unlink()
    assert not _staged_directory_looks_complete(companion, shard.model_card, expected)


@pytest.mark.parametrize("earlier_copy", ["missing_file", "missing_record", "invalid_record", "wrong_revision"])
def test_native_source_resolver_requires_one_complete_cache_generation(
    earlier_copy: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prefer an intact later cache, but never combine two complementary partial copies."""
    card, objects = native_music_fixture()
    first = tmp_path / "first"
    second = tmp_path / "second"
    _install(first, card, objects)
    _install(second, card, objects)
    monkeypatch.setattr(constants, "SKULK_MODELS_PATH", (first, second))
    monkeypatch.setattr(download_utils, "SKULK_MODELS_DIR", tmp_path / "unused")
    earlier = first / "test--native-components"
    if earlier_copy == "missing_file":
        (earlier / "Qwen3-Embedding-0.6B/config.json").unlink()
    elif earlier_copy == "missing_record":
        (earlier / ".skulk/installed-card.json").unlink()
    elif earlier_copy == "invalid_record":
        (earlier / ".skulk/installed-card.json").write_text("{}")
    else:
        download_utils.write_source_revision_marker(earlier, "c" * 40)
    assert download_utils.resolve_native_music_source_directory(
        card, "test/native-components", "b" * 40
    ) == second / "test--native-components"
    (second / "test--native-components/vae/config.json").unlink()
    assert download_utils.resolve_native_music_source_directory(
        card, "test/native-components", "b" * 40
    ) is None


@pytest.mark.parametrize("path", ["model.safetensors", "config.json"])
def test_native_installed_record_rejects_same_size_wrong_objects(path: str, tmp_path: Path) -> None:
    """Both tensor SHA-256 and metadata Git object pins bind actual captured bytes."""
    card, objects = native_music_fixture()
    primary = _install(tmp_path, card, objects)
    target = primary / path
    target.write_bytes(b"!" * target.stat().st_size)
    with pytest.raises(ValueError, match="complete exact source selection"):
        build_installed_card_record(primary, card)


@pytest.mark.parametrize("invalid_hint", ["escaped_path", "wrong_revision"])
def test_native_source_resolver_rejects_untrusted_store_index_hints(
    invalid_hint: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An index cannot redirect a native source outside its root or change its pin."""
    card, objects = native_music_fixture()
    first = tmp_path / "first"
    _install(first, card, objects)
    _install(tmp_path / "second", card, objects)
    store = ModelStore(first)
    entry = store.get_entry("test/native-components")
    assert entry is not None
    if invalid_hint == "escaped_path":
        (first / "test--native-components/.skulk/installed-card.json").unlink()
        update = {"store_path": "../second/test--native-components"}
    else:
        (first / "test--native-components").rename(first / "indexed-native-source")
        update = {"store_path": "indexed-native-source", "source_revision": "c" * 40}
    modified = entry.model_copy(update=update)
    (first / "registry.json").write_text(StoreRegistryIndex(entries={entry.model_id: modified}).model_dump_json())
    monkeypatch.setattr(constants, "SKULK_MODELS_PATH", (first,))
    monkeypatch.setattr(download_utils, "SKULK_MODELS_DIR", tmp_path / "unused")
    assert download_utils.resolve_native_music_source_directory(
        card, "test/native-components", "b" * 40
    ) is None


@pytest.mark.parametrize("extra", ["modeling_remote.py", ".foxlight-capture-notice.txt"])
def test_native_installed_record_rejects_unselected_or_forged_receiver_files(
    extra: str, tmp_path: Path
) -> None:
    """The preservation notice exception must not admit arbitrary repository content."""
    card, objects = native_music_fixture()
    primary = _install(tmp_path, card, objects)
    (primary / extra).write_text("unselected or forged content")
    with pytest.raises(ValueError, match="complete exact source selection"):
        build_installed_card_record(primary, card)
