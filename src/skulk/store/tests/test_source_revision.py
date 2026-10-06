# pyright: reportPrivateUsage=false
"""Immutable source-revision behavior for store downloads and staging."""

import asyncio
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import cast

import pytest

import skulk.shared.constants as constants
from skulk.download import download_utils
from skulk.shared.models.model_cards import (
    ArtifactBundleConfig,
    ArtifactBundleFile,
    ModelCard,
    ModelId,
    ModelTask,
)
from skulk.shared.types.memory import Memory
from skulk.shared.types.worker.downloads import FileListEntry
from skulk.store import model_store as model_store_module
from skulk.store.installed_cards import InstalledCardRecord
from skulk.store.model_store import ModelStore, StoreDownloadStatus
from skulk.store.model_store_client import (
    ModelStoreClient,
    _staged_source_revision_matches,
    _staging_dir,
)

_OLD_REVISION = "0" * 40
_NEW_REVISION = "1" * 40


@pytest.mark.parametrize("model_id", ["", ".", "..", "org\\..\\model"])
def test_staging_directory_rejects_path_like_model_ids(
    tmp_path: Path, model_id: str
) -> None:
    """Revision replacement can only delete a child of the staging root."""
    with pytest.raises(ValueError, match="safe staging directory"):
        _staging_dir(str(tmp_path / "staging"), model_id)


async def test_store_alias_download_reads_from_artifact_repository(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The canonical store key must not become the Hugging Face origin."""
    monkeypatch.setattr(model_store_module, "MINIMUM_STAGING_FREE_DISK_BYTES", 0)
    store = ModelStore(tmp_path)
    alias = "org/multi@q4-k-m"
    repository = ModelId("org/multi")
    observed: list[ModelId] = []
    store._active_downloads[alias] = StoreDownloadStatus(model_id=alias)

    async def file_list(
        model_id: ModelId,
        _revision: str,
        recursive: bool,
    ) -> list[FileListEntry]:
        assert recursive
        observed.append(model_id)
        return [FileListEntry(type="file", path="config.json", size=2)]

    async def download_file(
        model_id: ModelId,
        _revision: str,
        path: str,
        download_dir: Path,
        on_progress: Callable[[int, int, bool], None],
        *_: object,
        **__: object,
    ) -> Path:
        observed.append(model_id)
        target = download_dir / path
        target.write_bytes(b"{}")
        on_progress(2, 2, True)
        return target

    monkeypatch.setattr(download_utils, "fetch_file_list_with_cache", file_list)
    monkeypatch.setattr(download_utils, "download_file_with_retry", download_file)

    await store._do_download(alias, source_repository=str(repository))

    assert observed == [repository, repository]
    entry = store.get_entry(alias)
    assert entry is not None
    assert entry.source_repository == str(repository)
    assert store.get_entry(str(repository)) is None


async def test_exact_text_bundle_ignores_unselected_repository_projector(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Repo-wide legacy projector inference cannot invalidate a signed bundle."""

    monkeypatch.setattr(model_store_module, "MINIMUM_STAGING_FREE_DISK_BYTES", 0)
    store = ModelStore(tmp_path)
    model_id = "org/text-gguf@q4"
    revision = "a" * 40
    card = ModelCard(
        model_id=ModelId(model_id),
        source_repository=ModelId("org/text-gguf"),
        source_revision=revision,
        storage_size=Memory.from_bytes(7),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        gguf_file="model-Q4_K_M.gguf",
        artifact_bundle=ArtifactBundleConfig(
            bundle_id=f"bundle_{'a' * 52}",
            files=(
                ArtifactBundleFile(path="model-Q4_K_M.gguf", size_bytes=7),
            ),
            download_size=7,
        ),
    )
    store._active_downloads[model_id] = StoreDownloadStatus(model_id=model_id)

    async def file_list(
        _model_id: ModelId,
        _revision: str,
        recursive: bool,
    ) -> list[FileListEntry]:
        assert recursive
        return [
            FileListEntry(type="file", path="model-Q4_K_M.gguf", size=7),
            FileListEntry(type="file", path="mmproj-F16.gguf", size=5),
        ]

    async def download_file(
        _model_id: ModelId,
        _revision: str,
        path: str,
        download_dir: Path,
        on_progress: Callable[[int, int, bool], None],
        *_: object,
        **__: object,
    ) -> Path:
        target = download_dir / path
        target.write_bytes(b"weights")
        on_progress(7, 7, True)
        return target

    monkeypatch.setattr(download_utils, "fetch_file_list_with_cache", file_list)
    monkeypatch.setattr(download_utils, "download_file_with_retry", download_file)

    await store._do_download(
        model_id,
        pinned_gguf=card.gguf_file,
        source_revision=revision,
        source_repository=str(card.artifact_repository),
        model_card=card,
    )

    entry = store.get_entry(model_id)
    assert entry is not None
    assert set(entry.files) == {
        ".skulk-source-revision",
        ".skulk/installed-card.json",
        "model-Q4_K_M.gguf",
    }
    assert "mmproj-F16.gguf" not in entry.files


def test_canonical_capacity_counts_resumable_and_replaced_bytes(
    tmp_path: Path,
) -> None:
    target = tmp_path / "org--model"
    target.mkdir()
    (target / "complete.gguf").write_bytes(b"x" * 10)
    (target / "replace.gguf").write_bytes(b"x" * 3)
    (target / "resume.gguf.partial").write_bytes(b"x" * 4)
    files = [
        FileListEntry(type="file", path="complete.gguf", size=10),
        FileListEntry(type="file", path="replace.gguf", size=8),
        FileListEntry(type="file", path="resume.gguf", size=9),
        FileListEntry(type="file", path="new.gguf", size=7),
    ]

    assert (
        model_store_module._remaining_store_download_bytes(target, files) == 5 + 5 + 7
    )


def test_canonical_capacity_does_not_credit_hardlinked_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "org--model"
    staging = tmp_path / "staging"
    target.mkdir()
    staging.mkdir()
    canonical = target / "replace.gguf"
    canonical.write_bytes(b"x" * 3)
    (staging / "replace.gguf").hardlink_to(canonical)

    assert canonical.stat().st_nlink == 2
    assert (
        model_store_module._remaining_store_download_bytes(
            target,
            [FileListEntry(type="file", path="replace.gguf", size=8)],
        )
        == 8
    )


def test_canonical_capacity_rejects_unknown_manifest_sizes(tmp_path: Path) -> None:
    with pytest.raises(
        model_store_module.ModelStoreCapacityError, match="unknown size"
    ):
        model_store_module._remaining_store_download_bytes(
            tmp_path,
            [FileListEntry(type="file", path="missing.safetensors", size=None)],
        )


async def test_canonical_download_fails_before_transfer_without_headroom(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = ModelStore(tmp_path)
    model_id = "org/model"
    store._active_downloads[model_id] = StoreDownloadStatus(model_id=model_id)
    transfer_started = False

    async def file_list(
        _model_id: ModelId,
        _revision: str,
        recursive: bool,
    ) -> list[FileListEntry]:
        assert recursive
        return [FileListEntry(type="file", path="model.gguf", size=60)]

    async def download_file(*_args: object, **_kwargs: object) -> Path:
        nonlocal transfer_started
        transfer_started = True
        raise AssertionError("unsafe canonical transfer must not start")

    class _DiskUsage:
        free = 50

    def disk_usage(_path: Path) -> _DiskUsage:
        return _DiskUsage()

    monkeypatch.setattr(download_utils, "fetch_file_list_with_cache", file_list)
    monkeypatch.setattr(download_utils, "download_file_with_retry", download_file)
    monkeypatch.setattr(model_store_module, "MINIMUM_STAGING_FREE_DISK_BYTES", 100)
    monkeypatch.setattr(
        model_store_module.shutil,
        "disk_usage",
        disk_usage,
    )

    await store._do_download(model_id)

    status = store._active_downloads[model_id]
    assert status.status == "failed"
    assert status.error is not None
    assert "ModelStoreCapacityError" in status.error
    assert not transfer_started


async def test_canonical_download_reuses_complete_artifact_without_reserve(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = ModelStore(tmp_path)
    model_id = "org/model"
    target = tmp_path / "org--model"
    target.mkdir()
    (target / "model.gguf").write_bytes(b"complete")
    store._active_downloads[model_id] = StoreDownloadStatus(model_id=model_id)

    async def file_list(
        _model_id: ModelId,
        _revision: str,
        recursive: bool,
    ) -> list[FileListEntry]:
        assert recursive
        return [FileListEntry(type="file", path="model.gguf", size=len(b"complete"))]

    async def reuse_file(
        _model_id: ModelId,
        _revision: str,
        path: str,
        target_dir: Path,
        _on_progress: Callable[[int, int, bool], None],
        *_args: object,
        **_kwargs: object,
    ) -> Path:
        return target_dir / path

    def unexpected_disk_usage(_path: Path) -> object:
        raise AssertionError("zero-byte canonical reuse must not inspect free space")

    monkeypatch.setattr(download_utils, "fetch_file_list_with_cache", file_list)
    monkeypatch.setattr(download_utils, "download_file_with_retry", reuse_file)
    monkeypatch.setattr(
        model_store_module.shutil,
        "disk_usage",
        unexpected_disk_usage,
    )

    await store._do_download(model_id)

    assert store._active_downloads[model_id].status == "complete"
    assert store.get_entry(model_id) is not None


async def test_canonical_download_stays_pending_until_transfer_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = ModelStore(tmp_path)
    model_id = "org/queued"
    status = StoreDownloadStatus(model_id=model_id)
    store._active_downloads[model_id] = status
    manifest_resolved = asyncio.Event()

    async def file_list(
        _model_id: ModelId,
        _revision: str,
        recursive: bool,
    ) -> list[FileListEntry]:
        assert recursive
        manifest_resolved.set()
        return [FileListEntry(type="file", path="model.gguf", size=None)]

    monkeypatch.setattr(download_utils, "fetch_file_list_with_cache", file_list)
    await store._download_transfer_lock.acquire()
    download_task = asyncio.create_task(store._do_download(model_id))
    try:
        await asyncio.wait_for(manifest_resolved.wait(), timeout=1)
        await asyncio.sleep(0)
        assert status.status == "pending"
    finally:
        store._download_transfer_lock.release()
        await asyncio.wait_for(download_task, timeout=1)

    assert status.status == "failed"
    assert status.error is not None
    assert "ModelStoreCapacityError" in status.error


def test_external_pinned_registration_writes_loadable_revision_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Externally registered pins must satisfy runner path resolution."""

    model_dir = tmp_path / "org--model"
    model_dir.mkdir()
    (model_dir / "model.gguf").write_bytes(b"weights")
    monkeypatch.setattr(constants, "SKULK_MODELS_PATH", (tmp_path,))

    store = ModelStore(tmp_path)
    store.register_model(
        "org/model",
        model_dir,
        ["model.gguf"],
        7,
        source_revision=_NEW_REVISION,
    )

    assert (model_dir / ".skulk-source-revision").read_text().strip() == _NEW_REVISION
    assert (
        download_utils.build_model_path(ModelId("org/model"), _NEW_REVISION)
        == model_dir
    )


async def test_pinned_store_download_writes_revision_marker_and_offloads_hashing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A qualified store entry must remain discoverable by generic runners."""

    monkeypatch.setattr(model_store_module, "MINIMUM_STAGING_FREE_DISK_BYTES", 0)
    store = ModelStore(tmp_path)
    model_id = "org/model"
    store._active_downloads[model_id] = StoreDownloadStatus(
        model_id=model_id,
        source_revision=_NEW_REVISION,
    )
    card = ModelCard(
        model_id=ModelId(model_id),
        storage_size=Memory.from_bytes(7),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        source_revision=_NEW_REVISION,
    )
    event_loop_thread_id = threading.get_ident()
    hashing_thread_ids: list[int] = []
    original_builder = cast(
        "Callable[..., InstalledCardRecord]",
        model_store_module.build_installed_card_record,
    )

    def tracked_builder(*args: object, **kwargs: object) -> InstalledCardRecord:
        hashing_thread_ids.append(threading.get_ident())
        return original_builder(*args, **kwargs)

    async def file_list(
        _model_id: ModelId,
        revision: str,
        recursive: bool,
    ) -> list[FileListEntry]:
        assert revision == _NEW_REVISION
        assert recursive is True
        return [FileListEntry(type="file", path="model.gguf", size=7)]

    async def download_file(
        _model_id: ModelId,
        revision: str,
        path: str,
        target_dir: Path,
        on_progress: Callable[[int, int, bool], None],
        on_connection_lost: Callable[[], None] = lambda: None,
        skip_internet: bool = False,
    ) -> Path:
        del on_connection_lost, skip_internet
        assert revision == _NEW_REVISION
        target = target_dir / path
        target.write_bytes(b"weights")
        on_progress(7, 7, True)
        return target

    monkeypatch.setattr(download_utils, "fetch_file_list_with_cache", file_list)
    monkeypatch.setattr(download_utils, "download_file_with_retry", download_file)
    monkeypatch.setattr(
        model_store_module,
        "build_installed_card_record",
        tracked_builder,
    )

    await store._do_download(
        model_id,
        pinned_gguf="model.gguf",
        source_revision=_NEW_REVISION,
        model_card=card,
    )

    entry = store.get_entry(model_id)
    assert entry is not None
    model_dir = tmp_path / entry.store_path
    assert entry.source_revision == _NEW_REVISION
    assert (model_dir / ".skulk-source-revision").read_text().strip() == _NEW_REVISION
    assert hashing_thread_ids
    assert all(thread_id != event_loop_thread_id for thread_id in hashing_thread_ids)


async def test_store_redownloads_when_registered_revision_differs(
    tmp_path: Path,
) -> None:
    store = ModelStore(tmp_path)
    model_id = "org/model"
    model_dir = tmp_path / "org--model"
    model_dir.mkdir()
    (model_dir / "model.gguf").write_bytes(b"old")
    store.register_model(
        model_id,
        model_dir,
        ["model.gguf"],
        3,
        source_revision=_OLD_REVISION,
    )

    status = await store.request_download(
        model_id,
        pinned_gguf="model.gguf",
        source_revision=_NEW_REVISION,
    )

    assert status.status in {"pending", "downloading"}
    assert status.source_revision == _NEW_REVISION
    for task in tuple(store._download_tasks):
        task.cancel()
    await asyncio.gather(*store._download_tasks, return_exceptions=True)


async def test_store_redownloads_when_registered_repository_differs(
    tmp_path: Path,
) -> None:
    """A matching commit cannot make a different signed repository a cache hit."""
    store = ModelStore(tmp_path)
    model_id = "org/model@q4"
    model_dir = tmp_path / "org--model@q4"
    model_dir.mkdir()
    (model_dir / "model.gguf").write_bytes(b"old")
    store.register_model(
        model_id,
        model_dir,
        ["model.gguf"],
        3,
        source_revision=_NEW_REVISION,
        source_repository="org/old-source",
    )

    status = await store.request_download(
        model_id,
        pinned_gguf="model.gguf",
        source_revision=_NEW_REVISION,
        source_repository="org/new-source",
    )

    assert status.status in {"pending", "downloading"}
    assert status.source_revision == _NEW_REVISION
    assert status.source_repository == "org/new-source"
    for task in tuple(store._download_tasks):
        task.cancel()
    await asyncio.gather(*store._download_tasks, return_exceptions=True)


async def test_active_download_dedup_rejects_different_repository(
    tmp_path: Path,
) -> None:
    """Concurrent requests may deduplicate only the same complete artifact."""
    store = ModelStore(tmp_path)
    model_id = "org/model@q4"
    store._active_downloads[model_id] = StoreDownloadStatus(
        model_id=model_id,
        source_revision=_NEW_REVISION,
        source_repository="org/old-source",
        status="downloading",
    )

    with pytest.raises(ValueError, match="org/old-source"):
        await store.request_download(
            model_id,
            source_revision=_NEW_REVISION,
            source_repository="org/new-source",
        )


@pytest.mark.parametrize(
    ("pinned_gguf", "extra_pinned_gguf"),
    [
        ("model-Q5_K_M.gguf", ["draft-Q4_K_M.gguf"]),
        ("model-Q4_K_M.gguf", ["draft-Q5_K_M.gguf"]),
    ],
)
async def test_active_download_dedup_rejects_different_file_selection(
    tmp_path: Path,
    pinned_gguf: str,
    extra_pinned_gguf: list[str],
) -> None:
    """Concurrent requests may not reuse bytes selected by another card."""
    store = ModelStore(tmp_path)
    model_id = "org/model@quant"
    store._active_downloads[model_id] = StoreDownloadStatus(
        model_id=model_id,
        source_revision=_NEW_REVISION,
        source_repository="org/model",
        pinned_gguf="model-Q4_K_M.gguf",
        extra_pinned_gguf=("draft-Q4_K_M.gguf",),
        status="downloading",
    )

    with pytest.raises(ValueError, match="different artifact selection"):
        await store.request_download(
            model_id,
            pinned_gguf=pinned_gguf,
            extra_pinned_gguf=extra_pinned_gguf,
            source_revision=_NEW_REVISION,
            source_repository="org/model",
        )


async def test_staging_replaces_files_from_another_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "org--model"
    destination.mkdir()
    (destination / ".skulk-source-revision").write_text(f"{_OLD_REVISION}\n")
    (destination / "model.gguf").write_bytes(b"old")
    unrelated = tmp_path / "unrelated-cache-root"
    unrelated.mkdir()
    (unrelated / "sentinel").write_text("preserve")

    async def fake_stage_http(
        _self: ModelStoreClient,
        _model_id: str,
        dest_path: Path,
        _on_progress: Callable[[int, int], Awaitable[None]] | None,
        source_revision: str | None,
        _capacity_preflight: Callable[[int], Awaitable[None]] | None,
    ) -> Path:
        assert source_revision == _NEW_REVISION
        assert not (dest_path / "model.gguf").exists()
        (dest_path / "model.gguf").write_bytes(b"new")
        return dest_path

    monkeypatch.setattr(ModelStoreClient, "_stage_http", fake_stage_http)
    client = ModelStoreClient(store_host="store.local", store_port=58080)

    staged = await client.stage_shard(
        "org/model",
        tmp_path,
        source_revision=_NEW_REVISION,
    )

    assert (staged / "model.gguf").read_bytes() == b"new"
    assert (staged / ".skulk-source-revision").read_text().strip() == _NEW_REVISION
    assert (unrelated / "sentinel").read_text() == "preserve"


async def test_pinned_store_host_shared_root_uses_canonical_revision_path(
    tmp_path: Path,
) -> None:
    """Pinned store-host staging must not populate mutable-main's directory."""

    store = ModelStore(tmp_path)
    model_id = "org/model"
    canonical_path = tmp_path / f"org--model--revision-{_NEW_REVISION}"
    canonical_path.mkdir()
    (canonical_path / "model.gguf").write_bytes(b"pinned")
    store.register_model(
        model_id,
        canonical_path,
        ["model.gguf"],
        len(b"pinned"),
        source_revision=_NEW_REVISION,
    )
    client = ModelStoreClient("localhost", local_store_path=tmp_path)

    staged = await client.stage_shard(
        model_id,
        tmp_path,
        source_revision=_NEW_REVISION,
    )

    assert staged == canonical_path.resolve()
    assert not (tmp_path / "org--model").exists()
    assert (staged / "model.gguf").read_bytes() == b"pinned"


async def test_mutable_main_download_clears_old_pinned_staging_residue(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An upgrade must not reuse pinned bytes left in mutable-main's path."""

    monkeypatch.setattr(model_store_module, "MINIMUM_STAGING_FREE_DISK_BYTES", 0)
    store = ModelStore(tmp_path)
    model_id = "org/model"
    target_dir = tmp_path / "org--model"
    target_dir.mkdir()
    (target_dir / "model.gguf").write_bytes(b"pinned")
    (target_dir / ".skulk-source-revision").write_text(f"{_NEW_REVISION}\n")
    store._active_downloads[model_id] = StoreDownloadStatus(
        model_id=model_id,
        source_revision=None,
    )

    async def file_list(
        _model_id: ModelId,
        revision: str,
        recursive: bool,
    ) -> list[FileListEntry]:
        assert revision == "main"
        assert recursive is True
        return [FileListEntry(type="file", path="model.gguf", size=7)]

    async def download_file(
        _model_id: ModelId,
        revision: str,
        path: str,
        download_dir: Path,
        on_progress: Callable[[int, int, bool], None],
        on_connection_lost: Callable[[], None] = lambda: None,
        skip_internet: bool = False,
    ) -> Path:
        del on_connection_lost, skip_internet
        assert revision == "main"
        assert not (download_dir / path).exists()
        target = download_dir / path
        target.write_bytes(b"mutable")
        on_progress(7, 7, True)
        return target

    monkeypatch.setattr(download_utils, "fetch_file_list_with_cache", file_list)
    monkeypatch.setattr(download_utils, "download_file_with_retry", download_file)

    await store._do_download(model_id)

    assert (target_dir / "model.gguf").read_bytes() == b"mutable"
    assert not (target_dir / ".skulk-source-revision").exists()


async def test_staging_recovers_from_corrupted_revision_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A malformed staging marker is a mismatch, not a terminal load error."""

    destination = tmp_path / "org--model"
    destination.mkdir()
    (destination / ".skulk-source-revision").write_bytes(b"\xff")
    (destination / "model.gguf").write_bytes(b"old")

    async def fake_stage_http(
        _self: ModelStoreClient,
        _model_id: str,
        dest_path: Path,
        _on_progress: Callable[[int, int], Awaitable[None]] | None,
        _source_revision: str | None,
        _capacity_preflight: Callable[[int], Awaitable[None]] | None,
    ) -> Path:
        (dest_path / "model.gguf").write_bytes(b"new")
        return dest_path

    monkeypatch.setattr(ModelStoreClient, "_stage_http", fake_stage_http)
    client = ModelStoreClient(store_host="store.local", store_port=58080)

    staged = await client.stage_shard(
        "org/model",
        tmp_path,
        source_revision=_NEW_REVISION,
    )

    assert (staged / "model.gguf").read_bytes() == b"new"
    assert (staged / ".skulk-source-revision").read_text().strip() == _NEW_REVISION


async def test_pinned_staging_retry_preserves_partial_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A retry for the same pinned revision must retain resumable bytes."""

    attempts = 0

    async def fake_stage_http(
        _self: ModelStoreClient,
        _model_id: str,
        dest_path: Path,
        _on_progress: Callable[[int, int], Awaitable[None]] | None,
        source_revision: str | None,
        _capacity_preflight: Callable[[int], Awaitable[None]] | None,
    ) -> Path:
        nonlocal attempts
        attempts += 1
        assert source_revision == _NEW_REVISION
        partial = dest_path / "model.gguf.partial"
        if attempts == 1:
            partial.write_bytes(b"partial")
            raise ConnectionError("transfer interrupted")
        assert partial.read_bytes() == b"partial"
        partial.replace(dest_path / "model.gguf")
        return dest_path

    monkeypatch.setattr(ModelStoreClient, "_stage_http", fake_stage_http)
    client = ModelStoreClient(store_host="store.local", store_port=58080)

    with pytest.raises(ConnectionError, match="transfer interrupted"):
        await client.stage_shard(
            "org/model",
            tmp_path,
            source_revision=_NEW_REVISION,
        )

    destination = tmp_path / "org--model"
    assert (destination / "model.gguf.partial").read_bytes() == b"partial"
    assert (
        destination / ".skulk-source-revision-staging"
    ).read_text().strip() == _NEW_REVISION

    staged = await client.stage_shard(
        "org/model",
        tmp_path,
        source_revision=_NEW_REVISION,
    )

    assert attempts == 2
    assert (staged / "model.gguf").read_bytes() == b"partial"
    assert (staged / ".skulk-source-revision").read_text().strip() == _NEW_REVISION
    assert not (staged / ".skulk-source-revision-staging").exists()


async def test_unpinned_staging_rejects_interrupted_pinned_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Mutable-main staging must not reuse files from an interrupted pin."""

    destination = tmp_path / "org--model"
    destination.mkdir()
    (destination / "model.gguf").write_bytes(b"pinned")
    (destination / ".skulk-source-revision-staging").write_text(f"{_NEW_REVISION}\n")

    assert not _staged_source_revision_matches(destination, None)

    async def fake_stage_http(
        _self: ModelStoreClient,
        _model_id: str,
        dest_path: Path,
        _on_progress: Callable[[int, int], Awaitable[None]] | None,
        source_revision: str | None,
        _capacity_preflight: Callable[[int], Awaitable[None]] | None,
    ) -> Path:
        assert source_revision is None
        assert not (dest_path / "model.gguf").exists()
        assert not (dest_path / ".skulk-source-revision-staging").exists()
        (dest_path / "model.gguf").write_bytes(b"mutable-main")
        return dest_path

    monkeypatch.setattr(ModelStoreClient, "_stage_http", fake_stage_http)
    client = ModelStoreClient(store_host="store.local", store_port=58080)

    staged = await client.stage_shard("org/model", tmp_path)

    assert (staged / "model.gguf").read_bytes() == b"mutable-main"
    assert not (staged / ".skulk-source-revision").exists()
    assert not (staged / ".skulk-source-revision-staging").exists()


async def test_request_rechecks_revision_after_waiting_for_download_lock(
    tmp_path: Path,
) -> None:
    """A lock wait must not reuse a stale pre-lock revision comparison."""

    store = ModelStore(tmp_path)
    model_id = "org/model"
    model_dir = tmp_path / "org--model"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"weights")
    store.register_model(
        model_id,
        model_dir,
        ["model.safetensors"],
        len(b"weights"),
        source_revision=_NEW_REVISION,
    )

    await store._download_lock.acquire()
    request = asyncio.create_task(
        store.request_download(model_id, source_revision=_NEW_REVISION)
    )
    await asyncio.sleep(0)
    store.register_model(
        model_id,
        model_dir,
        ["model.safetensors"],
        len(b"weights"),
        source_revision=_OLD_REVISION,
    )
    store._download_lock.release()

    status = await request

    assert status.status in {"pending", "downloading"}
    assert status.source_revision == _NEW_REVISION
    for task in tuple(store._download_tasks):
        task.cancel()
    await asyncio.gather(*store._download_tasks, return_exceptions=True)
