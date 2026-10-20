# pyright: reportPrivateUsage=false
"""Overlapping signed packs share verified immutable store bytes safely."""

import asyncio
import base64
import errno
import hashlib
import json
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import pytest

import skulk.store.model_store as store_module
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
from skulk.store.model_store import ModelStore

_REVISION = "a" * 40
_DATA = b"immutable shared model bytes"
_PATH = "weights/model.safetensors"


class _DiskUsage(NamedTuple):
    total: int
    used: int
    free: int


def _card(*, extra: bool = False) -> ModelCard:
    files = [
        {
            "path": _PATH,
            "size_bytes": len(_DATA),
            "object_id": "sha256:" + hashlib.sha256(_DATA).hexdigest(),
        }
    ]
    if extra:
        files.append(
            {
                "path": "upscaler.safetensors",
                "size_bytes": 2,
                "object_id": "sha256:" + hashlib.sha256(b"up").hexdigest(),
            }
        )
    identity = [
        {
            "relative_path": item["path"],
            "size_bytes": item["size_bytes"],
            "object_id": item["object_id"],
        }
        for item in sorted(files, key=lambda item: str(item["path"]))
    ]
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).digest()
    bundle = ArtifactBundleConfig(
        bundle_id="bundle_" + base64.b32encode(digest).decode().rstrip("=").lower(),
        files=tuple(ArtifactBundleFile.model_validate(item) for item in files),
        download_size=len(_DATA) + (2 if extra else 0),
    )
    return ModelCard(
        model_id=ModelId("org/expanded-pack"),
        source_repository=ModelId("org/source"),
        source_revision=_REVISION,
        registry_card_id="card_" + "a" * 52,
        storage_size=Memory.from_bytes(bundle.download_size),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        artifact_bundle=bundle,
    )


def _register(store: ModelStore) -> Path:
    root = store.store_path / "original"
    source = root / _PATH
    source.parent.mkdir(parents=True)
    source.write_bytes(_DATA)
    store.register_model(
        "org/original-pack",
        root,
        [_PATH],
        len(_DATA),
        source_repository="org/source",
        source_revision=_REVISION,
    )
    return source


async def test_overlapping_pack_shares_inode_and_survives_original_deletion(
    tmp_path: Path,
) -> None:
    """Deleting one alias unlinks its name without damaging the other pack."""
    store = ModelStore(tmp_path)
    source = _register(store)
    destination = tmp_path / "expanded"
    destination.mkdir()
    assert await store._reuse_verified_bundle_files(destination, _card()) == frozenset(
        {_PATH}
    )
    assert source.stat().st_ino == (destination / _PATH).stat().st_ino
    assert source.stat().st_nlink == 2
    assert store.delete_model("org/original-pack")
    assert (destination / _PATH).read_bytes() == _DATA


@pytest.mark.parametrize(
    "fault",
    [
        "unsigned",
        "mutable",
        "revision",
        "repository",
        "unlisted",
        "missing",
        "size",
        "hash",
        "object",
        "no_object",
        "outside",
    ],
)
async def test_reuse_never_credits_unverified_or_unselected_bytes(
    tmp_path: Path, fault: str
) -> None:
    """A name or same-sized file is insufficient for signed capacity credit."""
    store = ModelStore(tmp_path / "store")
    source = _register(store)
    card = _card()
    if fault == "unsigned":
        card = card.model_copy(update={"registry_card_id": None})
    elif fault == "mutable":
        card = card.model_copy(update={"source_revision": None})
    elif fault == "revision":
        card = card.model_copy(update={"source_revision": "b" * 40})
    elif fault == "repository":
        card = card.model_copy(update={"source_repository": ModelId("other/source")})
    elif fault == "unlisted":
        store.register_model(
            "org/original-pack",
            source.parents[1],
            [],
            0,
            source_repository="org/source",
            source_revision=_REVISION,
        )
    elif fault == "missing":
        source.unlink()
    elif fault == "size":
        source.write_bytes(b"short")
    elif fault == "hash":
        source.write_bytes(b"x" * len(_DATA))
    elif fault in {"object", "no_object"}:
        assert card.artifact_bundle is not None
        member = card.artifact_bundle.files[0].model_copy(
            update={"object_id": "git-sha1:" + "b" * 40 if fault == "object" else None}
        )
        card = card.model_copy(
            update={
                "artifact_bundle": card.artifact_bundle.model_copy(
                    update={"files": (member,)}
                )
            }
        )
    else:
        outside = tmp_path / "outside"
        outside.write_bytes(_DATA)
        source.unlink()
        source.symlink_to(outside)
    destination = store.store_path / "expanded"
    destination.mkdir()
    assert not await store._reuse_verified_bundle_files(destination, card)
    assert not (destination / _PATH).exists()


async def test_link_failure_falls_back_without_copying_or_capacity_credit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cross-filesystem reuse needs a normal budgeted download, never a copy."""
    store = ModelStore(tmp_path)
    _register(store)
    destination = tmp_path / "expanded"
    destination.mkdir()

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise OSError(errno.EXDEV, "different filesystem")

    monkeypatch.setattr(store_module.os, "link", refuse)
    assert not await store._reuse_verified_bundle_files(destination, _card())
    assert not (destination / _PATH).exists()


async def test_destination_symlink_cannot_escape_store(tmp_path: Path) -> None:
    """An abandoned generation cannot redirect a reused member outside its root."""
    store = ModelStore(tmp_path / "store")
    _register(store)
    destination = store.store_path / "expanded"
    destination.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (destination / "weights").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes its destination"):
        await store._reuse_verified_bundle_files(destination, _card())
    assert not (outside / "model.safetensors").exists()


async def test_cancellation_while_hashing_cannot_link_after_lock_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only pure reads continue in the thread when the owning transfer cancels."""
    store = ModelStore(tmp_path)
    _register(store)
    destination = tmp_path / "expanded"
    destination.mkdir()
    started = asyncio.Event()
    finish = threading.Event()
    finished = threading.Event()
    loop = asyncio.get_running_loop()
    original_hash = store_module._sha256_path

    def blocked_hash(path: Path) -> str:
        loop.call_soon_threadsafe(started.set)
        finish.wait()
        try:
            return original_hash(path)
        finally:
            finished.set()

    monkeypatch.setattr(store_module, "_sha256_path", blocked_hash)
    task = asyncio.create_task(store._reuse_verified_bundle_files(destination, _card()))
    try:
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        finish.set()
        await asyncio.to_thread(finished.wait)
    assert not (destination / _PATH).exists()


@pytest.mark.parametrize("can_link", [True, False])
@pytest.mark.parametrize("source_mismatch", [None, "repository", "revision", "mutable"])
async def test_full_transfer_admits_only_new_physical_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    can_link: bool,
    source_mismatch: str | None,
) -> None:
    """A larger pack fits its new member plus reserve only after verified reuse."""
    store = ModelStore(tmp_path)
    source = _register(store)
    card = _card(extra=True)
    fetched: list[str] = []

    async def listing(
        _repository: ModelId, _revision: str, **_kwargs: object
    ) -> list[FileListEntry]:
        assert card.artifact_bundle is not None
        return [
            FileListEntry(type="file", path=item.path, size=item.size_bytes)
            for item in card.artifact_bundle.files
        ]

    async def download(
        _repository: ModelId,
        _revision: str,
        path: str,
        target: Path,
        _progress: Callable[[int, int, bool], None],
        **_kwargs: object,
    ) -> Path:
        fetched.append(path)
        assert path == "upscaler.safetensors"
        (target / path).write_bytes(b"up")
        return target / path

    def no_link(*_args: object, **_kwargs: object) -> None:
        raise OSError(errno.EXDEV, "different filesystem")

    monkeypatch.setattr(download_utils, "fetch_file_list_with_cache", listing)
    monkeypatch.setattr(download_utils, "download_file_with_retry", download)
    reserve = store_module.MINIMUM_STAGING_FREE_DISK_BYTES

    def disk_usage(_path: Path) -> _DiskUsage:
        return _DiskUsage(2 * reserve, reserve - 2, reserve + 2)

    monkeypatch.setattr(store_module.shutil, "disk_usage", disk_usage)
    if not can_link:
        monkeypatch.setattr(os, "link", no_link)
    requested_repository = (
        "other/source"
        if source_mismatch == "repository"
        else str(card.artifact_repository)
    )
    requested_revision = (
        None
        if source_mismatch == "mutable"
        else "b" * 40
        if source_mismatch == "revision"
        else card.source_revision
    )
    reusable = can_link and source_mismatch is None
    status = await store.request_download(
        str(card.model_id),
        source_repository=requested_repository,
        source_revision=requested_revision,
        model_card=card,
    )
    await asyncio.gather(*store._download_tasks)
    assert status.status == ("complete" if reusable else "failed")
    assert source.read_bytes() == _DATA
    if reusable:
        installed = store.get_store_path(str(card.model_id))
        assert installed is not None
        assert (installed / _PATH).stat().st_ino == source.stat().st_ino
        assert fetched == ["upscaler.safetensors"]
    else:
        assert fetched == []
        assert status.error is not None and "disk capacity" in status.error
