# pyright: reportPrivateUsage=false
"""Native image store transactions retain every pinned source-group component."""

import asyncio
import base64
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from skulk.download import download_utils
from skulk.shared.models.image import (
    ImageCardConfig,
    ImageComponent,
    ImageComponentFile,
    ImageComponentRole,
)
from skulk.shared.models.model_cards import (
    ArtifactBundleConfig,
    ArtifactBundleFile,
    ModelCard,
    ModelId,
    ModelTask,
)
from skulk.shared.types.memory import Memory
from skulk.shared.types.worker.downloads import FileListEntry, RepoDownloadProgress
from skulk.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata
from skulk.store import model_store_server as server_module
from skulk.store.installed_cards import companion_artifact_role
from skulk.store.model_store import ModelStore, select_store_image_files
from skulk.store.model_store_client import (
    _declared_companion_files,
    _staged_directory_looks_complete,
)
from skulk.store.model_store_server import ModelStoreServer

DATA = b"abcdefg"
IDENTITY = "sha256:" + hashlib.sha256(DATA).hexdigest()


def _owner(primary_file: str = "diffusion.gguf") -> ModelCard:
    """Return a synthetic complete card with base and grouped encoder artifacts."""
    sources: list[tuple[ImageComponentRole, str, str, str]] = [
        ("diffusion", "owner/base", "a" * 40, primary_file),
        ("vae", "owner/base", "a" * 40, "vae.safetensors"),
        ("text_encoder", "owner/encoders", "b" * 40, "encoder.gguf"),
        ("text_encoder_vision", "owner/encoders", "b" * 40, "mmproj-BF16.gguf"),
    ]
    return ModelCard(
        model_id=ModelId("test/image"),
        source_repository=ModelId("owner/base"),
        source_revision="a" * 40,
        gguf_file=primary_file if primary_file.endswith(".gguf") else None,
        storage_size=Memory.from_bytes(28),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextToImage, ModelTask.ImageToImage],
        is_custom=True,
        image=ImageCardConfig(
            family="qwen_image_21",
            max_reference_images=2,
            default_steps=20,
            components=tuple(
                ImageComponent(
                    role=role,
                    repository=repository,
                    revision=revision,
                    entry_file=path,
                    files=(
                        ImageComponentFile(path=path, size_bytes=7, object_id=IDENTITY),
                    ),
                )
                for role, repository, revision, path in sources
            ),
        ),
    )


@pytest.mark.parametrize("primary_file", ["diffusion.gguf", "diffusion.safetensors"])
@pytest.mark.parametrize("has_primary_bundle", [False, True])
async def test_direct_native_image_transfer_retains_root_components(
    primary_file: str, has_primary_bundle: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exact primary/VAE files survive downstream MLX filters, with or without a bundle."""
    owner = _owner(primary_file)
    if has_primary_bundle:
        manifest = [{"relative_path": primary_file, "size_bytes": 7, "object_id": IDENTITY}]
        digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).digest()
        owner = owner.model_copy(update={"artifact_bundle": ArtifactBundleConfig(
            bundle_id="bundle_" + base64.b32encode(digest).decode().rstrip("=").lower(),
            files=(ArtifactBundleFile(path=primary_file, size_bytes=7, object_id=IDENTITY),), download_size=7,
        )})
    shard = PipelineShardMetadata(model_card=owner, device_rank=0, world_size=1, start_layer=0, end_layer=1, n_layers=1)
    fetched: list[str] = []
    seen: list[RepoDownloadProgress] = []

    async def listing(repository: ModelId, revision: str, **_kwargs: object) -> list[FileListEntry]:
        assert str(repository) == "owner/base" and revision == "a" * 40
        return [FileListEntry(type="file", path=path, size=7) for path in (primary_file, "vae.safetensors", "unselected.safetensors", "unselected.gguf")]

    async def download(repository: ModelId, revision: str, path: str, target_dir: Path,
        on_progress: Callable[[int, int, bool], None], expected_size: int | None = None,
        expected_object_id: str | None = None, **_kwargs: object) -> Path:
        assert expected_size == 7 and expected_object_id == IDENTITY
        fetched.append(path)
        target = target_dir / path
        target.write_bytes(DATA)
        on_progress(7, 7, True)
        return target

    async def progress(_shard: ShardMetadata, value: RepoDownloadProgress) -> None:
        seen.append(value)

    monkeypatch.setattr(download_utils, "SKULK_MODELS_DIR", tmp_path)
    monkeypatch.setattr(download_utils, "fetch_file_list_with_cache", listing)
    monkeypatch.setattr(download_utils, "download_file_with_retry", download)
    result, terminal = await download_utils.download_shard(shard, progress)
    assert fetched == [primary_file, "vae.safetensors"]
    assert terminal.status == "complete" and terminal.total_files == 2
    assert seen[-1].status == "complete"
    root = result.parent if result.is_file() else result
    assert (root / primary_file).read_bytes() == DATA and (root / "vae.safetensors").read_bytes() == DATA


@pytest.mark.parametrize(
    "repository,revision,expected",
    [
        ("owner/base", "a" * 40, ["diffusion.gguf", "vae.safetensors"]),
        ("owner/encoders", "b" * 40, ["encoder.gguf", "mmproj-BF16.gguf"]),
    ],
)
async def test_store_fetches_complete_image_group_and_reuses_it(
    repository: str,
    revision: str,
    expected: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise actual store transfers, verified sizes/hashes and complete-cache reuse."""
    owner = _owner()
    fetched: list[str] = []

    async def listing(
        _model_id: ModelId, _revision: str, recursive: bool
    ) -> list[FileListEntry]:
        assert str(_model_id) == repository and _revision == revision and recursive
        return [
            FileListEntry(type="file", path=path, size=7)
            for path in [*expected, "unselected-Q8.gguf", "other-mmproj.gguf"]
        ]

    async def download(
        _model_id: ModelId,
        _revision: str,
        path: str,
        target_dir: Path,
        *_args: object,
        expected_size: int | None = None,
        expected_object_id: str | None = None,
        **_kwargs: object,
    ) -> Path:
        assert expected_size == 7 and expected_object_id == IDENTITY
        fetched.append(path)
        target = target_dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(DATA)
        return target

    monkeypatch.setattr(download_utils, "fetch_file_list_with_cache", listing)
    monkeypatch.setattr(download_utils, "download_file_with_retry", download)
    store = ModelStore(tmp_path)
    is_base = repository == "owner/base"
    model_id = str(owner.model_id) if is_base else repository

    async def request() -> None:
        status = await store.request_download(
            model_id,
            pinned_gguf=owner.gguf_file if is_base else None,
            source_revision=revision,
            source_repository=repository,
            model_card=owner,
            artifact_role="base" if is_base else "image_component",
            owner_model_id=None if is_base else str(owner.model_id),
        )
        for _ in range(200):
            if status.status in ("complete", "failed", "cancelled"):
                break
            await asyncio.sleep(0.01)
        assert status.status == "complete", status.error

    await request()
    assert fetched == expected
    await request()
    assert fetched == expected
    installed = store.get_store_path(model_id)
    assert installed is not None
    (installed / expected[-1]).unlink()
    await request()
    assert fetched == [*expected, *expected]
    assert (installed / expected[-1]).read_bytes() == DATA


async def test_image_companion_direct_fallback_retains_the_exact_manifest() -> None:
    """Store unavailability cannot widen a grouped encoder to its entire repository."""
    shard, patterns, required = download_utils.companion_download_specs(_owner())[0]
    assert required and patterns == ["encoder.gguf", "mmproj-BF16.gguf"]
    assert shard.model_card.gguf_file is None
    assert await download_utils.resolve_allow_patterns(shard) == patterns
    bundle = shard.model_card.artifact_bundle
    assert bundle is not None and bundle.download_size == 14
    assert all(
        item.object_id == IDENTITY and item.size_bytes == 7 for item in bundle.files
    )


async def test_store_verifies_image_companion_owner_and_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A companion must bind its actual immutable source group to the complete owner."""
    owner = _owner()

    async def cards() -> list[ModelCard]:
        return [owner]

    monkeypatch.setattr(server_module, "get_all_model_cards", cards)
    assert companion_artifact_role(owner, "owner/encoders") == "image_component"
    for revision in ["b" * 40, "c" * 40]:
        if revision == "b" * 40:
            assert (
                await ModelStoreServer._require_remote_code_download_approval(
                    "owner/encoders",
                    None,
                    source_repository="owner/encoders",
                    source_revision=revision,
                    pinned_gguf=None,
                    owner_model_id=str(owner.model_id),
                    artifact_role="image_component",
                )
                == owner
            )
        else:
            with pytest.raises(web.HTTPConflict):
                await ModelStoreServer._require_remote_code_download_approval(
                    "owner/encoders",
                    None,
                    source_repository="owner/encoders",
                    source_revision=revision,
                    pinned_gguf=None,
                    owner_model_id=str(owner.model_id),
                    artifact_role="image_component",
                )


@pytest.mark.parametrize("revision,expected_status", [("b" * 40, 200), ("c" * 40, 409)])
async def test_image_component_http_download_preserves_owner_and_manifest(
    revision: str,
    expected_status: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wire parser admits pinned image groups before verified store transfer."""
    owner = _owner().model_copy(update={
        "is_custom": False,
        "registry_card_id": "card_" + "a" * 52,
        "registry_snapshot_id": "snapshot_test",
        "registry_provenance": "foxlight",
    })
    fetched: list[str] = []

    async def cards() -> list[ModelCard]:
        return [owner]

    async def listing(
        repository: ModelId, source_revision: str, recursive: bool
    ) -> list[FileListEntry]:
        assert str(repository) == "owner/encoders" and source_revision == "b" * 40 and recursive
        return [FileListEntry(type="file", path=path, size=7)
            for path in ("encoder.gguf", "mmproj-BF16.gguf", "unselected.gguf")]

    async def download(
        repository: ModelId, source_revision: str, path: str, target_dir: Path,
        *_args: object, expected_size: int | None = None,
        expected_object_id: str | None = None, **_kwargs: object,
    ) -> Path:
        assert str(repository) == "owner/encoders" and source_revision == "b" * 40
        assert expected_size == 7 and expected_object_id == IDENTITY
        fetched.append(path)
        target = target_dir / path
        target.write_bytes(DATA)
        return target

    monkeypatch.setattr(server_module, "get_all_model_cards", cards)
    monkeypatch.setattr(download_utils, "fetch_file_list_with_cache", listing)
    monkeypatch.setattr(download_utils, "download_file_with_retry", download)
    store = ModelStore(tmp_path)
    server = ModelStoreServer(store, host="127.0.0.1", port=0)
    async with TestClient(TestServer(server._app)) as client:
        response = await client.post("/models/owner%2Fencoders/download", json={
            "source_repository": "owner/encoders", "source_revision": revision,
            "owner_model_id": str(owner.model_id), "owner_registry_card_id": owner.registry_card_id,
            "artifact_role": "image_component",
        })
        assert response.status == expected_status, await response.text()
        if expected_status != 200:
            assert not fetched and store.get_download_status("owner/encoders") is None
            return
        status = store.get_download_status("owner/encoders")
        assert status is not None
        for _ in range(200):
            if status.status in ("complete", "failed", "cancelled"):
                break
            await asyncio.sleep(0.01)
        assert status.status == "complete", status.error
    assert fetched == ["encoder.gguf", "mmproj-BF16.gguf"]
    installed = store.get_store_path("owner/encoders")
    assert installed is not None
    assert (installed / "encoder.gguf").read_bytes() == DATA
    assert (installed / "mmproj-BF16.gguf").read_bytes() == DATA


def test_staging_requires_all_image_files_even_after_a_base_gguf_exists(
    tmp_path: Path,
) -> None:
    """An old partial staging directory cannot bypass missing VAE/encoder recovery."""
    owner = _owner()
    specs = download_utils.companion_download_specs(owner)
    assert len(specs) == 1
    shard = specs[0][0]
    expected = _declared_companion_files(shard, owner, "image_component")
    assert expected == (("encoder.gguf", 7), ("mmproj-BF16.gguf", 7))
    (tmp_path / "encoder.gguf").write_bytes(DATA)
    assert not _staged_directory_looks_complete(tmp_path, shard.model_card, expected)
    (tmp_path / "mmproj-BF16.gguf").write_bytes(DATA)
    assert _staged_directory_looks_complete(tmp_path, shard.model_card, expected)


def test_image_selection_rejects_missing_or_wrong_size_files() -> None:
    """A source listing must contain the entire declared image transaction."""
    owner = _owner()
    with pytest.raises(FileNotFoundError):
        select_store_image_files(
            [FileListEntry(type="file", path="diffusion.gguf", size=7)],
            owner,
            "owner/base",
            "a" * 40,
        )
    with pytest.raises(ValueError, match="size mismatch"):
        select_store_image_files(
            [
                FileListEntry(type="file", path="diffusion.gguf", size=7),
                FileListEntry(type="file", path="vae.safetensors", size=8),
            ],
            owner,
            "owner/base",
            "a" * 40,
        )
