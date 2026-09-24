# pyright: reportPrivateUsage=false
"""A video companion downloads through the store as exactly its owner's named files."""

import asyncio
from pathlib import Path

import pytest

import skulk.store.model_store as model_store_module
from skulk.download import download_utils
from skulk.download.download_utils import FileListEntry
from skulk.shared.models.model_cards import (
    ArtifactBundleConfig,
    ArtifactBundleFile,
    ModelCard,
    ModelId,
    ModelTask,
    VideoCardConfig,
    VideoPreprocessorRole,
)
from skulk.shared.types.memory import Memory
from skulk.store.model_store import ModelStore

REVISION = "f" * 40
POSE_FILE = "checkpoints/pose.safetensors"


def _owner() -> ModelCard:
    """A video card whose own bundle lists files the companion repository lacks."""
    return ModelCard(
        model_id=ModelId("org/video"),
        source_revision="a" * 40,
        storage_size=Memory.from_bytes(4),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextToVideo],
        artifact_bundle=ArtifactBundleConfig(
            bundle_id=f"bundle_{'b' * 52}",
            files=(ArtifactBundleFile(path="diffusion_models/base.safetensors", size_bytes=4),),
            download_size=4,
        ),
        video=VideoCardConfig.model_validate(
            {
                "modes": ["t2va"],
                "companions": [
                    {
                        "kind": "preprocessor",
                        "name": "pose",
                        "role": "pose_estimator",
                        "path": POSE_FILE,
                        "repo": "org/pose",
                        "revision": REVISION,
                        "size_bytes": 5,
                    }
                ],
            }
        ),
    )


async def test_a_video_companion_fetches_its_named_files_not_the_owners_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fetched: list[tuple[str, int | None]] = []

    async def file_list(model_id: ModelId, revision: str, recursive: bool) -> list[FileListEntry]:
        assert (str(model_id), revision, recursive) == ("org/pose", REVISION, True)
        return [
            FileListEntry(type="file", path=POSE_FILE, size=5),
            FileListEntry(type="file", path="checkpoints/pose_fp32.safetensors", size=9),
        ]

    async def download(
        model_id: ModelId,
        revision: str,
        path: str,
        target_dir: Path,
        *_args: object,
        expected_size: int | None = None,
        **_kwargs: object,
    ) -> Path:
        fetched.append((path, expected_size))
        target = target_dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"12345")
        return target

    monkeypatch.setattr(download_utils, "fetch_file_list_with_cache", file_list)
    monkeypatch.setattr(download_utils, "download_file_with_retry", download)
    monkeypatch.setattr(model_store_module, "MINIMUM_STAGING_FREE_DISK_BYTES", 0)
    store = ModelStore(tmp_path)
    owner = _owner()

    async def request() -> str:
        status = await store.request_download(
            "org/pose",
            source_revision=REVISION,
            model_card=owner,
            artifact_role="video_companion",
            owner_model_id=str(owner.model_id),
        )
        for _ in range(200):
            if status.status in ("complete", "failed", "cancelled"):
                break
            await asyncio.sleep(0.01)
        return status.status

    assert await request() == "complete"
    # Only the named file, verified at its declared size; never the owner's bundle.
    assert fetched == [(POSE_FILE, 5)]
    # The stored companion matches its request, so asking again fetches nothing.
    assert await request() == "complete"
    assert fetched == [(POSE_FILE, 5)]


async def test_a_stored_companion_serves_another_card_only_with_its_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A second card naming more files from the same repository fetches them."""
    fetched: list[str] = []

    async def file_list(model_id: ModelId, revision: str, recursive: bool) -> list[FileListEntry]:
        return [
            FileListEntry(type="file", path=POSE_FILE, size=5),
            FileListEntry(type="file", path="diffusion_models/person.safetensors", size=5),
        ]

    async def download(
        model_id: ModelId, revision: str, path: str, target_dir: Path, *_args: object, **_kwargs: object
    ) -> Path:
        fetched.append(path)
        target = target_dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"12345")
        return target

    monkeypatch.setattr(download_utils, "fetch_file_list_with_cache", file_list)
    monkeypatch.setattr(download_utils, "download_file_with_retry", download)
    monkeypatch.setattr(model_store_module, "MINIMUM_STAGING_FREE_DISK_BYTES", 0)
    store = ModelStore(tmp_path)
    first = _owner()
    assert first.video is not None
    wider = first.model_copy(
        update={
            "model_id": ModelId("org/other-video"),
            "video": first.video.model_copy(
                update={
                    "companions": (
                        *first.video.companions,
                        first.video.companions[0].model_copy(
                            update={
                                "name": "person",
                                "role": VideoPreprocessorRole.PersonDetector,
                                "path": "diffusion_models/person.safetensors",
                            }
                        ),
                    )
                }
            ),
        }
    )

    async def request(owner: ModelCard) -> str:
        status = await store.request_download(
            "org/pose",
            source_revision=REVISION,
            model_card=owner,
            artifact_role="video_companion",
            owner_model_id=str(owner.model_id),
        )
        for _ in range(200):
            if status.status in ("complete", "failed", "cancelled"):
                break
            await asyncio.sleep(0.01)
        return status.status

    assert await request(first) == "complete"
    assert fetched == [POSE_FILE]
    # The stored entry lacks the wider card's detector, so it downloads again.
    assert await request(wider) == "complete"
    assert "diffusion_models/person.safetensors" in fetched
    # And the first card is still served from what is stored.
    count = len(fetched)
    assert await request(first) == "complete"
    assert len(fetched) == count
