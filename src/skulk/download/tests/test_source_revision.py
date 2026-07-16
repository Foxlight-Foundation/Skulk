# pyright: reportPrivateUsage=false
"""Direct-download cache identity tests for immutable source revisions."""

from collections.abc import Callable
from pathlib import Path

import pytest

import skulk.shared.constants as constants
from skulk.download import download_utils
from skulk.download.download_utils import (
    build_model_path,
    download_shard,
    resolve_model_in_path,
)
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.memory import Memory
from skulk.shared.types.worker.downloads import FileListEntry, RepoDownloadProgress
from skulk.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata

_OLD_REVISION = "0" * 40
_NEW_REVISION = "1" * 40


def _shard(source_revision: str) -> PipelineShardMetadata:
    model_id = ModelId("org/model")
    return PipelineShardMetadata(
        model_card=ModelCard(
            model_id=model_id,
            storage_size=Memory.from_bytes(7),
            n_layers=1,
            hidden_size=1,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
            gguf_file="model.gguf",
            source_revision=source_revision,
        ),
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=1,
        n_layers=1,
    )


def test_unpinned_resolution_rejects_pinned_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Mutable-main requests must not reuse bytes marked as a pinned revision."""

    model_id = ModelId("org/model")
    model_dir = tmp_path / model_id.normalize()
    model_dir.mkdir()
    (model_dir / "model.gguf").write_bytes(b"weights")
    marker = model_dir / ".skulk-source-revision"
    marker.write_text(f"{'0' * 40}\n")
    monkeypatch.setattr(constants, "SKULK_MODELS_PATH", (tmp_path,))

    assert resolve_model_in_path(model_id) is None
    assert build_model_path(model_id, _OLD_REVISION) == model_dir

    marker.unlink()
    assert resolve_model_in_path(model_id) == model_dir


def test_unpinned_resolution_rejects_interrupted_pinned_staging(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A staging marker keeps complete-looking pinned bytes non-loadable."""

    model_id = ModelId("org/model")
    model_dir = tmp_path / model_id.normalize()
    model_dir.mkdir()
    (model_dir / "model.gguf").write_bytes(b"weights")
    (model_dir / ".skulk-source-revision-staging").write_text(
        f"{_NEW_REVISION}\n"
    )
    monkeypatch.setattr(constants, "SKULK_MODELS_PATH", (tmp_path,))
    monkeypatch.setattr(download_utils, "SKULK_MODELS_DIR", tmp_path)

    assert resolve_model_in_path(model_id) is None
    with pytest.raises(FileNotFoundError):
        build_model_path(model_id)


@pytest.mark.asyncio
async def test_interrupted_first_pinned_download_resumes_without_becoming_mutable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Pinned bytes stay non-loadable until the final revision marker lands."""

    model_id = ModelId("org/model")
    canonical = tmp_path / model_id.normalize()
    attempts = 0

    async def file_list(*_args: object, **_kwargs: object) -> list[FileListEntry]:
        return [FileListEntry(type="file", path="model.gguf", size=7)]

    async def interrupt_then_complete(
        _model_id: ModelId,
        _revision: str,
        path: str,
        target_dir: Path,
        on_progress: Callable[[int, int, bool], None],
        on_connection_lost: Callable[[], None],
        skip_internet: bool,
    ) -> Path:
        nonlocal attempts
        del on_connection_lost, skip_internet
        attempts += 1
        assert target_dir == canonical
        assert (
            target_dir / ".skulk-source-revision-staging"
        ).read_text().strip() == _NEW_REVISION
        target = target_dir / path
        target.write_bytes(b"weights")
        on_progress(7, 7, True)
        if attempts == 1:
            raise RuntimeError("interrupted after rename")
        return target

    async def ignore_progress(
        _shard: ShardMetadata, _progress: RepoDownloadProgress
    ) -> None:
        pass

    monkeypatch.setattr(constants, "SKULK_MODELS_PATH", (tmp_path,))
    monkeypatch.setattr(download_utils, "SKULK_MODELS_DIR", tmp_path)
    monkeypatch.setattr(download_utils, "fetch_file_list_with_cache", file_list)
    monkeypatch.setattr(
        download_utils, "download_file_with_retry", interrupt_then_complete
    )

    with pytest.raises(RuntimeError, match="interrupted after rename"):
        await download_shard(
            _shard(_NEW_REVISION), ignore_progress, allow_patterns=["*"]
        )

    assert resolve_model_in_path(model_id) is None
    assert not (canonical / ".skulk-source-revision").exists()
    assert (canonical / ".skulk-source-revision-staging").exists()

    _, preflight = await download_shard(
        _shard(_NEW_REVISION),
        ignore_progress,
        skip_download=True,
        allow_patterns=["*"],
    )
    assert preflight.status == "in_progress"

    model_path, progress = await download_shard(
        _shard(_NEW_REVISION), ignore_progress, allow_patterns=["*"]
    )

    assert attempts == 2
    assert model_path == canonical / "model.gguf"
    assert progress.status == "complete"
    assert (
        canonical / ".skulk-source-revision"
    ).read_text().strip() == _NEW_REVISION
    assert not (canonical / ".skulk-source-revision-staging").exists()


def test_corrupted_revision_marker_is_a_cache_miss(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An unreadable marker must trigger recovery instead of aborting lookup."""

    model_id = ModelId("org/model")
    model_dir = tmp_path / model_id.normalize()
    model_dir.mkdir()
    (model_dir / "model.gguf").write_bytes(b"weights")
    (model_dir / ".skulk-source-revision").write_bytes(b"\xff")
    monkeypatch.setattr(constants, "SKULK_MODELS_PATH", (tmp_path,))

    assert resolve_model_in_path(model_id, _NEW_REVISION) is None


def test_pinned_store_revision_directory_is_discoverable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Runners must resolve qualified store-host directories without staging."""

    model_id = ModelId("org/model")
    model_dir = tmp_path / f"{model_id.normalize()}--revision-{_NEW_REVISION}"
    model_dir.mkdir()
    (model_dir / "model.gguf").write_bytes(b"weights")
    (model_dir / ".skulk-source-revision").write_text(f"{_NEW_REVISION}\n")
    monkeypatch.setattr(constants, "SKULK_MODELS_PATH", (tmp_path,))

    assert resolve_model_in_path(model_id, _NEW_REVISION) == model_dir
    assert build_model_path(model_id, _NEW_REVISION) == model_dir


@pytest.mark.asyncio
async def test_preflight_rejects_complete_cache_from_another_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Status probes must not report stale same-sized artifacts as complete."""

    canonical = tmp_path / "org--model"
    canonical.mkdir()
    (canonical / "model.gguf").write_bytes(b"weights")
    (canonical / ".skulk-source-revision").write_text(f"{_OLD_REVISION}\n")

    async def file_list(*_args: object, **_kwargs: object) -> list[FileListEntry]:
        return [FileListEntry(type="file", path="model.gguf", size=7)]

    terminal_progress: list[RepoDownloadProgress] = []

    async def collect_progress(
        _shard: ShardMetadata, progress: RepoDownloadProgress
    ) -> None:
        terminal_progress.append(progress)

    monkeypatch.setattr(download_utils, "SKULK_MODELS_DIR", tmp_path)
    monkeypatch.setattr(download_utils, "fetch_file_list_with_cache", file_list)

    model_path, progress = await download_shard(
        _shard(_NEW_REVISION),
        collect_progress,
        skip_download=True,
        allow_patterns=["*"],
    )

    assert model_path != canonical / "model.gguf"
    assert progress.status == "not_started"
    assert progress.downloaded.in_bytes == 0
    assert terminal_progress[-1].status == "not_started"
    assert (canonical / "model.gguf").read_bytes() == b"weights"


@pytest.mark.asyncio
async def test_preflight_keeps_complete_replacement_nonterminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A downloaded replacement is incomplete until its atomic installation."""

    canonical = tmp_path / "org--model"
    canonical.mkdir()
    (canonical / "model.gguf").write_bytes(b"old-old")
    (canonical / ".skulk-source-revision").write_text(f"{_OLD_REVISION}\n")
    replacement = download_utils._replacement_model_dir(canonical, _NEW_REVISION)
    replacement.mkdir()
    (replacement / "model.gguf").write_bytes(b"new-new")
    (replacement / ".skulk-source-revision").write_text(f"{_NEW_REVISION}\n")

    async def file_list(*_args: object, **_kwargs: object) -> list[FileListEntry]:
        return [FileListEntry(type="file", path="model.gguf", size=7)]

    terminal_progress: list[RepoDownloadProgress] = []

    async def collect_progress(
        _shard: ShardMetadata, progress: RepoDownloadProgress
    ) -> None:
        terminal_progress.append(progress)

    monkeypatch.setattr(download_utils, "SKULK_MODELS_DIR", tmp_path)
    monkeypatch.setattr(download_utils, "fetch_file_list_with_cache", file_list)

    model_path, progress = await download_shard(
        _shard(_NEW_REVISION),
        collect_progress,
        skip_download=True,
        allow_patterns=["*"],
    )

    assert model_path == replacement / "model.gguf"
    assert progress.status == "in_progress"
    assert progress.downloaded == progress.total
    assert terminal_progress[-1].status == "in_progress"
    with pytest.raises(FileNotFoundError):
        build_model_path(ModelId("org/model"), _NEW_REVISION)
    assert (canonical / "model.gguf").read_bytes() == b"old-old"


@pytest.mark.asyncio
async def test_revision_download_failure_preserves_previous_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A failed replacement must leave the last complete revision untouched."""

    canonical = tmp_path / "org--model"
    canonical.mkdir()
    (canonical / "model.gguf").write_bytes(b"old")
    (canonical / ".skulk-source-revision").write_text(f"{_OLD_REVISION}\n")

    async def file_list(*_args: object, **_kwargs: object) -> list[FileListEntry]:
        return [FileListEntry(type="file", path="model.gguf", size=7)]

    async def fail_download(
        _model_id: ModelId,
        _revision: str,
        _path: str,
        target_dir: Path,
        _on_progress: Callable[[int, int, bool], None],
        on_connection_lost: Callable[[], None],
        skip_internet: bool,
    ) -> Path:
        del on_connection_lost, skip_internet
        assert target_dir != canonical
        assert (canonical / "model.gguf").read_bytes() == b"old"
        raise RuntimeError("replacement failed")

    async def ignore_progress(
        _shard: ShardMetadata, _progress: RepoDownloadProgress
    ) -> None:
        pass

    monkeypatch.setattr(download_utils, "SKULK_MODELS_DIR", tmp_path)
    monkeypatch.setattr(download_utils, "fetch_file_list_with_cache", file_list)
    monkeypatch.setattr(download_utils, "download_file_with_retry", fail_download)

    with pytest.raises(RuntimeError, match="replacement failed"):
        await download_shard(
            _shard(_NEW_REVISION), ignore_progress, allow_patterns=["*"]
        )

    assert (canonical / "model.gguf").read_bytes() == b"old"
    assert (
        canonical / ".skulk-source-revision"
    ).read_text().strip() == _OLD_REVISION


@pytest.mark.asyncio
async def test_revision_download_commits_complete_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A complete replacement atomically takes over the canonical cache path."""

    canonical = tmp_path / "org--model"
    canonical.mkdir()
    (canonical / "model.gguf").write_bytes(b"old")
    (canonical / ".skulk-source-revision").write_text(f"{_OLD_REVISION}\n")

    async def file_list(*_args: object, **_kwargs: object) -> list[FileListEntry]:
        return [FileListEntry(type="file", path="model.gguf", size=7)]

    async def complete_download(
        _model_id: ModelId,
        _revision: str,
        path: str,
        target_dir: Path,
        on_progress: Callable[[int, int, bool], None],
        on_connection_lost: Callable[[], None],
        skip_internet: bool,
    ) -> Path:
        del on_connection_lost, skip_internet
        assert target_dir != canonical
        assert (canonical / "model.gguf").read_bytes() == b"old"
        target = target_dir / path
        target.write_bytes(b"new-new")
        on_progress(7, 7, True)
        return target

    terminal_progress: list[RepoDownloadProgress] = []

    async def collect_progress(
        _shard: ShardMetadata, progress: RepoDownloadProgress
    ) -> None:
        terminal_progress.append(progress)

    monkeypatch.setattr(download_utils, "SKULK_MODELS_DIR", tmp_path)
    monkeypatch.setattr(download_utils, "fetch_file_list_with_cache", file_list)
    monkeypatch.setattr(
        download_utils, "download_file_with_retry", complete_download
    )

    model_path, progress = await download_shard(
        _shard(_NEW_REVISION), collect_progress, allow_patterns=["*"]
    )

    assert model_path == canonical / "model.gguf"
    assert (canonical / "model.gguf").read_bytes() == b"new-new"
    assert (
        canonical / ".skulk-source-revision"
    ).read_text().strip() == _NEW_REVISION
    assert progress.status == "complete"
    assert terminal_progress[-1].status == "complete"
