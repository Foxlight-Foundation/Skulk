# pyright: reportAny=false
"""Companion-repo handling in ModelStoreDownloader.

The launch smoke (2026-06-06) found that three of the store downloader's
four base-resolution paths returned without fetching the card's companion
repos (MTP sidecar / assistant model / vision weights) — a staged base
model would load and silently run without speculative decoding. These
tests pin the fixed contract: every ``ensure_shard`` resolution also
ensures the declared companions, and a companion failure never fails the
base load.
"""

from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import cast

import pytest

from exo.download.download_utils import RepoDownloadProgress
from exo.download.shard_downloader import ShardDownloader
from exo.shared.models.model_cards import (
    ModelCard,
    ModelTask,
    RuntimeCapabilityCardConfig,
)
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory
from exo.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata
from exo.store.config import StagingNodeConfig
from exo.store.model_store_client import ModelStoreClient, ModelStoreDownloader

_BASE_MODEL = "mlx-community/Qwen-test-9B-4bit"
_SIDECAR_REPO = "FoxlightAI/qwen-test-9b-mtp"


class _UnusedInnerDownloader(ShardDownloader):
    """Inner downloader that must not be reached in store-served tests."""

    async def ensure_shard(
        self, shard: ShardMetadata, config_only: bool = False
    ) -> Path:
        raise AssertionError(
            f"inner downloader should not be used (model {shard.model_card.model_id})"
        )

    def on_progress(
        self,
        callback: Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]],
    ) -> None:
        pass

    async def get_shard_download_status(
        self,
    ) -> AsyncIterator[tuple[Path, RepoDownloadProgress]]:
        if False:
            yield (Path("/unused"), cast(RepoDownloadProgress, object()))

    async def get_shard_download_status_for_shard(
        self, shard: ShardMetadata
    ) -> RepoDownloadProgress:
        raise AssertionError("status queries are not used in this test")


class _RecordingStoreClient:
    """Store client stub that records availability probes and staging calls."""

    def __init__(self, available: set[str], fail_staging: set[str] | None = None):
        self.available = available
        self.fail_staging = fail_staging or set()
        self.staged: list[str] = []

    async def is_model_available(self, model_id: str) -> bool:
        return model_id in self.available

    async def stage_shard(
        self,
        model_id: str,
        dest_path: Path,
        on_progress: Callable[[int, int], Awaitable[None]] | None = None,
    ) -> Path:
        if model_id in self.fail_staging:
            raise RuntimeError(f"simulated staging failure for {model_id}")
        self.staged.append(model_id)
        dest_path.mkdir(parents=True, exist_ok=True)
        (dest_path / "weights.safetensors").write_bytes(b"fake")
        return dest_path


def _shard_with_sidecar() -> PipelineShardMetadata:
    return PipelineShardMetadata(
        model_card=ModelCard(
            model_id=ModelId(_BASE_MODEL),
            storage_size=Memory.from_bytes(4),
            n_layers=2,
            hidden_size=8,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
            runtime=RuntimeCapabilityCardConfig(
                mtp_heads=True,
                mtp_sidecar_repo=_SIDECAR_REPO,
            ),
        ),
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=2,
        n_layers=2,
    )


def _downloader(
    store_client: _RecordingStoreClient, staging_root: Path
) -> ModelStoreDownloader:
    return ModelStoreDownloader(
        inner=_UnusedInnerDownloader(),
        store_client=cast(ModelStoreClient, cast(object, store_client)),
        staging_config=StagingNodeConfig(
            enabled=True,
            node_cache_path=str(staging_root),
        ),
    )


@pytest.mark.anyio
async def test_already_staged_base_still_ensures_companion(tmp_path: Path) -> None:
    """The fast path (base already staged) must still fetch the sidecar.

    This is the exact kite1 failure: base weights present in staging, fast
    path returns, sidecar never fetched, MTP silently disabled.
    """
    base_dir = tmp_path / "mlx-community--Qwen-test-9B-4bit"
    base_dir.mkdir(parents=True)
    (base_dir / "config.json").write_text("{}")
    (base_dir / "model.safetensors").write_bytes(b"fake-weights")

    store = _RecordingStoreClient(available={_SIDECAR_REPO})
    downloader = _downloader(store, tmp_path)

    path = await downloader.ensure_shard(_shard_with_sidecar())

    assert path == base_dir
    assert store.staged == [_SIDECAR_REPO]
    assert (tmp_path / "FoxlightAI--qwen-test-9b-mtp").is_dir()


@pytest.mark.anyio
async def test_store_staged_base_also_stages_companion(tmp_path: Path) -> None:
    """Staging the base from the store must stage the sidecar alongside it."""
    store = _RecordingStoreClient(available={_BASE_MODEL, _SIDECAR_REPO})
    downloader = _downloader(store, tmp_path)

    await downloader.ensure_shard(_shard_with_sidecar())

    assert store.staged == [_BASE_MODEL, _SIDECAR_REPO]


@pytest.mark.anyio
async def test_companion_failure_does_not_fail_base_load(tmp_path: Path) -> None:
    """A sidecar that cannot be fetched logs loudly but the base still loads.

    The runner treats a missing companion as "run without speculation"; a
    fetch failure must not turn a loadable model into a download error.
    """
    base_dir = tmp_path / "mlx-community--Qwen-test-9B-4bit"
    base_dir.mkdir(parents=True)
    (base_dir / "config.json").write_text("{}")
    (base_dir / "model.safetensors").write_bytes(b"fake-weights")

    # Sidecar is reported available but staging it always fails, and the
    # inner downloader (HF fallback) raises if reached — the base load must
    # survive both.
    store = _RecordingStoreClient(
        available={_SIDECAR_REPO}, fail_staging={_SIDECAR_REPO}
    )
    downloader = _downloader(store, tmp_path)

    path = await downloader.ensure_shard(_shard_with_sidecar())

    assert path == base_dir
    assert store.staged == []


@pytest.mark.anyio
async def test_companion_recursion_terminates_on_bare_cards(tmp_path: Path) -> None:
    """Companion shards carry bare cards, so ensuring them must not recurse."""
    store = _RecordingStoreClient(available={_BASE_MODEL, _SIDECAR_REPO})
    downloader = _downloader(store, tmp_path)

    await downloader.ensure_shard(_shard_with_sidecar())
    # One staging call per repo — no repeated/looping companion fetches.
    assert store.staged.count(_SIDECAR_REPO) == 1


@pytest.mark.anyio
async def test_terminal_progress_waits_for_companions(tmp_path: Path) -> None:
    """The base's "complete" progress becomes cluster-visible download state
    that gates model loads — it must not fire until companions are ensured,
    or a runner can load while the sidecar is still staging and silently
    run without speculation (codex review, #213)."""
    base_dir = tmp_path / "mlx-community--Qwen-test-9B-4bit"
    base_dir.mkdir(parents=True)
    (base_dir / "config.json").write_text("{}")
    (base_dir / "model.safetensors").write_bytes(b"fake-weights")

    store = _RecordingStoreClient(available={_SIDECAR_REPO})
    downloader = _downloader(store, tmp_path)

    ordering: list[str] = []

    async def _on_progress(shard: ShardMetadata, progress: object) -> None:
        status = getattr(progress, "status", "?")
        if status == "complete":
            ordering.append(f"complete:{shard.model_card.model_id}")

    original_stage = store.stage_shard

    async def _recording_stage(
        model_id: str,
        dest_path: Path,
        on_progress: Callable[[int, int], Awaitable[None]] | None = None,
    ) -> Path:
        ordering.append(f"staged:{model_id}")
        return await original_stage(model_id, dest_path, on_progress=on_progress)

    store.stage_shard = _recording_stage
    downloader.on_progress(_on_progress)

    await downloader.ensure_shard(_shard_with_sidecar())

    base_complete = ordering.index(f"complete:{_BASE_MODEL}")
    sidecar_staged = ordering.index(f"staged:{_SIDECAR_REPO}")
    assert sidecar_staged < base_complete, (
        f"base completed before its sidecar staged: {ordering}"
    )


@pytest.mark.anyio
async def test_partial_staged_dir_is_restaged(tmp_path: Path) -> None:
    """An interrupted staging (partial file present) must not satisfy the
    fast path — the model gets re-staged (resume via Range) instead of
    being handed to MLX broken (codex round 6 on #213)."""
    base_dir = tmp_path / "mlx-community--Qwen-test-9B-4bit"
    base_dir.mkdir(parents=True)
    (base_dir / "model.safetensors.partial").write_bytes(b"half")

    store = _RecordingStoreClient(available={_BASE_MODEL, _SIDECAR_REPO})
    downloader = _downloader(store, tmp_path)

    await downloader.ensure_shard(_shard_with_sidecar())

    assert _BASE_MODEL in store.staged
