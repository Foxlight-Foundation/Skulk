# pyright: reportPrivateUsage=false
"""Pinned-GGUF completeness checks for node-local staging reuse."""

from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import cast
from unittest.mock import ANY

import pytest

from skulk.download.download_utils import RepoDownloadProgress
from skulk.download.shard_downloader import ShardDownloader
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.memory import Memory
from skulk.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata
from skulk.store.config import StagingNodeConfig
from skulk.store.installed_cards import (
    build_installed_card_record,
    write_installed_card,
)
from skulk.store.model_store import ModelStore
from skulk.store.model_store_client import (
    ModelStoreClient,
    ModelStoreDownloader,
    _staged_generation_matches,
    _staged_pinned_gguf_missing,
)

_MODEL_ID = "org/multi-quant-GGUF"


class _UnusedInnerDownloader(ShardDownloader):
    async def ensure_shard(
        self, shard: ShardMetadata, config_only: bool = False
    ) -> Path:
        raise AssertionError("store-backed test must not use the inner downloader")

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
    def __init__(self) -> None:
        self.availability_checks = 0
        self.download_requests: list[tuple[str, str | None]] = []
        self.stage_calls = 0

    async def is_model_available(
        self, model_id: str, source_revision: str | None = None
    ) -> bool:
        assert model_id == _MODEL_ID
        assert source_revision is None
        self.availability_checks += 1
        return True

    async def request_and_wait_for_download(
        self,
        model_id: str,
        *,
        pinned_gguf: str | None = None,
        extra_pinned_gguf: list[str] | None = None,
        source_revision: str | None = None,
        source_repository: str | None = None,
        registry_card_id: str | None = None,
    ) -> bool:
        assert not extra_pinned_gguf
        assert source_revision is None
        assert source_repository is None or "/" in source_repository
        assert registry_card_id is None
        self.download_requests.append((model_id, pinned_gguf))
        return True

    async def stage_shard(
        self,
        model_id: str,
        staging_root: Path,
        on_progress: Callable[[int, int], Awaitable[None]] | None = None,
        source_revision: str | None = None,
        capacity_preflight: Callable[[int], Awaitable[None]] | None = None,
    ) -> Path:
        assert model_id == _MODEL_ID
        assert source_revision is None
        if capacity_preflight is not None:
            await capacity_preflight(8)
        self.stage_calls += 1
        dest_path = staging_root / model_id.replace("/", "--")
        dest_path.mkdir(parents=True, exist_ok=True)
        (dest_path / "model-IQ3_XXS.gguf").write_bytes(b"selected")
        return dest_path


def _shard(gguf_file: str) -> PipelineShardMetadata:
    return PipelineShardMetadata(
        model_card=ModelCard(
            model_id=ModelId(_MODEL_ID),
            storage_size=Memory.from_bytes(8),
            n_layers=1,
            hidden_size=1,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
            gguf_file=gguf_file,
        ),
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=1,
        n_layers=1,
    )


def _aliased_shard() -> PipelineShardMetadata:
    return PipelineShardMetadata(
        model_card=ModelCard(
            model_id=ModelId("org/multi-quant@iq3-xxs"),
            source_repository=ModelId(_MODEL_ID),
            source_revision="a" * 40,
            storage_size=Memory.from_bytes(8),
            n_layers=1,
            hidden_size=1,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
            gguf_file="model-IQ3_XXS.gguf",
            trust_remote_code=False,
            registry_card_id=f"card_{'a' * 52}",
            registry_snapshot_id="snapshot_1_test",
        ),
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=1,
        n_layers=1,
    )


def test_pinned_shard_group_requires_every_sibling(tmp_path: Path) -> None:
    shard = _shard("weights/model-IQ3-00001-of-00002.gguf")
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "model-IQ3-00001-of-00002.gguf").write_bytes(b"one")

    assert _staged_pinned_gguf_missing(shard, tmp_path) is True

    (weights / "model-IQ3-00002-of-00002.gguf").write_bytes(b"two")
    assert _staged_pinned_gguf_missing(shard, tmp_path) is False


def test_sidecar_mismatch_cannot_fall_back_to_matching_revision(
    tmp_path: Path,
) -> None:
    requested = _aliased_shard().model_card
    staged = tmp_path / "org--multi-quant@iq3-xxs"
    staged.mkdir()
    (staged / "model-IQ3_XXS.gguf").write_bytes(b"old-generation")
    (staged / ".skulk-source-revision").write_text(f"{'a' * 40}\n")
    old_card = requested.model_copy(
        update={"registry_card_id": f"card_{'b' * 52}"}
    )
    write_installed_card(
        staged,
        build_installed_card_record(staged, old_card),
    )

    assert not _staged_generation_matches(
        staged,
        artifact_model_id=str(requested.model_id),
        requested_card=requested,
        owner_card=None,
        artifact_role="base",
    )


@pytest.mark.anyio
async def test_different_staged_quant_is_replaced_from_store(tmp_path: Path) -> None:
    staged = tmp_path / "org--multi-quant-GGUF"
    staged.mkdir()
    (staged / "model-Q4_K_M.gguf").write_bytes(b"old")
    store = _RecordingStoreClient()
    downloader = ModelStoreDownloader(
        inner=_UnusedInnerDownloader(),
        store_client=cast(ModelStoreClient, cast(object, store)),
        staging_config=StagingNodeConfig(
            enabled=True,
            node_cache_path=str(tmp_path),
        ),
    )

    path = await downloader.ensure_shard(_shard("model-IQ3_XXS.gguf"))

    assert path == staged
    assert store.availability_checks == 1
    assert store.download_requests == [(_MODEL_ID, "model-IQ3_XXS.gguf")]
    assert store.stage_calls == 1
    assert (staged / "model-IQ3_XXS.gguf").is_file()


@pytest.mark.anyio
async def test_store_download_keeps_alias_but_fetches_artifact_repository(
    tmp_path: Path,
) -> None:
    """Store identity and upstream byte source remain independent."""
    observed: dict[str, object] = {}

    class AliasStoreClient(_RecordingStoreClient):
        async def is_model_available(
            self, model_id: str, source_revision: str | None = None
        ) -> bool:
            observed["availability"] = (model_id, source_revision)
            return False

        async def request_and_wait_for_download(
            self,
            model_id: str,
            **kwargs: object,
        ) -> bool:
            observed["download"] = (model_id, kwargs)
            return True

        async def stage_shard(
            self,
            model_id: str,
            staging_root: Path,
            on_progress: Callable[[int, int], Awaitable[None]] | None = None,
            source_revision: str | None = None,
            capacity_preflight: Callable[[int], Awaitable[None]] | None = None,
        ) -> Path:
            del on_progress, source_revision, capacity_preflight
            observed["stage"] = model_id
            path = staging_root / model_id.replace("/", "--")
            path.mkdir(parents=True, exist_ok=True)
            (path / "model-IQ3_XXS.gguf").write_bytes(b"selected")
            return path

    store = AliasStoreClient()
    downloader = ModelStoreDownloader(
        inner=_UnusedInnerDownloader(),
        store_client=cast(ModelStoreClient, cast(object, store)),
        staging_config=StagingNodeConfig(
            enabled=True,
            node_cache_path=str(tmp_path),
        ),
    )

    await downloader.ensure_shard(_aliased_shard())

    alias = "org/multi-quant@iq3-xxs"
    assert observed["availability"] == (alias, "a" * 40)
    assert observed["stage"] == alias
    assert observed["download"] == (
        alias,
        {
            "on_progress": ANY,
            "pinned_gguf": "model-IQ3_XXS.gguf",
            "extra_pinned_gguf": [],
            "source_revision": "a" * 40,
            "source_repository": _MODEL_ID,
            "registry_card_id": f"card_{'a' * 52}",
        },
    )


@pytest.mark.anyio
async def test_staging_disabled_direct_load_does_not_probe_http(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A complete local canonical model must not depend on store HTTP health."""

    direct_path = tmp_path / "canonical"
    direct_path.mkdir()
    (direct_path / "model-IQ3_XXS.gguf").write_bytes(b"weights")
    ModelStore(tmp_path).register_model(
        _MODEL_ID,
        direct_path,
        ["model-IQ3_XXS.gguf"],
        len(b"weights"),
    )
    client = ModelStoreClient("localhost", local_store_path=tmp_path)

    async def unexpected_registry_fetch() -> list[dict[str, object]]:
        raise AssertionError("local direct loads must read the local registry")

    async def unexpected_availability_probe(
        _model_id: str, _source_revision: str | None = None
    ) -> bool:
        raise AssertionError("local direct loads must not probe store HTTP")

    monkeypatch.setattr(client, "fetch_registry", unexpected_registry_fetch)
    monkeypatch.setattr(client, "is_model_available", unexpected_availability_probe)
    downloader = ModelStoreDownloader(
        inner=_UnusedInnerDownloader(),
        store_client=client,
        staging_config=StagingNodeConfig(enabled=False),
    )

    path = await downloader.ensure_shard(_shard("model-IQ3_XXS.gguf"))

    assert path == direct_path
