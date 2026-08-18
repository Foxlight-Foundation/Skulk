# pyright: reportPrivateUsage=false
"""Pinned-GGUF completeness checks for node-local staging reuse."""

from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import cast
from unittest.mock import ANY

import pytest

from skulk.download.download_utils import RepoDownloadProgress
from skulk.download.shard_downloader import ShardDownloader
from skulk.shared.models.model_cards import (
    ModelCard,
    ModelId,
    ModelTask,
    VisionCardConfig,
)
from skulk.shared.types.memory import Memory
from skulk.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata
from skulk.store.config import StagingNodeConfig
from skulk.store.installed_cards import (
    InstalledCardRecord,
    build_installed_card_record,
    read_installed_card,
    write_installed_card,
)
from skulk.store.model_store import ModelStore
from skulk.store.model_store_client import (
    ModelStoreClient,
    ModelStoreDownloader,
    _publish_staged_generation,
    _staged_generation_matches,
    _staged_pinned_gguf_missing,
)

_MODEL_ID = "org/multi-quant-GGUF"


def test_signed_staged_fast_path_rejects_local_legacy_sidecar(
    tmp_path: Path,
) -> None:
    """Signed loads must refresh legacy evidence before reusing staged bytes."""

    card = ModelCard(
        model_id=ModelId("org/model"),
        storage_size=Memory.from_mb(1),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        source_revision="a" * 40,
        registry_card_id=f"card_{'a' * 52}",
        registry_snapshot_id="snapshot_1_test",
        registry_provenance="foxlight",
    )
    artifact = tmp_path / "org--model"
    artifact.mkdir()
    (artifact / "config.json").write_text("{}")
    (artifact / "model.safetensors").write_bytes(b"weights")
    legacy = build_installed_card_record(artifact, card)
    write_installed_card(artifact, legacy)
    (artifact / ".skulk-source-revision").write_text(f"{card.source_revision}\n")

    assert legacy.verification == "local_legacy"
    assert not _staged_generation_matches(
        artifact,
        artifact_model_id=str(card.model_id),
        requested_card=card,
        owner_card=None,
        artifact_role="base",
    )


def test_published_generation_ignores_shared_parent_cleanup_failure(
    tmp_path: Path,
) -> None:
    """A committed replacement remains successful while another generation exists."""

    destination = tmp_path / "org--model"
    destination.mkdir()
    (destination / "model.gguf").write_bytes(b"old")
    generation_root = tmp_path / ".replacement-generations" / "org--model"
    replacement = generation_root / "requested"
    replacement.mkdir(parents=True)
    (replacement / "model.gguf").write_bytes(b"new")
    leftover = generation_root / "interrupted"
    leftover.mkdir()

    _publish_staged_generation(replacement, destination)

    assert (destination / "model.gguf").read_bytes() == b"new"
    assert leftover.is_dir()


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
            trust_remote_code=False,
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
async def test_same_revision_card_change_replaces_staged_bytes_transactionally(
    tmp_path: Path,
) -> None:
    requested_card = _aliased_shard().model_card.model_copy(
        update={"source_repository": ModelId("org/new-artifact")}
    )
    old_card = requested_card.model_copy(
        update={
            "source_repository": ModelId("org/old-artifact"),
            "registry_card_id": f"card_{'b' * 52}",
        }
    )
    staged = tmp_path / "org--multi-quant@iq3-xxs"
    staged.mkdir()
    (staged / "model-IQ3_XXS.gguf").write_bytes(b"old-generation")
    write_installed_card(staged, build_installed_card_record(staged, old_card))

    class GenerationStoreClient(_RecordingStoreClient):
        async def is_model_available(
            self, model_id: str, source_revision: str | None = None
        ) -> bool:
            del model_id, source_revision
            return True

        async def request_and_wait_for_download(
            self, model_id: str, **kwargs: object
        ) -> bool:
            del model_id, kwargs
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
            destination = staging_root / model_id.replace("/", "--")
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "model-IQ3_XXS.gguf").write_bytes(b"new-generation")
            write_installed_card(
                destination,
                build_installed_card_record(destination, requested_card),
            )
            return destination

    downloader = ModelStoreDownloader(
        inner=_UnusedInnerDownloader(),
        store_client=cast(
            ModelStoreClient,
            cast(object, GenerationStoreClient()),
        ),
        staging_config=StagingNodeConfig(
            enabled=True,
            node_cache_path=str(tmp_path),
        ),
    )
    requested_shard = _aliased_shard().model_copy(
        update={"model_card": requested_card}
    )

    path = await downloader.ensure_shard(requested_shard)

    assert path == staged
    assert (staged / "model-IQ3_XXS.gguf").read_bytes() == b"new-generation"
    assert not any(tmp_path.glob(".org--multi-quant@iq3-xxs.previous"))


@pytest.mark.anyio
async def test_failed_same_revision_replacement_preserves_old_generation(
    tmp_path: Path,
) -> None:
    requested_card = _aliased_shard().model_card.model_copy(
        update={"source_repository": ModelId("org/new-artifact")}
    )
    old_card = requested_card.model_copy(
        update={
            "source_repository": ModelId("org/old-artifact"),
            "registry_card_id": f"card_{'b' * 52}",
        }
    )
    staged = tmp_path / "org--multi-quant@iq3-xxs"
    staged.mkdir()
    (staged / "model-IQ3_XXS.gguf").write_bytes(b"old-generation")
    write_installed_card(staged, build_installed_card_record(staged, old_card))

    class FailingGenerationStoreClient(_RecordingStoreClient):
        async def is_model_available(
            self, model_id: str, source_revision: str | None = None
        ) -> bool:
            del model_id, source_revision
            return True

        async def request_and_wait_for_download(
            self, model_id: str, **kwargs: object
        ) -> bool:
            del model_id, kwargs
            return True

        async def stage_shard(self, *args: object, **kwargs: object) -> Path:
            del args, kwargs
            raise RuntimeError("transfer interrupted")

    downloader = ModelStoreDownloader(
        inner=_UnusedInnerDownloader(),
        store_client=cast(
            ModelStoreClient,
            cast(object, FailingGenerationStoreClient()),
        ),
        staging_config=StagingNodeConfig(
            enabled=True,
            node_cache_path=str(tmp_path),
        ),
    )
    requested_shard = _aliased_shard().model_copy(
        update={"model_card": requested_card}
    )

    with pytest.raises(RuntimeError, match="transfer interrupted"):
        await downloader.ensure_shard(requested_shard)

    assert (staged / "model-IQ3_XXS.gguf").read_bytes() == b"old-generation"


@pytest.mark.anyio
async def test_store_host_reloads_canonical_path_after_generation_replacement(
    tmp_path: Path,
) -> None:
    requested_card = _aliased_shard().model_card
    old_card = requested_card.model_copy(
        update={
            "source_repository": ModelId("org/old-artifact"),
            "registry_card_id": f"card_{'b' * 52}",
        }
    )
    old_path = tmp_path / "old-generation"
    new_path = tmp_path / "new-generation"
    for directory, card, payload in (
        (old_path, old_card, b"old"),
        (new_path, requested_card, b"new"),
    ):
        directory.mkdir()
        (directory / "model-IQ3_XXS.gguf").write_bytes(payload)
        (directory / ".skulk-source-revision").write_text(
            f"{requested_card.source_revision}\n"
        )
        write_installed_card(
            directory,
            build_installed_card_record(directory, card),
        )

    class ReplacingCanonicalStoreClient(_RecordingStoreClient):
        def __init__(self) -> None:
            super().__init__()
            self.local_store_path = tmp_path
            self.replaced = False

        async def local_model_path(
            self,
            model_id: str,
            source_revision: str | None = None,
        ) -> Path:
            del model_id, source_revision
            return new_path if self.replaced else old_path

        async def request_and_wait_for_download(
            self, model_id: str, **kwargs: object
        ) -> bool:
            del model_id, kwargs
            self.replaced = True
            return True

    downloader = ModelStoreDownloader(
        inner=_UnusedInnerDownloader(),
        store_client=cast(
            ModelStoreClient,
            cast(object, ReplacingCanonicalStoreClient()),
        ),
        staging_config=StagingNodeConfig(
            enabled=False,
            node_cache_path=str(tmp_path / "unused-staging"),
        ),
    )

    path = await downloader.ensure_shard(_aliased_shard())

    assert path == new_path
    assert (path / "model-IQ3_XXS.gguf").read_bytes() == b"new"


@pytest.mark.anyio
async def test_store_host_repairs_corrupt_canonical_projector(
    tmp_path: Path,
) -> None:
    """A direct-store load must repair invalid projector bytes before serving."""

    card = _aliased_shard().model_card.model_copy(
        update={
            "registry_provenance": "foxlight",
            "vision": VisionCardConfig(
                projector_file="mmproj-F16.gguf",
                projector_size=1,
            )
        }
    )
    shard = _aliased_shard().model_copy(update={"model_card": card})
    direct_path = tmp_path / "canonical"
    direct_path.mkdir()
    (direct_path / "model-IQ3_XXS.gguf").write_bytes(b"weights")
    (direct_path / "mmproj-F16.gguf").write_bytes(b"x")
    (direct_path / ".skulk-source-revision").write_text(f"{card.source_revision}\n")
    write_installed_card(
        direct_path,
        build_installed_card_record(direct_path, card),
    )
    (direct_path / "mmproj-F16.gguf").write_bytes(b"y")

    class RepairingCanonicalStoreClient(_RecordingStoreClient):
        def __init__(self) -> None:
            super().__init__()
            self.local_store_path = tmp_path
            self.repaired = False

        async def local_model_path(
            self,
            model_id: str,
            source_revision: str | None = None,
        ) -> Path:
            del model_id, source_revision
            return direct_path

        async def request_and_wait_for_download(
            self,
            model_id: str,
            **kwargs: object,
        ) -> bool:
            del model_id, kwargs
            (direct_path / "mmproj-F16.gguf").write_bytes(b"x")
            write_installed_card(
                direct_path,
                build_installed_card_record(direct_path, card),
            )
            self.repaired = True
            return True

    store_client = RepairingCanonicalStoreClient()
    downloader = ModelStoreDownloader(
        inner=_UnusedInnerDownloader(),
        store_client=cast(ModelStoreClient, cast(object, store_client)),
        staging_config=StagingNodeConfig(enabled=False),
    )

    path = await downloader.ensure_shard(shard)

    assert path == direct_path
    assert store_client.repaired is True
    assert (path / "mmproj-F16.gguf").read_bytes() == b"x"


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
    requested_card = _shard("model-IQ3_XXS.gguf").model_card
    old_card = requested_card.model_copy(update={"family": "stale-family"})
    store = ModelStore(tmp_path)
    store.register_model(
        _MODEL_ID,
        direct_path,
        ["model-IQ3_XXS.gguf"],
        len(b"weights"),
        installed_card=build_installed_card_record(direct_path, old_card),
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

    async def refresh_through_authoritative_store(
        model_id: str,
        **request: object,
    ) -> bool:
        status = await store.request_download(
            model_id,
            pinned_gguf=cast(str | None, request["pinned_gguf"]),
            source_revision=cast(str | None, request["source_revision"]),
            source_repository=cast(str | None, request["source_repository"]),
            model_card=requested_card,
        )
        return status.status == "complete"

    monkeypatch.setattr(
        client,
        "request_and_wait_for_download",
        refresh_through_authoritative_store,
    )
    installed_records: list[InstalledCardRecord] = []
    downloader = ModelStoreDownloader(
        inner=_UnusedInnerDownloader(),
        store_client=client,
        staging_config=StagingNodeConfig(enabled=False),
        installed_card_callback=installed_records.append,
    )

    path = await downloader.ensure_shard(_shard("model-IQ3_XXS.gguf"))

    assert path == direct_path
    refreshed = read_installed_card(direct_path)
    assert refreshed is not None
    assert refreshed.model_card == requested_card
    assert installed_records == [refreshed]
