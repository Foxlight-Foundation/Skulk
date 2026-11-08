"""Public retry preserves native ACE's complete grouped artifact contract."""

# pyright: reportPrivateUsage=false

from collections.abc import Callable
from pathlib import Path

import anyio
import httpx
import pytest

from skulk.api.main import API
from skulk.download import coordinator as coordinator_module
from skulk.download import download_utils
from skulk.download.coordinator import DownloadCoordinator
from skulk.download.impl_shard_downloader import ResumableShardDownloader
from skulk.shared import constants
from skulk.shared.election import ElectionMessage
from skulk.shared.models.model_cards import ModelCard
from skulk.shared.tests.native_music_fixtures import native_music_fixture
from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.common import ModelId, NodeId
from skulk.shared.types.events import Event, IndexedEvent, NodeDownloadProgress
from skulk.shared.types.telemetry import NodeTelemetry
from skulk.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    FileListEntry,
)
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.utils.channels import channel


@pytest.mark.parametrize("shared_primary", [False, True])
async def test_public_retry_recovers_complete_native_sources_without_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shared_primary: bool
) -> None:
    """Retry a source refusal, retain every pin, and reopen entirely offline."""
    card, objects = native_music_fixture(shared_primary=shared_primary)
    node = NodeId("native-retry-node")
    shard = PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=card.n_layers,
        n_layers=card.n_layers,
    )
    refuse_listing = True
    fetched: list[tuple[str, str, str]] = []
    original_listing = download_utils.fetch_file_list_with_cache
    original_download = download_utils.download_file_with_retry

    async def listing(
        repository: ModelId, revision: str, **_kwargs: object
    ) -> list[FileListEntry]:
        """Refuse this source until the operator corrects its availability."""
        if refuse_listing:
            raise FileNotFoundError("Transient native source listing refusal")
        return [
            FileListEntry(type="file", path=path, size=len(data))
            for (repo, rev, path), data in objects.items()
            if (repo, rev) == (str(repository), revision)
        ]

    async def download(
        repository: ModelId,
        revision: str,
        path: str,
        target_dir: Path,
        on_progress: Callable[[int, int, bool], None],
        expected_size: int | None = None,
        expected_object_id: str | None = None,
        **_kwargs: object,
    ) -> Path:
        """Supply tiny exact original data through the real downloader boundary."""
        key = (str(repository), revision, path)
        data = objects[key]
        assert expected_size == len(data) and expected_object_id is not None
        fetched.append(key)
        target = target_dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        on_progress(len(data), len(data), True)
        return target

    async def load(model_id: ModelId) -> ModelCard:
        """Authorize exactly the same immutable native fixture submitted by HTTP."""
        assert model_id == card.model_id
        return card

    monkeypatch.setattr(ModelCard, "load", staticmethod(load))
    monkeypatch.setattr(constants, "SKULK_MODELS_PATH", (tmp_path,))
    monkeypatch.setattr(download_utils, "SKULK_MODELS_DIR", tmp_path)
    monkeypatch.setattr(coordinator_module, "SKULK_MODELS_DIR", tmp_path)
    monkeypatch.setattr(
        "skulk.download.impl_shard_downloader.SKULK_MODELS_DIR", tmp_path
    )
    monkeypatch.setattr(download_utils, "fetch_file_list_with_cache", listing)
    monkeypatch.setattr(download_utils, "download_file_with_retry", download)
    commands, incoming = channel[ForwarderDownloadCommand]()
    events, received = channel[Event]()
    telemetry, _ = channel[NodeTelemetry]()
    forwarded, _ = channel[ForwarderCommand]()
    _, indexed = channel[IndexedEvent]()
    _, election = channel[ElectionMessage]()
    api = API(
        node,
        port=52415,
        event_receiver=indexed,
        command_sender=forwarded,
        download_command_sender=commands,
        election_receiver=election,
        enable_event_log=False,
        mount_dashboard=False,
    )
    coordinator = DownloadCoordinator(
        node,
        ResumableShardDownloader(),
        incoming,
        events,
        telemetry,
    )

    async def terminal() -> DownloadFailed | DownloadCompleted:
        """Observe the actual command actor's ordered terminal publication."""
        while True:
            event = await received.receive()
            if isinstance(event, NodeDownloadProgress) and isinstance(
                event.download_progress, (DownloadFailed, DownloadCompleted)
            ):
                return event.download_progress

    transport = httpx.ASGITransport(app=api.app, client=("127.0.0.1", 50000))
    completed: DownloadCompleted | DownloadFailed | None = None
    with anyio.fail_after(20):
        async with (
            coordinator._tg,
            httpx.AsyncClient(
                transport=transport, base_url="http://localhost"
            ) as client,
        ):
            coordinator._tg.start_soon(coordinator._command_processor)
            body = {
                "targetNodeId": str(node),
                "shardMetadata": shard.model_dump(mode="json"),
            }
            response = await client.post("/download/start", json=body)
            assert response.status_code == 200, response.text
            failed = await terminal()
            assert isinstance(failed, DownloadFailed) and not fetched
            while coordinator.active_downloads:
                await anyio.sleep(0)

            refuse_listing = False
            response = await client.post("/download/start", json=body)
            assert response.status_code == 200, response.text
            completed = await terminal()
            assert isinstance(completed, DownloadCompleted)
            assert completed.attempt_id != failed.attempt_id
            assert set(fetched) == set(objects) and len(fetched) == 14
            while coordinator.active_downloads:
                await anyio.sleep(0)

            response = await client.post("/download/start", json=body)
            assert response.status_code == 200, response.text
            assert await terminal() == completed
            assert len(fetched) == 14
            coordinator._tg.cancel_tasks()

    assert isinstance(completed, DownloadCompleted)
    primary = tmp_path / card.model_id.normalize()
    assert completed.model_directory == str(primary)
    for source in card.native_music_artifact_sources():
        if (source.repository, source.revision) == (
            card.source_repository,
            card.source_revision,
        ):
            root = primary
        else:
            alias, _ = download_utils.companion_artifact_location(
                card, source.repository, source.revision
            )
            root = tmp_path / alias.normalize()
        for item in source.files:
            assert (root / item.path).read_bytes() == objects[
                str(source.repository), source.revision, item.path
            ]

    async def no_network(*_args: object, **_kwargs: object) -> None:
        """An offline reopen may only reuse the complete verified closure."""
        raise AssertionError("Offline native reopen attempted network access")

    monkeypatch.setattr(download_utils, "fetch_file_list_with_cache", original_listing)
    monkeypatch.setattr(download_utils, "download_file_with_retry", original_download)
    monkeypatch.setattr(download_utils, "file_meta", no_network)
    monkeypatch.setattr(httpx.AsyncClient, "send", no_network)
    assert await ResumableShardDownloader(offline=True).ensure_shard(shard) == primary
