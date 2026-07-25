# pyright: reportPrivateUsage=false
"""Runtime model-store wiring follows cluster-synchronized configuration."""

from pathlib import Path
from typing import cast

import pytest

from skulk import main as main_module
from skulk.api.main import API
from skulk.download.coordinator import DownloadCoordinator
from skulk.download.shard_downloader import ShardDownloader
from skulk.main import Node
from skulk.shared.types.common import NodeId
from skulk.store.config import SkulkConfig, StagingNodeConfig
from skulk.store.model_store_client import ModelStoreClient
from skulk.worker.main import Worker


class _RecordingCoordinator:
    """Capture the downloader wiring installed after config synchronization."""

    def __init__(self) -> None:
        self.received: tuple[ShardDownloader, Path | None] | None = None

    def reconfigure_store(
        self,
        shard_downloader: ShardDownloader,
        staging_cache_path: Path | None,
    ) -> None:
        self.received = (shard_downloader, staging_cache_path)


class _RecordingWorker:
    """Capture worker store wiring installed after config synchronization."""

    def __init__(self) -> None:
        self.received: tuple[
            ModelStoreClient | None,
            StagingNodeConfig | None,
        ] | None = None

    def reconfigure_store(
        self,
        store_client: ModelStoreClient | None,
        staging_config: StagingNodeConfig | None,
    ) -> None:
        self.received = (store_client, staging_config)


class _RecordingAPI:
    """Capture API store wiring installed after config synchronization."""

    def __init__(self) -> None:
        self.received: tuple[
            SkulkConfig | None,
            ModelStoreClient | None,
        ] | None = None

    def apply_synced_config(
        self,
        skulk_config: SkulkConfig | None,
        store_client: ModelStoreClient | None,
    ) -> None:
        self.received = (skulk_config, store_client)


def test_cluster_config_sync_rebinds_live_store_consumers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A routable store advertisement must replace the startup loopback client."""

    config_path = tmp_path / "skulk.yaml"
    config_path.write_text(
        "model_store:\n"
        "  enabled: true\n"
        "  store_host: store-node\n"
        "  store_http_host: 10.0.0.5\n"
        "  store_port: 12415\n"
        "  store_path: /tmp/model-store\n",
        encoding="utf-8",
    )
    refreshed_client = ModelStoreClient("10.0.0.5", 12415)
    refreshed_downloader = cast(ShardDownloader, object())
    refreshed_staging_path = Path("~/.skulk/staging")

    def configure_client(
        _node_id: NodeId,
        _config: SkulkConfig | None,
    ) -> ModelStoreClient:
        return refreshed_client

    def configure_downloader(
        _node_id: NodeId,
        _config: SkulkConfig | None,
        _store_client: ModelStoreClient | None,
        *,
        offline: bool,
    ) -> tuple[ShardDownloader, Path | None]:
        assert offline is False
        return refreshed_downloader, refreshed_staging_path

    monkeypatch.setattr(main_module, "resolve_config_path", lambda: config_path)
    monkeypatch.setattr(
        main_module,
        "_configure_model_store_client",
        configure_client,
    )
    monkeypatch.setattr(
        main_module,
        "_configure_store_download_runtime",
        configure_downloader,
    )

    coordinator = _RecordingCoordinator()
    worker = _RecordingWorker()
    api = _RecordingAPI()
    node = Node.__new__(Node)
    node.node_id = NodeId("local-node")
    node.offline = False
    node.skulk_config = None
    node.store_client = ModelStoreClient("127.0.0.1", 12415)
    node.download_coordinator = cast(DownloadCoordinator, cast(object, coordinator))
    node.worker = cast(Worker, cast(object, worker))
    node.api = cast(API, cast(object, api))

    node._refresh_runtime_config_from_disk()

    assert node.store_client is refreshed_client
    assert node.skulk_config is not None
    assert node.skulk_config.model_store is not None
    assert node.skulk_config.model_store.store_http_host == "10.0.0.5"
    assert coordinator.received == (
        refreshed_downloader,
        refreshed_staging_path,
    )
    assert worker.received is not None
    assert worker.received[0] is refreshed_client
    assert worker.received[1] == node.skulk_config.model_store.staging
    assert api.received == (node.skulk_config, refreshed_client)
