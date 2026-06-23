"""Tests for the derived per-node health summary (#388)."""

from datetime import datetime, timedelta, timezone

from skulk.api.node_health import (
    DISK_FULL_THRESHOLD,
    DISK_LOW_THRESHOLD,
    UNREACHABLE_WARN_AFTER,
    compute_node_health,
)
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.common import NodeId
from skulk.shared.types.memory import Memory
from skulk.shared.types.profiling import DiskUsage
from skulk.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadProgress,
)
from skulk.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata

_NOW = datetime(2026, 6, 23, 12, 0, 0, tzinfo=timezone.utc)
_NODE = NodeId("node-a")


def _card(model_id: str) -> ModelCard:
    return ModelCard(
        model_id=ModelId(model_id),
        storage_size=Memory.from_gb(1.0),
        n_layers=1,
        hidden_size=1,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    )


def _shard(model_id: str) -> ShardMetadata:
    return PipelineShardMetadata(
        model_card=_card(model_id),
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=1,
        n_layers=1,
    )


def _failed(model_id: str, error: str) -> DownloadFailed:
    return DownloadFailed(
        node_id=_NODE, shard_metadata=_shard(model_id), error_message=error
    )


def _disk(available_gb: float, total_gb: float = 500.0) -> DiskUsage:
    return DiskUsage(
        total=Memory.from_gb(total_gb), available=Memory.from_gb(available_gb)
    )


def test_healthy_node_is_ok_with_no_reasons() -> None:
    health = compute_node_health(
        live_nodes={_NODE: _NOW},
        downloads={},
        node_disk={_NODE: _disk(available_gb=400.0)},
        now=_NOW,
    )
    assert set(health) == {"node-a"}
    assert health["node-a"].level == "ok"
    assert list(health["node-a"].reasons) == []


def test_download_failed_is_error_and_names_model_and_error() -> None:
    downloads: dict[NodeId, list[DownloadProgress]] = {
        _NODE: [_failed("org/Big-Model", "No space left on device")]
    }
    health = compute_node_health(
        live_nodes={_NODE: _NOW}, downloads=downloads, node_disk={}, now=_NOW
    )
    node = health["node-a"]
    assert node.level == "error"
    assert len(node.reasons) == 1
    reason = node.reasons[0]
    assert reason.code == "download_failed"
    assert "org/Big-Model" in reason.message
    assert "No space left on device" in reason.message
    assert reason.remediation


def test_completed_download_does_not_flag() -> None:
    downloads: dict[NodeId, list[DownloadProgress]] = {
        _NODE: [
            DownloadCompleted(
                node_id=_NODE, shard_metadata=_shard("org/ok"), total=Memory.from_gb(1)
            )
        ]
    }
    health = compute_node_health(
        live_nodes={_NODE: _NOW}, downloads=downloads, node_disk={}, now=_NOW
    )
    assert health["node-a"].level == "ok"


def test_disk_low_is_warn() -> None:
    # Just under the low threshold, comfortably above full.
    available = (DISK_LOW_THRESHOLD.in_gb + DISK_FULL_THRESHOLD.in_gb) / 2
    health = compute_node_health(
        live_nodes={_NODE: _NOW},
        downloads={},
        node_disk={_NODE: _disk(available_gb=available)},
        now=_NOW,
    )
    node = health["node-a"]
    assert node.level == "warn"
    assert node.reasons[0].code == "disk_low"


def test_disk_full_is_error() -> None:
    health = compute_node_health(
        live_nodes={_NODE: _NOW},
        downloads={},
        node_disk={_NODE: _disk(available_gb=DISK_FULL_THRESHOLD.in_gb / 2)},
        now=_NOW,
    )
    node = health["node-a"]
    assert node.level == "error"
    assert node.reasons[0].code == "disk_full"


def test_disk_above_threshold_is_ok() -> None:
    health = compute_node_health(
        live_nodes={_NODE: _NOW},
        downloads={},
        node_disk={_NODE: _disk(available_gb=DISK_LOW_THRESHOLD.in_gb + 1.0)},
        now=_NOW,
    )
    assert health["node-a"].level == "ok"


def test_stale_heartbeat_is_unreachable_warn() -> None:
    stale = _NOW - (UNREACHABLE_WARN_AFTER + timedelta(seconds=1))
    health = compute_node_health(
        live_nodes={_NODE: stale}, downloads={}, node_disk={}, now=_NOW
    )
    node = health["node-a"]
    assert node.level == "warn"
    assert node.reasons[0].code == "unreachable"


def test_recent_heartbeat_is_not_unreachable() -> None:
    recent = _NOW - timedelta(seconds=1)
    health = compute_node_health(
        live_nodes={_NODE: recent}, downloads={}, node_disk={}, now=_NOW
    )
    assert health["node-a"].level == "ok"


def test_error_dominates_warn_when_multiple_reasons() -> None:
    # A node both out of disk AND with a failed download reports error overall,
    # but surfaces every reason so the operator sees the full picture.
    downloads: dict[NodeId, list[DownloadProgress]] = {
        _NODE: [_failed("org/m", "boom")]
    }
    health = compute_node_health(
        live_nodes={_NODE: _NOW - (UNREACHABLE_WARN_AFTER + timedelta(seconds=1))},
        downloads=downloads,
        node_disk={_NODE: _disk(available_gb=0.5)},
        now=_NOW,
    )
    node = health["node-a"]
    assert node.level == "error"
    codes = {reason.code for reason in node.reasons}
    assert codes == {"download_failed", "disk_full", "unreachable"}


def test_only_live_nodes_get_entries() -> None:
    other = NodeId("node-b")
    health = compute_node_health(
        live_nodes={_NODE: _NOW},
        downloads={other: [_failed("org/m", "boom")]},
        node_disk={other: _disk(available_gb=0.1)},
        now=_NOW,
    )
    # node-b is not live, so it gets no health entry even though it has signals.
    assert set(health) == {"node-a"}
    assert health["node-a"].level == "ok"
