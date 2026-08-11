# pyright: reportPrivateUsage=false
"""Fabric-native node artifact availability and registry projection tests."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MethodType
from typing import cast

import anyio
import pytest

import skulk.api.main as api_main
import skulk.main as skulk_main
import skulk.store.artifact_inventory as artifact_inventory
from skulk.api.main import API
from skulk.api.types.api import ReconciliationStatus
from skulk.main import Node
from skulk.routing.router import Router
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.topology import Topology
from skulk.shared.types.artifact_inventory import (
    ARTIFACT_INVENTORY_STALE_SECONDS,
    NodeArtifactAvailability,
    NodeArtifactInventory,
)
from skulk.shared.types.common import NodeId
from skulk.shared.types.events import IndexedEvent, StagedModelEvicted
from skulk.shared.types.memory import Memory
from skulk.shared.types.state import State
from skulk.shared.types.telemetry import NodeTelemetry, TelemetryView
from skulk.store.config import ModelStoreConfig, SkulkConfig, StagingNodeConfig
from skulk.store.installed_cards import (
    InstalledCardRecord,
    VerifiedDetachedInstalledCardCache,
    build_installed_card_record,
    write_installed_card,
)
from skulk.store.model_store_client import ModelStoreClient
from skulk.utils.channels import channel


class _RecordingTelemetrySender:
    """Minimal telemetry sender used by projection and publication tests."""

    def __init__(self) -> None:
        self.messages: list[NodeTelemetry] = []

    async def send(self, item: NodeTelemetry) -> None:
        """Record one offered telemetry message."""

        self.messages.append(item)


class _RecordingRouter:
    """Expose one stable sender through the node's router-shaped surface."""

    def __init__(self, sender: _RecordingTelemetrySender) -> None:
        self.sender = sender

    def telemetry_sender(self) -> _RecordingTelemetrySender:
        """Return the sender used by the node-owned publisher."""

        return self.sender


def _publisher_node(sender: _RecordingTelemetrySender) -> Node:
    """Build the node fields required by inventory publication tests."""

    node = object.__new__(Node)
    node.router = cast(Router, cast(object, _RecordingRouter(sender)))
    node.node_id = NodeId("cache-node")
    node.worker = None
    node.api = None
    node.master = None
    node.skulk_config = None
    node.store_client = None
    node._artifact_inventory_detached_cache = VerifiedDetachedInstalledCardCache()
    (
        node._artifact_inventory_trigger_sender,
        node._artifact_inventory_trigger_receiver,
    ) = channel[None](1)
    return node


def _model_card() -> ModelCard:
    """Return a minimal signed-generation-shaped card for filesystem scans."""

    return ModelCard(
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


def _write_artifact(root: Path, card: ModelCard) -> InstalledCardRecord:
    """Create one complete installed artifact beneath a test root."""

    artifact = root / "org--model"
    artifact.mkdir(parents=True)
    (artifact / "model.safetensors").write_bytes(b"weights")
    (artifact / ".skulk-source-revision").write_text(f"{'a' * 40}\n")
    record = build_installed_card_record(artifact, card)
    write_installed_card(artifact, record)
    return record


def _inventory(
    identity: str,
    *,
    store_host: bool = False,
    truncated: bool = False,
    in_use: bool = False,
) -> NodeArtifactInventory:
    """Build one compact availability reading for projection tests."""

    return NodeArtifactInventory(
        artifacts=[
            NodeArtifactAvailability(
                model_id="org/model",
                installed_identity=identity,
                size_bytes=1024,
                last_used_epoch_seconds=42,
                in_use=in_use,
                manifest_complete=True,
            ),
        ],
        store_host=store_host,
        truncated=truncated,
    )


def _projection_api(*node_ids: str) -> API:
    """Build only the API state required by the inventory projection."""

    local_node = NodeId(node_ids[0])
    topology = Topology()
    for node_id in node_ids:
        topology.add_node(NodeId(node_id))
    api = object.__new__(API)
    api.node_id = local_node
    api.state = State().model_copy(update={"topology": topology})
    api._telemetry_view = TelemetryView()
    api._telemetry_sender = _RecordingTelemetrySender()
    api._artifact_inventory_expected_since = {}
    api._artifact_inventory_nodes_seen = set()
    return api


def _apply_inventory(
    api: API,
    node_id: str,
    inventory: NodeArtifactInventory,
    *,
    age_seconds: float = 0,
) -> None:
    """Apply one reading with a deterministic local receipt age."""

    api._telemetry_view.apply(
        NodeTelemetry(node_id=NodeId(node_id), info=inventory),
        received_at=datetime.now(tz=timezone.utc) - timedelta(seconds=age_seconds),
    )


def test_registry_inventory_is_current_only_when_every_node_is_fresh() -> None:
    """Complete fresh readings establish a current fleet-wide location set."""

    api = _projection_api("node-a", "node-b")
    _apply_inventory(api, "node-a", _inventory("generation", store_host=True))
    _apply_inventory(api, "node-b", _inventory("generation", in_use=True))

    status, locations, store_hosts = api._cache_inventory_projection()

    assert status.state == "current"
    assert status.observed_nodes == 2
    assert status.expected_nodes == 2
    assert {location.node_id for location in locations["generation"]} == {
        "node-a",
        "node-b",
    }
    assert store_hosts == (NodeId("node-a"),)


def test_healthy_api_nodes_project_the_same_location_set() -> None:
    """Every API derives availability from the same fabric telemetry truth."""

    apis = (
        _projection_api("node-a", "node-b"),
        _projection_api("node-b", "node-a"),
    )
    for api in apis:
        _apply_inventory(api, "node-a", _inventory("generation", store_host=True))
        _apply_inventory(api, "node-b", _inventory("generation", in_use=True))

    projections = [api._cache_inventory_projection() for api in apis]

    assert projections[0][0] == projections[1][0]
    assert projections[0][1] == projections[1][1]
    assert projections[0][2] == projections[1][2]


def test_registry_inventory_syncs_while_a_new_node_has_not_published() -> None:
    """Initial convergence is not mislabeled as degraded truth."""

    api = _projection_api("node-a", "node-b")
    _apply_inventory(api, "node-a", _inventory("generation"))

    status, locations, _store_hosts = api._cache_inventory_projection()

    assert status.state == "syncing"
    assert status.observed_nodes == 1
    assert [location.node_id for location in locations["generation"]] == ["node-a"]


def test_registry_inventory_treats_a_rejoined_node_as_new_convergence() -> None:
    """Membership pruning forgets prior publication history for a later rejoin."""

    api = _projection_api("node-a", "node-b")
    _apply_inventory(api, "node-a", _inventory("generation"))
    _apply_inventory(api, "node-b", _inventory("generation"))
    assert api._cache_inventory_projection()[0].state == "current"

    local_only = Topology()
    local_only.add_node(NodeId("node-a"))
    api.state = api.state.model_copy(update={"topology": local_only})
    api._telemetry_view.prune(NodeId("node-b"))
    assert api._cache_inventory_projection()[0].state == "current"

    rejoined = Topology()
    rejoined.add_node(NodeId("node-a"))
    rejoined.add_node(NodeId("node-b"))
    api.state = api.state.model_copy(update={"topology": rejoined})

    assert api._cache_inventory_projection()[0].state == "syncing"


def test_registry_inventory_keeps_stale_locations_as_degraded_partial_truth() -> None:
    """Staleness changes confidence without erasing the last known location."""

    api = _projection_api("node-a", "node-b")
    _apply_inventory(
        api,
        "node-a",
        _inventory("generation", store_host=True),
        age_seconds=ARTIFACT_INVENTORY_STALE_SECONDS + 1,
    )
    _apply_inventory(api, "node-b", _inventory("generation"))

    status, locations, store_hosts = api._cache_inventory_projection()

    assert status.state == "degraded"
    assert status.observed_nodes == 2
    assert {location.node_id for location in locations["generation"]} == {
        "node-a",
        "node-b",
    }
    assert store_hosts == (NodeId("node-a"),)


def test_registry_inventory_marks_truncated_truth_degraded() -> None:
    """A bounded reading remains visible but can never claim completeness."""

    api = _projection_api("node-a")
    _apply_inventory(api, "node-a", _inventory("generation", truncated=True))

    status, locations, _store_hosts = api._cache_inventory_projection()

    assert status.state == "degraded"
    assert [location.node_id for location in locations["generation"]] == ["node-a"]


def test_registry_inventory_becomes_unavailable_without_any_usable_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A node that never publishes cannot leave the API syncing forever."""

    api = _projection_api("node-a")
    _status, _locations, _store_hosts = api._cache_inventory_projection()
    expected_since = api._artifact_inventory_expected_since[NodeId("node-a")]
    monkeypatch.setattr(
        api_main.time,
        "monotonic",
        lambda: expected_since + ARTIFACT_INVENTORY_STALE_SECONDS + 1,
    )

    status, locations, store_hosts = api._cache_inventory_projection()

    assert status.state == "unavailable"
    assert status.observed_nodes == 0
    assert locations == {}
    assert store_hosts == ()


async def test_inventory_loop_publishes_at_startup_periodically_and_after_a_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The repair cadence and debounced mutation path both publish readings."""

    node = _publisher_node(_RecordingTelemetrySender())
    publications: list[int] = []

    async def record_publication(_self: Node) -> None:
        publications.append(len(publications) + 1)

    node._publish_artifact_inventory = MethodType(record_publication, node)
    monkeypatch.setattr(skulk_main, "ARTIFACT_INVENTORY_REFRESH_SECONDS", 0.02)
    monkeypatch.setattr(skulk_main, "ARTIFACT_INVENTORY_DEBOUNCE_SECONDS", 0.001)

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(node._artifact_inventory_loop)
        with anyio.fail_after(0.25):
            while len(publications) < 2:
                await anyio.sleep(0.001)
        node._mark_artifact_inventory_dirty()
        with anyio.fail_after(0.25):
            while len(publications) < 3:
                await anyio.sleep(0.001)
        task_group.cancel_scope.cancel()

    assert len(publications) >= 3


async def test_inventory_loop_contains_scan_failure_and_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient filesystem failure cannot cancel the node lifetime group."""

    node = _publisher_node(_RecordingTelemetrySender())
    attempts = 0

    async def fail_then_publish(_self: Node) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("temporary model root failure")

    node._publish_artifact_inventory = MethodType(fail_then_publish, node)
    monkeypatch.setattr(skulk_main, "ARTIFACT_INVENTORY_REFRESH_SECONDS", 0.01)

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(node._artifact_inventory_loop)
        with anyio.fail_after(0.25):
            while attempts < 2:
                await anyio.sleep(0.001)
        task_group.cancel_scope.cancel()

    assert attempts >= 2


async def test_inventory_publisher_excludes_canonical_store_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The store host advertises its role without gossiping canonical entries."""

    store_root = tmp_path / "store"
    cache_root = tmp_path / "cache"
    store_root.mkdir()
    cache_root.mkdir()
    card = _model_card()
    _write_artifact(store_root, card)
    monkeypatch.setattr(
        artifact_inventory.shared_constants,
        "SKULK_MODELS_DIR",
        store_root,
    )
    monkeypatch.setattr(
        artifact_inventory.shared_constants,
        "SKULK_MODELS_PATH",
        (store_root,),
    )

    async def cards() -> tuple[ModelCard, ...]:
        return (card,)

    monkeypatch.setattr(skulk_main, "get_all_model_cards", cards)
    sender = _RecordingTelemetrySender()
    node = _publisher_node(sender)
    node.node_id = NodeId("store-node")
    node.skulk_config = SkulkConfig(
        model_store=ModelStoreConfig(
            store_host="store-node",
            store_path=str(store_root),
            staging=StagingNodeConfig(node_cache_path=str(cache_root)),
        )
    )
    node.store_client = ModelStoreClient(
        store_host="store-node",
        local_store_path=store_root,
    )

    await node._publish_artifact_inventory()

    assert len(sender.messages) == 1
    published = sender.messages[0].info
    assert isinstance(published, NodeArtifactInventory)
    assert published.store_host
    assert published.artifacts == []


async def test_node_event_tap_requests_inventory_refresh_without_an_api() -> None:
    """A no-API node still turns storage transitions into publication hints."""

    node = _publisher_node(_RecordingTelemetrySender())
    event_sender, event_receiver = channel[IndexedEvent](1)
    node.artifact_inventory_event_receiver = event_receiver

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(node._observe_artifact_inventory_events)
        await event_sender.send(
            IndexedEvent(
                idx=1,
                event=StagedModelEvicted(model_id=ModelId("org/model")),
            )
        )
        with anyio.fail_after(0.25):
            await node._artifact_inventory_trigger_receiver.receive()
        task_group.cancel_scope.cancel()


async def test_registry_synthesizes_store_local_and_keeps_node_cache_distinct(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canonical locality is synthesized while cache locations come from telemetry."""

    store_root = tmp_path / "store"
    store_root.mkdir()
    card = _model_card()
    record = _write_artifact(store_root, card)
    registry_entry: dict[str, object] = {
        "model_id": str(card.model_id),
        "store_path": "org--model",
        "files": ["model.safetensors"],
        "downloaded_at": "2026-08-11T12:00:00+00:00",
        "total_bytes": 7,
        "source_revision": card.source_revision,
        "source_repository": None,
        "repo_has_projector": None,
        "installed_card": record.model_dump(mode="json"),
    }
    client = ModelStoreClient(
        store_host="store-node",
        local_store_path=store_root,
    )

    async def fetch_registry() -> list[dict[str, object]]:
        return [registry_entry]

    async def cards() -> tuple[ModelCard, ...]:
        return (card,)

    def current_card(_model_id: ModelId) -> ModelCard:
        return card

    def current_card_id(_model_id: ModelId) -> str | None:
        return card.registry_card_id

    monkeypatch.setattr(client, "fetch_registry", fetch_registry)
    monkeypatch.setattr(api_main, "get_all_model_cards", cards)
    monkeypatch.setattr(api_main, "get_current_registry_card", current_card)
    monkeypatch.setattr(
        api_main,
        "get_current_registry_card_id",
        current_card_id,
    )
    api = _projection_api("store-node", "worker-node")
    api._store_client = client
    api._reconciliation_status = ReconciliationStatus()
    _apply_inventory(
        api,
        "store-node",
        NodeArtifactInventory(artifacts=[], store_host=True, truncated=False),
    )
    _apply_inventory(
        api,
        "worker-node",
        NodeArtifactInventory(
            artifacts=[
                NodeArtifactAvailability(
                    model_id=str(card.model_id),
                    installed_identity=record.installed_identity,
                    size_bytes=7,
                    last_used_epoch_seconds=42,
                    in_use=True,
                    manifest_complete=True,
                )
            ],
            store_host=False,
            truncated=False,
        ),
    )

    response = await api.get_store_registry()

    assert response.cache_inventory.state == "current"
    locations = response.entries[0].cached_on_nodes
    assert {
        (location.node_id, location.location_kind, location.in_use)
        for location in locations
    } == {
        ("store-node", "store_local", False),
        ("worker-node", "node_cache", True),
    }
