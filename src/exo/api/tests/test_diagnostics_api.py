"""Tests for read-only node and cluster diagnostics endpoints."""

from typing import cast

import httpx
import pytest
from fastapi.testclient import TestClient

import exo.api.main as api_main
from exo.api.main import API
from exo.shared.election import ElectionMessage
from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from exo.shared.types.common import ModelId, NodeId
from exo.shared.types.diagnostics import NodeDiagnostics
from exo.shared.types.events import IndexedEvent
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.tasks import StartWarmup, TaskStatus
from exo.shared.types.worker.instances import InstanceId, MlxRingInstance
from exo.shared.types.worker.runners import RunnerId, RunnerWarmingUp, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.utils.channels import channel


def _json_object(response: httpx.Response) -> dict[str, object]:
    """Return a JSON response payload as a typed object mapping."""

    return cast(dict[str, object], cast(object, response.json()))


def _json_list(value: object) -> list[object]:
    """Narrow a nested JSON array from a response payload."""

    return cast(list[object], value)


def _json_mapping(value: object) -> dict[str, object]:
    """Narrow one nested JSON object from a response payload."""

    return cast(dict[str, object], value)


def _build_api(node_id: str = "local-node") -> API:
    """Create a minimal API instance for diagnostics endpoint testing."""

    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    _, event_receiver = channel[IndexedEvent]()
    _, election_receiver = channel[ElectionMessage]()
    return API(
        NodeId(node_id),
        port=52415,
        event_receiver=event_receiver,
        command_sender=command_sender,
        download_command_sender=download_sender,
        election_receiver=election_receiver,
        enable_event_log=False,
        mount_dashboard=False,
    )


def _running_state_without_master_placement() -> State:
    """Build state where the master is outside a warming placement."""

    model_card = ModelCard(
        model_id=ModelId("mlx-community/gemma-4-26b-a4b-it-4bit"),
        storage_size=Memory.from_mb(100),
        n_layers=30,
        hidden_size=2816,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )
    instance_id = InstanceId("instance-1")
    runner_1 = RunnerId("runner-1")
    runner_2 = RunnerId("runner-2")
    shard_1 = PipelineShardMetadata(
        model_card=model_card,
        device_rank=0,
        world_size=2,
        start_layer=0,
        end_layer=15,
        n_layers=30,
    )
    shard_2 = PipelineShardMetadata(
        model_card=model_card,
        device_rank=1,
        world_size=2,
        start_layer=15,
        end_layer=30,
        n_layers=30,
    )
    warmup_task = StartWarmup(
        instance_id=instance_id,
        task_status=TaskStatus.Running,
    )
    return State(
        instances={
            instance_id: MlxRingInstance(
                instance_id=instance_id,
                shard_assignments=ShardAssignments(
                    model_id=model_card.model_id,
                    runner_to_shard={runner_1: shard_1, runner_2: shard_2},
                    node_to_runner={
                        NodeId("local-node"): runner_1,
                        NodeId("peer-node"): runner_2,
                    },
                ),
                hosts_by_node={},
                ephemeral_port=58484,
            )
        },
        runners={runner_1: RunnerWarmingUp(), runner_2: RunnerWarmingUp()},
        tasks={warmup_task.task_id: warmup_task},
    )


def test_node_diagnostics_marks_master_outside_placement() -> None:
    """Local diagnostics should expose master-vs-placement mismatch warnings."""

    api = _build_api("local-node")
    api._master_node_id = NodeId("master-node")  # pyright: ignore[reportPrivateUsage]
    api.state = _running_state_without_master_placement()
    client = TestClient(api.app)

    response = client.get("/v1/diagnostics/node")

    assert response.status_code == 200
    body = _json_object(response)
    runtime = _json_mapping(body["runtime"])
    assert runtime["masterNodeId"] == "master-node"
    assert runtime["isMaster"] is False
    placements = _json_list(body["placements"])
    placement = _json_mapping(placements[0])
    assert placement["masterIsPlacementNode"] is False
    assert placement["localNodeIsPlacementNode"] is True
    warnings = _json_list(placement["warnings"])
    assert "Current master is not a placement node for this instance." in warnings


def test_cluster_diagnostics_returns_local_and_peer_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cluster diagnostics should fan out and tolerate reachable peers."""

    api = _build_api("local-node")
    client = TestClient(api.app)

    async def _reachable_peer_api_urls() -> dict[str, str]:
        return {"peer-node": "http://peer-node:52415"}

    class _FakeAsyncClient:
        def __init__(self, responses: dict[str, httpx.Response]) -> None:
            self._responses = responses

        async def __aenter__(self) -> "_FakeAsyncClient":
            return self

        async def __aexit__(
            self,
            _exc_type: object,
            _exc: object,
            _tb: object,
        ) -> None:
            return None

        async def get(self, url: str) -> httpx.Response:
            return self._responses[url]

    runtime = api._runtime_diagnostics()  # pyright: ignore[reportPrivateUsage]
    peer_diagnostics = NodeDiagnostics(
        generated_at="2026-04-23T00:00:00+00:00",
        runtime=runtime.model_copy(
            update={"node_id": "peer-node", "hostname": "peer-node.local"}
        ),
        resources=api._resource_diagnostics(),  # pyright: ignore[reportPrivateUsage]
    )

    def _build_async_client(*_args: object, **_kwargs: object) -> _FakeAsyncClient:
        return _FakeAsyncClient(
            responses={
                "http://peer-node:52415/v1/diagnostics/node": httpx.Response(
                    200,
                    json=peer_diagnostics.model_dump(mode="json", by_alias=True),
                    request=httpx.Request(
                        "GET",
                        "http://peer-node:52415/v1/diagnostics/node",
                    ),
                )
            }
        )

    monkeypatch.setattr(api, "_reachable_peer_api_urls", _reachable_peer_api_urls)
    monkeypatch.setattr(api_main.httpx, "AsyncClient", _build_async_client)

    response = client.get("/v1/diagnostics/cluster")

    assert response.status_code == 200
    nodes = [
        _json_mapping(node) for node in _json_list(_json_object(response)["nodes"])
    ]
    assert {node["nodeId"] for node in nodes} == {"local-node", "peer-node"}
    assert all(node["ok"] is True for node in nodes)
