"""Tests for read-only node and cluster diagnostics endpoints."""

from datetime import datetime, timezone
from typing import cast

import httpx
import httpx2
import pytest
from fastapi.testclient import TestClient

import skulk.api.main as api_main
from skulk.api.main import API
from skulk.shared.election import ElectionMessage
from skulk.shared.models.model_cards import ModelCard, ModelTask
from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.common import ModelId, NodeId
from skulk.shared.types.diagnostics import (
    DiagnosticCaptureResponse,
    DiagnosticProcessSample,
    MlxMemorySnapshot,
    NodeDiagnostics,
    RunnerDiagnosticContext,
    RunnerFlightRecorderEntry,
    RunnerLifecycleMilestone,
    RunnerSupervisorDiagnostics,
    RunnerTaskCancelResponse,
    RunnerTaskDiagnostics,
    TelemetryPlaneDiagnostics,
)
from skulk.shared.types.events import IndexedEvent
from skulk.shared.types.memory import Memory
from skulk.shared.types.profiling import NodeIdentity, NodeResources
from skulk.shared.types.state import State
from skulk.shared.types.tasks import StartWarmup, TaskId, TaskStatus
from skulk.shared.types.telemetry import NodeTelemetry
from skulk.shared.types.worker.instances import InstanceId, MlxRingInstance
from skulk.shared.types.worker.runners import (
    RunnerId,
    RunnerWarmingUp,
    ShardAssignments,
)
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.utils.channels import channel


def _json_object(response: httpx2.Response) -> dict[str, object]:
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


def test_telemetry_diagnostics_are_exposed_without_changing_node_bundle() -> None:
    """Telemetry pressure uses a separate endpoint safe for older node schemas."""

    api = _build_api("local-node")
    api._telemetry_plane_provider = lambda: TelemetryPlaneDiagnostics(  # pyright: ignore[reportPrivateUsage]
        transport="isolated_gossipsub",
        admission_capacity=256,
        pending_readings=3,
        network_queue_capacity=1,
        network_queue_depth=1,
        max_queue_depth=4,
        readings_offered=20,
        readings_coalesced=15,
        readings_dropped=1,
        readings_published=3,
        publish_failures=1,
        bytes_published=512,
        oldest_pending_age_seconds=0.25,
        last_successful_publish_age_seconds=0.1,
    )
    client = TestClient(api.app)

    telemetry_response = client.get("/v1/diagnostics/telemetry")
    node_response = client.get("/v1/diagnostics/node")

    assert telemetry_response.status_code == 200
    assert telemetry_response.json()["readingsCoalesced"] == 15
    assert telemetry_response.json()["networkQueueDepth"] == 1
    assert node_response.status_code == 200
    assert "telemetryPlane" not in node_response.json()


def test_node_diagnostics_warns_about_split_data_transports() -> None:
    """Diagnostics should fail loudly when live nodes advertise both transports."""
    api = _build_api("local-node")
    management_node = NodeId("remote-management-node")
    worker_node = NodeId("worker-node")
    now = datetime.now(tz=timezone.utc)
    api.state = api.state.model_copy(update={"last_seen": {worker_node: now}})
    api._telemetry_view.apply(  # pyright: ignore[reportPrivateUsage]
        NodeTelemetry(
            node_id=management_node,
            info=NodeResources(
                backends=frozenset(),
                participation="management",
                data_transport="zenoh",
            ),
        ),
        received_at=now,
    )
    api._telemetry_view.apply(  # pyright: ignore[reportPrivateUsage]
        NodeTelemetry(
            node_id=worker_node,
            info=NodeResources(data_transport="gossipsub"),
        ),
        received_at=now,
    )
    client = TestClient(api.app)

    response = client.get("/v1/diagnostics/node")

    assert response.status_code == 200
    warnings = _json_list(_json_object(response)["warnings"])
    assert any(
        "Fleet DATA transport mismatch" in cast(str, warning)
        and "gossipsub, zenoh" in cast(str, warning)
        for warning in warnings
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
    provider = _json_mapping(body["provider"])
    assert provider["activeStreams"] == 0
    assert provider["activeUnaryCalls"] == 0
    assert provider["streamSlotsInUse"] == 0
    assert provider["unaryConcurrencyLimit"] == 8
    assert provider["streamConcurrencyLimit"] == 8
    assert provider["capabilities"] == {}
    placements = _json_list(body["placements"])
    placement = _json_mapping(placements[0])
    assert placement["masterIsPlacementNode"] is False
    assert placement["localNodeIsPlacementNode"] is True
    # A master that hosts no rank is the NORMAL multi-node topology, not a
    # problem; the old warning here fired for virtually every instance and
    # was removed as noise.
    warnings = _json_list(placement["warnings"])
    assert "Current master is not a placement node for this instance." not in warnings


def test_node_diagnostics_flags_orphaned_live_runner_tasks() -> None:
    """Live supervisor tasks missing from state should surface divergence warnings."""

    api = _build_api("local-node")
    api._master_node_id = NodeId("master-node")  # pyright: ignore[reportPrivateUsage]
    api.state = _running_state_without_master_placement()
    warmup_task_id = next(iter(api.state.tasks.keys()))
    api.set_runner_diagnostics_provider(
        lambda: [
            RunnerSupervisorDiagnostics(
                runner_id="runner-1",
                instance_id="instance-1",
                node_id="local-node",
                model_id="mlx-community/gemma-4-26b-a4b-it-4bit",
                device_rank=0,
                world_size=2,
                start_layer=0,
                end_layer=15,
                n_layers=30,
                pid=1234,
                process_alive=True,
                exit_code=None,
                status_kind="RunnerRunning",
                status_since="2026-04-23T00:00:00+00:00",
                seconds_in_status=42.0,
                phase="decode_wait_first_token",
                phase_started_at="2026-04-23T00:00:02+00:00",
                seconds_in_phase=40.0,
                last_progress_at="2026-04-23T00:00:02+00:00",
                active_task_id="orphan-task",
                active_command_id="command-1",
                phase_detail="stream_generate_first_token",
                last_mlx_memory=MlxMemorySnapshot(
                    generated_at="2026-04-23T00:00:02+00:00",
                    active=Memory.from_mb(1024),
                    cache=Memory.from_mb(128),
                    peak=Memory.from_mb(2048),
                    wired_limit=None,
                    source="mlx.core",
                ),
                flight_recorder=[
                    RunnerFlightRecorderEntry(
                        at="2026-04-23T00:00:02+00:00",
                        phase="decode_wait_first_token",
                        event="enter",
                        detail="stream_generate_first_token",
                        attrs={"prompt_tokens": 123},
                        context=RunnerDiagnosticContext(
                            node_id="local-node",
                            runner_id="runner-1",
                            pid=1234,
                            instance_id="instance-1",
                            model_id="mlx-community/gemma-4-26b-a4b-it-4bit",
                            rank=0,
                            world_size=2,
                            start_layer=0,
                            end_layer=15,
                            n_layers=30,
                        ),
                        task_id="orphan-task",
                        command_id="command-1",
                    )
                ],
                pending_task_ids=[],
                in_progress_tasks=[
                    RunnerTaskDiagnostics(
                        task_id="orphan-task",
                        task_kind="TextGeneration",
                        task_status="Pending",
                        instance_id="instance-1",
                        command_id="command-1",
                        runner_id="runner-1",
                        model_id="mlx-community/gemma-4-26b-a4b-it-4bit",
                    ),
                    RunnerTaskDiagnostics(
                        task_id=str(warmup_task_id),
                        task_kind="StartWarmup",
                        task_status="Running",
                        instance_id="instance-1",
                        command_id=None,
                        runner_id="runner-1",
                        model_id="mlx-community/gemma-4-26b-a4b-it-4bit",
                    ),
                ],
                completed_task_count=0,
                cancelled_task_ids=[],
                last_task_sent_at="2026-04-23T00:00:00+00:00",
                last_event_received_at="2026-04-23T00:00:01+00:00",
                last_event_type="TaskAcknowledged",
                milestones=[
                    RunnerLifecycleMilestone(
                        at="2026-04-23T00:00:00+00:00",
                        name="task_sent",
                        detail="TextGeneration:orphan-task",
                    )
                ],
            )
        ]
    )
    client = TestClient(api.app)

    response = client.get("/v1/diagnostics/node")

    assert response.status_code == 200
    body = _json_object(response)
    warnings = _json_list(body["warnings"])
    assert any(
        "still reports TextGeneration:orphan-t…task in progress"
        in cast(str, warning)
        or "still reports TextGeneration:orphan-task in progress"
        in cast(str, warning)
        for warning in warnings
    )
    placements = _json_list(body["placements"])
    placement = _json_mapping(placements[0])
    placement_warnings = _json_list(placement["warnings"])
    assert any(
        "cluster state no longer tracks that task" in cast(str, warning)
        for warning in placement_warnings
    )


def test_capture_local_node_diagnostics_returns_runner_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local capture should include flight recorder and partial process samples."""

    api = _build_api("local-node")
    runner = RunnerSupervisorDiagnostics(
        runner_id="runner-1",
        instance_id="instance-1",
        node_id="local-node",
        model_id="mlx-community/gemma-4-26b-a4b-it-4bit",
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=30,
        n_layers=30,
        pid=1234,
        process_alive=True,
        exit_code=None,
        status_kind="RunnerRunning",
        status_since="2026-04-23T00:00:00+00:00",
        seconds_in_status=12.0,
        phase="decode_wait_first_token",
        phase_started_at="2026-04-23T00:00:01+00:00",
        seconds_in_phase=11.0,
        last_progress_at="2026-04-23T00:00:01+00:00",
        active_task_id="task-1",
        active_command_id="command-1",
        phase_detail="stream_generate_first_token",
        last_mlx_memory=MlxMemorySnapshot(
            generated_at="2026-04-23T00:00:01+00:00",
            active=Memory.from_mb(512),
            cache=Memory.from_mb(64),
            peak=Memory.from_mb(1024),
            wired_limit=None,
            source="mlx.core",
        ),
        flight_recorder=[
            RunnerFlightRecorderEntry(
                at="2026-04-23T00:00:01+00:00",
                phase="decode_wait_first_token",
                event="enter",
                detail="stream_generate_first_token",
                attrs={"prompt_tokens": 42},
                context=RunnerDiagnosticContext(
                    node_id="local-node",
                    runner_id="runner-1",
                    pid=1234,
                    instance_id="instance-1",
                    model_id="mlx-community/gemma-4-26b-a4b-it-4bit",
                    rank=0,
                    world_size=1,
                    start_layer=0,
                    end_layer=30,
                    n_layers=30,
                ),
                task_id="task-1",
                command_id="command-1",
            )
        ],
        pending_task_ids=[],
        in_progress_tasks=[
            RunnerTaskDiagnostics(
                task_id="task-1",
                task_kind="TextGeneration",
                task_status="Running",
                instance_id="instance-1",
                command_id="command-1",
                runner_id="runner-1",
                model_id="mlx-community/gemma-4-26b-a4b-it-4bit",
            )
        ],
        completed_task_count=0,
        cancelled_task_ids=[],
        last_task_sent_at="2026-04-23T00:00:00+00:00",
        last_event_received_at="2026-04-23T00:00:01+00:00",
        last_event_type="ChunkGenerated",
        milestones=[],
    )
    api.set_runner_diagnostics_provider(lambda: [runner])

    async def _sample_runner(_pid: int, _duration: float) -> list[DiagnosticProcessSample]:
        return [
            DiagnosticProcessSample(
                name="sample",
                command=["sample", "1234", "3"],
                ok=False,
                exit_code=1,
                duration_seconds=0.1,
                stderr="permission denied",
                error="Command exited non-zero",
            )
        ]

    monkeypatch.setattr(api, "_collect_process_samples", _sample_runner)
    client = TestClient(api.app)

    response = client.post(
        "/v1/diagnostics/node/capture",
        json={"runnerId": "runner-1", "taskId": "task-1"},
    )

    assert response.status_code == 200
    body = _json_object(response)
    assert _json_mapping(body["runner"])["phase"] == "decode_wait_first_token"
    assert _json_mapping(body["mlxMemory"])["source"] == "mlx.core"
    assert len(_json_list(body["flightRecorder"])) == 1
    samples = _json_list(body["processSamples"])
    assert _json_mapping(samples[0])["ok"] is False


def test_capture_local_node_diagnostics_rejects_unknown_runner() -> None:
    """Capture should return a clear 404 for unknown focused runner IDs."""

    api = _build_api("local-node")
    client = TestClient(api.app)

    response = client.post(
        "/v1/diagnostics/node/capture",
        json={"runnerId": "runner-missing"},
    )

    assert response.status_code == 404
    assert "No local runner matched" in response.text


def test_cluster_diagnostics_returns_local_and_peer_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cluster diagnostics should fan out and tolerate reachable peers."""

    api = _build_api("local-node")
    client = TestClient(api.app)

    async def _reachable_peer_api_urls(fail_fast: bool = False) -> dict[str, str]:
        del fail_fast
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
        data_plane=api._data_plane_observer.snapshot(),  # pyright: ignore[reportPrivateUsage]
    )

    peer_payload = peer_diagnostics.model_dump(mode="json", by_alias=True)
    peer_payload["futureNodeCounter"] = 7
    data_plane = _json_mapping(cast(object, peer_payload["dataPlane"]))
    data_plane["futureLifecycleCounter"] = 11
    egress = _json_mapping(data_plane["egress"])
    egress.pop("idleStreamReclaims")
    egress["futureEgressCounter"] = 13
    egress["owners"] = {
        "owner-a": {
            "queueDepth": 0,
            "activeStreams": 0,
            "maxQueueDepth": 0,
            "framesEnqueued": 0,
            "framesPublished": 0,
            "framesDropped": 0,
            "publishFailures": 0,
            "futureOwnerCounter": 17,
        }
    }

    def _build_async_client(*_args: object, **_kwargs: object) -> _FakeAsyncClient:
        return _FakeAsyncClient(
            responses={
                "http://peer-node:52415/v1/diagnostics/node": httpx.Response(
                    200,
                    json=peer_payload,
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
    body = _json_object(response)
    assert body["versionStatus"] == "consistent"
    nodes = [
        _json_mapping(node) for node in _json_list(body["nodes"])
    ]
    assert {node["nodeId"] for node in nodes} == {"local-node", "peer-node"}
    assert all(node["ok"] is True for node in nodes)
    assert all(node["versionStatus"] == "current" for node in nodes)

    peer = next(node for node in nodes if node["nodeId"] == "peer-node")
    parsed = _json_mapping(peer["diagnostics"])
    assert "futureNodeCounter" not in parsed
    parsed_data_plane = _json_mapping(parsed["dataPlane"])
    assert "futureLifecycleCounter" not in parsed_data_plane
    parsed_egress = _json_mapping(parsed_data_plane["egress"])
    assert parsed_egress["idleStreamReclaims"] == 0
    assert "futureEgressCounter" not in parsed_egress
    owner = _json_mapping(_json_mapping(parsed_egress["owners"])["owner-a"])
    assert owner["idleStreamReclaims"] == 0
    assert "futureOwnerCounter" not in owner


def test_cluster_diagnostics_marks_mixed_source_builds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A staged same-version/different-commit window stays visible and degraded."""

    api = _build_api("local-node")
    api._telemetry_view.node_identities[api.node_id] = NodeIdentity(  # pyright: ignore[reportPrivateUsage]
        skulk_version="1.4.3",
        skulk_commit="aaaaaaaa",
    )
    client = TestClient(api.app)

    async def _reachable_peer_api_urls(fail_fast: bool = False) -> dict[str, str]:
        del fail_fast
        return {"peer-node": "http://peer-node:52415"}

    class _FakeAsyncClient:
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
            runtime = api._runtime_diagnostics().model_copy(  # pyright: ignore[reportPrivateUsage]
                update={
                    "node_id": "peer-node",
                    "hostname": "peer-node.local",
                    "skulk_commit": "bbbbbbbb",
                }
            )
            diagnostics = NodeDiagnostics(
                generated_at="2026-07-15T00:00:00+00:00",
                runtime=runtime,
                resources=api._resource_diagnostics(),  # pyright: ignore[reportPrivateUsage]
                data_plane=api._data_plane_observer.snapshot(),  # pyright: ignore[reportPrivateUsage]
            )
            return httpx.Response(
                200,
                json=diagnostics.model_dump(mode="json", by_alias=True),
                request=httpx.Request("GET", url),
            )

    def _build_async_client(*_args: object, **_kwargs: object) -> _FakeAsyncClient:
        return _FakeAsyncClient()

    monkeypatch.setattr(api, "_reachable_peer_api_urls", _reachable_peer_api_urls)
    monkeypatch.setattr(api_main.httpx, "AsyncClient", _build_async_client)

    response = client.get("/v1/diagnostics/cluster")

    assert response.status_code == 200
    body = _json_object(response)
    assert body["versionStatus"] == "mixed"
    nodes = {
        str(node["nodeId"]): node
        for node in (
            _json_mapping(item) for item in _json_list(body["nodes"])
        )
    }
    assert nodes["local-node"]["versionStatus"] == "current"
    assert nodes["peer-node"]["versionStatus"] == "version_mismatch"
    assert nodes["peer-node"]["ok"] is True


def test_cluster_diagnostics_marks_failed_routed_peer_version_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A known route that fails collection must make build status uncertain."""

    api = _build_api("local-node")
    client = TestClient(api.app)

    async def _reachable_peer_api_urls(fail_fast: bool = False) -> dict[str, str]:
        del fail_fast
        return {"peer-node": "http://peer-node:52415"}

    class _FakeAsyncClient:
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
            return httpx.Response(
                503,
                request=httpx.Request("GET", url),
            )

    def _build_async_client(*_args: object, **_kwargs: object) -> _FakeAsyncClient:
        return _FakeAsyncClient()

    monkeypatch.setattr(api, "_reachable_peer_api_urls", _reachable_peer_api_urls)
    monkeypatch.setattr(api_main.httpx, "AsyncClient", _build_async_client)

    response = client.get("/v1/diagnostics/cluster")

    assert response.status_code == 200
    body = _json_object(response)
    assert body["versionStatus"] == "unknown"
    nodes = {
        str(node["nodeId"]): node
        for node in (
            _json_mapping(item) for item in _json_list(body["nodes"])
        )
    }
    peer = nodes["peer-node"]
    assert peer["ok"] is False
    assert peer["url"] == "http://peer-node:52415"
    assert peer["versionStatus"] == "unknown"


def _build_supervisor_runner(
    *,
    node_id: str,
    rank: int,
    runner_id: str,
    flight_at: list[str],
) -> RunnerSupervisorDiagnostics:
    """Build a minimal supervisor diagnostics record for cluster-timeline tests."""

    context = RunnerDiagnosticContext(
        node_id=node_id,
        runner_id=runner_id,
        pid=1234 + rank,
        instance_id="instance-1",
        model_id="mlx-community/gemma-4-26b-a4b-it-4bit",
        rank=rank,
        world_size=2,
        start_layer=0,
        end_layer=15,
        n_layers=30,
    )
    return RunnerSupervisorDiagnostics(
        runner_id=runner_id,
        instance_id="instance-1",
        node_id=node_id,
        model_id="mlx-community/gemma-4-26b-a4b-it-4bit",
        device_rank=rank,
        world_size=2,
        start_layer=0,
        end_layer=15,
        n_layers=30,
        pid=1234 + rank,
        process_alive=True,
        exit_code=None,
        status_kind="RunnerRunning",
        status_since="2026-04-23T00:00:00+00:00",
        seconds_in_status=12.0,
        phase="decode_stream",
        phase_started_at="2026-04-23T00:00:01+00:00",
        seconds_in_phase=11.0,
        last_progress_at=flight_at[-1] if flight_at else None,
        active_task_id="task-1",
        active_command_id="command-1",
        phase_detail="pipeline_last_eval_output",
        last_mlx_memory=None,
        flight_recorder=[
            RunnerFlightRecorderEntry(
                at=ts,
                phase="decode_stream",
                event="enter",
                detail="pipeline_last_eval_output",
                attrs={"rank": rank},
                context=context,
                task_id="task-1",
                command_id="command-1",
            )
            for ts in flight_at
        ],
        pending_task_ids=[],
        in_progress_tasks=[],
        completed_task_count=0,
        cancelled_task_ids=[],
        last_task_sent_at=None,
        last_event_received_at=None,
        last_event_type=None,
        milestones=[],
    )


def test_cluster_timeline_merges_flight_recorders_by_wall_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cluster timeline should merge per-rank flight recorders chronologically.

    The intended consumer is a human staring at a distributed deadlock: they
    need to see "rank 0 entered phase X at T while rank 1 was still in phase Y"
    by reading top to bottom. Verify that ranks are stitched into one list
    sorted by `at`, with `node_id` / `device_rank` lifted onto each entry.
    """

    api = _build_api("local-node")
    client = TestClient(api.app)

    # Local rank 0: two flight-recorder entries, the later one being the
    # eval site that hangs in production.
    local_runner = _build_supervisor_runner(
        node_id="local-node",
        rank=0,
        runner_id="runner-0",
        flight_at=[
            "2026-04-24T22:03:50.000000+00:00",
            "2026-04-24T22:03:51.508000+00:00",
        ],
    )
    api.set_runner_diagnostics_provider(lambda: [local_runner])

    # Peer rank 1: two entries that interleave with the local timeline so
    # chronological merge is observable in the response order.
    peer_runner = _build_supervisor_runner(
        node_id="peer-node",
        rank=1,
        runner_id="runner-1",
        flight_at=[
            "2026-04-24T22:03:50.500000+00:00",
            "2026-04-24T22:03:52.000000+00:00",
        ],
    )

    runtime = api._runtime_diagnostics()  # pyright: ignore[reportPrivateUsage]
    peer_diagnostics = NodeDiagnostics(
        generated_at="2026-04-24T22:03:53+00:00",
        runtime=runtime.model_copy(
            update={"node_id": "peer-node", "hostname": "peer-node.local"}
        ),
        resources=api._resource_diagnostics(),  # pyright: ignore[reportPrivateUsage]
        data_plane=api._data_plane_observer.snapshot(),  # pyright: ignore[reportPrivateUsage]
        supervisor_runners=[peer_runner],
    )

    async def _reachable_peer_api_urls(fail_fast: bool = False) -> dict[str, str]:
        del fail_fast
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

    response = client.get("/v1/diagnostics/cluster/timeline")

    assert response.status_code == 200
    body = _json_object(response)

    runners = [_json_mapping(r) for r in _json_list(body["runners"])]
    assert [r["deviceRank"] for r in runners] == [0, 1]
    assert [r["nodeId"] for r in runners] == ["local-node", "peer-node"]

    timeline = [_json_mapping(e) for e in _json_list(body["timeline"])]
    assert [e["at"] for e in timeline] == [
        "2026-04-24T22:03:50.000000+00:00",
        "2026-04-24T22:03:50.500000+00:00",
        "2026-04-24T22:03:51.508000+00:00",
        "2026-04-24T22:03:52.000000+00:00",
    ]
    assert [e["deviceRank"] for e in timeline] == [0, 1, 0, 1]
    assert [e["nodeId"] for e in timeline] == [
        "local-node",
        "peer-node",
        "local-node",
        "peer-node",
    ]
    # Every entry must carry world_size + runner identity for downstream tools.
    assert all(e["worldSize"] == 2 for e in timeline)
    assert {e["runnerId"] for e in timeline} == {"runner-0", "runner-1"}


def test_cluster_timeline_records_unreachable_peers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unreachable peers should appear in unreachableNodes, not break the response."""

    api = _build_api("local-node")
    client = TestClient(api.app)

    api.set_runner_diagnostics_provider(list)

    async def _reachable_peer_api_urls(fail_fast: bool = False) -> dict[str, str]:
        del fail_fast
        return {"peer-node": "http://peer-node:52415"}

    class _FakeAsyncClient:
        async def __aenter__(self) -> "_FakeAsyncClient":
            return self

        async def __aexit__(
            self,
            _exc_type: object,
            _exc: object,
            _tb: object,
        ) -> None:
            return None

        async def get(self, _url: str) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

    def _build_async_client(*_args: object, **_kwargs: object) -> _FakeAsyncClient:
        return _FakeAsyncClient()

    monkeypatch.setattr(api, "_reachable_peer_api_urls", _reachable_peer_api_urls)
    monkeypatch.setattr(api_main.httpx, "AsyncClient", _build_async_client)

    response = client.get("/v1/diagnostics/cluster/timeline")

    assert response.status_code == 200
    body = _json_object(response)
    unreachable = [_json_mapping(u) for u in _json_list(body["unreachableNodes"])]
    assert len(unreachable) == 1
    assert unreachable[0]["nodeId"] == "peer-node"
    assert "connection refused" in str(unreachable[0]["error"])
    # Local node has no runners — timeline is empty but well-formed.
    assert _json_list(body["timeline"]) == []
    assert _json_list(body["runners"]) == []


def test_cancel_local_runner_task_calls_worker_provider() -> None:
    """Local runner control should call the attached worker provider."""

    api = _build_api("local-node")
    captured: list[tuple[str, str]] = []
    expected_task_id = TaskId("task-1")

    async def _cancel_provider(runner_id: object, task_id: object) -> RunnerTaskCancelResponse:
        captured.append((str(runner_id), str(task_id)))
        return RunnerTaskCancelResponse(
            node_id=NodeId("local-node"),
            runner_id=RunnerId(str(runner_id)),
            task_id=expected_task_id,
            status="cancel_requested",
            message="cancelled",
        )

    api.set_runner_cancel_provider(_cancel_provider)
    client = TestClient(api.app)

    response = client.post(
        "/v1/diagnostics/node/runners/runner-1/cancel",
        json={"taskId": "task-1"},
    )

    assert response.status_code == 200
    assert captured == [("runner-1", "task-1")]
    body = _json_object(response)
    assert body["status"] == "cancel_requested"
    assert body["runnerId"] == "runner-1"
    assert body["taskId"] == "task-1"


def test_cancel_cluster_runner_task_proxies_to_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cluster runner control should proxy to a reachable peer node."""

    api = _build_api("local-node")
    client = TestClient(api.app)

    async def _reachable_peer_api_urls(fail_fast: bool = False) -> dict[str, str]:
        del fail_fast
        return {"peer-node": "http://peer-node:52415"}

    class _FakeAsyncClient:
        async def __aenter__(self) -> "_FakeAsyncClient":
            return self

        async def __aexit__(
            self,
            _exc_type: object,
            _exc: object,
            _tb: object,
        ) -> None:
            return None

        async def post(self, url: str, json: object) -> httpx.Response:
            assert (
                url
                == "http://peer-node:52415/v1/diagnostics/node/runners/runner-1/cancel"
            )
            assert json == {"taskId": "task-1"}
            return httpx.Response(
                200,
                json=RunnerTaskCancelResponse(
                    node_id=NodeId("peer-node"),
                    runner_id=RunnerId("runner-1"),
                    task_id=TaskId("task-1"),
                    status="cancel_requested",
                    message="proxied",
                ).model_dump(mode="json", by_alias=True),
                request=httpx.Request("POST", url),
            )

    def _build_async_client(
        *_args: object,
        **_kwargs: object,
    ) -> _FakeAsyncClient:
        return _FakeAsyncClient()

    monkeypatch.setattr(api, "_reachable_peer_api_urls", _reachable_peer_api_urls)
    monkeypatch.setattr(api_main.httpx, "AsyncClient", _build_async_client)

    response = client.post(
        "/v1/diagnostics/cluster/peer-node/runners/runner-1/cancel",
        json={"taskId": "task-1"},
    )

    assert response.status_code == 200
    body = _json_object(response)
    assert body["nodeId"] == "peer-node"
    assert body["status"] == "cancel_requested"


def test_capture_cluster_node_diagnostics_proxies_to_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cluster capture should proxy the request body to a reachable peer."""

    api = _build_api("local-node")
    client = TestClient(api.app)

    async def _reachable_peer_api_urls(fail_fast: bool = False) -> dict[str, str]:
        del fail_fast
        return {"peer-node": "http://peer-node:52415"}

    capture = DiagnosticCaptureResponse(
        generated_at="2026-04-23T00:00:00+00:00",
        node_id=NodeId("peer-node"),
        node_diagnostics=NodeDiagnostics(
            generated_at="2026-04-23T00:00:00+00:00",
            runtime=api._runtime_diagnostics().model_copy(  # pyright: ignore[reportPrivateUsage]
                update={"node_id": "peer-node", "hostname": "peer-node.local"}
            ),
            resources=api._resource_diagnostics(),  # pyright: ignore[reportPrivateUsage]
            data_plane=api._data_plane_observer.snapshot(),  # pyright: ignore[reportPrivateUsage]
        ),
        flight_recorder=[],
        process_samples=[],
        warnings=[],
    )

    class _FakeAsyncClient:
        async def __aenter__(self) -> "_FakeAsyncClient":
            return self

        async def __aexit__(
            self,
            _exc_type: object,
            _exc: object,
            _tb: object,
        ) -> None:
            return None

        async def post(self, url: str, json: object) -> httpx.Response:
            assert url == "http://peer-node:52415/v1/diagnostics/node/capture"
            assert json == {
                "runnerId": "runner-1",
                "taskId": "task-1",
                "includeProcessSamples": True,
                "sampleDurationSeconds": 3.0,
            }
            return httpx.Response(
                200,
                json=capture.model_dump(mode="json", by_alias=True),
                request=httpx.Request("POST", url),
            )

    def _build_async_client(
        *_args: object,
        **_kwargs: object,
    ) -> _FakeAsyncClient:
        return _FakeAsyncClient()

    monkeypatch.setattr(api, "_reachable_peer_api_urls", _reachable_peer_api_urls)
    monkeypatch.setattr(api_main.httpx, "AsyncClient", _build_async_client)

    response = client.post(
        "/v1/diagnostics/cluster/peer-node/capture",
        json={"runnerId": "runner-1", "taskId": "task-1"},
    )

    assert response.status_code == 200
    body = _json_object(response)
    assert body["nodeId"] == "peer-node"


async def test_node_diagnostics_includes_tailscale_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bundle carries this node's own tailnet state (per-node via proxy)."""

    from skulk.connectivity.tailscale import TailscaleStatus

    async def fake_status() -> TailscaleStatus:
        return TailscaleStatus(
            running=True,
            self_ip="100.64.0.9",
            hostname="kite-test",
            dns_name="kite-test.tailnet.ts.net",
        )

    monkeypatch.setattr("skulk.api.main.query_tailscale_status", fake_status)
    api = _build_api("local-node")
    client = TestClient(api.app)

    body = _json_object(client.get("/v1/diagnostics/node"))
    tailscale = _json_mapping(body["tailscale"])
    assert tailscale["running"] is True
    assert tailscale["selfIp"] == "100.64.0.9"
    assert tailscale["dnsName"] == "kite-test.tailnet.ts.net"


async def test_node_diagnostics_tailscale_probe_exception_is_null(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unexpected probe exception serializes as null, never fails the bundle."""

    async def exploding_status() -> object:
        raise RuntimeError("probe blew up")

    monkeypatch.setattr("skulk.api.main.query_tailscale_status", exploding_status)
    api = _build_api("local-node")
    client = TestClient(api.app)

    response = client.get("/v1/diagnostics/node")
    assert response.status_code == 200
    assert _json_object(response)["tailscale"] is None


def test_cluster_diagnostics_reports_unreachable_topology_members(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A topology member with no reachable API route must appear as an
    explicit ok=false entry, not vanish: an overlay-joined node otherwise has
    no observability presence at all (#558)."""

    api = _build_api("local-node")
    client = TestClient(api.app)

    api.state.topology.add_node(NodeId("local-node"))
    api.state.topology.add_node(NodeId("unroutable-node"))

    async def _reachable_peer_api_urls(fail_fast: bool = False) -> dict[str, str]:
        del fail_fast
        return {}

    monkeypatch.setattr(api, "_reachable_peer_api_urls", _reachable_peer_api_urls)

    response = client.get("/v1/diagnostics/cluster")

    assert response.status_code == 200
    body = _json_object(response)
    assert body["versionStatus"] == "consistent"
    nodes = {
        str(node["nodeId"]): node
        for node in (
            _json_mapping(item) for item in _json_list(body["nodes"])
        )
    }
    assert nodes["local-node"]["ok"] is True
    unreachable = nodes["unroutable-node"]
    assert unreachable["ok"] is False
    assert unreachable["url"] is None
    assert "no reachable API route" in str(unreachable["error"])
