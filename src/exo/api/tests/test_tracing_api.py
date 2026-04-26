"""Tests for tracing control and trace-browsing API endpoints."""

from pathlib import Path
from typing import Literal, cast
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient
from httpx import Response

import exo.api.main as api_main
from exo.api.main import API
from exo.shared.election import ElectionMessage
from exo.shared.tracing import TraceEvent, export_trace
from exo.shared.types.commands import (
    ForwarderCommand,
    ForwarderDownloadCommand,
    SetTracingEnabled,
)
from exo.shared.types.common import NodeId
from exo.shared.types.events import IndexedEvent
from exo.shared.types.profiling import NodeIdentity
from exo.utils.channels import channel


def _json_object(response: Response) -> dict[str, object]:
    """Return a JSON response payload as a typed object mapping."""

    return cast(dict[str, object], cast(object, response.json()))


def _json_list(value: object) -> list[object]:
    """Narrow a nested JSON array from a response payload."""

    return cast(list[object], value)


def _json_mapping(value: object) -> dict[str, object]:
    """Narrow one nested JSON object from a response payload."""

    return cast(dict[str, object], value)


def _build_api(node_id: str = "test-node") -> API:
    """Create a minimal API instance for tracing endpoint testing."""

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


def _write_trace(
    trace_dir: Path,
    task_id: str,
    *,
    node_id: str,
    model_id: str,
    task_kind: Literal["image", "text", "embedding"],
    category: str,
    tags: tuple[str, ...] = (),
) -> None:
    """Write one trace artifact to the given directory."""

    export_trace(
        [
            TraceEvent(
                name="decode_step",
                start_us=100,
                duration_us=50,
                rank=0,
                category=category,
                node_id=node_id,
                model_id=model_id,
                task_kind=task_kind,
                tags=tags,
                attrs={"shared_span": False},
            )
        ],
        trace_dir / f"trace_{task_id}.json",
    )


def test_get_tracing_state_returns_cluster_state() -> None:
    """GET /v1/tracing should expose the current cluster tracing toggle."""

    api = _build_api()
    api.state = api.state.model_copy(update={"tracing_enabled": True})
    client = TestClient(api.app)

    response = client.get("/v1/tracing")

    assert response.status_code == 200
    assert response.json() == {"enabled": True}


def test_update_tracing_state_sends_cluster_toggle_command() -> None:
    """PUT /v1/tracing should send SetTracingEnabled and reflect the new state."""

    api = _build_api()
    client = TestClient(api.app)

    async def _send_and_apply(command: object) -> None:
        assert isinstance(command, SetTracingEnabled)
        api.state = api.state.model_copy(update={"tracing_enabled": command.enabled})

    with patch.object(
        api,
        "_send",
        new=AsyncMock(side_effect=_send_and_apply),
    ) as mock_send:
        response = client.put("/v1/tracing", json={"enabled": True})

    assert response.status_code == 200
    assert response.json() == {"enabled": True}
    assert mock_send.await_args is not None
    sent_command = cast(object, mock_send.await_args.args[0])
    assert isinstance(sent_command, SetTracingEnabled)
    assert sent_command.enabled is True


def test_list_traces_returns_metadata_rich_items(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GET /v1/traces should expose model, task kind, tags, and source nodes."""

    monkeypatch.setattr(api_main, "EXO_TRACING_CACHE_DIR", tmp_path)
    _write_trace(
        tmp_path,
        "task-123",
        node_id="peer-node",
        model_id="mlx-community/gemma-4-26b-a4b-it-4bit",
        task_kind="text",
        category="decode",
        tags=("tool_call",),
    )

    api = _build_api("local-node")
    api.state = api.state.model_copy(
        update={
            "node_identities": {
                NodeId("peer-node"): NodeIdentity(friendly_name="Kite 2")
            }
        }
    )
    client = TestClient(api.app)

    response = client.get("/v1/traces")

    assert response.status_code == 200
    trace = _json_mapping(_json_list(_json_object(response)["traces"])[0])
    assert trace["taskId"] == "task-123"
    assert trace["modelId"] == "mlx-community/gemma-4-26b-a4b-it-4bit"
    assert trace["taskKind"] == "text"
    assert trace["categories"] == ["decode"]
    assert trace["tags"] == ["tool_call"]
    assert trace["hasToolActivity"] is True
    source_node = _json_mapping(_json_list(trace["sourceNodes"])[0])
    assert source_node["nodeId"] == "peer-node"
    assert source_node["friendlyName"] == "Kite 2"


def test_list_cluster_traces_dedupes_local_and_peer_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GET /v1/traces/cluster should dedupe matching task IDs across peers."""

    monkeypatch.setattr(api_main, "EXO_TRACING_CACHE_DIR", tmp_path)
    _write_trace(
        tmp_path,
        "task-123",
        node_id="local-node",
        model_id="mlx-community/local-model",
        task_kind="text",
        category="decode",
    )

    api = _build_api("local-node")
    client = TestClient(api.app)

    async def _reachable_peer_trace_urls() -> list[str]:
        return ["http://peer-node:52415"]

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

    peer_payload = {
        "traces": [
            {
                "taskId": "task-123",
                "createdAt": "2026-04-22T12:00:00+00:00",
                "fileSize": 999,
                "modelId": "mlx-community/peer-model",
                "taskKind": "text",
                "categories": ["decode", "prefill"],
                "tags": ["tool_call"],
                "hasToolActivity": True,
                "sourceNodes": [
                    {"nodeId": "peer-node", "friendlyName": "Kite 3"}
                ],
            },
            {
                "taskId": "task-999",
                "createdAt": "2026-04-22T12:01:00+00:00",
                "fileSize": 111,
                "modelId": "mlx-community/embedding-model",
                "taskKind": "embedding",
                "categories": ["embedding"],
                "tags": [],
                "hasToolActivity": False,
                "sourceNodes": [
                    {"nodeId": "peer-node", "friendlyName": "Kite 3"}
                ],
            },
        ]
    }

    def _build_async_client(*_args: object, **_kwargs: object) -> _FakeAsyncClient:
        return _FakeAsyncClient(
            responses={
                "http://peer-node:52415/v1/traces": httpx.Response(
                    200,
                    json=peer_payload,
                )
            }
        )

    monkeypatch.setattr(api, "_reachable_peer_trace_urls", _reachable_peer_trace_urls)
    monkeypatch.setattr(api_main.httpx, "AsyncClient", _build_async_client)

    response = client.get("/v1/traces/cluster")

    assert response.status_code == 200
    trace_items = [
        _json_mapping(trace_item)
        for trace_item in _json_list(_json_object(response)["traces"])
    ]
    traces: dict[str, dict[str, object]] = {
        cast(str, trace["taskId"]): trace for trace in trace_items
    }
    assert set(traces) == {"task-123", "task-999"}
    merged = traces["task-123"]
    assert merged["hasToolActivity"] is True
    assert merged["modelId"] == "mlx-community/local-model"
    assert merged["taskKind"] == "text"
    assert merged["categories"] == ["decode", "prefill"]
    assert merged["tags"] == ["tool_call"]
    source_nodes = _json_list(merged["sourceNodes"])
    source_node_ids = {
        cast(str, _json_mapping(source_node)["nodeId"]) for source_node in source_nodes
    }
    assert source_node_ids == {"local-node", "peer-node"}
