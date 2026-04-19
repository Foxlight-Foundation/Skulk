"""Tests for the POST /admin/restart API endpoint."""

from typing import cast
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from httpx import Response

from exo.api.main import API
from exo.shared.election import ElectionMessage
from exo.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from exo.shared.types.common import NodeId
from exo.shared.types.events import IndexedEvent
from exo.utils.channels import channel


def _json_object(response: Response) -> dict[str, object]:
    """Return a JSON response payload as a typed object mapping."""
    return cast(dict[str, object], cast(object, response.json()))


def _build_api(node_id: str = "test-node") -> API:
    """Create a minimal API instance for testing."""
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


def test_restart_local_node() -> None:
    """POST /admin/restart without node_id should trigger a local restart."""
    api = _build_api()
    client = TestClient(api.app)

    with patch("exo.utils.restart.schedule_restart", return_value=True) as mock:
        response = client.post("/admin/restart")

    assert response.status_code == 200
    data = _json_object(response)
    assert data["status"] == "restarting"
    assert data["node_id"] == "test-node"
    mock.assert_called_once()


def test_restart_local_node_with_explicit_id() -> None:
    """POST /admin/restart?node_id=<self> should also trigger local restart."""
    api = _build_api()
    client = TestClient(api.app)

    with patch("exo.utils.restart.schedule_restart", return_value=True) as mock:
        response = client.post("/admin/restart?node_id=test-node")

    assert response.status_code == 200
    data = _json_object(response)
    assert data["status"] == "restarting"
    mock.assert_called_once()


def test_restart_idempotent_returns_409() -> None:
    """If schedule_restart returns False (already pending), API returns 409."""
    api = _build_api()
    client = TestClient(api.app)

    with patch("exo.utils.restart.schedule_restart", return_value=False):
        response = client.post("/admin/restart")

    assert response.status_code == 409
    data = _json_object(response)
    assert data["status"] == "restart_already_pending"


def test_restart_remote_node() -> None:
    """POST /admin/restart?node_id=<other> should send RestartNode via pub/sub."""
    api = _build_api("local-node")
    client = TestClient(api.app)

    with patch.object(api, "_send_download", new_callable=AsyncMock) as mock_send:
        response = client.post("/admin/restart?node_id=remote-node")

    assert response.status_code == 200
    data = _json_object(response)
    assert data["status"] == "restart_sent"
    assert data["node_id"] == "remote-node"
    mock_send.assert_called_once()

    # Verify the command type and target
    from exo.shared.types.commands import RestartNode

    cmd = cast(object, mock_send.call_args[0][0])
    assert isinstance(cmd, RestartNode)
    assert cmd.target_node_id == NodeId("remote-node")
