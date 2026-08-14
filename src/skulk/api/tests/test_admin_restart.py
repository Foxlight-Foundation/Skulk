"""Tests for the POST /admin/restart API endpoint."""

from typing import cast
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from httpx2 import Response

from skulk.api.main import API
from skulk.shared.election import ElectionMessage
from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.common import NodeId
from skulk.shared.types.events import IndexedEvent
from skulk.shared.types.telemetry import NodeTelemetry
from skulk.utils.channels import channel
from skulk.utils.info_gatherer.info_gatherer import StaticNodeInformation


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


def _report_stable_identity(api: API, node_id: str, node_install_id: UUID) -> None:
    """Publish one live static-identity reading into the API telemetry view."""
    api._telemetry_view.apply(  # pyright: ignore[reportPrivateUsage]
        NodeTelemetry(
            node_id=NodeId(node_id),
            info=StaticNodeInformation(
                node_install_id=node_install_id,
                model="Test host",
                chip="Test chip",
                os_version="1",
                os_build_version="1",
                skulk_version="1",
                skulk_commit="abc123",
            ),
        )
    )


def test_restart_local_node() -> None:
    """POST /admin/restart without node_id should trigger a local restart."""
    api = _build_api()
    client = TestClient(api.app)

    with patch("skulk.utils.restart.schedule_restart", return_value=True) as mock:
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

    with patch("skulk.utils.restart.schedule_restart", return_value=True) as mock:
        response = client.post("/admin/restart?node_id=test-node")

    assert response.status_code == 200
    data = _json_object(response)
    assert data["status"] == "restarting"
    mock.assert_called_once()


def test_restart_idempotent_returns_409() -> None:
    """If schedule_restart returns False (already pending), API returns 409."""
    api = _build_api()
    client = TestClient(api.app)

    with patch("skulk.utils.restart.schedule_restart", return_value=False):
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
    from skulk.shared.types.commands import RestartNode

    cmd = cast(object, mock_send.call_args[0][0])
    assert isinstance(cmd, RestartNode)
    assert cmd.target_node_id == NodeId("remote-node")


def test_restart_resolves_stable_remote_node_identity() -> None:
    """Stable installation identity resolves to the current live runtime node."""
    api = _build_api("local-node")
    node_install_id = uuid4()
    _report_stable_identity(api, "remote-session", node_install_id)
    client = TestClient(api.app)

    with patch.object(api, "_send_download", new_callable=AsyncMock) as mock_send:
        response = client.post(
            "/admin/restart",
            params={"node_install_id": str(node_install_id)},
        )

    assert response.status_code == 200
    assert _json_object(response) == {
        "status": "restart_sent",
        "node_id": "remote-session",
        "node_install_id": str(node_install_id),
    }
    from skulk.shared.types.commands import RestartNode

    command = cast(object, mock_send.call_args[0][0])
    assert isinstance(command, RestartNode)
    assert command.target_node_id == NodeId("remote-session")


def test_restart_resolves_stable_local_node_identity() -> None:
    """A stable identity may safely target the API node itself."""
    api = _build_api("local-session")
    node_install_id = uuid4()
    _report_stable_identity(api, "local-session", node_install_id)
    client = TestClient(api.app)

    with patch("skulk.utils.restart.schedule_restart", return_value=True) as mock:
        response = client.post(
            "/admin/restart",
            params={"node_install_id": str(node_install_id)},
        )

    assert response.status_code == 200
    assert _json_object(response) == {
        "status": "restarting",
        "node_id": "local-session",
        "node_install_id": str(node_install_id),
    }
    mock.assert_called_once()


def test_restart_rejects_missing_or_conflicting_stable_targets() -> None:
    """Stable targeting fails closed when current live truth cannot resolve it."""
    api = _build_api("local-node")
    client = TestClient(api.app)
    missing_identity = uuid4()

    missing = client.post(
        "/admin/restart",
        params={"node_install_id": str(missing_identity)},
    )
    conflicting = client.post(
        "/admin/restart",
        params={"node_id": "local-node", "node_install_id": str(missing_identity)},
    )
    _report_stable_identity(api, "first-session", missing_identity)
    _report_stable_identity(api, "second-session", missing_identity)
    ambiguous = client.post(
        "/admin/restart",
        params={"node_install_id": str(missing_identity)},
    )

    assert missing.status_code == 404
    assert conflicting.status_code == 400
    assert ambiguous.status_code == 409


def test_state_projects_stable_node_identity() -> None:
    """Canonical cluster truth exposes the stable ID beside runtime node truth."""
    api = _build_api("local-node")
    node_install_id = uuid4()
    _report_stable_identity(api, "remote-session", node_install_id)

    response = TestClient(api.app).get("/state")

    assert response.status_code == 200
    node_identities = cast(dict[str, object], _json_object(response)["nodeIdentities"])
    remote_identity = cast(dict[str, object], node_identities["remote-session"])
    assert remote_identity["nodeInstallId"] == str(node_install_id)
