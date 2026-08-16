"""Tests for the /config API endpoint."""

from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from httpx2 import Response

from skulk.api.main import API
from skulk.shared.election import ElectionMessage
from skulk.shared.types.commands import (
    ForwarderCommand,
    ForwarderDownloadCommand,
    SetModelTrustApproval,
)
from skulk.shared.types.common import NodeId
from skulk.shared.types.events import IndexedEvent
from skulk.store.config import load_skulk_config
from skulk.utils.channels import channel


def _json_object(response: Response) -> dict[str, object]:
    """Return a JSON response payload as a typed object mapping."""
    return cast(dict[str, object], cast(object, response.json()))


def _json_mapping(value: object) -> dict[str, object]:
    """Narrow one nested JSON object from a response payload."""
    return cast(dict[str, object], value)


def _build_api(node_id: str = "test-node") -> API:
    """Create a minimal API instance for config endpoint testing."""
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


def test_get_config_reports_effective_kv_backend_when_file_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "skulk.yaml"
    config_path.write_text(
        "inference:\n  kv_cache_backend: default\nhf_token: secret-token\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SKULK_KV_CACHE_BACKEND", "optiq")

    api = _build_api()
    object.__setattr__(api, "_config_path", config_path)
    client = TestClient(api.app)

    response = client.get("/config")

    assert response.status_code == 200
    data = _json_object(response)
    config = _json_mapping(data["config"])
    inference = _json_mapping(config["inference"])
    effective = _json_mapping(data["effective"])
    assert data["fileExists"] is True
    assert data["configPath"] == str(config_path)
    assert inference["kv_cache_backend"] == "default"
    assert config.get("hf_token") is None
    assert effective["kv_cache_backend"] == "optiq"
    assert effective["has_hf_token"] is True


def test_get_config_treats_blank_skulk_kv_backend_as_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "skulk.yaml"
    config_path.write_text("inference:\n  kv_cache_backend: default\n", encoding="utf-8")
    monkeypatch.setenv("SKULK_KV_CACHE_BACKEND", "")

    api = _build_api()
    object.__setattr__(api, "_config_path", config_path)
    client = TestClient(api.app)

    response = client.get("/config")

    assert response.status_code == 200
    data = _json_object(response)
    effective = _json_mapping(data["effective"])
    assert effective["kv_cache_backend"] == "default"


def test_get_config_treats_invalid_skulk_kv_backend_as_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "skulk.yaml"
    config_path.write_text("inference:\n  kv_cache_backend: default\n", encoding="utf-8")
    monkeypatch.setenv("SKULK_KV_CACHE_BACKEND", "typo-backend")

    api = _build_api()
    object.__setattr__(api, "_config_path", config_path)
    client = TestClient(api.app)

    response = client.get("/config")

    assert response.status_code == 200
    data = _json_object(response)
    effective = _json_mapping(data["effective"])
    assert effective["kv_cache_backend"] == "default"


def test_update_config_rejects_non_object_request_body(tmp_path: Path) -> None:
    api = _build_api()
    object.__setattr__(api, "_config_path", tmp_path / "skulk.yaml")
    client = TestClient(api.app)

    response = client.put("/config", json=["not", "an", "object"])

    assert response.status_code == 422
    assert response.json()["error"]["message"] == "Request body must be a JSON object."


def test_update_config_rejects_non_object_config_field(tmp_path: Path) -> None:
    api = _build_api()
    object.__setattr__(api, "_config_path", tmp_path / "skulk.yaml")
    client = TestClient(api.app)

    response = client.put("/config", json={"config": True})

    assert response.status_code == 422
    assert response.json()["error"]["message"] == "'config' field must be a JSON object."


def test_update_config_preserves_existing_experiments_when_omitted(
    tmp_path: Path,
) -> None:
    """Saving unrelated settings must not clear hidden experiment toggles."""

    config_path = tmp_path / "skulk.yaml"
    config_path.write_text(
        "experiments:\n  tts_streaming: true\n  stt_realtime: true\n",
        encoding="utf-8",
    )
    api = _build_api()
    object.__setattr__(api, "_config_path", config_path)
    client = TestClient(api.app)

    response = client.put(
        "/config",
        json={"config": {"inference": {"kv_cache_backend": "default"}}},
    )

    assert response.status_code == 200
    config = load_skulk_config(config_path)
    assert config is not None
    assert config.experiments is not None
    assert config.experiments.tts_streaming is True
    assert config.experiments.stt_realtime is True


def test_update_config_preserves_existing_model_trust_when_omitted(
    tmp_path: Path,
) -> None:
    """An older Settings client cannot silently revoke model approvals."""
    card_id = f"card_{'a' * 52}"
    config_path = tmp_path / "skulk.yaml"
    config_path.write_text(
        "model_trust:\n"
        "  approved_remote_code_identities:\n"
        f"    - {card_id}\n",
        encoding="utf-8",
    )
    api = _build_api()
    object.__setattr__(api, "_config_path", config_path)
    client = TestClient(api.app)

    response = client.put(
        "/config",
        json={"config": {"inference": {"kv_cache_backend": "default"}}},
    )

    assert response.status_code == 200
    config = load_skulk_config(config_path)
    assert config is not None
    assert config.model_trust is not None
    assert config.model_trust.approved_remote_code_identities == [card_id]


def test_update_config_rejects_model_trust_snapshot_on_loopback(
    tmp_path: Path,
) -> None:
    """Even an operator must use the master-ordered trust endpoints."""

    api = _build_api()
    object.__setattr__(api, "_config_path", tmp_path / "skulk.yaml")
    client = TestClient(api.app, client=("127.0.0.1", 50000))

    response = client.put(
        "/config",
        json={"config": {"model_trust": {"approved_remote_code_identities": []}}},
    )

    assert response.status_code == 409
    assert "master-ordered" in response.json()["error"]["message"]


@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        (
            "PUT",
            "/config",
            {"config": {"model_trust": {"approved_remote_code_identities": []}}},
        ),
        (
            "POST",
            f"/models/remote-code-approvals/card_{'a' * 52}",
            None,
        ),
        (
            "DELETE",
            f"/models/remote-code-approvals/card_{'a' * 52}",
            None,
        ),
        ("POST", "/models/add", {"model_id": "org/model"}),
    ],
)
def test_sensitive_model_mutations_reject_unauthenticated_network_client(
    tmp_path: Path,
    method: str,
    path: str,
    payload: dict[str, object] | None,
) -> None:
    """The unauthenticated fabric listener cannot grant model-code authority."""

    api = _build_api()
    object.__setattr__(api, "_config_path", tmp_path / "skulk.yaml")
    client = TestClient(api.app)

    response = client.request(method, path, json=payload)

    assert response.status_code == 403
    assert "loopback" in response.json()["error"]["message"]


@pytest.mark.asyncio
async def test_model_trust_decision_is_submitted_to_master(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One model approval becomes a master-serialized command."""
    config_path = tmp_path / "skulk.yaml"
    config_path.write_text(
        "hf_token: keep-secret\n",
        encoding="utf-8",
    )
    api = _build_api()
    object.__setattr__(api, "_config_path", config_path)
    send = AsyncMock()
    monkeypatch.setattr(api, "_send", send)
    card_id = f"card_{'a' * 52}"

    await api.set_cluster_remote_code_approval(card_id, approved=True)

    send.assert_awaited_once()
    assert send.await_args is not None
    command = cast(SetModelTrustApproval, cast(object, send.await_args.args[0]))
    assert command.trust_identity == card_id
    assert command.approved is True
    assert config_path.read_text() == "hf_token: keep-secret\n"
