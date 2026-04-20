"""Tests for the /config API endpoint."""

from pathlib import Path
from typing import cast

import pytest
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
    config_path = tmp_path / "exo.yaml"
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
    config_path = tmp_path / "exo.yaml"
    config_path.write_text("inference:\n  kv_cache_backend: default\n", encoding="utf-8")
    monkeypatch.setenv("SKULK_KV_CACHE_BACKEND", "")
    monkeypatch.setenv("EXO_KV_CACHE_BACKEND", "optiq")

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
    config_path = tmp_path / "exo.yaml"
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
    object.__setattr__(api, "_config_path", tmp_path / "exo.yaml")
    client = TestClient(api.app)

    response = client.put("/config", json=["not", "an", "object"])

    assert response.status_code == 422
    assert response.json()["error"]["message"] == "Request body must be a JSON object."


def test_update_config_rejects_non_object_config_field(tmp_path: Path) -> None:
    api = _build_api()
    object.__setattr__(api, "_config_path", tmp_path / "exo.yaml")
    client = TestClient(api.app)

    response = client.put("/config", json={"config": True})

    assert response.status_code == 422
    assert response.json()["error"]["message"] == "'config' field must be a JSON object."
