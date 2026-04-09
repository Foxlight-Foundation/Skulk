# pyright: reportUnusedFunction=false, reportPrivateUsage=false
"""Tests for the GET /config API endpoint."""

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from exo.api.main import API
from exo.shared.types.common import NodeId
from exo.utils.channels import channel


def _build_api(node_id: str = "test-node") -> API:
    """Create a minimal API instance for config endpoint testing."""
    command_sender, _ = channel()
    download_sender, _ = channel()
    _, event_receiver = channel()
    _, election_receiver = channel()
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
    api._config_path = config_path  # pyright: ignore[reportPrivateUsage]
    client = TestClient(api.app)

    response = client.get("/config")

    assert response.status_code == 200
    data: dict[str, Any] = response.json()
    assert data["fileExists"] is True
    assert data["configPath"] == str(config_path)
    assert data["config"]["inference"]["kv_cache_backend"] == "default"
    assert data["config"].get("hf_token") is None
    assert data["effective"]["kv_cache_backend"] == "optiq"
    assert data["effective"]["has_hf_token"] is True


def test_get_config_treats_blank_skulk_kv_backend_as_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "exo.yaml"
    config_path.write_text("inference:\n  kv_cache_backend: default\n", encoding="utf-8")
    monkeypatch.setenv("SKULK_KV_CACHE_BACKEND", "")
    monkeypatch.setenv("EXO_KV_CACHE_BACKEND", "optiq")

    api = _build_api()
    api._config_path = config_path  # pyright: ignore[reportPrivateUsage]
    client = TestClient(api.app)

    response = client.get("/config")

    assert response.status_code == 200
    data: dict[str, Any] = response.json()
    assert data["effective"]["kv_cache_backend"] == "default"
