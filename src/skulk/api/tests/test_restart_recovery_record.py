"""The API serves this node's restart recovery record to fabric peers only."""

from pathlib import Path

import pytest
from fastapi import HTTPException, Request

import skulk.api.main as api_main
from skulk.api.main import API
from skulk.recovery.record import derive_cluster_intent, write_record
from skulk.recovery.tests.fixtures import INSTALL_IDS, NOW, SESSION, sample_state
from skulk.shared.types.common import NodeId


def _request(host: str) -> Request:
    return Request({"type": "http", "headers": [], "client": (host, 52415)})


async def test_a_fabric_peer_reads_the_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "cluster-intent.json"
    record = derive_cluster_intent(
        sample_state(), install_ids=INSTALL_IDS, session_id=SESSION, written_at=NOW
    )
    write_record(path, record)
    monkeypatch.setattr(api_main, "RECOVERY_RECORD_PATH", path)
    api = object.__new__(API)
    api.node_id = NodeId("serving-node")

    served = await api.get_restart_recovery_record(_request("192.168.1.20"))

    assert served.node_id == NodeId("serving-node")
    assert served.record == record


async def test_a_node_without_a_record_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api_main, "RECOVERY_RECORD_PATH", tmp_path / "missing.json")
    api = object.__new__(API)
    api.node_id = NodeId("serving-node")

    served = await api.get_restart_recovery_record(_request("127.0.0.1"))

    assert served.record is None


async def test_a_public_caller_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api_main, "RECOVERY_RECORD_PATH", tmp_path / "missing.json")
    api = object.__new__(API)
    api.node_id = NodeId("serving-node")

    with pytest.raises(HTTPException) as refused:
        await api.get_restart_recovery_record(_request("198.51.100.10"))

    assert refused.value.status_code == 403
