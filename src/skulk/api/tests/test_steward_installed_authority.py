"""Steward installation reads require separately granted, current HTTP authority."""

from pathlib import Path

import pytest
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse

from skulk.api.main import API
from skulk.api.steward import STEWARD_VIRTUAL_MODEL_ID, InstalledMetadataReader
from skulk.api.tests.test_operator_gateway import paired_service
from skulk.api.types import ChatCompletionRequest
from skulk.operator.pairing import PluginGrantUpdate
from skulk.shared.election import ElectionMessage
from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.common import ModelId, NodeId
from skulk.shared.types.events import IndexedEvent
from skulk.utils.channels import channel


async def test_steward_reader_is_omitted_without_grant_and_rechecks_revocation(
    tmp_path: "Path", monkeypatch: pytest.MonkeyPatch
) -> None:
    """Captured tool callbacks cannot outlive their caller's plugin-read grant."""

    pairing, exchange = paired_service(tmp_path)
    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    _, event_receiver = channel[IndexedEvent]()
    _, election_receiver = channel[ElectionMessage]()
    api = API(
        NodeId("fixture"),
        port=52415,
        event_receiver=event_receiver,
        command_sender=command_sender,
        download_command_sender=download_sender,
        election_receiver=election_receiver,
        enable_event_log=False,
        mount_dashboard=False,
        operator_pairing_service=pairing,
    )
    readers: list[InstalledMetadataReader | None] = []

    async def capture(
        _self: API,
        _payload: ChatCompletionRequest,
        *,
        steward_proposals_allowed: bool = True,
        steward_installation_reader: InstalledMetadataReader | None = None,
    ) -> StreamingResponse:
        readers.append(steward_installation_reader)
        return StreamingResponse(iter(()))

    monkeypatch.setattr(API, "chat_completions", capture)
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "scheme": "https",
            "server": ("localhost", 443),
            "client": ("127.0.0.1", 52000),
            "headers": [(b"authorization", f"Bearer {exchange.access_token}".encode())],
        }
    )
    payload = ChatCompletionRequest(
        model=ModelId(STEWARD_VIRTUAL_MODEL_ID), messages=[]
    )
    await api.chat_completions_route(payload, request)
    assert readers == [None]
    pairing.set_plugin_grant(
        exchange.device_id,
        PluginGrantUpdate(expected_revision=0, scopes=("plugins:read",)),
    )
    await api.chat_completions_route(payload, request)
    reader = readers[-1]
    assert reader is not None
    assert "unavailable" in str(await reader(None, None, None))
    pairing.set_plugin_grant(
        exchange.device_id, PluginGrantUpdate(expected_revision=1, scopes=())
    )
    with pytest.raises(HTTPException) as refused:
        await reader(None, None, None)
    assert refused.value.status_code == 403
