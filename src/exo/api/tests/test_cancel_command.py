# pyright: reportUnusedFunction=false, reportAny=false
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from exo.api.main import API
from exo.shared.types.common import CommandId
from exo.utils.channels import Sender


def _make_api() -> Any:
    """Create a minimal API instance with cancel route and error handler."""

    app = FastAPI()
    api = object.__new__(API)
    api.app = app
    api._text_generation_queues = {}  # pyright: ignore[reportPrivateUsage]
    api._image_generation_queues = {}  # pyright: ignore[reportPrivateUsage]
    api._embedding_queues = {}  # pyright: ignore[reportPrivateUsage]
    api._cancelled_command_ids = set()  # pyright: ignore[reportPrivateUsage]
    api._send = AsyncMock()  # pyright: ignore[reportPrivateUsage]
    api._setup_exception_handlers()  # pyright: ignore[reportPrivateUsage]
    app.post("/v1/cancel/{command_id}")(api.cancel_command)
    return api


def test_cancel_nonexistent_command_returns_404() -> None:
    """Cancel for an unknown command_id returns 404 in OpenAI error format."""
    api = _make_api()
    client = TestClient(api.app)

    response = client.post("/v1/cancel/nonexistent-id")
    assert response.status_code == 404
    data: dict[str, Any] = response.json()
    assert "error" in data
    assert data["error"]["message"] == "Command not found or already completed"
    assert data["error"]["type"] == "Not Found"
    assert data["error"]["code"] == 404


def test_cancel_active_text_generation() -> None:
    """Cancel an active text generation command: returns 200, sender.close() called."""
    api = _make_api()
    client = TestClient(api.app)

    cid = CommandId("text-cmd-123")
    sender = MagicMock()
    api._text_generation_queues[cid] = sender

    response = client.post(f"/v1/cancel/{cid}")
    assert response.status_code == 200
    data: dict[str, Any] = response.json()
    assert data["message"] == "Command cancelled."
    assert data["command_id"] == str(cid)
    sender.close.assert_called_once()
    api._send.assert_called_once()
    assert cid in api._cancelled_command_ids
    task_cancelled = api._send.call_args[0][0]
    assert task_cancelled.cancelled_command_id == cid


def test_cancel_active_image_generation() -> None:
    """Cancel an active image generation command: returns 200, sender.close() called."""
    api = _make_api()
    client = TestClient(api.app)

    cid = CommandId("img-cmd-456")
    sender = MagicMock()
    api._image_generation_queues[cid] = sender

    response = client.post(f"/v1/cancel/{cid}")
    assert response.status_code == 200
    data: dict[str, Any] = response.json()
    assert data["message"] == "Command cancelled."
    assert data["command_id"] == str(cid)
    sender.close.assert_called_once()
    api._send.assert_called_once()
    assert cid in api._cancelled_command_ids
    task_cancelled = api._send.call_args[0][0]
    assert task_cancelled.cancelled_command_id == cid


@pytest.mark.asyncio
async def test_finalize_command_stream_suppresses_task_finished_for_cancelled_command() -> None:
    """Local cancellation should skip TaskFinished so workers can observe Cancelled."""

    api = _make_api()
    cid = CommandId("cancelled-cmd")
    sender = MagicMock()
    queue: dict[CommandId, Sender[object]] = {
        cid: cast(Sender[object], cast(object, sender))
    }
    api._cancelled_command_ids.add(cid)

    await api._finalize_command_stream(cid, queue)

    api._send.assert_not_called()
    assert cid not in queue
    assert cid not in api._cancelled_command_ids


@pytest.mark.asyncio
async def test_finalize_command_stream_reports_natural_completion() -> None:
    """Natural completion should still emit TaskFinished."""

    api = _make_api()
    cid = CommandId("finished-cmd")
    sender = MagicMock()
    queue: dict[CommandId, Sender[object]] = {
        cid: cast(Sender[object], cast(object, sender))
    }

    await api._finalize_command_stream(cid, queue)

    api._send.assert_called_once()
    task_finished = api._send.call_args[0][0]
    assert task_finished.finished_command_id == cid
    assert cid not in queue
