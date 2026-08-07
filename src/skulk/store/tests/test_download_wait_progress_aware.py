"""The worker's wait for a store download is progress-aware, not a total cap.

With the store host's uncapped file-body transfer, a very large model can take
hours. ``request_and_wait_for_download`` must not give up on a live,
still-progressing download (which would make the master tear the placement
down); it fails only on a genuine stall (no progress for ``timeout`` seconds).
"""

from collections.abc import Iterator
from typing import Any

import pytest

from skulk.store import model_store_client
from skulk.store.model_store_client import ModelStoreClient


class _FakeResp:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.status = 200

    async def __aenter__(self) -> "_FakeResp":
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def json(self) -> dict[str, Any]:
        return self._payload


class _FakeSession:
    """Returns the POST ack once, then walks a scripted list of status polls."""

    def __init__(
        self,
        polls: Iterator[dict[str, Any]],
        post_payload: dict[str, Any],
    ) -> None:
        self._polls = polls
        self._post_payload = post_payload

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    def post(self, *_: object, **__: object) -> _FakeResp:
        return _FakeResp(self._post_payload)

    def get(self, *_: object, **__: object) -> _FakeResp:
        return _FakeResp(next(self._polls))


def _install(
    monkeypatch: pytest.MonkeyPatch,
    polls: list[dict[str, Any]],
    *,
    post_payload: dict[str, Any] | None = None,
) -> None:
    it = iter(polls)
    response = post_payload or {"status": "pending", "progress": 0.0}

    def _fake(*_: object, **__: object) -> _FakeSession:
        return _FakeSession(it, response)

    monkeypatch.setattr(model_store_client, "create_http_session", _fake)


async def test_wait_does_not_time_out_while_progress_advances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Progress creeps up over many polls, far more than `timeout`/`poll_interval`
    # would allow if the wait were a fixed total. It must still complete.
    polls = [{"status": "downloading", "progress": p / 100} for p in range(1, 40)]
    polls.append({"status": "complete", "progress": 1.0})
    _install(monkeypatch, polls)
    client = ModelStoreClient(store_host="h", store_port=1)
    ok = await client.request_and_wait_for_download(
        "org/big", timeout=0.05, poll_interval=0.0
    )
    assert ok is True


async def test_wait_does_not_count_pending_queue_time_as_a_stall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    polls = [{"status": "pending", "progress": 0.0} for _ in range(4)]
    polls.extend(
        [
            {"status": "downloading", "progress": 0.0},
            {"status": "complete", "progress": 1.0},
        ]
    )
    _install(monkeypatch, polls)
    client = ModelStoreClient(store_host="h", store_port=1)

    ok = await client.request_and_wait_for_download(
        "org/queued",
        timeout=0.02,
        poll_interval=0.01,
    )

    assert ok is True


async def test_wait_times_out_on_a_genuine_stall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Progress never advances: this is a real stall and must time out.
    polls = [{"status": "downloading", "progress": 0.5} for _ in range(1000)]
    _install(monkeypatch, polls)
    client = ModelStoreClient(store_host="h", store_port=1)
    with pytest.raises(TimeoutError):
        await client.request_and_wait_for_download(
            "org/stuck", timeout=0.02, poll_interval=0.01
        )


async def test_wait_rejects_immediate_completion_for_other_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_revision = "1" * 40
    _install(
        monkeypatch,
        [],
        post_payload={
            "status": "complete",
            "progress": 1.0,
            "sourceRevision": "0" * 40,
        },
    )
    client = ModelStoreClient(store_host="h", store_port=1)

    with pytest.raises(RuntimeError, match="but revision .* was requested"):
        await client.request_and_wait_for_download(
            "org/model",
            source_revision=requested_revision,
        )


async def test_wait_rejects_polled_completion_for_other_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_revision = "1" * 40
    _install(
        monkeypatch,
        [
            {
                "status": "complete",
                "progress": 1.0,
                "sourceRevision": "0" * 40,
            }
        ],
    )
    client = ModelStoreClient(store_host="h", store_port=1)

    with pytest.raises(RuntimeError, match="but revision .* was requested"):
        await client.request_and_wait_for_download(
            "org/model",
            poll_interval=0.0,
            source_revision=requested_revision,
        )
