"""Exercise owned-job cleanup and producer failures without loading GPU weights."""

import io
import json
import signal
import subprocess
import threading
from collections.abc import Iterator

import httpx
import pytest

from skulk.worker.runner.stable_diffusion_cpp.server import (
    NativeImageCancelledError,
    NativeImageServer,
    NativePreview,
    NativeResult,
)


class _ReapProcess(subprocess.Popen[bytes]):
    """Model delayed driver exit without creating an operating-system process."""

    def __init__(self, outcomes: list[int | subprocess.TimeoutExpired]) -> None:
        self.pid = 123
        self.returncode = None
        self._child_created = False
        self.outcomes = outcomes
        self.waits: list[float | None] = []

    def wait(self, timeout: float | None = None) -> int:
        self.waits.append(timeout)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, subprocess.TimeoutExpired):
            raise outcome
        self.returncode = outcome
        return outcome


def test_slow_native_reap_keeps_ownership_and_closes_log(
    producer: NativeImageServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = _ReapProcess([
        subprocess.TimeoutExpired("native", 30),
        subprocess.TimeoutExpired("native", 15),
        0,
    ])
    log = io.BytesIO()
    signals: list[int] = []

    def send_signal(_pid: int, sent: int) -> None:
        signals.append(sent)

    monkeypatch.setattr(producer, "_process", process)
    monkeypatch.setattr(producer, "_log", log)
    monkeypatch.setattr(
        "skulk.worker.runner.stable_diffusion_cpp.server.os.killpg",
        send_signal,
    )
    with pytest.raises(subprocess.TimeoutExpired):
        producer.close()
    assert log.closed
    producer.close()
    assert process.waits == [30, 15, 30]
    assert signals == [signal.SIGTERM, signal.SIGKILL, signal.SIGTERM]


@pytest.fixture
def producer(monkeypatch: pytest.MonkeyPatch) -> Iterator[NativeImageServer]:
    server = NativeImageServer.__new__(NativeImageServer)
    monkeypatch.setattr(server, "_lock", threading.Lock(), raising=False)
    monkeypatch.setattr(server, "_process", None, raising=False)
    monkeypatch.setattr(server, "_client", None, raising=False)
    monkeypatch.setattr(server, "_log", io.BytesIO(), raising=False)
    monkeypatch.setattr(server, "check_alive", lambda: None)

    def no_sleep(_seconds: float) -> None:
        pass

    monkeypatch.setattr(
        "skulk.worker.runner.stable_diffusion_cpp.server.time.sleep", no_sleep
    )
    yield server
    server.close()


def _job(status: str, *, identifier: str = "owned", preview: bool = False) -> bytes:
    document: dict[str, object] = {
        "id": identifier,
        "kind": "img_gen",
        "status": status,
    }
    if preview:
        document["preview"] = {
            "pass": 1,
            "step": 2,
            "total_steps": 4,
            "b64_json": "YQ==",
        }
    if status == "completed":
        document["result"] = {
            "output_format": "png",
            "images": [{"index": 0, "b64_json": "YQ=="}],
        }
    return json.dumps(document).encode()


def test_terminal_collection_deletes_job_and_deduplicates_preview(
    producer: NativeImageServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshots = iter(
        [
            _job("generating", preview=True),
            _job("generating", preview=True),
            _job("completed"),
        ]
    )
    calls: list[tuple[str, str]] = []

    def request(method: str, path: str, _body: object = None) -> bytes:
        calls.append((method, path))
        return next(snapshots) if method == "GET" else b'{"id":"owned"}'

    monkeypatch.setattr(producer, "_request", request)
    outputs = list(producer.generate({"prompt": "literal"}, threading.Event()))
    assert len(outputs) == 2
    assert isinstance(outputs[0], NativePreview)
    assert isinstance(outputs[1], NativeResult)
    assert calls[-1] == ("DELETE", "/sdcpp/v1/jobs/owned")


def test_active_cancellation_waits_for_producer_then_deletes(
    producer: NativeImageServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    cancelled = threading.Event()
    calls: list[tuple[str, str]] = []

    def request(method: str, path: str, _body: object = None) -> bytes:
        calls.append((method, path))
        if method == "GET":
            return _job("cancelled")
        if path == "/sdcpp/v1/img_gen":
            cancelled.set()
        return b'{"id":"owned"}'

    monkeypatch.setattr(producer, "_request", request)
    with pytest.raises(NativeImageCancelledError):
        list(producer.generate({}, cancelled))
    assert ("POST", "/sdcpp/v1/jobs/owned/cancel") in calls
    assert calls[-1] == ("DELETE", "/sdcpp/v1/jobs/owned")


def test_changed_job_ownership_closes_producer(
    producer: NativeImageServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed: list[bool] = []

    def request(method: str, _path: str, _body: object = None) -> bytes:
        return (
            _job("completed", identifier="other")
            if method == "GET"
            else b'{"id":"owned"}'
        )

    monkeypatch.setattr(producer, "_request", request)
    monkeypatch.setattr(producer, "close", lambda: closed.append(True))
    with pytest.raises(ValueError, match="ownership"):
        list(producer.generate({}, threading.Event()))
    assert closed == [True]


def test_abandoned_preview_closes_active_producer(
    producer: NativeImageServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed: list[bool] = []

    def request(method: str, _path: str, _body: object = None) -> bytes:
        return (
            _job("generating", preview=True) if method == "GET" else b'{"id":"owned"}'
        )

    monkeypatch.setattr(producer, "_request", request)
    monkeypatch.setattr(producer, "close", lambda: closed.append(True))
    outputs = producer.generate({}, threading.Event())
    assert isinstance(next(outputs), NativePreview)
    outputs.close()
    assert closed == [True]


def test_native_failure_is_terminal_and_collected(
    producer: NativeImageServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def request(method: str, _path: str, _body: object = None) -> bytes:
        calls.append(method)
        return _job("failed") if method == "GET" else b'{"id":"owned"}'

    monkeypatch.setattr(producer, "_request", request)
    with pytest.raises(RuntimeError, match="generation failed"):
        list(producer.generate({}, threading.Event()))
    assert calls[-1] == "DELETE"


def test_private_http_response_has_a_byte_limit(
    producer: NativeImageServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "skulk.worker.runner.stable_diffusion_cpp.server._MAX_RESPONSE_BYTES", 5
    )
    client = httpx.Client(
        base_url="http://127.0.0.1",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, content=b"123456")
        ),
    )
    monkeypatch.setattr(producer, "_client", client)
    with pytest.raises(ValueError, match="response exceeds"):
        producer._request("GET", "/bounded")  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize("reply", [b"[]", b'{"id":"../other"}', b"invalid"])
def test_uncertain_submission_closes_producer(
    producer: NativeImageServer, monkeypatch: pytest.MonkeyPatch, reply: bytes
) -> None:
    closed: list[bool] = []

    def request(_method: str, _path: str, _body: object = None) -> bytes:
        return reply

    monkeypatch.setattr(producer, "_request", request)
    monkeypatch.setattr(producer, "close", lambda: closed.append(True))
    with pytest.raises(ValueError):
        list(producer.generate({}, threading.Event()))
    assert closed == [True]
