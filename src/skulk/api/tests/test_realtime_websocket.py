# pyright: reportPrivateUsage=false
"""Realtime transcription WebSocket compatibility-edge coverage."""

import base64
from collections.abc import AsyncIterator
from threading import Event
from typing import cast

import anyio
import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession
from starlette.websockets import WebSocketDisconnect

from skulk.api.main import API
from skulk.extensions import (
    CapabilityResult,
    CapabilityStreamError,
    CapabilityStreamFrame,
    CapabilityStreamInput,
    CapabilityStreamSession,
    InlineMediaAttachment,
    call_failure,
)
from skulk.shared.election import ElectionMessage
from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.common import NodeId
from skulk.shared.types.events import IndexedEvent
from skulk.utils.channels import channel


def _build_api() -> API:
    """Build an API whose WebSocket provider opener can be replaced per test."""

    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    _, event_receiver = channel[IndexedEvent]()
    _, election_receiver = channel[ElectionMessage]()
    return API(
        NodeId("api-node"),
        port=52415,
        event_receiver=event_receiver,
        command_sender=command_sender,
        download_command_sender=download_sender,
        election_receiver=election_receiver,
        enable_event_log=False,
        mount_dashboard=False,
    )


def _receive_json(websocket: WebSocketTestSession) -> dict[str, object]:
    """Narrow one WebSocket JSON message to an object for strict tests."""

    return cast(dict[str, object], cast(object, websocket.receive_json()))


def _mapping(value: object) -> dict[str, object]:
    """Narrow one nested JSON object."""

    return cast(dict[str, object], value)


async def _empty_frames() -> AsyncIterator[CapabilityStreamFrame]:
    if False:
        yield CapabilityStreamFrame(
            call_id="unused",
            direction="provider_to_caller",
            sequence=0,
            kind="started",
        )


def _install_waiting_session(
    api: API,
    input_frames: list[CapabilityStreamFrame],
    *,
    call_id: str,
) -> None:
    """Install a provider session that remains live until the bridge cancels it."""

    async def open_session(model: str, sample_rate: int) -> CapabilityStreamSession:
        del model, sample_rate

        async def send_input(frame: CapabilityStreamFrame) -> None:
            input_frames.append(frame)

        input_stream = CapabilityStreamInput(
            call_id=call_id,
            deadline_at=anyio.current_time() + 10.0,
            send_frame=send_input,
        )
        await input_stream.start()

        async def output_frames() -> AsyncIterator[CapabilityStreamFrame]:
            yield CapabilityStreamFrame(
                call_id=call_id,
                direction="provider_to_caller",
                sequence=0,
                kind="started",
            )
            await anyio.sleep_forever()

        return CapabilityStreamSession(
            open_result=CapabilityResult(
                call_id=call_id,
                ok=True,
                result={"admitted": True},
            ),
            frames=output_frames(),
            input=input_stream,
        )

    api._open_realtime_transcription_session = open_session


def test_realtime_websocket_translates_pcm_and_transcript_lifecycle() -> None:
    """The compatibility edge maps one committed utterance onto provider frames."""

    api = _build_api()
    input_frames: list[CapabilityStreamFrame] = []

    async def open_session(model: str, sample_rate: int) -> CapabilityStreamSession:
        assert model == "org/realtime-stt"
        assert sample_rate == 24_000
        input_completed = anyio.Event()

        async def send_input(frame: CapabilityStreamFrame) -> None:
            input_frames.append(frame)
            if frame.kind == "completed":
                input_completed.set()

        input_stream = CapabilityStreamInput(
            call_id="ws-call",
            deadline_at=anyio.current_time() + 10.0,
            send_frame=send_input,
        )
        await input_stream.start()

        async def output_frames() -> AsyncIterator[CapabilityStreamFrame]:
            yield CapabilityStreamFrame(
                call_id="ws-call",
                direction="provider_to_caller",
                sequence=0,
                kind="started",
            )
            await input_completed.wait()
            yield CapabilityStreamFrame(
                call_id="ws-call",
                direction="provider_to_caller",
                sequence=1,
                kind="chunk",
                payload={"model": model, "text": "hello ", "is_partial": True},
            )
            yield CapabilityStreamFrame(
                call_id="ws-call",
                direction="provider_to_caller",
                sequence=2,
                kind="completed",
                payload={"model": model, "text": "hello world", "is_partial": False},
            )

        return CapabilityStreamSession(
            open_result=CapabilityResult(
                call_id="ws-call",
                ok=True,
                result={"admitted": True},
            ),
            frames=output_frames(),
            input=input_stream,
        )

    api._open_realtime_transcription_session = open_session
    client = TestClient(api.app)
    audio = b"\x01\x00\x02\x00"

    with client.websocket_connect(
        "/v1/realtime?model=org%2Frealtime-stt",
        headers={"origin": "http://testserver"},
    ) as websocket:
        created = _receive_json(websocket)
        assert created["type"] == "session.created"
        created_session = _mapping(created["session"])
        created_audio = _mapping(created_session["audio"])
        created_input = _mapping(created_audio["input"])
        created_format = _mapping(created_input["format"])
        created_transcription = _mapping(created_input["transcription"])
        assert created_session["type"] == "transcription"
        assert created_format == {"type": "audio/pcm", "rate": 24_000}
        assert created_transcription["model"] == (
            "org/realtime-stt"
        )

        websocket.send_json(
            {
                "type": "session.update",
                "event_id": "update-1",
                "session": {
                    "type": "transcription",
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": 24_000},
                            "transcription": {"model": "org/realtime-stt"},
                            "turn_detection": None,
                            "noise_reduction": None,
                        }
                    },
                    "include": [],
                },
            }
        )
        assert _receive_json(websocket)["type"] == "session.updated"
        websocket.send_json(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(audio).decode("ascii"),
            }
        )
        websocket.send_json({"type": "input_audio_buffer.commit"})

        assert _receive_json(websocket)["type"] == "input_audio_buffer.committed"
        delta = _receive_json(websocket)
        assert delta["type"] == "conversation.item.input_audio_transcription.delta"
        assert delta["delta"] == "hello "
        completed = _receive_json(websocket)
        assert completed["type"] == (
            "conversation.item.input_audio_transcription.completed"
        )
        assert completed["transcript"] == "hello world"

    assert [frame.kind for frame in input_frames] == ["started", "chunk", "completed"]
    media = input_frames[1].media
    assert isinstance(media, InlineMediaAttachment)
    assert media.data == audio
    assert media.sample_rate == 24_000


def test_realtime_websocket_rejects_invalid_audio_without_forwarding() -> None:
    """Malformed base64 terminates only the socket and never reaches the provider."""

    api = _build_api()
    input_frames: list[CapabilityStreamFrame] = []

    _install_waiting_session(api, input_frames, call_id="invalid-audio")
    client = TestClient(api.app)

    with client.websocket_connect("/v1/realtime?model=org%2Frealtime-stt") as websocket:
        assert _receive_json(websocket)["type"] == "session.created"
        websocket.send_json(
            {"type": "input_audio_buffer.append", "event_id": "bad", "audio": "%%%"}
        )
        error = _receive_json(websocket)
        assert error["type"] == "error"
        error_detail = _mapping(error["error"])
        assert error_detail["code"] == "invalid_audio"
        assert error_detail["event_id"] == "bad"
        try:
            _receive_json(websocket)
        except WebSocketDisconnect as exc:
            assert exc.code == 1008

    assert [frame.kind for frame in input_frames] == ["started", "cancelled"]


def test_realtime_websocket_rejects_binary_compatibility_events() -> None:
    """The OpenAI-compatible edge requires JSON text frames."""

    api = _build_api()
    input_frames: list[CapabilityStreamFrame] = []
    _install_waiting_session(api, input_frames, call_id="binary-event")
    client = TestClient(api.app)

    with client.websocket_connect("/v1/realtime?model=org%2Frealtime-stt") as websocket:
        assert _receive_json(websocket)["type"] == "session.created"
        websocket.send_bytes(b"\x00\x01")
        error = _receive_json(websocket)
        assert _mapping(error["error"])["code"] == "unsupported_frame"
        with pytest.raises(WebSocketDisconnect) as disconnect:
            _receive_json(websocket)
        assert disconnect.value.code == 1003

    assert [frame.kind for frame in input_frames] == ["started", "cancelled"]


def test_realtime_websocket_rejects_unimplemented_vad_configuration() -> None:
    """The edge does not advertise or silently ignore unsupported server VAD."""

    api = _build_api()
    input_frames: list[CapabilityStreamFrame] = []
    _install_waiting_session(api, input_frames, call_id="vad-update")
    client = TestClient(api.app)

    with client.websocket_connect("/v1/realtime?model=org%2Frealtime-stt") as websocket:
        assert _receive_json(websocket)["type"] == "session.created"
        websocket.send_json(
            {
                "type": "transcription_session.update",
                "session": {
                    "input_audio_format": "pcm16",
                    "input_audio_transcription": {"model": "org/realtime-stt"},
                    "turn_detection": {"type": "server_vad"},
                },
            }
        )
        error = _receive_json(websocket)
        assert _mapping(error["error"])["code"] == "invalid_event"

    assert [frame.kind for frame in input_frames] == ["started", "cancelled"]


def test_realtime_websocket_denies_cross_origin_browser_connection() -> None:
    """Browser sockets cannot cross origins even though SDK clients omit Origin."""

    api = _build_api()
    opened = Event()

    async def open_session(model: str, sample_rate: int) -> CapabilityStreamSession:
        del model, sample_rate
        opened.set()
        return CapabilityStreamSession(
            open_result=call_failure("unexpected", "provider_error", "unexpected"),
            frames=_empty_frames(),
        )

    api._open_realtime_transcription_session = open_session
    client = TestClient(api.app)

    with (
        pytest.raises(WebSocketDisconnect) as disconnect,
        client.websocket_connect(
            "/v1/realtime?model=org%2Frealtime-stt",
            headers={"origin": "https://untrusted.example"},
        ),
    ):
        pass

    assert disconnect.value.code == 1008
    assert not opened.is_set()


def test_realtime_websocket_surfaces_provider_admission_failure() -> None:
    """A typed provider rejection becomes an error event and retryable close."""

    api = _build_api()

    async def open_session(model: str, sample_rate: int) -> CapabilityStreamSession:
        del model, sample_rate
        return CapabilityStreamSession(
            open_result=call_failure("rejected", "overloaded", "all runners busy"),
            frames=_empty_frames(),
        )

    api._open_realtime_transcription_session = open_session
    client = TestClient(api.app)

    with client.websocket_connect("/v1/realtime?model=org%2Frealtime-stt") as websocket:
        error = _receive_json(websocket)
        assert error["type"] == "error"
        assert _mapping(error["error"])["code"] == "overloaded"
        try:
            _receive_json(websocket)
        except WebSocketDisconnect as exc:
            assert exc.code == 1013


def test_realtime_websocket_surfaces_provider_failure_before_commit() -> None:
    """Runner loss is reported immediately even while client input remains open."""

    api = _build_api()
    input_frames: list[CapabilityStreamFrame] = []

    async def open_session(model: str, sample_rate: int) -> CapabilityStreamSession:
        del model, sample_rate

        async def send_input(frame: CapabilityStreamFrame) -> None:
            input_frames.append(frame)

        input_stream = CapabilityStreamInput(
            call_id="provider-failed",
            deadline_at=anyio.current_time() + 10.0,
            send_frame=send_input,
        )
        await input_stream.start()

        async def output_frames() -> AsyncIterator[CapabilityStreamFrame]:
            yield CapabilityStreamFrame(
                call_id="provider-failed",
                direction="provider_to_caller",
                sequence=0,
                kind="started",
            )
            yield CapabilityStreamFrame(
                call_id="provider-failed",
                direction="provider_to_caller",
                sequence=1,
                kind="failed",
                error=CapabilityStreamError(
                    code="transport_error",
                    message="serving runner disconnected",
                ),
            )

        return CapabilityStreamSession(
            open_result=CapabilityResult(
                call_id="provider-failed",
                ok=True,
                result={"admitted": True},
            ),
            frames=output_frames(),
            input=input_stream,
        )

    api._open_realtime_transcription_session = open_session
    client = TestClient(api.app)

    with client.websocket_connect("/v1/realtime?model=org%2Frealtime-stt") as websocket:
        assert _receive_json(websocket)["type"] == "session.created"
        failed = _receive_json(websocket)
        assert failed["type"] == "conversation.item.input_audio_transcription.failed"
        assert _mapping(failed["error"])["code"] == "transport_error"
        with pytest.raises(WebSocketDisconnect) as disconnect:
            _receive_json(websocket)
        assert disconnect.value.code == 1011

    assert [frame.kind for frame in input_frames] == ["started", "cancelled"]


def test_realtime_websocket_drains_partial_output_before_commit() -> None:
    """Pre-commit transcript output stays bounded without blocking Fabric drain."""

    api = _build_api()
    input_frames: list[CapabilityStreamFrame] = []
    partials_drained = Event()

    async def open_session(model: str, sample_rate: int) -> CapabilityStreamSession:
        del model, sample_rate
        input_completed = anyio.Event()

        async def send_input(frame: CapabilityStreamFrame) -> None:
            input_frames.append(frame)
            if frame.kind == "completed":
                input_completed.set()

        input_stream = CapabilityStreamInput(
            call_id="precommit-partials",
            deadline_at=anyio.current_time() + 10.0,
            send_frame=send_input,
        )
        await input_stream.start()

        async def output_frames() -> AsyncIterator[CapabilityStreamFrame]:
            yield CapabilityStreamFrame(
                call_id="precommit-partials",
                direction="provider_to_caller",
                sequence=0,
                kind="started",
            )
            for sequence in range(1, 301):
                yield CapabilityStreamFrame(
                    call_id="precommit-partials",
                    direction="provider_to_caller",
                    sequence=sequence,
                    kind="chunk",
                    payload={"model": "org/realtime-stt", "text": "x"},
                )
            partials_drained.set()
            await input_completed.wait()
            yield CapabilityStreamFrame(
                call_id="precommit-partials",
                direction="provider_to_caller",
                sequence=301,
                kind="completed",
                payload={"model": "org/realtime-stt", "text": "x" * 300},
            )

        return CapabilityStreamSession(
            open_result=CapabilityResult(
                call_id="precommit-partials",
                ok=True,
                result={"admitted": True},
            ),
            frames=output_frames(),
            input=input_stream,
        )

    api._open_realtime_transcription_session = open_session
    client = TestClient(api.app)

    with client.websocket_connect("/v1/realtime?model=org%2Frealtime-stt") as websocket:
        assert _receive_json(websocket)["type"] == "session.created"
        assert partials_drained.wait(timeout=1.0)
        websocket.send_json(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(b"\x01\x00").decode("ascii"),
            }
        )
        websocket.send_json({"type": "input_audio_buffer.commit"})

        assert _receive_json(websocket)["type"] == "input_audio_buffer.committed"
        for _ in range(300):
            delta = _receive_json(websocket)
            assert delta["type"] == (
                "conversation.item.input_audio_transcription.delta"
            )
            assert delta["delta"] == "x"
        completed = _receive_json(websocket)
        assert completed["type"] == (
            "conversation.item.input_audio_transcription.completed"
        )
        assert completed["transcript"] == "x" * 300

    assert [frame.kind for frame in input_frames] == ["started", "chunk", "completed"]


def test_realtime_websocket_surfaces_provider_input_transport_failure() -> None:
    """A failed Fabric input send becomes a typed socket error, not an ASGI escape."""

    api = _build_api()

    async def open_session(model: str, sample_rate: int) -> CapabilityStreamSession:
        del model, sample_rate

        async def send_input(frame: CapabilityStreamFrame) -> None:
            if frame.kind == "chunk":
                raise anyio.ClosedResourceError

        input_stream = CapabilityStreamInput(
            call_id="input-failed",
            deadline_at=anyio.current_time() + 10.0,
            send_frame=send_input,
        )
        await input_stream.start()

        async def output_frames() -> AsyncIterator[CapabilityStreamFrame]:
            yield CapabilityStreamFrame(
                call_id="input-failed",
                direction="provider_to_caller",
                sequence=0,
                kind="started",
            )
            await anyio.sleep_forever()

        return CapabilityStreamSession(
            open_result=CapabilityResult(
                call_id="input-failed",
                ok=True,
                result={"admitted": True},
            ),
            frames=output_frames(),
            input=input_stream,
        )

    api._open_realtime_transcription_session = open_session
    client = TestClient(api.app)

    with client.websocket_connect("/v1/realtime?model=org%2Frealtime-stt") as websocket:
        assert _receive_json(websocket)["type"] == "session.created"
        websocket.send_json(
            {
                "type": "input_audio_buffer.append",
                "event_id": "audio-1",
                "audio": base64.b64encode(b"\x01\x00").decode("ascii"),
            }
        )
        error = _receive_json(websocket)
        error_detail = _mapping(error["error"])
        assert error_detail["code"] == "input_transport_error"
        assert error_detail["event_id"] == "audio-1"
        with pytest.raises(WebSocketDisconnect) as disconnect:
            _receive_json(websocket)
        assert disconnect.value.code == 1011


def test_realtime_websocket_disconnect_cancels_provider_input() -> None:
    """Closing the browser socket promptly cancels the provider direction."""

    api = _build_api()
    input_frames: list[CapabilityStreamFrame] = []
    output_cancelled = Event()

    async def open_session(model: str, sample_rate: int) -> CapabilityStreamSession:
        del model, sample_rate
        async def send_input(frame: CapabilityStreamFrame) -> None:
            input_frames.append(frame)

        input_stream = CapabilityStreamInput(
            call_id="disconnect",
            deadline_at=anyio.current_time() + 10.0,
            send_frame=send_input,
        )
        await input_stream.start()

        async def output_frames() -> AsyncIterator[CapabilityStreamFrame]:
            try:
                yield CapabilityStreamFrame(
                    call_id="disconnect",
                    direction="provider_to_caller",
                    sequence=0,
                    kind="started",
                )
                await anyio.sleep_forever()
            finally:
                output_cancelled.set()

        return CapabilityStreamSession(
            open_result=CapabilityResult(
                call_id="disconnect",
                ok=True,
                result={"admitted": True},
            ),
            frames=output_frames(),
            input=input_stream,
        )

    api._open_realtime_transcription_session = open_session
    client = TestClient(api.app)

    with client.websocket_connect("/v1/realtime?model=org%2Frealtime-stt") as websocket:
        assert _receive_json(websocket)["type"] == "session.created"

    assert output_cancelled.is_set()
    assert [frame.kind for frame in input_frames] == ["started", "cancelled"]
