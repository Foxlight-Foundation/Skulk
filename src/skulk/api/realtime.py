"""OpenAI-compatible WebSocket edge for realtime transcription providers."""

import base64
import binascii
from collections.abc import Awaitable, Callable
from typing import Annotated, Literal, cast, final
from urllib.parse import urlsplit
from uuid import uuid4

import anyio
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from skulk.extensions import (
    MAX_INLINE_MEDIA_BYTES,
    CapabilityStreamFrame,
    CapabilityStreamSession,
    InlineMediaAttachment,
)

_PCM_SAMPLE_RATE = 24_000
_MAX_SESSION_AUDIO_BYTES = 64 * 1024 * 1024
_MAX_PENDING_TRANSCRIPT_BYTES = 1024 * 1024
REALTIME_WEBSOCKET_MAX_MESSAGE_BYTES = 2 * MAX_INLINE_MEDIA_BYTES
"""Maximum encoded JSON event size accepted by the WebSocket edge."""


class _RealtimeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class InputAudioTranscriptionConfig(_RealtimeModel):
    """Transcription model selection accepted by the compatibility edge."""

    model: str = Field(
        min_length=1,
        max_length=512,
        description="Mounted realtime STT model ID.",
    )
    language: str | None = Field(
        default=None,
        description="Optional language hint; currently accepted only when null.",
    )
    prompt: str = Field(
        default="",
        description="Optional prompt hint; currently accepted only when empty.",
    )


class TranscriptionSessionConfig(_RealtimeModel):
    """Supported transcription-session configuration subset."""

    input_audio_format: Literal["pcm16"] = Field(
        default="pcm16",
        description="OpenAI PCM16 compatibility format: 24 kHz mono little-endian.",
    )
    input_audio_transcription: InputAudioTranscriptionConfig = Field(
        description="Mounted realtime STT model selection."
    )
    turn_detection: None = Field(
        default=None,
        description="Server VAD is not implemented in the initial edge.",
    )
    input_audio_noise_reduction: None = Field(
        default=None,
        description="API-side noise reduction is not implemented.",
    )
    include: tuple[()] = Field(
        default=(),
        description="Additional transcription fields are not implemented.",
    )


class TranscriptionSessionUpdate(_RealtimeModel):
    """Client request to confirm the effective transcription session."""

    type: Literal["transcription_session.update"]
    event_id: str | None = Field(default=None, max_length=256)
    session: TranscriptionSessionConfig


class InputAudioBufferAppend(_RealtimeModel):
    """One base64-encoded PCM16 audio frame from a compatibility client."""

    type: Literal["input_audio_buffer.append"]
    event_id: str | None = Field(default=None, max_length=256)
    audio: str = Field(min_length=1)


class InputAudioBufferCommit(_RealtimeModel):
    """Client half-close for the current single-utterance audio stream."""

    type: Literal["input_audio_buffer.commit"]
    event_id: str | None = Field(default=None, max_length=256)


class InputAudioBufferClear(_RealtimeModel):
    """Compatibility event rejected after audio reaches the provider."""

    type: Literal["input_audio_buffer.clear"]
    event_id: str | None = Field(default=None, max_length=256)


RealtimeClientEvent = Annotated[
    TranscriptionSessionUpdate
    | InputAudioBufferAppend
    | InputAudioBufferCommit
    | InputAudioBufferClear,
    Field(discriminator="type"),
]
_CLIENT_EVENT_ADAPTER: TypeAdapter[RealtimeClientEvent] = TypeAdapter(
    RealtimeClientEvent
)

OpenRealtimeSession = Callable[[str, int], Awaitable[CapabilityStreamSession]]


def websocket_origin_allowed(websocket: WebSocket) -> bool:
    """Return whether a browser Origin matches the WebSocket Host.

    Non-browser SDK clients normally omit ``Origin`` and remain allowed. Browser
    clients must be same-origin so future cookie or token authentication cannot
    silently introduce cross-site WebSocket hijacking.

    Args:
        websocket: Pending FastAPI WebSocket connection.

    Returns:
        True for absent origins or an HTTP(S) origin matching the Host header.
    """

    origin = websocket.headers.get("origin")
    if origin is None:
        return True
    parsed = urlsplit(origin)
    host = websocket.headers.get("host")
    return (
        parsed.scheme in ("http", "https")
        and bool(parsed.netloc)
        and host is not None
        and parsed.netloc.casefold() == host.casefold()
    )


@final
class RealtimeTranscriptionBridge:
    """Translate one WebSocket utterance onto the Fabric STT provider contract."""

    def __init__(
        self,
        *,
        websocket: WebSocket,
        model: str,
        open_session: OpenRealtimeSession,
    ) -> None:
        """Create a bridge for one pending WebSocket connection.

        Args:
            websocket: FastAPI WebSocket accepted and owned by this bridge.
            model: Mounted realtime STT model selected by the URL query.
            open_session: Fabric-provider session opener.
        """

        self._websocket = websocket
        self._model = model
        self._open_session = open_session
        self._session_id = f"sess_{uuid4().hex}"
        self._item_id = f"item_{uuid4().hex}"
        self._send_lock = anyio.Lock()
        self._commit_announced = anyio.Event()
        self._audio_bytes = 0
        self._committed = False

    async def serve(self) -> None:
        """Accept, run, and clean up one transcription WebSocket session."""

        if not websocket_origin_allowed(self._websocket):
            await self._websocket.close(code=1008, reason="cross-origin WebSocket denied")
            return
        await self._websocket.accept()
        session = await self._open_session(self._model, _PCM_SAMPLE_RATE)
        if not session.open_result.ok:
            error = session.open_result.error
            await self._send_error(
                code=error.code if error is not None else "provider_error",
                message=(
                    error.message
                    if error is not None
                    else "realtime transcription provider rejected the session"
                ),
            )
            await self._close(
                self._provider_rejection_close_code(
                    error.code if error is not None else "provider_error"
                )
            )
            return
        if session.input is None:
            await self._send_error(
                code="invalid_result",
                message="realtime transcription provider returned no input stream",
            )
            await self._close(1011)
            return

        await self._send_json(
            {
                "event_id": self._event_id(),
                "type": "transcription_session.created",
                "session": self._session_payload(),
            }
        )
        try:
            async with anyio.create_task_group() as task_group:
                task_group.start_soon(
                    self._send_provider_output,
                    session,
                    task_group.cancel_scope,
                )
                try:
                    await self._receive_client_input(session)
                except WebSocketDisconnect:
                    pass
                finally:
                    task_group.cancel_scope.cancel()
        finally:
            if not session.input.closed:
                with anyio.CancelScope(shield=True):
                    with anyio.move_on_after(1.0):
                        await session.input.cancel("WebSocket client disconnected")

    async def _receive_client_input(self, session: CapabilityStreamSession) -> None:
        assert session.input is not None
        while True:
            message = await self._websocket.receive()
            if message["type"] == "websocket.disconnect":
                raw_code = cast(object, message.get("code"))
                raw_reason = cast(object, message.get("reason"))
                raise WebSocketDisconnect(
                    code=raw_code if isinstance(raw_code, int) else 1000,
                    reason=raw_reason if isinstance(raw_reason, str) else "",
                )
            text = message.get("text")
            if not isinstance(text, str):
                await self._send_error(
                    code="unsupported_frame",
                    message="realtime compatibility events must use JSON text frames",
                )
                await self._close(1003)
                return
            if len(text.encode("utf-8")) > REALTIME_WEBSOCKET_MAX_MESSAGE_BYTES:
                await self._send_error(
                    code="event_too_large",
                    message=(
                        "realtime client event exceeds the bounded WebSocket "
                        f"limit of {REALTIME_WEBSOCKET_MAX_MESSAGE_BYTES} bytes"
                    ),
                )
                await self._close(1009)
                return
            try:
                event = _CLIENT_EVENT_ADAPTER.validate_json(text)
            except ValidationError as exc:
                await self._send_error(
                    code="invalid_event",
                    message=f"invalid realtime client event: {exc.errors()[0]['msg']}",
                )
                await self._close(1008)
                return

            if isinstance(event, TranscriptionSessionUpdate):
                if not self._session_update_matches(event.session):
                    await self._send_error(
                        code="unsupported_session_update",
                        message=(
                            "Skulk realtime transcription currently requires the URL "
                            "model, pcm16, 24 kHz mono audio, no VAD/noise reduction, "
                            "and no additional fields"
                        ),
                        client_event_id=event.event_id,
                    )
                    await self._close(1008)
                    return
                await self._send_json(
                    {
                        "event_id": self._event_id(),
                        "type": "transcription_session.updated",
                        "session": self._session_payload(),
                    }
                )
                continue

            if isinstance(event, InputAudioBufferClear):
                await self._send_error(
                    code="unsupported_event",
                    message=(
                        "input_audio_buffer.clear is unavailable because Skulk "
                        "forwards audio incrementally without retaining a replay buffer"
                    ),
                    client_event_id=event.event_id,
                )
                await self._close(1008)
                return

            if isinstance(event, InputAudioBufferAppend):
                if self._committed:
                    await self._send_error(
                        code="input_closed",
                        message="audio cannot be appended after input commit",
                        client_event_id=event.event_id,
                    )
                    await self._close(1008)
                    return
                try:
                    audio = base64.b64decode(event.audio, validate=True)
                except (binascii.Error, ValueError):
                    await self._send_error(
                        code="invalid_audio",
                        message="audio must be valid base64-encoded pcm16 bytes",
                        client_event_id=event.event_id,
                    )
                    await self._close(1008)
                    return
                if not audio or len(audio) % 2 != 0:
                    await self._send_error(
                        code="invalid_audio",
                        message="pcm16 audio frames must contain a positive even byte count",
                        client_event_id=event.event_id,
                    )
                    await self._close(1008)
                    return
                if len(audio) > MAX_INLINE_MEDIA_BYTES:
                    await self._send_error(
                        code="audio_frame_too_large",
                        message=f"audio frame exceeds {MAX_INLINE_MEDIA_BYTES} bytes",
                        client_event_id=event.event_id,
                    )
                    await self._close(1009)
                    return
                self._audio_bytes += len(audio)
                if self._audio_bytes > _MAX_SESSION_AUDIO_BYTES:
                    await self._send_error(
                        code="audio_session_too_large",
                        message=(
                            "session audio exceeds the bounded compatibility-edge "
                            f"limit of {_MAX_SESSION_AUDIO_BYTES} bytes"
                        ),
                        client_event_id=event.event_id,
                    )
                    await self._close(1009)
                    return
                try:
                    await session.input.send_chunk(
                        payload={
                            "format": "pcm_s16le",
                            "sample_rate": _PCM_SAMPLE_RATE,
                            "channels": 1,
                        },
                        media=InlineMediaAttachment(
                            data=audio,
                            media_type="audio/pcm",
                            codec="pcm_s16le",
                            sample_rate=_PCM_SAMPLE_RATE,
                            channels=1,
                        ),
                    )
                except (
                    TimeoutError,
                    anyio.BrokenResourceError,
                    anyio.ClosedResourceError,
                    RuntimeError,
                ) as exc:
                    await self._send_error(
                        code="input_transport_error",
                        message=f"provider input transport rejected audio: {exc}",
                        client_event_id=event.event_id,
                    )
                    await self._close(1011)
                    return
                continue

            assert isinstance(event, InputAudioBufferCommit)
            if self._committed:
                await self._send_error(
                    code="input_closed",
                    message="input_audio_buffer.commit may be sent only once",
                    client_event_id=event.event_id,
                )
                await self._close(1008)
                return
            if self._audio_bytes == 0:
                await self._send_error(
                    code="empty_audio_buffer",
                    message="cannot commit an empty audio buffer",
                    client_event_id=event.event_id,
                )
                continue
            try:
                await session.input.complete()
            except (
                TimeoutError,
                anyio.BrokenResourceError,
                anyio.ClosedResourceError,
                RuntimeError,
            ) as exc:
                await self._send_error(
                    code="input_transport_error",
                    message=f"provider input transport could not commit audio: {exc}",
                    client_event_id=event.event_id,
                )
                await self._close(1011)
                return
            self._committed = True
            await self._send_json(
                {
                    "event_id": self._event_id(),
                    "type": "input_audio_buffer.committed",
                    "previous_item_id": None,
                    "item_id": self._item_id,
                }
            )
            self._commit_announced.set()

    async def _send_provider_output(
        self,
        session: CapabilityStreamSession,
        cancel_scope: anyio.CancelScope,
    ) -> None:
        terminal = False
        pending_deltas: list[str] = []
        pending_delta_bytes = 0
        try:
            async for frame in session.frames:
                if frame.kind == "started":
                    continue
                if frame.kind == "chunk":
                    text = self._frame_text(frame)
                    if not self._commit_announced.is_set():
                        pending_delta_bytes += len(text.encode("utf-8"))
                        if pending_delta_bytes > _MAX_PENDING_TRANSCRIPT_BYTES:
                            await self._send_transcription_failed(
                                code="provider_output_too_large",
                                message=(
                                    "provider produced more than "
                                    f"{_MAX_PENDING_TRANSCRIPT_BYTES} bytes of "
                                    "transcript before input commit"
                                ),
                            )
                            await self._close(1011)
                            cancel_scope.cancel()
                            return
                        pending_deltas.append(text)
                        continue
                    await self._send_transcript_deltas(pending_deltas)
                    pending_deltas.clear()
                    await self._send_json(
                        {
                            "event_id": self._event_id(),
                            "type": (
                                "conversation.item.input_audio_transcription.delta"
                            ),
                            "item_id": self._item_id,
                            "content_index": 0,
                            "delta": text,
                        }
                    )
                    continue
                terminal = True
                if frame.kind == "completed":
                    await self._commit_announced.wait()
                    await self._send_transcript_deltas(pending_deltas)
                    await self._send_json(
                        {
                            "event_id": self._event_id(),
                            "type": (
                                "conversation.item.input_audio_transcription.completed"
                            ),
                            "item_id": self._item_id,
                            "content_index": 0,
                            "transcript": self._frame_text(frame),
                        }
                    )
                    await self._close(1000)
                else:
                    error = frame.error
                    await self._send_transcription_failed(
                        code=(
                            error.code
                            if error is not None
                            else ("cancelled" if frame.kind == "cancelled" else "provider_error")
                        ),
                        message=(
                            error.message
                            if error is not None
                            else "realtime transcription ended without a result"
                        ),
                    )
                    await self._close(1011 if frame.kind == "failed" else 1000)
                cancel_scope.cancel()
                return
        except (WebSocketDisconnect, RuntimeError):
            cancel_scope.cancel()
            return
        if not terminal:
            await self._send_transcription_failed(
                code="provider_error",
                message="realtime transcription stream ended without a terminal frame",
            )
            await self._close(1011)
            cancel_scope.cancel()

    async def _send_transcript_deltas(self, deltas: list[str]) -> None:
        """Send buffered provider deltas in their original order after commit."""

        for text in deltas:
            await self._send_json(
                {
                    "event_id": self._event_id(),
                    "type": "conversation.item.input_audio_transcription.delta",
                    "item_id": self._item_id,
                    "content_index": 0,
                    "delta": text,
                }
            )

    def _session_update_matches(self, update: TranscriptionSessionConfig) -> bool:
        return (
            update.input_audio_format == "pcm16"
            and update.input_audio_transcription.model == self._model
            and update.input_audio_transcription.language is None
            and update.input_audio_transcription.prompt == ""
            and update.turn_detection is None
            and update.input_audio_noise_reduction is None
            and not update.include
        )

    def _session_payload(self) -> dict[str, object]:
        return {
            "id": self._session_id,
            "object": "realtime.transcription_session",
            "input_audio_format": "pcm16",
            "input_audio_transcription": {
                "model": self._model,
                "language": None,
                "prompt": "",
            },
            "turn_detection": None,
            "input_audio_noise_reduction": None,
            "include": [],
        }

    @staticmethod
    def _frame_text(frame: CapabilityStreamFrame) -> str:
        payload = frame.payload or {}
        text = payload.get("text")
        return text if isinstance(text, str) else ""

    async def _send_transcription_failed(self, *, code: str, message: str) -> None:
        await self._send_json(
            {
                "event_id": self._event_id(),
                "type": "conversation.item.input_audio_transcription.failed",
                "item_id": self._item_id,
                "content_index": 0,
                "error": {
                    "type": "transcription_error",
                    "code": code,
                    "message": message,
                    "param": None,
                },
            }
        )

    async def _send_error(
        self,
        *,
        code: str,
        message: str,
        client_event_id: str | None = None,
    ) -> None:
        await self._send_json(
            {
                "event_id": self._event_id(),
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "code": code,
                    "message": message,
                    "param": None,
                    "event_id": client_event_id,
                },
            }
        )

    async def _send_json(self, payload: dict[str, object]) -> None:
        async with self._send_lock:
            await self._websocket.send_json(payload)

    async def _close(self, code: int) -> None:
        try:
            await self._websocket.close(code=code)
        except RuntimeError:
            return

    @staticmethod
    def _provider_rejection_close_code(code: str) -> int:
        if code in ("not_found", "overloaded", "timeout", "unreachable"):
            return 1013
        if code in (
            "invalid_payload",
            "payload_too_large",
            "revision_mismatch",
            "version_mismatch",
        ):
            return 1008
        return 1011

    @staticmethod
    def _event_id() -> str:
        return f"event_{uuid4().hex}"
