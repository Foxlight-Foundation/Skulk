"""OpenAI-compatible WebSocket edge for realtime transcription providers."""

import base64
import binascii
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from typing import Annotated, Literal, Self, cast, final
from urllib.parse import urlsplit
from uuid import uuid4

import anyio
from anyio.abc import TaskGroup
from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from skulk.extensions import (
    MAX_INLINE_MEDIA_BYTES,
    CapabilityStreamFrame,
    CapabilityStreamSession,
    InlineMediaAttachment,
    StreamingPcm16Resampler,
    VadConfig,
    VoiceActivityDetector,
)

_PCM_SAMPLE_RATE = 24_000
_VAD_SOURCE_FRAME_BYTES = _PCM_SAMPLE_RATE * 20 // 1000 * 2
_MAX_SESSION_AUDIO_BYTES = 64 * 1024 * 1024
_MAX_TRANSCRIPT_TEXT_BYTES = 1024 * 1024
_MAX_CONVERSATION_TEXT_BYTES = 4 * 1024 * 1024
_MAX_RESPONSE_AUDIO_BYTES = 64 * 1024 * 1024
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


class ServerVadConfig(_RealtimeModel):
    """Server-owned WebRTC voice activity detection settings."""

    type: Literal["server_vad"] = Field(
        description="Select server-owned voice activity detection."
    )
    aggressiveness: int = Field(
        default=2,
        ge=0,
        le=3,
        description="WebRTC VAD aggressiveness from least to most restrictive.",
    )
    prefix_padding_ms: int = Field(
        default=200,
        ge=0,
        le=2000,
        description=(
            "Speech preroll duration subtracted from the reported audio_start_ms "
            "boundary timestamp."
        ),
    )
    silence_duration_ms: int = Field(
        default=400,
        ge=20,
        le=5000,
        description="Trailing silence required to end the current utterance.",
    )
    minimum_speech_ms: int = Field(
        default=120,
        ge=20,
        le=5000,
        description="Continuous speech required before announcing a turn start.",
    )
    maximum_utterance_ms: int = Field(
        default=30000,
        ge=100,
        le=120000,
        description="Hard duration limit that ends a continuous utterance.",
    )

    @model_validator(mode="after")
    def validate_turn_durations(self) -> Self:
        """Reject thresholds that cannot complete within the utterance bound."""

        if self.minimum_speech_ms > self.maximum_utterance_ms:
            raise ValueError("minimum_speech_ms must not exceed maximum_utterance_ms")
        if self.silence_duration_ms > self.maximum_utterance_ms:
            raise ValueError("silence_duration_ms must not exceed maximum_utterance_ms")
        return self


class TranscriptionSessionConfig(_RealtimeModel):
    """Supported transcription-session configuration subset."""

    input_audio_format: Literal["pcm16"] = Field(
        default="pcm16",
        description="OpenAI PCM16 compatibility format: 24 kHz mono little-endian.",
    )
    input_audio_transcription: InputAudioTranscriptionConfig = Field(
        description="Mounted realtime STT model selection."
    )
    turn_detection: ServerVadConfig | None = Field(
        default=None,
        description="Optional server-owned WebRTC voice activity detection.",
    )
    input_audio_noise_reduction: None = Field(
        default=None,
        description="API-side noise reduction is not implemented.",
    )
    include: tuple[()] = Field(
        default=(),
        description="Additional transcription fields are not implemented.",
    )
    response: "RealtimeResponseConfig | None" = Field(
        default=None,
        description="Optional mounted chat and TTS participants for automatic replies.",
    )


class RealtimeResponseConfig(_RealtimeModel):
    """Mounted participants used to answer completed realtime transcripts."""

    model: str = Field(min_length=1, max_length=512)
    tts_model: str | None = Field(default=None, min_length=1, max_length=512)
    voice: str | None = Field(default=None, min_length=1, max_length=256)


class TranscriptionSessionUpdate(_RealtimeModel):
    """Legacy client request to confirm the effective transcription session."""

    type: Literal["transcription_session.update"]
    event_id: str | None = Field(default=None, max_length=256)
    session: TranscriptionSessionConfig


class RealtimePcmAudioFormat(_RealtimeModel):
    """Current Realtime API PCM input-format declaration."""

    type: Literal["audio/pcm"]
    rate: Literal[24000]


class RealtimeInputTranscriptionConfig(_RealtimeModel):
    """Current Realtime API transcription selection accepted by Skulk."""

    model: str = Field(min_length=1, max_length=512)
    language: str | None = Field(default=None, max_length=64)
    delay: Literal["medium"] | None = None


class RealtimeAudioInputConfig(_RealtimeModel):
    """Current Realtime API input-audio configuration subset."""

    format: RealtimePcmAudioFormat | None = None
    transcription: RealtimeInputTranscriptionConfig
    turn_detection: ServerVadConfig | None = None
    noise_reduction: None = None


class RealtimeAudioConfig(_RealtimeModel):
    """Current Realtime API audio configuration subset."""

    input: RealtimeAudioInputConfig


class RealtimeTranscriptionSessionConfig(_RealtimeModel):
    """Current Realtime transcription-session configuration subset."""

    type: Literal["transcription"]
    audio: RealtimeAudioConfig
    include: tuple[()] = ()
    response: RealtimeResponseConfig | None = None


class RealtimeSessionUpdate(_RealtimeModel):
    """Current OpenAI Realtime transcription session update."""

    type: Literal["session.update"]
    event_id: str | None = Field(default=None, max_length=256)
    session: RealtimeTranscriptionSessionConfig


class InputAudioBufferAppend(_RealtimeModel):
    """One base64-encoded PCM16 audio frame from a compatibility client."""

    type: Literal["input_audio_buffer.append"]
    event_id: str | None = Field(default=None, max_length=256)
    audio: str = Field(min_length=1)


class InputAudioBufferCommit(_RealtimeModel):
    """Client half-close for the current utterance."""

    type: Literal["input_audio_buffer.commit"]
    event_id: str | None = Field(default=None, max_length=256)


class InputAudioBufferClear(_RealtimeModel):
    """Compatibility event rejected after audio reaches the provider."""

    type: Literal["input_audio_buffer.clear"]
    event_id: str | None = Field(default=None, max_length=256)


class ResponseCancel(_RealtimeModel):
    """Cancel the active assistant generation or speech stream."""

    type: Literal["response.cancel"]
    event_id: str | None = Field(default=None, max_length=256)


RealtimeClientEvent = Annotated[
    TranscriptionSessionUpdate
    | RealtimeSessionUpdate
    | InputAudioBufferAppend
    | InputAudioBufferCommit
    | InputAudioBufferClear
    | ResponseCancel,
    Field(discriminator="type"),
]
_CLIENT_EVENT_ADAPTER: TypeAdapter[RealtimeClientEvent] = TypeAdapter(
    RealtimeClientEvent
)

OpenRealtimeSession = Callable[[str, int], Awaitable[CapabilityStreamSession]]
ConversationMessage = tuple[Literal["user", "assistant"], str]
GenerateAssistant = Callable[
    [str, tuple[ConversationMessage, ...]], AsyncIterator[str]
]
OpenSpeechSession = Callable[
    [str, str, str | None], Awaitable[CapabilityStreamSession]
]
ValidateResponseConfig = Callable[[RealtimeResponseConfig], Awaitable[None]]


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
    """Translate serialized WebSocket utterances onto Fabric STT provider calls."""

    def __init__(
        self,
        *,
        websocket: WebSocket,
        model: str,
        open_session: OpenRealtimeSession,
        generate_assistant: GenerateAssistant | None = None,
        open_speech_session: OpenSpeechSession | None = None,
        validate_response_config: ValidateResponseConfig | None = None,
    ) -> None:
        """Create a bridge for one pending WebSocket connection.

        Args:
            websocket: FastAPI WebSocket accepted and owned by this bridge.
            model: Mounted realtime STT model selected by the URL query.
            open_session: Fabric-provider session opener.
            generate_assistant: Optional mounted text-model stream opener.
            open_speech_session: Optional mounted TTS provider stream opener.
            validate_response_config: Optional participant admission validator.
        """

        self._websocket = websocket
        self._model = model
        self._open_session = open_session
        self._generate_assistant = generate_assistant
        self._open_speech_session = open_speech_session
        self._validate_response_config = validate_response_config
        self._session_id = f"sess_{uuid4().hex}"
        self._item_id = f"item_{uuid4().hex}"
        self._previous_item_id: str | None = None
        self._send_lock = anyio.Lock()
        self._commit_announced = anyio.Event()
        self._session_audio_bytes = 0
        self._turn_audio_bytes = 0
        self._committed = False
        self._current_session: CapabilityStreamSession | None = None
        self._vad_auto_committed = False
        self._turn_detection: ServerVadConfig | None = None
        self._vad_detector: VoiceActivityDetector | None = None
        self._vad_resampler: StreamingPcm16Resampler | None = None
        self._vad_pcm = bytearray()
        self._response_config: RealtimeResponseConfig | None = None
        self._conversation: list[ConversationMessage] = []
        self._response_cancel_scope: anyio.CancelScope | None = None

    async def serve(self) -> None:
        """Accept, run, and clean up one transcription WebSocket session."""

        if not websocket_origin_allowed(self._websocket):
            await self._websocket.close(code=1008, reason="cross-origin WebSocket denied")
            return
        await self._websocket.accept()
        await self._send_json(
            {
                "event_id": self._event_id(),
                "type": "session.created",
                "session": self._session_payload(),
            }
        )
        try:
            async with anyio.create_task_group() as task_group:
                try:
                    await self._receive_client_input(task_group)
                except WebSocketDisconnect:
                    pass
                finally:
                    task_group.cancel_scope.cancel()
        finally:
            await self._cancel_current_session()

    async def _cancel_current_session(self) -> None:
        """Cancel the provider call still owned when the WebSocket exits."""

        current_session = self._current_session
        if (
            current_session is None
            or current_session.input is None
            or current_session.input.closed
        ):
            return
        with anyio.CancelScope(shield=True):
            with anyio.move_on_after(1.0):
                await current_session.input.cancel("WebSocket client disconnected")

    async def _receive_client_input(self, task_group: TaskGroup) -> None:
        """Receive client events while rotating one provider call per utterance."""

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

            if isinstance(event, (TranscriptionSessionUpdate, RealtimeSessionUpdate)):
                update_matches = (
                    self._session_update_matches(event.session)
                    if isinstance(event, TranscriptionSessionUpdate)
                    else self._realtime_session_update_matches(event.session)
                )
                if not update_matches:
                    await self._send_error(
                        code="unsupported_session_update",
                        message=(
                            "Skulk realtime transcription currently requires the URL "
                            "model, pcm16, 24 kHz mono audio, optional supported "
                            "server VAD, "
                            "no noise reduction, "
                            "and no additional fields"
                        ),
                        client_event_id=event.event_id,
                    )
                    await self._close(1008)
                    return
                requested_turn_detection = (
                    event.session.turn_detection
                    if isinstance(event, TranscriptionSessionUpdate)
                    else event.session.audio.input.turn_detection
                )
                requested_response = event.session.response
                if requested_response is not None and self._generate_assistant is None:
                    await self._send_error(
                        code="unsupported_session_update",
                        message="automatic realtime responses are unavailable",
                        client_event_id=event.event_id,
                    )
                    continue
                if (
                    requested_response is not None
                    and requested_response.tts_model is not None
                    and self._open_speech_session is None
                ):
                    await self._send_error(
                        code="unsupported_session_update",
                        message="realtime speech synthesis is unavailable",
                        client_event_id=event.event_id,
                    )
                    continue
                if (
                    requested_response is not None
                    and self._validate_response_config is not None
                ):
                    try:
                        await self._validate_response_config(requested_response)
                    except Exception as exc:
                        logger.opt(exception=exc).warning(
                            "Realtime response participant validation failed"
                        )
                        await self._send_error(
                            code="unsupported_session_update",
                            message=(
                                "selected realtime response participants are not "
                                "ready"
                            ),
                            client_event_id=event.event_id,
                        )
                        continue
                if (
                    self._turn_audio_bytes > 0
                    and requested_turn_detection != self._turn_detection
                ):
                    await self._send_error(
                        code="turn_detection_locked",
                        message="turn detection cannot change after audio is appended",
                        client_event_id=event.event_id,
                    )
                    continue
                if (
                    self._turn_audio_bytes > 0
                    and requested_response != self._response_config
                ):
                    await self._send_error(
                        code="response_config_locked",
                        message="response configuration cannot change after audio is appended",
                        client_event_id=event.event_id,
                    )
                    continue
                if self._turn_audio_bytes == 0:
                    self._configure_turn_detection(requested_turn_detection)
                    self._response_config = requested_response
                await self._send_json(
                    {
                        "event_id": self._event_id(),
                        "type": (
                            "transcription_session.updated"
                            if isinstance(event, TranscriptionSessionUpdate)
                            else "session.updated"
                        ),
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

            if isinstance(event, ResponseCancel):
                if self._response_cancel_scope is None:
                    await self._send_error(
                        code="response_not_active",
                        message="there is no active realtime response to cancel",
                        client_event_id=event.event_id,
                    )
                else:
                    self._cancel_assistant_response()
                continue

            if isinstance(event, InputAudioBufferAppend):
                if self._committed:
                    await self._send_error(
                        code="turn_in_progress",
                        message="audio cannot be appended until the committed turn completes",
                        client_event_id=event.event_id,
                    )
                    continue
                if (
                    self._response_cancel_scope is not None
                    and self._vad_detector is None
                ):
                    self._cancel_assistant_response()
                session = self._current_session
                if session is None:
                    session = await self._open_next_turn(task_group)
                    if session is None:
                        return
                assert session.input is not None
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
                self._session_audio_bytes += len(audio)
                self._turn_audio_bytes += len(audio)
                if self._session_audio_bytes > _MAX_SESSION_AUDIO_BYTES:
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
                segment_size = (
                    _VAD_SOURCE_FRAME_BYTES
                    if self._vad_detector is not None
                    else len(audio)
                )
                for offset in range(0, len(audio), segment_size):
                    segment = audio[offset : offset + segment_size]
                    try:
                        await session.input.send_chunk(
                            payload={
                                "format": "pcm_s16le",
                                "sample_rate": _PCM_SAMPLE_RATE,
                                "channels": 1,
                            },
                            media=InlineMediaAttachment(
                                data=segment,
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
                    if await self._process_vad_audio(
                        session,
                        segment,
                        client_event_id=event.event_id,
                    ):
                        if not self._committed:
                            return
                        break
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
            if self._turn_audio_bytes == 0:
                await self._send_error(
                    code="empty_audio_buffer",
                    message="cannot commit an empty audio buffer",
                    client_event_id=event.event_id,
                )
                continue
            session = self._current_session
            if session is None:
                await self._send_error(
                    code="provider_error",
                    message="the current transcription turn has no provider session",
                    client_event_id=event.event_id,
                )
                await self._close(1011)
                return
            await self._finish_vad_on_manual_commit()
            if not await self._commit_input(
                session,
                client_event_id=event.event_id,
            ):
                return

    async def _finish_vad_on_manual_commit(self) -> None:
        """Emit a terminal VAD boundary before a client-owned input commit."""

        detector = self._vad_detector
        if detector is None:
            return
        for event in detector.finish():
            await self._send_json(
                {
                    "event_id": self._event_id(),
                    "type": "input_audio_buffer.speech_stopped",
                    "audio_end_ms": event.timestamp_ms,
                    "item_id": self._item_id,
                }
            )

    async def _open_next_turn(
        self,
        task_group: TaskGroup,
    ) -> CapabilityStreamSession | None:
        """Open and start the provider call for the next utterance."""

        try:
            session = await self._open_session(self._model, _PCM_SAMPLE_RATE)
        except Exception as exc:
            logger.opt(exception=exc).warning(
                "Realtime transcription provider session failed to reopen"
            )
            await self._send_internal_error(
                code="provider_open_error",
                message="realtime transcription provider session could not be opened",
            )
            return None
        if not session.open_result.ok:
            error = session.open_result.error
            await self._send_error(
                code=error.code if error is not None else "provider_error",
                message=(
                    error.message
                    if error is not None
                    else "realtime transcription provider rejected the next turn"
                ),
            )
            await self._close(
                self._provider_rejection_close_code(
                    error.code if error is not None else "provider_error"
                )
            )
            return None
        if session.input is None:
            await self._send_error(
                code="invalid_result",
                message="realtime transcription provider returned no input stream",
            )
            await self._close(1011)
            return None
        self._current_session = session
        item_id = self._item_id
        commit_announced = self._commit_announced
        task_group.start_soon(
            self._send_provider_output,
            session,
            task_group,
            task_group.cancel_scope,
            item_id,
            commit_announced,
        )
        return session

    def _configure_turn_detection(
        self,
        config: ServerVadConfig | None,
    ) -> None:
        """Replace server VAD state before any audio is committed."""

        self._turn_detection = config
        self._vad_pcm.clear()
        if config is None:
            self._vad_detector = None
            self._vad_resampler = None
            return
        self._vad_detector = VoiceActivityDetector(
            VadConfig(
                sample_rate=16000,
                aggressiveness=config.aggressiveness,
                frame_ms=20,
                minimum_speech_ms=config.minimum_speech_ms,
                silence_hangover_ms=config.silence_duration_ms,
                preroll_ms=config.prefix_padding_ms,
                maximum_utterance_ms=config.maximum_utterance_ms,
            )
        )
        self._vad_resampler = StreamingPcm16Resampler(_PCM_SAMPLE_RATE, 16000)

    async def _process_vad_audio(
        self,
        session: CapabilityStreamSession,
        audio: bytes,
        *,
        client_event_id: str | None,
    ) -> bool:
        """Feed one 24 kHz chunk to server VAD and auto-commit on turn end."""

        detector = self._vad_detector
        resampler = self._vad_resampler
        if detector is None or resampler is None or self._committed:
            return False
        self._vad_pcm.extend(resampler.process(audio))
        while len(self._vad_pcm) >= detector.frame_bytes:
            frame_bytes = detector.frame_bytes
            frame = bytes(self._vad_pcm[:frame_bytes])
            del self._vad_pcm[:frame_bytes]
            for event in detector.process(frame):
                if event.kind == "speech_started":
                    self._cancel_assistant_response()
                    await self._send_json(
                        {
                            "event_id": self._event_id(),
                            "type": "input_audio_buffer.speech_started",
                            "audio_start_ms": event.timestamp_ms,
                            "item_id": self._item_id,
                        }
                    )
                    continue
                await self._send_json(
                    {
                        "event_id": self._event_id(),
                        "type": "input_audio_buffer.speech_stopped",
                        "audio_end_ms": event.timestamp_ms,
                        "item_id": self._item_id,
                    }
                )
                await self._commit_input(
                    session,
                    client_event_id=client_event_id,
                )
                self._vad_auto_committed = self._committed
                return True
        return False

    async def _commit_input(
        self,
        session: CapabilityStreamSession,
        *,
        client_event_id: str | None,
    ) -> bool:
        """Half-close provider input and publish one compatibility commit."""

        assert session.input is not None
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
                client_event_id=client_event_id,
            )
            await self._close(1011)
            return False
        self._committed = True
        await self._send_json(
            {
                "event_id": self._event_id(),
                "type": "input_audio_buffer.committed",
                "previous_item_id": self._previous_item_id,
                "item_id": self._item_id,
            }
        )
        self._commit_announced.set()
        return True

    async def _send_provider_output(
        self,
        session: CapabilityStreamSession,
        task_group: TaskGroup,
        cancel_scope: anyio.CancelScope,
        item_id: str,
        commit_announced: anyio.Event,
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
                    text_bytes = len(text.encode("utf-8"))
                    if text_bytes > _MAX_TRANSCRIPT_TEXT_BYTES:
                        await self._fail_oversized_transcript(cancel_scope)
                        return
                    if not commit_announced.is_set():
                        pending_delta_bytes += text_bytes
                        if pending_delta_bytes > _MAX_TRANSCRIPT_TEXT_BYTES:
                            await self._fail_oversized_transcript(cancel_scope)
                            return
                        pending_deltas.append(text)
                        continue
                    await self._send_transcript_deltas(pending_deltas, item_id=item_id)
                    pending_deltas.clear()
                    await self._send_json(
                        {
                            "event_id": self._event_id(),
                            "type": (
                                "conversation.item.input_audio_transcription.delta"
                            ),
                            "item_id": item_id,
                            "content_index": 0,
                            "delta": text,
                        }
                    )
                    continue
                terminal = True
                if frame.kind == "completed":
                    transcript = self._frame_text(frame)
                    if len(transcript.encode("utf-8")) > _MAX_TRANSCRIPT_TEXT_BYTES:
                        await self._fail_oversized_transcript(cancel_scope)
                        return
                    await commit_announced.wait()
                    await self._send_transcript_deltas(pending_deltas, item_id=item_id)
                    await self._send_json(
                        {
                            "event_id": self._event_id(),
                            "type": (
                                "conversation.item.input_audio_transcription.completed"
                            ),
                            "item_id": item_id,
                            "content_index": 0,
                            "transcript": transcript,
                        }
                    )
                    self._finish_turn(session, item_id)
                    response_config = self._response_config
                    if response_config is not None:
                        self._cancel_assistant_response()
                        task_group.start_soon(
                            self._run_assistant_response,
                            transcript,
                            item_id,
                            response_config,
                        )
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
                if frame.kind != "completed":
                    cancel_scope.cancel()
                return
        except WebSocketDisconnect:
            cancel_scope.cancel()
            return
        except Exception as exc:
            logger.opt(exception=exc).warning(
                "Realtime transcription provider output stream failed"
            )
            await self._send_internal_transcription_failure(cancel_scope)
            return
        if not terminal:
            await self._send_transcription_failed(
                code="provider_error",
                message="realtime transcription stream ended without a terminal frame",
            )
            await self._close(1011)
            cancel_scope.cancel()

    def _cancel_assistant_response(self) -> None:
        """Cancel active model/TTS work when a new VAD turn starts."""

        if self._response_cancel_scope is not None:
            self._response_cancel_scope.cancel()

    async def _run_assistant_response(
        self,
        transcript: str,
        input_item_id: str,
        config: RealtimeResponseConfig,
    ) -> None:
        """Generate visible assistant text and optional streamed speech audio."""

        generate = self._generate_assistant
        if generate is None:
            return
        response_id = f"resp_{uuid4().hex}"
        output_item_id = f"item_{uuid4().hex}"
        user_message: ConversationMessage = ("user", transcript)
        messages = (*self._conversation, user_message)
        cancel_scope = anyio.CancelScope()
        self._response_cancel_scope = cancel_scope
        try:
            await self._send_json(
                {
                    "event_id": self._event_id(),
                    "type": "response.created",
                    "response": {"id": response_id, "status": "in_progress"},
                }
            )
        except BaseException:
            if self._response_cancel_scope is cancel_scope:
                self._response_cancel_scope = None
            raise
        completed = False
        failed = False
        text_limit_exceeded = False
        text_parts: list[str] = []
        text_bytes = 0
        try:
            with cancel_scope:
                async for delta in generate(config.model, messages):
                    text_bytes += len(delta.encode("utf-8"))
                    if text_bytes > _MAX_TRANSCRIPT_TEXT_BYTES:
                        text_limit_exceeded = True
                        cancel_scope.cancel()
                        await anyio.sleep(0)
                    text_parts.append(delta)
                    await self._send_json(
                        {
                            "event_id": self._event_id(),
                            "type": "response.output_text.delta",
                            "response_id": response_id,
                            "item_id": output_item_id,
                            "output_index": 0,
                            "content_index": 0,
                            "delta": delta,
                        }
                    )
                text = "".join(text_parts)
                if not text:
                    raise RuntimeError("assistant model produced no visible text")
                await self._send_json(
                    {
                        "event_id": self._event_id(),
                        "type": "response.output_text.done",
                        "response_id": response_id,
                        "item_id": output_item_id,
                        "output_index": 0,
                        "content_index": 0,
                        "text": text,
                    }
                )
                self._append_conversation(transcript, text)
                if config.tts_model is not None:
                    await self._stream_assistant_audio(
                        response_id=response_id,
                        item_id=output_item_id,
                        model=config.tts_model,
                        text=text,
                        voice=config.voice,
                    )
                completed = True
            if text_limit_exceeded:
                raise RuntimeError("assistant response exceeds the bounded text limit")
        except WebSocketDisconnect:
            cancel_scope.cancel()
            return
        except anyio.get_cancelled_exc_class():
            if not cancel_scope.cancel_called:
                raise
        except Exception as exc:
            logger.opt(exception=exc).warning("Realtime assistant response failed")
            failed = True
            await self._send_error(
                code="response_error",
                message="realtime assistant response failed",
            )
        finally:
            if self._response_cancel_scope is cancel_scope:
                self._response_cancel_scope = None
        await self._send_json(
            {
                "event_id": self._event_id(),
                "type": "response.done",
                "response": {
                    "id": response_id,
                    "status": (
                        "completed"
                        if completed
                        else ("failed" if failed else "cancelled")
                    ),
                    "input_item_id": input_item_id,
                    "output": (
                        [{"id": output_item_id, "text": "".join(text_parts)}]
                        if text_parts
                        else []
                    ),
                },
            }
        )

    def _append_conversation(self, transcript: str, response: str) -> None:
        """Append one text turn while evicting oldest complete turns by bytes."""

        self._conversation.extend((("user", transcript), ("assistant", response)))
        conversation_bytes = sum(
            len(content.encode("utf-8")) for _, content in self._conversation
        )
        while conversation_bytes > _MAX_CONVERSATION_TEXT_BYTES:
            if len(self._conversation) <= 2:
                raise RuntimeError("conversation turn exceeds the bounded text limit")
            removed = self._conversation[:2]
            del self._conversation[:2]
            conversation_bytes -= sum(
                len(content.encode("utf-8")) for _, content in removed
            )

    async def _stream_assistant_audio(
        self,
        *,
        response_id: str,
        item_id: str,
        model: str,
        text: str,
        voice: str | None,
    ) -> None:
        """Forward one mounted TTS provider stream as realtime audio events."""

        open_speech = self._open_speech_session
        if open_speech is None:
            raise RuntimeError("realtime speech synthesis is unavailable")
        session = await open_speech(model, text, voice)
        if not session.open_result.ok:
            error = session.open_result.error
            raise RuntimeError(
                error.message if error is not None else "TTS provider rejected response"
            )
        audio_bytes = 0
        terminal = False
        frames = session.frames
        try:
            async for frame in frames:
                if frame.kind == "started":
                    continue
                if frame.kind == "completed":
                    terminal = True
                    break
                if (
                    frame.kind != "chunk"
                    or not isinstance(frame.media, InlineMediaAttachment)
                ):
                    error = frame.error
                    raise RuntimeError(
                        error.message if error is not None else "TTS stream failed"
                    )
                audio_bytes += len(frame.media.data)
                if audio_bytes > _MAX_RESPONSE_AUDIO_BYTES:
                    raise RuntimeError("TTS response exceeds the bounded audio limit")
                payload = frame.payload or {}
                await self._send_json(
                    {
                        "event_id": self._event_id(),
                        "type": "response.audio.delta",
                        "response_id": response_id,
                        "item_id": item_id,
                        "output_index": 0,
                        "content_index": 1,
                        "delta": base64.b64encode(frame.media.data).decode("ascii"),
                        "format": payload.get("format", "mp3"),
                    }
                )
        finally:
            if isinstance(frames, AsyncGenerator):
                with anyio.CancelScope(shield=True):
                    await frames.aclose()
        if not terminal:
            raise RuntimeError("TTS provider ended without a terminal frame")
        if audio_bytes == 0:
            raise RuntimeError("TTS provider produced no audio")
        await self._send_json(
            {
                "event_id": self._event_id(),
                "type": "response.audio.done",
                "response_id": response_id,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 1,
            }
        )

    def _finish_turn(
        self,
        session: CapabilityStreamSession,
        item_id: str,
    ) -> None:
        """Rotate mutable turn state after one provider call completes."""

        if self._current_session is not session:
            return
        self._previous_item_id = item_id
        self._item_id = f"item_{uuid4().hex}"
        self._current_session = None
        self._turn_audio_bytes = 0
        self._committed = False
        self._vad_auto_committed = False
        self._commit_announced = anyio.Event()
        self._configure_turn_detection(self._turn_detection)

    async def _send_transcript_deltas(
        self,
        deltas: list[str],
        *,
        item_id: str,
    ) -> None:
        """Send buffered provider deltas in their original order after commit."""

        for text in deltas:
            await self._send_json(
                {
                    "event_id": self._event_id(),
                    "type": "conversation.item.input_audio_transcription.delta",
                    "item_id": item_id,
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
            and update.input_audio_noise_reduction is None
            and not update.include
        )

    def _realtime_session_update_matches(
        self,
        update: RealtimeTranscriptionSessionConfig,
    ) -> bool:
        input_config = update.audio.input
        return (
            update.type == "transcription"
            and input_config.transcription.model == self._model
            and input_config.transcription.language is None
            and input_config.transcription.delay in (None, "medium")
            and (
                input_config.format is None
                or (
                    input_config.format.type == "audio/pcm"
                    and input_config.format.rate == _PCM_SAMPLE_RATE
                )
            )
            and input_config.noise_reduction is None
            and not update.include
        )

    def _session_payload(self) -> dict[str, object]:
        return {
            "id": self._session_id,
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": _PCM_SAMPLE_RATE},
                    "transcription": {
                        "model": self._model,
                        "language": None,
                        "delay": "medium",
                    },
                    "turn_detection": (
                        None
                        if self._turn_detection is None
                        else self._turn_detection.model_dump(mode="json")
                    ),
                    "noise_reduction": None,
                },
            },
            "response": (
                None
                if self._response_config is None
                else self._response_config.model_dump(mode="json")
            ),
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

    async def _send_internal_error(self, *, code: str, message: str) -> None:
        try:
            await self._send_error(code=code, message=message)
            await self._close(1011)
        except (
            WebSocketDisconnect,
            RuntimeError,
            anyio.BrokenResourceError,
            anyio.ClosedResourceError,
        ):
            return

    async def _send_internal_transcription_failure(
        self,
        cancel_scope: anyio.CancelScope,
    ) -> None:
        try:
            await self._send_transcription_failed(
                code="provider_output_error",
                message="realtime transcription provider output failed unexpectedly",
            )
            await self._close(1011)
        except (
            WebSocketDisconnect,
            RuntimeError,
            anyio.BrokenResourceError,
            anyio.ClosedResourceError,
        ):
            pass
        finally:
            cancel_scope.cancel()

    async def _fail_oversized_transcript(
        self,
        cancel_scope: anyio.CancelScope,
    ) -> None:
        await self._send_transcription_failed(
            code="provider_output_too_large",
            message=(
                "provider transcript exceeds the bounded text limit of "
                f"{_MAX_TRANSCRIPT_TEXT_BYTES} bytes"
            ),
        )
        await self._close(1011)
        cancel_scope.cancel()

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
