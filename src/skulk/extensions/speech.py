"""First-party provider facades for mounted Skulk speech models."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import final

from skulk.extensions.calls import CapabilityCall, CapabilityError
from skulk.extensions.capabilities import CapabilityDescriptor
from skulk.extensions.streams import CapabilityStreamFrame
from skulk.extensions.types import ExtensionContext

TTS_CAPABILITY_DESCRIPTOR = CapabilityDescriptor(
    id="tts",
    version="1.0.0",
    title="Text to speech",
    description=(
        "Generate progressive encoded speech audio with a mounted Skulk "
        "text-to-speech model."
    ),
    input_schema={
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "model": {"type": "string", "minLength": 1},
            "text": {"type": "string", "minLength": 1},
            "voice": {"type": "string", "minLength": 1},
            "response_format": {"const": "mp3"},
            "streaming_interval": {"type": "number", "exclusiveMinimum": 0},
            "speed": {"type": "number", "exclusiveMinimum": 0},
            "instruct": {"type": "string"},
            "lang_code": {"type": "string"},
            "temperature": {"type": "number"},
            "top_p": {"type": "number"},
            "top_k": {"type": "integer", "minimum": 0},
            "repetition_penalty": {
                "type": "number",
                "exclusiveMinimum": 0,
            },
            "max_tokens": {"type": "integer", "minimum": 1},
        },
        "required": ["model", "text"],
        "additionalProperties": False,
    },
    io_mode="server_streaming",
    output_chunk_schema={
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "model": {"type": "string"},
            "format": {"const": "mp3"},
            "chunk_index": {"type": "integer", "minimum": 0},
            "sample_rate": {"type": "integer", "minimum": 1},
            "is_partial": {"type": "boolean"},
        },
        "required": ["model", "format", "chunk_index", "is_partial"],
        "additionalProperties": False,
    },
    annotations={
        "modality": "audio",
        "latency": "interactive",
        "runtime": "mlx_audio",
        "stability": "experimental",
    },
)
"""Generic provider descriptor for Skulk's mounted-model TTS facade."""

STT_CAPABILITY_DESCRIPTOR = CapabilityDescriptor(
    id="stt",
    version="1.0.0",
    title="Speech to text",
    description=(
        "Transcribe a bounded encoded audio clip with a mounted Skulk "
        "speech-to-text model. Audio is supplied as ordered binary input "
        "frames and inference begins after input half-close."
    ),
    input_schema={
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "model": {"type": "string", "minLength": 1},
            "filename": {"type": "string", "minLength": 1},
            "content_type": {"type": "string", "minLength": 1},
            "language": {"type": "string", "minLength": 1},
            "prompt": {"type": "string"},
            "temperature": {"type": "number"},
            "max_tokens": {"type": "integer", "minimum": 1},
            "chunk_duration": {"type": "number", "exclusiveMinimum": 0},
            "frame_threshold": {"type": "integer", "minimum": 1},
            "context": {"type": "string"},
            "prefill_step_size": {"type": "integer", "minimum": 1},
            "text": {"type": "string"},
            "word_timestamps": {"type": "boolean"},
            "timestamp_granularities": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
            },
        },
        "required": ["model"],
        "additionalProperties": False,
    },
    output_schema={
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "model": {"type": "string"},
            "text": {"type": "string"},
            "language": {"type": "string"},
            "segments": {"type": "array", "items": {"type": "object"}},
        },
        "required": ["model", "text"],
        "additionalProperties": False,
    },
    io_mode="client_streaming",
    input_chunk_schema={
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
    },
    annotations={
        "modality": "audio",
        "latency": "batch",
        "runtime": "mlx_audio",
        "stability": "beta",
        "input_media": "inline",
        "max_input_bytes": str(25 * 1024 * 1024),
    },
)
"""Provider descriptor for bounded, non-realtime mounted-model STT."""

REALTIME_STT_CAPABILITY_DESCRIPTOR = CapabilityDescriptor(
    id="stt.realtime",
    version="1.0.0",
    title="Realtime speech to text",
    description=(
        "Transcribe ordered mono PCM16 audio frames with a mounted Skulk "
        "model that exposes a true incremental streaming session."
    ),
    input_schema={
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "model": {"type": "string", "minLength": 1},
            "sample_rate": {"type": "integer", "minimum": 8000, "maximum": 96000},
            "temperature": {"type": "number"},
            "transcription_delay_ms": {
                "type": "integer",
                "minimum": 80,
                "maximum": 2400,
                "multipleOf": 80,
            },
        },
        "required": ["model", "sample_rate"],
        "additionalProperties": False,
    },
    output_schema={
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "model": {"type": "string"},
            "text": {"type": "string"},
            "is_partial": {"const": False},
        },
        "required": ["model", "text", "is_partial"],
        "additionalProperties": False,
    },
    io_mode="bidirectional",
    input_chunk_schema={
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "format": {"const": "pcm_s16le"},
            "sample_rate": {"type": "integer", "minimum": 8000, "maximum": 96000},
            "channels": {"const": 1},
        },
        "required": ["format", "sample_rate", "channels"],
        "additionalProperties": False,
    },
    output_chunk_schema={
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "model": {"type": "string"},
            "text": {"type": "string"},
            "is_partial": {"const": True},
        },
        "required": ["model", "text", "is_partial"],
        "additionalProperties": False,
    },
    annotations={
        "modality": "audio",
        "latency": "realtime",
        "runtime": "mlx_audio",
        "stability": "stable",
        "input_codec": "pcm_s16le",
    },
)
"""Provider descriptor for true incremental mounted-model STT sessions."""

TtsAdmission = Callable[[CapabilityCall], Awaitable[CapabilityError | None]]
TtsStream = Callable[[CapabilityCall], AsyncIterator[CapabilityStreamFrame]]
RealtimeSttAdmission = Callable[[CapabilityCall], Awaitable[CapabilityError | None]]
RealtimeSttStream = Callable[
    [CapabilityCall, AsyncIterator[CapabilityStreamFrame]],
    AsyncIterator[CapabilityStreamFrame],
]
BatchSttAdmission = Callable[[CapabilityCall], Awaitable[CapabilityError | None]]
BatchSttStream = Callable[
    [CapabilityCall, AsyncIterator[CapabilityStreamFrame]],
    AsyncIterator[CapabilityStreamFrame],
]


@final
class BuiltinSpeechProvider:
    """Expose core speech serving through the generic provider contract."""

    name = "skulk-builtin-speech"
    skulk_requires = ">=0"

    def __init__(
        self,
        *,
        admit_tts: TtsAdmission,
        stream_tts: TtsStream,
        admit_stt: BatchSttAdmission,
        stream_stt: BatchSttStream,
        admit_realtime_stt: RealtimeSttAdmission,
        stream_realtime_stt: RealtimeSttStream,
    ) -> None:
        """Create the facade around API-owned admission and stream adapters."""

        self._admit_tts = admit_tts
        self._stream_tts = stream_tts
        self._admit_stt = admit_stt
        self._stream_stt = stream_stt
        self._admit_realtime_stt = admit_realtime_stt
        self._stream_realtime_stt = stream_realtime_stt

    def chat_middleware(self) -> None:
        """Return no chat middleware; this provider is a fabric service."""

        return None

    def capabilities(self) -> Sequence[CapabilityDescriptor]:
        """Describe the first-party TTS, batch STT, and realtime STT services."""

        return (
            TTS_CAPABILITY_DESCRIPTOR,
            STT_CAPABILITY_DESCRIPTOR,
            REALTIME_STT_CAPABILITY_DESCRIPTOR,
        )

    def on_start(self, context: ExtensionContext) -> None:
        """Keep discovery withdrawn until an eligible model is mounted."""

        context.withdraw_capability(TTS_CAPABILITY_DESCRIPTOR.id)
        context.withdraw_capability(STT_CAPABILITY_DESCRIPTOR.id)
        context.withdraw_capability(REALTIME_STT_CAPABILITY_DESCRIPTOR.id)

    async def admit_stream(
        self,
        context: ExtensionContext,
        call: CapabilityCall,
    ) -> CapabilityError | None:
        """Validate dynamic mounted-model requirements before ``started``."""

        del context
        if call.capability_id == TTS_CAPABILITY_DESCRIPTOR.id:
            return await self._admit_tts(call)
        if call.capability_id == STT_CAPABILITY_DESCRIPTOR.id:
            return await self._admit_stt(call)
        if call.capability_id == REALTIME_STT_CAPABILITY_DESCRIPTOR.id:
            return await self._admit_realtime_stt(call)
        return CapabilityError(
            code="not_found",
            message=f"unsupported built-in speech capability {call.capability_id!r}",
        )

    def handle_stream(
        self,
        context: ExtensionContext,
        call: CapabilityCall,
    ) -> AsyncIterator[CapabilityStreamFrame]:
        """Translate core ``AudioChunk`` output into provider media frames."""

        del context
        return self._stream_tts(call)

    def handle_input_stream(
        self,
        context: ExtensionContext,
        call: CapabilityCall,
        input_frames: AsyncIterator[CapabilityStreamFrame],
    ) -> AsyncIterator[CapabilityStreamFrame]:
        """Translate provider audio input into batch or realtime core STT."""

        del context
        if call.capability_id == STT_CAPABILITY_DESCRIPTOR.id:
            return self._stream_stt(call, input_frames)
        return self._stream_realtime_stt(call, input_frames)
