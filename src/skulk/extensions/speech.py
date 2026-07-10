"""First-party provider facade for mounted Skulk speech synthesis models."""

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

TtsAdmission = Callable[[CapabilityCall], Awaitable[CapabilityError | None]]
TtsStream = Callable[[CapabilityCall], AsyncIterator[CapabilityStreamFrame]]


@final
class BuiltinSpeechProvider:
    """Expose core speech serving through the generic provider contract."""

    name = "skulk-builtin-speech"
    skulk_requires = ">=0"

    def __init__(self, *, admit_tts: TtsAdmission, stream_tts: TtsStream) -> None:
        """Create the facade around API-owned admission and stream adapters."""

        self._admit_tts = admit_tts
        self._stream_tts = stream_tts

    def chat_middleware(self) -> None:
        """Return no chat middleware; this provider is a fabric service."""

        return None

    def capabilities(self) -> Sequence[CapabilityDescriptor]:
        """Describe the first-party server-streaming TTS capability."""

        return (TTS_CAPABILITY_DESCRIPTOR,)

    def on_start(self, context: ExtensionContext) -> None:
        """Keep discovery withdrawn until an eligible model is mounted."""

        context.withdraw_capability(TTS_CAPABILITY_DESCRIPTOR.id)

    async def admit_stream(
        self,
        context: ExtensionContext,
        call: CapabilityCall,
    ) -> CapabilityError | None:
        """Validate dynamic mounted-model requirements before ``started``."""

        del context
        return await self._admit_tts(call)

    def handle_stream(
        self,
        context: ExtensionContext,
        call: CapabilityCall,
    ) -> AsyncIterator[CapabilityStreamFrame]:
        """Translate core ``AudioChunk`` output into provider media frames."""

        del context
        return self._stream_tts(call)
