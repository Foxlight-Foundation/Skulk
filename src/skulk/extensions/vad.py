"""First-party voice activity detection capability."""

from __future__ import annotations

import importlib
import sys
from array import array
from collections import deque
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, cast, final

from pydantic import BaseModel, ConfigDict, Field

from skulk.extensions.calls import CapabilityCall
from skulk.extensions.capabilities import CapabilityDescriptor
from skulk.extensions.streams import (
    CapabilityStreamError,
    CapabilityStreamFrame,
    InlineMediaAttachment,
)
from skulk.extensions.types import ExtensionContext

_VALID_SAMPLE_RATES = frozenset({8000, 16000, 32000, 48000})
_VALID_FRAME_DURATIONS_MS = frozenset({10, 20, 30})


class _WebRtcVad(Protocol):
    """Typed subset of the untyped WebRTC VAD binding."""

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        """Classify one exact-duration PCM16 frame."""

        ...


class _WebRtcVadModule(Protocol):
    """Factory surface exported by the untyped WebRTC VAD binding."""

    def Vad(self, aggressiveness: int) -> _WebRtcVad:  # noqa: N802
        """Create a classifier with aggressiveness from zero through three."""

        ...


class VadConfig(BaseModel):
    """Validated turn-detection settings for one provider call."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    sample_rate: int = Field(ge=8000, le=48000)
    aggressiveness: int = Field(default=2, ge=0, le=3)
    frame_ms: int = Field(default=20)
    minimum_speech_ms: int = Field(default=120, ge=10, le=5000)
    silence_hangover_ms: int = Field(default=400, ge=10, le=5000)
    preroll_ms: int = Field(default=200, ge=0, le=2000)
    maximum_utterance_ms: int = Field(default=30000, ge=100, le=120000)

    @property
    def frame_bytes(self) -> int:
        """Return the exact PCM16 byte count consumed per classifier frame."""

        return self.sample_rate * self.frame_ms // 1000 * 2


@dataclass(frozen=True)
class VadTurnEvent:
    """One speech boundary emitted by the VAD state machine."""

    kind: str
    timestamp_ms: int
    reason: str
    preroll_ms: int = 0


@final
class StreamingPcm16Resampler:
    """Stateful mono PCM16 linear resampler preserving chunk boundaries."""

    def __init__(self, input_rate: int, output_rate: int) -> None:
        """Create a resampler between positive integer sample rates."""

        if input_rate <= 0 or output_rate <= 0:
            raise ValueError("sample rates must be positive")
        self._step = input_rate / output_rate
        self._carry: list[int] = []
        self._position = 0.0

    def process(self, pcm16: bytes) -> bytes:
        """Resample one ordered little-endian PCM16 chunk."""

        if len(pcm16) % 2 != 0:
            raise ValueError("PCM16 input must contain an even byte count")
        decoded = array("h")
        decoded.frombytes(pcm16)
        if sys.byteorder != "little":
            decoded.byteswap()
        combined = self._carry + decoded.tolist()
        output = array("h")
        while self._position + 1 < len(combined):
            left_index = int(self._position)
            fraction = self._position - left_index
            value = round(
                combined[left_index]
                + (combined[left_index + 1] - combined[left_index]) * fraction
            )
            output.append(max(-32768, min(32767, value)))
            self._position += self._step
        consumed = min(int(self._position), len(combined))
        self._carry = combined[consumed:]
        self._position -= consumed
        if sys.byteorder != "little":
            output.byteswap()
        return output.tobytes()


@final
class VoiceActivityDetector:
    """Convert exact PCM16 frames into stable speech turn boundaries."""

    def __init__(
        self,
        config: VadConfig,
        *,
        classify: Callable[[bytes, int], bool] | None = None,
    ) -> None:
        """Create one stateful detector with an injectable frame classifier."""

        if config.sample_rate not in _VALID_SAMPLE_RATES:
            raise ValueError("WebRTC VAD supports 8, 16, 32, or 48 kHz PCM")
        if config.frame_ms not in _VALID_FRAME_DURATIONS_MS:
            raise ValueError("WebRTC VAD frames must be 10, 20, or 30 ms")
        if classify is None:
            module = cast(
                _WebRtcVadModule,
                cast(object, importlib.import_module("webrtcvad")),
            )
            classifier = module.Vad(config.aggressiveness)
            classify = classifier.is_speech
        self._config = config
        self._classify = classify
        self._elapsed_ms = 0
        self._speech_run_ms = 0
        self._silence_run_ms = 0
        self._utterance_ms = 0
        self._active = False
        self._preroll: deque[bytes] = deque(
            maxlen=max(1, config.preroll_ms // config.frame_ms)
        )

    @property
    def frame_bytes(self) -> int:
        """Return the exact PCM byte count required by one detector frame."""

        return self._config.frame_bytes

    def process(self, frame: bytes) -> tuple[VadTurnEvent, ...]:
        """Consume one exact PCM16 frame and emit zero or more boundaries."""

        if len(frame) != self._config.frame_bytes:
            raise ValueError(
                f"VAD frame must contain exactly {self._config.frame_bytes} bytes"
            )
        frame_start_ms = self._elapsed_ms
        self._elapsed_ms += self._config.frame_ms
        speech = self._classify(frame, self._config.sample_rate)
        if not self._active:
            self._preroll.append(frame)
            self._speech_run_ms = (
                self._speech_run_ms + self._config.frame_ms if speech else 0
            )
            if self._speech_run_ms < self._config.minimum_speech_ms:
                return ()
            self._active = True
            self._silence_run_ms = 0
            self._utterance_ms = self._speech_run_ms
            detected_start_ms = self._elapsed_ms - self._speech_run_ms
            available_preroll_ms = min(
                self._config.preroll_ms,
                len(self._preroll) * self._config.frame_ms,
                detected_start_ms,
            )
            return (
                VadTurnEvent(
                    kind="speech_started",
                    timestamp_ms=detected_start_ms - available_preroll_ms,
                    reason="minimum_speech",
                    preroll_ms=available_preroll_ms,
                ),
            )

        self._utterance_ms += self._config.frame_ms
        self._silence_run_ms = 0 if speech else self._silence_run_ms + self._config.frame_ms
        if self._utterance_ms >= self._config.maximum_utterance_ms:
            return self._stop(self._elapsed_ms, "maximum_duration")
        if self._silence_run_ms >= self._config.silence_hangover_ms:
            speech_end_ms = frame_start_ms - self._silence_run_ms + self._config.frame_ms
            return self._stop(max(0, speech_end_ms), "silence")
        return ()

    def finish(self) -> tuple[VadTurnEvent, ...]:
        """Close an active turn when the caller half-closes input."""

        if not self._active:
            return ()
        return self._stop(self._elapsed_ms, "input_completed")

    def _stop(self, timestamp_ms: int, reason: str) -> tuple[VadTurnEvent, ...]:
        event = VadTurnEvent(
            kind="speech_stopped",
            timestamp_ms=timestamp_ms,
            reason=reason,
        )
        self._active = False
        self._speech_run_ms = 0
        self._silence_run_ms = 0
        self._utterance_ms = 0
        self._preroll.clear()
        return (event,)


VAD_CAPABILITY_DESCRIPTOR = CapabilityDescriptor(
    id="vad",
    version="1.0.0",
    title="Voice activity detection",
    description=(
        "Detect speech turn boundaries in ordered mono PCM16 audio without "
        "retaining media."
    ),
    input_schema={
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "sample_rate": {"enum": sorted(_VALID_SAMPLE_RATES)},
            "aggressiveness": {"type": "integer", "minimum": 0, "maximum": 3},
            "frame_ms": {"enum": sorted(_VALID_FRAME_DURATIONS_MS)},
            "minimum_speech_ms": {"type": "integer", "minimum": 10, "maximum": 5000},
            "silence_hangover_ms": {"type": "integer", "minimum": 10, "maximum": 5000},
            "preroll_ms": {"type": "integer", "minimum": 0, "maximum": 2000},
            "maximum_utterance_ms": {"type": "integer", "minimum": 100, "maximum": 120000},
        },
        "required": ["sample_rate"],
        "additionalProperties": False,
    },
    output_schema={
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {"turns": {"type": "integer", "minimum": 0}},
        "required": ["turns"],
        "additionalProperties": False,
    },
    io_mode="bidirectional",
    input_chunk_schema={
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "format": {"const": "pcm_s16le"},
            "sample_rate": {"enum": sorted(_VALID_SAMPLE_RATES)},
            "channels": {"const": 1},
        },
        "required": ["format", "sample_rate", "channels"],
        "additionalProperties": False,
    },
    output_chunk_schema={
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "event": {"enum": ["speech_started", "speech_stopped"]},
            "timestamp_ms": {"type": "integer", "minimum": 0},
            "reason": {"type": "string"},
            "preroll_ms": {"type": "integer", "minimum": 0},
        },
        "required": ["event", "timestamp_ms", "reason", "preroll_ms"],
        "additionalProperties": False,
    },
    annotations={
        "modality": "audio",
        "latency": "realtime",
        "runtime": "webrtcvad",
        "stability": "stable",
        "input_codec": "pcm_s16le",
        "retains_media": "false",
    },
)


@final
class BuiltinVadProvider:
    """Serve stateless-per-call VAD through the generic provider contract."""

    name = "skulk-builtin-vad"
    skulk_requires = ">=0"

    def __init__(
        self,
        *,
        detector_factory: Callable[[VadConfig], VoiceActivityDetector] = VoiceActivityDetector,
    ) -> None:
        """Create a provider with an injectable detector factory for tests."""

        self._detector_factory = detector_factory

    def chat_middleware(self) -> None:
        """Return no chat middleware; VAD is a Fabric service."""

        return None

    def capabilities(self) -> Sequence[CapabilityDescriptor]:
        """Describe the reusable voice activity detector."""

        return (VAD_CAPABILITY_DESCRIPTOR,)

    def on_start(self, context: ExtensionContext) -> None:
        """Advertise VAD because it has no mounted-model dependency."""

        context.advertise_capability(VAD_CAPABILITY_DESCRIPTOR.id)

    def handle_input_stream(
        self,
        context: ExtensionContext,
        call: CapabilityCall,
        input_frames: AsyncIterator[CapabilityStreamFrame],
    ) -> AsyncIterator[CapabilityStreamFrame]:
        """Classify caller PCM frames and emit ordered speech boundaries."""

        del context
        return self._stream(call, input_frames)

    async def _stream(
        self,
        call: CapabilityCall,
        input_frames: AsyncIterator[CapabilityStreamFrame],
    ) -> AsyncIterator[CapabilityStreamFrame]:
        config = VadConfig.model_validate(call.payload)
        detector = self._detector_factory(config)
        sequence = 1
        turns = 0
        buffered_pcm = bytearray()
        async for frame in input_frames:
            if frame.kind != "chunk":
                continue
            attachment = frame.media
            if not isinstance(attachment, InlineMediaAttachment):
                yield self._invalid_frame(
                    call.call_id,
                    sequence,
                    "VAD requires one inline PCM media attachment",
                )
                return
            if (
                attachment.codec != "pcm_s16le"
                or attachment.sample_rate != config.sample_rate
                or attachment.channels != 1
            ):
                yield self._invalid_frame(
                    call.call_id,
                    sequence,
                    "VAD media metadata must match negotiated mono PCM16",
                )
                return
            buffered_pcm.extend(attachment.data)
            while len(buffered_pcm) >= config.frame_bytes:
                pcm_frame = bytes(buffered_pcm[: config.frame_bytes])
                del buffered_pcm[: config.frame_bytes]
                for event in detector.process(pcm_frame):
                    if event.kind == "speech_started":
                        turns += 1
                    yield self._event_frame(call.call_id, sequence, event)
                    sequence += 1
        if buffered_pcm:
            yield self._invalid_frame(
                call.call_id,
                sequence,
                "VAD input ended with a partial PCM frame",
            )
            return
        for event in detector.finish():
            yield self._event_frame(call.call_id, sequence, event)
            sequence += 1
        yield CapabilityStreamFrame(
            call_id=call.call_id,
            direction="provider_to_caller",
            sequence=sequence,
            kind="completed",
            payload={"turns": turns},
        )

    @staticmethod
    def _event_frame(
        call_id: str,
        sequence: int,
        event: VadTurnEvent,
    ) -> CapabilityStreamFrame:
        return CapabilityStreamFrame(
            call_id=call_id,
            direction="provider_to_caller",
            sequence=sequence,
            kind="chunk",
            payload={
                "event": event.kind,
                "timestamp_ms": event.timestamp_ms,
                "reason": event.reason,
                "preroll_ms": event.preroll_ms,
            },
        )

    @staticmethod
    def _invalid_frame(
        call_id: str,
        sequence: int,
        message: str,
    ) -> CapabilityStreamFrame:
        """Return a caller-actionable terminal for malformed input media."""

        return CapabilityStreamFrame(
            call_id=call_id,
            direction="provider_to_caller",
            sequence=sequence,
            kind="failed",
            error=CapabilityStreamError(code="invalid_frame", message=message),
        )
