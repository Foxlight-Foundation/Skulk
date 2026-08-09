"""Canonical internal types for speech-serving task parameters and media."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from skulk.shared.models.model_cards import AudioResponseFormat
from skulk.shared.types.common import CommandId, ModelId


class SpeechSynthesisTaskParams(BaseModel, frozen=True):
    """Internal task params for text-to-speech inference.

    The API endpoint validates the OpenAI-compatible wire request and converts it
    into this compact runner contract before handing it to the master/worker
    pipeline.
    """

    model: ModelId
    input_text: str
    response_format: AudioResponseFormat = AudioResponseFormat.Mp3
    voice: str | None = None
    speed: float | None = Field(default=None, gt=0)
    instruct: str | None = None
    lang_code: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    max_tokens: int | None = None
    seed: int | None = Field(
        default=None,
        ge=0,
        le=2**32 - 1,
        description="Optional deterministic MLX sampling seed for this request.",
    )
    reference_audio: str | None = None
    reference_text: str | None = None
    reference_voice_profile: str | None = None
    """Verified bundled profile ID resolved locally by the selected worker."""
    reference_audio_present: bool = False
    reference_audio_filename: str | None = None
    reference_audio_content_type: str | None = None
    reference_audio_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    reference_audio_data: bytes | None = None
    stream: bool = False
    streaming_interval: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_reference_audio(self) -> "SpeechSynthesisTaskParams":
        if self.reference_audio_present != (self.reference_audio_sha256 is not None):
            raise ValueError(
                "reference_audio_present and reference_audio_sha256 must agree"
            )
        if self.reference_audio_data is not None and not self.reference_audio_present:
            raise ValueError("reference_audio_data requires reference_audio_present")
        if self.reference_voice_profile is not None and (
            self.reference_audio is not None
            or self.reference_text is not None
            or self.reference_audio_present
            or self.reference_audio_data is not None
        ):
            raise ValueError(
                "bundled reference voice profiles cannot be combined with "
                "request-scoped reference audio"
            )
        return self


class AudioTranscriptionTaskParams(BaseModel, frozen=True):
    """Internal task params for speech-to-text inference.

    The API retains multipart audio until authoritative task placement, sends
    raw bytes over the node-addressed speech-media data topic, and the selected
    worker injects the verified base64 payload before runner dispatch.
    """

    model: ModelId
    filename: str | None = None
    content_type: str | None = None
    total_input_chunks: int = Field(default=0, ge=0)
    audio_sha256: str
    audio_data: str | None = None
    language: str | None = None
    prompt: str | None = None
    temperature: float | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    chunk_duration: float | None = Field(default=None, gt=0)
    frame_threshold: int | None = Field(default=None, gt=0)
    context: str | None = None
    prefill_step_size: int | None = Field(default=None, gt=0)
    text: str | None = None
    word_timestamps: bool = False
    timestamp_granularities: tuple[str, ...] = ()
    translate_to_english: bool = False
    stream: bool = False


class RealtimeAudioTranscriptionTaskParams(BaseModel, frozen=True):
    """Parameters for one true incremental STT runner session.

    The model card and provider admission establish that the mounted runner
    implements the upstream ``create_streaming_session`` protocol. Audio bytes
    arrive separately through :class:`RealtimeAudioInputFrame` so they never
    enter commands, State, or the event log.
    """

    model: ModelId
    input_sample_rate: int = Field(ge=8000, le=96000)
    temperature: float = 0.0
    transcription_delay_ms: int = Field(default=480, ge=80, le=2400)

    @model_validator(mode="after")
    def _validate_delay_step(self) -> "RealtimeAudioTranscriptionTaskParams":
        if self.transcription_delay_ms % 80 != 0:
            raise ValueError("transcription_delay_ms must be a multiple of 80")
        return self


class RealtimeAudioInputFrame(BaseModel, frozen=True):
    """One bounded PCM frame or terminal sent from the API to a local worker."""

    command_id: CommandId
    sequence: int = Field(ge=0)
    kind: Literal["chunk", "completed", "cancelled"]
    data: bytes = b""

    @model_validator(mode="after")
    def _validate_payload(self) -> "RealtimeAudioInputFrame":
        if self.kind == "chunk" and not self.data:
            raise ValueError("realtime audio chunk must carry PCM bytes")
        if self.kind != "chunk" and self.data:
            raise ValueError("realtime audio terminal must not carry bytes")
        if len(self.data) % 2 != 0:
            raise ValueError("pcm_s16le audio must contain complete 16-bit samples")
        return self
