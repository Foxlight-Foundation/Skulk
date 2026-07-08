"""Canonical internal types for speech-serving task parameters."""

from pydantic import BaseModel, Field

from skulk.shared.models.model_cards import AudioResponseFormat
from skulk.shared.types.common import ModelId


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
    reference_audio: str | None = None
    reference_text: str | None = None


class AudioTranscriptionTaskParams(BaseModel, frozen=True):
    """Internal task params for speech-to-text inference.

    The API receives multipart audio bytes, sends them through command-owned
    input chunks, and the worker injects the assembled base64 payload before
    dispatching the task to the speech runner.
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
