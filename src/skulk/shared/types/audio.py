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
