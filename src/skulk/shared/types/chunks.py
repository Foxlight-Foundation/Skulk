from collections.abc import Generator
from typing import Any, Literal

from pydantic import Field, model_validator

from skulk.api.types import (
    FinishReason,
    GenerationStats,
    ImageGenerationStats,
    ToolCallItem,
    TopLogprobItem,
    Usage,
)
from skulk.shared.models.model_cards import AudioResponseFormat, ModelId
from skulk.utils.pydantic_ext import CamelCaseModel, TaggedModel

from .common import CommandId, NodeId
from .streaming import StreamFrameKind, is_terminal_stream_frame_kind


class BaseChunk(TaggedModel):
    model: ModelId


class TokenChunk(BaseChunk):
    text: str
    token_id: int
    usage: Usage | None
    finish_reason: Literal["stop", "length", "content_filter"] | None = None
    stats: GenerationStats | None = None
    logprob: float | None = None
    top_logprobs: list[TopLogprobItem] | None = None
    is_thinking: bool = False


class ErrorChunk(BaseChunk):
    error_message: str
    finish_reason: Literal["error"] = "error"


class ToolCallChunk(BaseChunk):
    tool_calls: list[ToolCallItem]
    usage: Usage | None
    finish_reason: Literal["tool_calls"] = "tool_calls"
    stats: GenerationStats | None = None


class ImageChunk(BaseChunk):
    data: str
    chunk_index: int
    total_chunks: int
    image_index: int
    is_partial: bool = False
    partial_index: int | None = None
    total_partials: int | None = None
    stats: ImageGenerationStats | None = None
    format: Literal["png", "jpeg", "webp"] | None = None
    finish_reason: FinishReason | None = None
    error_message: str | None = None

    def __repr_args__(self) -> Generator[tuple[str, Any], None, None]:
        for name, value in super().__repr_args__():  # pyright: ignore[reportAny]
            if name == "data" and hasattr(value, "__len__"):  # pyright: ignore[reportAny]
                yield name, f"<{len(self.data)} chars>"
            elif name is not None:
                yield name, value


class InputImageChunk(BaseChunk):
    command_id: CommandId
    data: str
    chunk_index: int
    total_chunks: int
    image_index: int = 0

    def __repr_args__(self) -> Generator[tuple[str, Any], None, None]:
        for name, value in super().__repr_args__():  # pyright: ignore[reportAny]
            if name == "data" and hasattr(value, "__len__"):  # pyright: ignore[reportAny]
                yield name, f"<{len(self.data)} chars>"
            elif name is not None:
                yield name, value


class AudioInputChunk(BaseChunk):
    """Base64-encoded input audio chunk for speech-to-text requests."""

    command_id: CommandId
    data: str
    chunk_index: int
    total_chunks: int
    filename: str | None = None
    content_type: str | None = None
    audio_sha256: str

    def __repr_args__(self) -> Generator[tuple[str, Any], None, None]:
        for name, value in super().__repr_args__():  # pyright: ignore[reportAny]
            if name == "data" and hasattr(value, "__len__"):  # pyright: ignore[reportAny]
                yield name, f"<{len(self.data)} chars>"
            elif name is not None:
                yield name, value


InputChunk = InputImageChunk | AudioInputChunk


class EmbeddingChunk(BaseChunk):
    """Response chunk for embedding inference — single batch response, no streaming."""

    embeddings: list[list[float]]
    token_count: int


class AudioChunk(BaseChunk):
    """Encoded audio output chunk for speech-synthesis responses."""

    data: str
    """Base64-encoded audio bytes for this chunk."""
    chunk_index: int
    """Zero-based chunk index within the response."""
    total_chunks: int | None = None
    """Total number of chunks when known for non-streaming responses."""
    format: AudioResponseFormat
    """Encoded audio format for ``data``."""
    sample_rate: int | None = None
    """Audio sample rate in hertz when the runner reports it."""
    is_partial: bool = False
    """Whether this chunk is a partial streaming result."""
    finish_reason: FinishReason | None = None
    """Terminal reason when this is the last output chunk."""
    error_message: str | None = None
    """Error text when a speech runner fails after beginning output."""

    def __repr_args__(self) -> Generator[tuple[str, Any], None, None]:
        for name, value in super().__repr_args__():  # pyright: ignore[reportAny]
            if name == "data" and hasattr(value, "__len__"):  # pyright: ignore[reportAny]
                yield name, f"<{len(self.data)} chars>"
            elif name is not None:
                yield name, value


class TranscriptionChunk(BaseChunk):
    """Text output chunk for speech-to-text and speech-translation responses."""

    text: str
    """Transcript or translated text carried by this chunk."""
    segment_index: int | None = None
    """Model-provided segment index when the runner reports segmented output."""
    is_partial: bool = False
    """Whether this chunk is an interim streaming transcript."""
    language: str | None = None
    """Detected or requested language code when available."""
    segments: list[dict[str, str | int | float | bool | None]] = Field(
        default_factory=list
    )
    """Model-provided segment metadata normalized for verbose transcript formats."""
    finish_reason: FinishReason | None = None
    """Terminal reason when this is the last output chunk."""
    error_message: str | None = None
    """Error text when a speech runner fails after beginning output."""


class PrefillProgressChunk(BaseChunk):
    """Data class for prefill progress events during streaming."""

    processed_tokens: int
    total_tokens: int


GenerationChunk = (
    TokenChunk
    | ImageChunk
    | ToolCallChunk
    | ErrorChunk
    | EmbeddingChunk
    | AudioChunk
    | TranscriptionChunk
    | PrefillProgressChunk
)


class DataChunk(CamelCaseModel):
    """One ordered lifecycle frame on the serving data plane.

    Frames travel directly from the serving rank-0 worker to the owning API
    node: they are never indexed by the master, written to the event log, or
    rebroadcast cluster-wide. ``kind`` makes stream start and termination
    explicit while payload-bearing frames retain the existing
    ``GenerationChunk`` contract used by endpoint queues.

    ``sequence`` is monotonic across every lifecycle and payload frame for one
    command. The API reorders whole frames before observing or dispatching them,
    so lifecycle state and output remain coherent on transports that can reorder.
    """

    command_id: CommandId
    kind: StreamFrameKind = Field(
        default="chunk",
        description=(
            "Explicit stream lifecycle kind. Producers emit started first, then "
            "zero or more chunk frames, and exactly one completed, failed, or "
            "cancelled terminal frame."
        ),
    )
    chunk: GenerationChunk | None = Field(
        default=None,
        description=(
            "Endpoint payload for chunk frames and terminal frames that carry a "
            "final result or ErrorChunk. Lifecycle-only frames omit it."
        ),
    )
    sequence: int
    owner_node: NodeId | None = Field(
        default=None,
        description=(
            "The API node that owns this command (#279 Phase 2). The API stamps "
            "it on the serving command, the master carries it onto the task, and "
            "the producing rank-0 supervisor stamps it here. On the Zenoh data "
            "plane the Router publishes to the key data/<owner_node> so only the "
            "owning API node receives the chunk (killing the cluster-wide "
            "fan-out). On gossipsub it is ignored (the bare topic broadcasts), "
            "and it is None only for output with no recorded owner."
        ),
    )

    @model_validator(mode="after")
    def _validate_lifecycle_payload(self) -> "DataChunk":
        """Reject lifecycle frames that cannot be interpreted consistently."""

        if self.kind == "started" and self.chunk is not None:
            raise ValueError("started data-plane frames must not carry a chunk")
        if self.kind == "chunk" and self.chunk is None:
            raise ValueError("chunk data-plane frames must carry a chunk")
        if self.kind in ("failed", "cancelled") and not isinstance(
            self.chunk, ErrorChunk
        ):
            raise ValueError(
                f"{self.kind} data-plane frames must carry an ErrorChunk"
            )
        return self

    @property
    def is_terminal(self) -> bool:
        """Whether this frame closes its command stream."""

        return is_terminal_stream_frame_kind(self.kind)
