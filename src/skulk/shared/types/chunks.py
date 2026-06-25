from collections.abc import Generator
from typing import Any, Literal

from pydantic import Field

from skulk.api.types import (
    FinishReason,
    GenerationStats,
    ImageGenerationStats,
    ToolCallItem,
    TopLogprobItem,
    Usage,
)
from skulk.shared.models.model_cards import ModelId
from skulk.utils.pydantic_ext import CamelCaseModel, TaggedModel

from .common import CommandId, NodeId


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


class EmbeddingChunk(BaseChunk):
    """Response chunk for embedding inference — single batch response, no streaming."""

    embeddings: list[list[float]]
    token_count: int


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
    | PrefillProgressChunk
)


class DataChunk(CamelCaseModel):
    """A generation output chunk on the data plane (#279 Phase 2).

    Carries the same ``{command_id, chunk}`` payload as the ``ChunkGenerated``
    event, but travels the ``DATA`` topic directly from the serving rank-0
    worker to the owning API node: it is never indexed by the master, written to
    the event log, or rebroadcast cluster-wide. The owning API node demuxes by
    ``command_id`` into its per-command stream queue, exactly as it did for the
    event. Output chunks never mutate ``State`` (apply was a no-op), so removing
    them from the ordered log is loss-free for correctness while eliminating the
    per-token master hop + disk write that dominated event-log volume (#278).

    ``sequence`` is a per-command monotonic counter stamped by the producing
    rank-0 supervisor (0, 1, 2, ...). The control plane this replaced gave every
    chunk a total order via the master's event ``idx``; the ``DATA`` gossip topic
    has no such order, and when the producing worker and the owning API node are
    different nodes the mesh can deliver a command's chunks out of order, which
    silently transposed multi-node generation output. The API reorders by
    ``sequence`` before yielding (see ``API._reorder_and_dispatch``).
    """

    command_id: CommandId
    chunk: GenerationChunk
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
