import time
from collections.abc import Generator
from typing import Annotated, Any, Literal, get_args
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from exo.shared.models.capabilities import ResolvedCapabilityProfile
from exo.shared.models.model_cards import ModelCard, ModelId
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.text_generation import ReasoningEffort
from exo.shared.types.worker.instances import Instance, InstanceId, InstanceMeta
from exo.shared.types.worker.shards import Sharding, ShardMetadata
from exo.utils.pydantic_ext import CamelCaseModel

FinishReason = Literal[
    "stop", "length", "tool_calls", "content_filter", "function_call", "error"
]


class ErrorInfo(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: int


class ErrorResponse(BaseModel):
    error: ErrorInfo


class ModelListModel(BaseModel):
    """Public model-catalog entry returned by the models endpoints."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "exo"
    # openwebui fields
    hugging_face_id: str = Field(default="")
    name: str = Field(default="")
    description: str = Field(default="")
    context_length: int = Field(default=0)
    tags: list[str] = Field(default=[])
    storage_size_megabytes: int = Field(default=0)
    supports_tensor: bool = Field(default=False)
    tasks: list[str] = Field(default=[])
    is_custom: bool = Field(default=False)
    family: str = Field(default="")
    quantization: str = Field(default="")
    base_model: str = Field(default="")
    capabilities: list[str] = Field(
        default_factory=list,
        description="Coarse catalog capability labels such as text, vision, thinking, or embedding.",
    )
    reasoning: "ReasoningCapabilitySection | None" = Field(
        default=None,
        description="Optional declarative reasoning controls from the model card.",
    )
    modalities: "ModalitiesCapabilitySection | None" = Field(
        default=None,
        description="Optional declarative modality support details from the model card.",
    )
    tooling: "ToolingCapabilitySection | None" = Field(
        default=None,
        description="Optional declarative tool-calling metadata from the model card.",
    )
    runtime: "RuntimeCapabilitySection | None" = Field(
        default=None,
        description="Optional declarative runtime integration hints from the model card.",
    )
    resolved_capabilities: "ResolvedModelCapabilities | None" = Field(
        default=None,
        description="Normalized runtime capabilities resolved from the model card and model-family defaults.",
    )


class ResolvedModelCapabilities(BaseModel):
    """Normalized runtime behavior that UI and API consumers can safely inspect."""

    family: str = Field(default="", description="Resolved model family used for runtime behavior decisions.")
    supports_thinking: bool = Field(
        default=False,
        description="Whether the runtime expects the model to expose a reasoning or thinking mode.",
    )
    supports_thinking_toggle: bool = Field(
        default=False,
        description="Whether thinking can be explicitly enabled or disabled for requests.",
    )
    supports_thinking_budget: bool = Field(
        default=False,
        description="Whether the runtime expects the model to accept a thinking or reasoning budget control.",
    )
    default_reasoning_effort: ReasoningEffort = Field(
        default="medium",
        description="Reasoning effort used when thinking is enabled without an explicit effort override.",
    )
    disabled_reasoning_effort: ReasoningEffort = Field(
        default="none",
        description="Reasoning effort used when thinking is explicitly disabled.",
    )
    thinking_format: str = Field(
        default="none",
        description="Resolved reasoning marker format expected from this model family.",
    )
    supports_image_input: bool = Field(
        default=False,
        description="Whether the runtime should treat the model as accepting image inputs.",
    )
    supports_audio_input: bool = Field(
        default=False,
        description="Whether the runtime should treat the model as accepting audio inputs.",
    )
    supports_tool_calling: bool = Field(
        default=False,
        description="Whether the runtime expects the model to support structured tool calling.",
    )
    tool_call_format: str = Field(
        default="generic",
        description="Resolved tool-call output format family used for parsing.",
    )
    prompt_renderer: str = Field(
        default="tokenizer",
        description="Resolved prompt renderer strategy used to prepare requests for this model.",
    )
    output_parser: str = Field(
        default="generic",
        description="Resolved output parser strategy used to interpret model responses.",
    )
    supports_native_multimodal: bool = Field(
        default=False,
        description="Whether the runtime can use a native multimodal execution path for the model.",
    )

    @classmethod
    def from_profile(
        cls, profile: ResolvedCapabilityProfile
    ) -> "ResolvedModelCapabilities":
        return cls(
            family=profile.family,
            supports_thinking=profile.supports_thinking,
            supports_thinking_toggle=profile.supports_thinking_toggle,
            supports_thinking_budget=profile.supports_thinking_budget,
            default_reasoning_effort=profile.default_reasoning_effort,
            disabled_reasoning_effort=profile.disabled_reasoning_effort,
            thinking_format=profile.thinking_format.value,
            supports_image_input=profile.supports_image_input,
            supports_audio_input=profile.supports_audio_input,
            supports_tool_calling=profile.supports_tool_calling,
            tool_call_format=profile.tool_call_format.value,
            prompt_renderer=profile.prompt_renderer.value,
            output_parser=profile.output_parser.value,
            supports_native_multimodal=profile.supports_native_multimodal,
        )


class ReasoningCapabilitySection(BaseModel):
    """Snake-case reasoning metadata exposed by the models API."""

    supports_toggle: bool | None = None
    supports_budget: bool | None = None
    format: str | None = None
    default_effort: ReasoningEffort | None = None
    disabled_effort: ReasoningEffort | None = None

    @classmethod
    def from_model_card(cls, model_card: ModelCard) -> "ReasoningCapabilitySection | None":
        config = model_card.reasoning
        if config is None:
            return None
        return cls(
            supports_toggle=config.supports_toggle,
            supports_budget=config.supports_budget,
            format=config.format.value if config.format is not None else None,
            default_effort=config.default_effort,
            disabled_effort=config.disabled_effort,
        )


class ModalitiesCapabilitySection(BaseModel):
    """Snake-case modality metadata exposed by the models API."""

    supports_audio_input: bool | None = None
    supports_native_multimodal: bool | None = None

    @classmethod
    def from_model_card(cls, model_card: ModelCard) -> "ModalitiesCapabilitySection | None":
        config = model_card.modalities
        if config is None:
            return None
        return cls(
            supports_audio_input=config.supports_audio_input,
            supports_native_multimodal=config.supports_native_multimodal,
        )


class ToolingCapabilitySection(BaseModel):
    """Snake-case tool-calling metadata exposed by the models API."""

    supports_tool_calling: bool | None = None
    tool_call_format: str | None = None

    @classmethod
    def from_model_card(cls, model_card: ModelCard) -> "ToolingCapabilitySection | None":
        config = model_card.tooling
        if config is None:
            return None
        return cls(
            supports_tool_calling=config.supports_tool_calling,
            tool_call_format=(
                config.tool_call_format.value
                if config.tool_call_format is not None
                else None
            ),
        )


class RuntimeCapabilitySection(BaseModel):
    """Snake-case runtime metadata exposed by the models API."""

    prompt_renderer: str | None = None
    output_parser: str | None = None

    @classmethod
    def from_model_card(cls, model_card: ModelCard) -> "RuntimeCapabilitySection | None":
        config = model_card.runtime
        if config is None:
            return None
        return cls(
            prompt_renderer=(
                config.prompt_renderer.value
                if config.prompt_renderer is not None
                else None
            ),
            output_parser=(
                config.output_parser.value if config.output_parser is not None else None
            ),
        )


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelListModel]


class ChatCompletionMessageText(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ChatCompletionMessageImageUrl(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: dict[str, str]  # {"url": "data:image/png;base64,..."}


ChatCompletionContentPart = ChatCompletionMessageText | ChatCompletionMessageImageUrl


class ToolCallItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    index: int | None = None
    type: Literal["function"] = "function"
    function: ToolCallItem


class ChatCompletionMessage(BaseModel):
    role: Literal["system", "user", "assistant", "developer", "tool", "function"]
    content: (
        str | ChatCompletionContentPart | list[ChatCompletionContentPart] | None
    ) = None
    reasoning_content: str | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    function_call: dict[str, Any] | None = None


class BenchChatCompletionMessage(ChatCompletionMessage):
    pass


class TopLogprobItem(BaseModel):
    token: str
    logprob: float
    bytes: list[int] | None = None


class LogprobsContentItem(BaseModel):
    token: str
    logprob: float
    bytes: list[int] | None = None
    top_logprobs: list[TopLogprobItem]


class Logprobs(BaseModel):
    content: list[LogprobsContentItem] | None = None


class PromptTokensDetails(BaseModel):
    cached_tokens: int = 0
    audio_tokens: int = 0


class CompletionTokensDetails(BaseModel):
    reasoning_tokens: int = 0
    audio_tokens: int = 0
    accepted_prediction_tokens: int = 0
    rejected_prediction_tokens: int = 0


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: PromptTokensDetails
    completion_tokens_details: CompletionTokensDetails


class StreamingChoiceResponse(BaseModel):
    index: int
    delta: ChatCompletionMessage
    logprobs: Logprobs | None = None
    finish_reason: FinishReason | None = None
    usage: Usage | None = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    logprobs: Logprobs | None = None
    finish_reason: FinishReason | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice | StreamingChoiceResponse]
    usage: Usage | None = None
    service_tier: str | None = None


class GenerationStats(BaseModel):
    prompt_tps: float
    generation_tps: float
    prompt_tokens: int
    generation_tokens: int
    peak_memory_usage: Memory


class ImageGenerationStats(BaseModel):
    seconds_per_step: float
    total_generation_time: float

    num_inference_steps: int
    num_images: int

    image_width: int
    image_height: int

    peak_memory_usage: Memory


class NodePowerStats(BaseModel, frozen=True):
    node_id: NodeId
    samples: int
    avg_sys_power: float


class PowerUsage(BaseModel, frozen=True):
    elapsed_seconds: float
    nodes: list[NodePowerStats]
    total_avg_sys_power_watts: float
    total_energy_joules: float


class BenchChatCompletionResponse(ChatCompletionResponse):
    generation_stats: GenerationStats | None = None
    power_usage: PowerUsage | None = None


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
                "messages": [{"role": "user", "content": "Hello from Skulk"}],
                "stream": False,
                "temperature": 0.7,
            }
        }
    )

    model: ModelId
    frequency_penalty: float | None = None
    messages: list[ChatCompletionMessage]
    logit_bias: dict[str, int] | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    max_tokens: int | None = None
    n: int | None = None
    presence_penalty: float | None = None
    response_format: dict[str, Any] | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    tools: list[dict[str, Any]] | None = None
    reasoning_effort: ReasoningEffort | None = None
    enable_thinking: bool | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    repetition_context_size: int | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    user: str | None = None


class BenchChatCompletionRequest(ChatCompletionRequest):
    pass


class AddCustomModelParams(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"example": {"model_id": "mlx-community/my-custom-model"}}
    )

    model_id: ModelId


class HuggingFaceSearchResult(BaseModel):
    id: str
    author: str = ""
    downloads: int = 0
    likes: int = 0
    last_modified: str = ""
    tags: list[str] = Field(default_factory=list)


class PlaceInstanceParams(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_id": "mlx-community/Llama-3.2-1B-Instruct-4bit",
                "sharding": "Pipeline",
                "instance_meta": "MlxRing",
                "min_nodes": 1,
            }
        }
    )

    model_id: ModelId
    sharding: Sharding = Sharding.Pipeline
    instance_meta: InstanceMeta = InstanceMeta.MlxRing
    min_nodes: int = 1


class CreateInstanceParams(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "instance": {
                    "MlxRingInstance": {
                        "instanceId": "00000000-0000-0000-0000-000000000000",
                        "shardAssignments": {
                            "modelId": "mlx-community/Llama-3.2-1B-Instruct-4bit",
                            "runnerToShard": {
                                "runner-1": {
                                    "PipelineShardMetadata": {
                                        "modelCard": {
                                            "modelId": "mlx-community/Llama-3.2-1B-Instruct-4bit",
                                            "storageSize": {"inBytes": 2147483648},
                                            "nLayers": 32,
                                            "hiddenSize": 2048,
                                            "supportsTensor": False,
                                            "tasks": ["TextGeneration"],
                                        },
                                        "deviceRank": 0,
                                        "worldSize": 1,
                                        "startLayer": 0,
                                        "endLayer": 32,
                                        "nLayers": 32,
                                    }
                                }
                            },
                            "nodeToRunner": {"node-1": "runner-1"},
                        },
                        "hostsByNode": {"node-1": []},
                        "ephemeralPort": 52416,
                    }
                }
            }
        }
    )

    instance: Instance


class PlacementPreview(BaseModel):
    model_id: ModelId
    sharding: Sharding
    instance_meta: InstanceMeta
    instance: Instance | None = None
    # Keys are NodeId strings, values are additional bytes that would be used on that node
    memory_delta_by_node: dict[str, int] | None = None
    error: str | None = None


class PlacementPreviewResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "previews": [
                    {
                        "model_id": "mlx-community/Llama-3.2-1B-Instruct-4bit",
                        "sharding": "Pipeline",
                        "instance_meta": "MlxRing",
                        "instance": None,
                        "memory_delta_by_node": None,
                        "error": None,
                    }
                ]
            }
        }
    )

    previews: list[PlacementPreview]


class DeleteInstanceTaskParams(BaseModel):
    instance_id: str


class CreateInstanceResponse(BaseModel):
    message: str
    command_id: CommandId
    model_card: ModelCard


class DeleteInstanceResponse(BaseModel):
    message: str
    command_id: CommandId
    instance_id: InstanceId


class CancelCommandResponse(BaseModel):
    message: str
    command_id: CommandId


ImageSize = Literal[
    "auto",
    "512x512",
    "768x768",
    "1024x768",
    "768x1024",
    "1024x1024",
    "1024x1536",
    "1536x1024",
]


def normalize_image_size(v: object) -> ImageSize:
    """Shared validator for ImageSize fields: maps None → "auto" and rejects invalid values."""
    if v is None:
        return "auto"
    if v not in get_args(ImageSize):
        raise ValueError(f"Invalid size: {v!r}. Must be one of {get_args(ImageSize)}")
    return v  # pyright: ignore[reportReturnType]


class AdvancedImageParams(BaseModel):
    seed: Annotated[int, Field(ge=0)] | None = None
    num_inference_steps: Annotated[int, Field(ge=1, le=100)] | None = None
    guidance: Annotated[float, Field(ge=1.0, le=20.0)] | None = None
    negative_prompt: str | None = None
    num_sync_steps: Annotated[int, Field(ge=1, le=100)] | None = None


class ImageGenerationTaskParams(BaseModel):
    prompt: str
    background: str | None = None
    model: str
    moderation: str | None = None
    n: int | None = 1
    output_compression: int | None = None
    output_format: Literal["png", "jpeg", "webp"] = "png"
    partial_images: int | None = 0
    quality: Literal["high", "medium", "low"] | None = "medium"
    response_format: Literal["url", "b64_json"] | None = "b64_json"
    size: ImageSize = "auto"
    stream: bool | None = False
    style: str | None = "vivid"
    user: str | None = None
    advanced_params: AdvancedImageParams | None = None
    # Internal flag for benchmark mode - set by API, preserved through serialization
    bench: bool = False

    @field_validator("size", mode="before")
    @classmethod
    def normalize_size(cls, v: object) -> ImageSize:
        return normalize_image_size(v)


class BenchImageGenerationTaskParams(ImageGenerationTaskParams):
    bench: bool = True


class ImageEditsTaskParams(BaseModel):
    """Internal task params for image-editing requests."""

    image_data: str = ""  # Base64-encoded image (empty when using chunked transfer)
    total_input_chunks: int = 0
    prompt: str
    model: str
    n: int | None = 1
    quality: Literal["high", "medium", "low"] | None = "medium"
    output_format: Literal["png", "jpeg", "webp"] = "png"
    response_format: Literal["url", "b64_json"] | None = "b64_json"
    size: ImageSize = "auto"
    image_strength: float | None = 0.7
    stream: bool = False
    partial_images: int | None = 0
    advanced_params: AdvancedImageParams | None = None
    bench: bool = False

    @field_validator("size", mode="before")
    @classmethod
    def normalize_size(cls, v: object) -> ImageSize:
        return normalize_image_size(v)

    def __repr_args__(self) -> Generator[tuple[str, Any], None, None]:
        for name, value in super().__repr_args__():  # pyright: ignore[reportAny]
            if name == "image_data":
                yield name, f"<{len(self.image_data)} chars>"
            elif name is not None:
                yield name, value


class ImageData(BaseModel):
    b64_json: str | None = None
    url: str | None = None
    revised_prompt: str | None = None

    def __repr_args__(self) -> Generator[tuple[str, Any], None, None]:
        for name, value in super().__repr_args__():  # pyright: ignore[reportAny]
            if name == "b64_json" and self.b64_json is not None:
                yield name, f"<{len(self.b64_json)} chars>"
            elif name is not None:
                yield name, value


class ImageGenerationResponse(BaseModel):
    created: int = Field(default_factory=lambda: int(time.time()))
    data: list[ImageData]


# ── Embeddings ──────────────────────────────────────────


class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: int | None = None
    user: str | None = None


class EmbeddingObject(BaseModel):
    object: str = "embedding"
    index: int
    embedding: list[float] | str


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingObject]
    model: str
    usage: EmbeddingUsage


class BenchImageGenerationResponse(ImageGenerationResponse):
    generation_stats: ImageGenerationStats | None = None
    power_usage: PowerUsage | None = None


class ImageListItem(BaseModel, frozen=True):
    image_id: str
    url: str
    content_type: str
    expires_at: float


class ImageListResponse(BaseModel, frozen=True):
    data: list[ImageListItem]


class StartDownloadParams(CamelCaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "targetNodeId": "12D3KooWExampleNodeId",
                "shardMetadata": {
                    "TensorShardMetadata": {
                        "modelCard": {
                            "modelId": "mlx-community/Llama-3.2-1B-Instruct-4bit",
                            "storageSize": {"inBytes": 2147483648},
                            "nLayers": 32,
                            "hiddenSize": 2048,
                            "supportsTensor": True,
                            "tasks": ["TextGeneration"],
                        },
                        "deviceRank": 0,
                        "worldSize": 1,
                        "startLayer": 0,
                        "endLayer": 32,
                        "nLayers": 32,
                    }
                },
            }
        }
    )

    target_node_id: NodeId
    shard_metadata: ShardMetadata


class StartDownloadResponse(CamelCaseModel):
    command_id: CommandId


class DeleteDownloadResponse(CamelCaseModel):
    command_id: CommandId


class PurgeStagingRequest(CamelCaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {"modelId": "mlx-community/Llama-3.2-1B-Instruct-4bit"}
        }
    )

    model_id: str | None = None


class PurgeStagingResponse(CamelCaseModel):
    command_id: CommandId
    message: str


class TraceEventResponse(CamelCaseModel):
    name: str
    start_us: int
    duration_us: int
    rank: int
    category: str


class TraceResponse(CamelCaseModel):
    task_id: str
    traces: list[TraceEventResponse]


class TraceCategoryStats(CamelCaseModel):
    total_us: int
    count: int
    min_us: int
    max_us: int
    avg_us: float


class TraceRankStats(CamelCaseModel):
    by_category: dict[str, TraceCategoryStats]


class TraceStatsResponse(CamelCaseModel):
    task_id: str
    total_wall_time_us: int
    by_category: dict[str, TraceCategoryStats]
    by_rank: dict[int, TraceRankStats]


class TraceListItem(CamelCaseModel):
    task_id: str
    created_at: str
    file_size: int


class TraceListResponse(CamelCaseModel):
    traces: list[TraceListItem]


class DeleteTracesRequest(CamelCaseModel):
    model_config = ConfigDict(
        json_schema_extra={"example": {"taskIds": ["chatcmpl-123", "chatcmpl-456"]}}
    )

    task_ids: list[str]


class DeleteTracesResponse(CamelCaseModel):
    deleted: list[str]
    not_found: list[str]
