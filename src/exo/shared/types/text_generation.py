"""Canonical internal type for text generation task parameters.

All external API formats (Chat Completions, Claude Messages, OpenAI Responses)
are converted to TextGenerationTaskParams at the API boundary via adapters.
"""

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from exo.shared.types.common import ModelId

if TYPE_CHECKING:
    from exo.shared.models.capabilities import ResolvedCapabilityProfile

MessageRole = Literal["user", "assistant", "system", "developer"]
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]


def resolve_reasoning_params(
    reasoning_effort: ReasoningEffort | None,
    enable_thinking: bool | None,
    capability_profile: "ResolvedCapabilityProfile | None" = None,
) -> tuple[ReasoningEffort | None, bool | None]:
    """
    enable_thinking=True -> use the profile's default enabled effort
    enable_thinking=False -> use the profile's disabled effort
    reasoning_effort="none" -> normalize to the profile's disabled effort and disable thinking
    reasoning_effort=<anything else> -> enable thinking unless it already matches the disabled effort
    """
    resolved_effort: ReasoningEffort | None = reasoning_effort
    resolved_thinking: bool | None = enable_thinking
    enabled_effort = (
        capability_profile.default_reasoning_effort
        if capability_profile is not None
        else "medium"
    )
    disabled_effort = (
        capability_profile.disabled_reasoning_effort
        if capability_profile is not None
        else "none"
    )

    if reasoning_effort is None and enable_thinking is not None:
        resolved_effort = enabled_effort if enable_thinking else disabled_effort

    if enable_thinking is None and reasoning_effort is not None:
        if reasoning_effort == "none":
            resolved_effort = disabled_effort
            resolved_thinking = False
        else:
            resolved_thinking = reasoning_effort != disabled_effort

    return resolved_effort, resolved_thinking


class InputMessage(BaseModel, frozen=True):
    """Internal message for text generation pipelines."""

    role: MessageRole
    content: str


class TextGenerationTaskParams(BaseModel, frozen=True):
    """Canonical internal task params for text generation.

    Every API adapter converts its wire type into this before handing
    off to the master/worker pipeline.
    """

    model: ModelId
    input: list[InputMessage]
    instructions: str | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    bench: bool = False
    top_k: int | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    chat_template_messages: list[dict[str, Any]] | None = None
    reasoning_effort: ReasoningEffort | None = None
    enable_thinking: bool | None = None
    logprobs: bool = False
    top_logprobs: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    repetition_context_size: int | None = None
    images: list[str] = Field(default_factory=list)
    image_hashes: dict[int, str] = Field(default_factory=dict)
    total_input_chunks: int = 0
    image_count: int = 0
