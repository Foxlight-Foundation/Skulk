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
    Resolve public thinking controls into the canonical internal request form.

    The Phase 2 contract is:
    - models without thinking support ignore reasoning controls entirely
    - models without toggle support ignore explicit thinking/toggle overrides,
      but still preserve explicit non-disabled reasoning-effort hints
    - ``reasoning_effort="none"`` always normalizes to the profile's disabled effort
      and disables thinking for toggleable models
    - ``enable_thinking=False`` disables thinking for toggleable models
    - ``enable_thinking=True`` enables thinking using either the explicit
      non-disabled effort or the profile default effort
    - when only ``reasoning_effort`` is provided, it determines thinking on/off
      relative to the profile's disabled effort
    - when neither value is provided, the runtime uses the model's default behavior
    """
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
    supports_thinking = (
        capability_profile.supports_thinking
        if capability_profile is not None
        else True
    )
    supports_toggle = (
        capability_profile.supports_thinking_toggle
        if capability_profile is not None
        else True
    )

    if not supports_thinking:
        return None, None

    if not supports_toggle:
        if reasoning_effort is not None and reasoning_effort != "none":
            return reasoning_effort, None
        return None, None

    if reasoning_effort == "none" or enable_thinking is False:
        return disabled_effort, False

    if enable_thinking is True:
        if reasoning_effort is not None and reasoning_effort != disabled_effort:
            return reasoning_effort, True
        return enabled_effort, True

    if reasoning_effort is not None:
        return reasoning_effort, reasoning_effort != disabled_effort

    return None, None


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
