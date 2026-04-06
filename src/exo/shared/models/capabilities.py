"""Runtime capability resolution derived from model cards.

This module keeps model cards as the persisted declarative source of truth while
providing a normalized runtime profile that inference code can consume without
sprinkling optional-field checks throughout the hot path.
"""

from typing import TYPE_CHECKING

from exo.shared.models.model_cards import (
    ModelCard,
    ModelId,
    OutputParserType,
    PromptRendererType,
    ReasoningFormat,
    ToolCallFormat,
    get_card,
)
from exo.shared.types.text_generation import ReasoningEffort, TextGenerationTaskParams
from exo.utils.pydantic_ext import FrozenModel

if TYPE_CHECKING:
    from mlx_lm.tokenizer_utils import TokenizerWrapper


class ResolvedCapabilityProfile(FrozenModel):
    """Normalized runtime behavior derived from a model card and runtime facts."""

    family: str = "generic"
    supports_thinking: bool = False
    supports_thinking_toggle: bool = False
    supports_thinking_budget: bool = False
    default_reasoning_effort: ReasoningEffort = "medium"
    disabled_reasoning_effort: ReasoningEffort = "none"
    thinking_format: ReasoningFormat = ReasoningFormat.None_
    supports_image_input: bool = False
    supports_audio_input: bool = False
    supports_tool_calling: bool = False
    tool_call_format: ToolCallFormat = ToolCallFormat.Generic
    prompt_renderer: PromptRendererType = PromptRendererType.Tokenizer
    output_parser: OutputParserType = OutputParserType.Generic
    supports_native_multimodal: bool = False


def _infer_family(model_id: ModelId, model_card: ModelCard | None) -> str:
    if model_card is not None and model_card.family:
        return model_card.family
    return model_id.normalize().split("-", 1)[0]


def _is_gemma4_family(profile_family: str, normalized_model_id: str) -> bool:
    return (
        "gemma-4" in normalized_model_id
        or "gemma4" in normalized_model_id
        or profile_family in {"gemma4", "gemma-4"}
    )


def _is_deepseek_v32_family(profile_family: str, normalized_model_id: str) -> bool:
    return (
        "deepseek-v3.2" in normalized_model_id
        or profile_family in {"deepseek-v3.2", "deepseek_v32"}
    )


def _is_gpt_oss_family(profile_family: str, normalized_model_id: str) -> bool:
    return profile_family == "gpt-oss" or any(
        marker in normalized_model_id for marker in ("gpt-oss", "gpt_oss")
    )


def resolve_model_capability_profile(
    model_id: ModelId,
    *,
    model_card: ModelCard | None = None,
    tokenizer: "TokenizerWrapper | None" = None,
    task_params: TextGenerationTaskParams | None = None,
) -> ResolvedCapabilityProfile:
    """Resolve runtime capabilities for one request.

    The resolver is intentionally conservative: if a card does not declare an
    advanced capability, we fall back to generic runtime behavior and only
    preserve the broad support we can infer from the existing card fields.
    """

    card = model_card or get_card(model_id)
    normalized_model_id = model_id.normalize().lower()
    profile_family = _infer_family(model_id, card)

    supports_image_input = bool(
        card is not None
        and (
            "vision" in card.capabilities
            or card.vision is not None
            or (
                card.modalities is not None
                and card.modalities.supports_native_multimodal is True
            )
        )
    )
    supports_thinking = bool(
        card is not None
        and ("thinking" in card.capabilities or card.reasoning is not None)
    )
    supports_tool_calling = bool(
        (tokenizer is not None and getattr(tokenizer, "has_tool_calling", False))
        or (
            card is not None
            and (
                (card.tooling is not None and card.tooling.supports_tool_calling is True)
                or (
                    card.tooling is not None
                    and card.tooling.tool_call_format is not None
                    and card.tooling.tool_call_format != ToolCallFormat.Generic
                )
            )
        )
    )
    thinking_format = (
        ReasoningFormat.TokenDelimited
        if tokenizer is not None and getattr(tokenizer, "has_thinking", False)
        else ReasoningFormat.None_
    )

    profile = ResolvedCapabilityProfile(
        family=profile_family,
        supports_thinking=supports_thinking,
        supports_image_input=supports_image_input,
        supports_tool_calling=supports_tool_calling,
        thinking_format=thinking_format,
        supports_native_multimodal=supports_image_input,
    )

    # Family-specific defaults preserve current behavior until cards opt in to
    # richer declarations. Explicit card fields override these defaults below.
    if _is_gemma4_family(profile.family, normalized_model_id):
        prompt_renderer = PromptRendererType.Tokenizer
        if task_params is None or (
            task_params.chat_template_messages is not None and not task_params.tools
        ):
            prompt_renderer = PromptRendererType.Gemma4

        profile = profile.model_copy(
            update={
                "supports_thinking": True,
                "supports_thinking_toggle": True,
                "thinking_format": ReasoningFormat.ChannelDelimited,
                "prompt_renderer": prompt_renderer,
                "output_parser": OutputParserType.Gemma4,
                "tool_call_format": ToolCallFormat.Gemma4,
                "supports_native_multimodal": supports_image_input,
            }
        )
    elif _is_deepseek_v32_family(profile.family, normalized_model_id):
        profile = profile.model_copy(
            update={
                "supports_thinking": True,
                "supports_thinking_toggle": True,
                "prompt_renderer": PromptRendererType.Dsml,
                "output_parser": OutputParserType.DeepseekV32,
                "tool_call_format": ToolCallFormat.Dsml,
            }
        )
    elif _is_gpt_oss_family(profile.family, normalized_model_id):
        profile = profile.model_copy(
            update={
                "supports_tool_calling": True,
                "output_parser": OutputParserType.GptOss,
                "tool_call_format": ToolCallFormat.GptOss,
            }
        )

    if card is not None and card.reasoning is not None:
        updates: dict[str, object] = {}
        if card.reasoning.supports_toggle is not None:
            updates["supports_thinking_toggle"] = card.reasoning.supports_toggle
        if card.reasoning.supports_budget is not None:
            updates["supports_thinking_budget"] = card.reasoning.supports_budget
        if card.reasoning.format is not None:
            updates["thinking_format"] = card.reasoning.format
        if card.reasoning.default_effort is not None:
            updates["default_reasoning_effort"] = card.reasoning.default_effort
        if card.reasoning.disabled_effort is not None:
            updates["disabled_reasoning_effort"] = card.reasoning.disabled_effort
        if updates:
            profile = profile.model_copy(update=updates)

    if card is not None and card.modalities is not None:
        updates = {}
        if card.modalities.supports_audio_input is not None:
            updates["supports_audio_input"] = card.modalities.supports_audio_input
        if card.modalities.supports_native_multimodal is not None:
            updates["supports_native_multimodal"] = (
                card.modalities.supports_native_multimodal
            )
        if updates:
            profile = profile.model_copy(update=updates)

    if card is not None and card.tooling is not None:
        updates = {}
        if card.tooling.supports_tool_calling is not None:
            updates["supports_tool_calling"] = card.tooling.supports_tool_calling
        if card.tooling.tool_call_format is not None:
            updates["tool_call_format"] = card.tooling.tool_call_format
        if updates:
            profile = profile.model_copy(update=updates)

    if card is not None and card.runtime is not None:
        updates = {}
        if card.runtime.prompt_renderer is not None:
            updates["prompt_renderer"] = card.runtime.prompt_renderer
        if card.runtime.output_parser is not None:
            updates["output_parser"] = card.runtime.output_parser
        if updates:
            profile = profile.model_copy(update=updates)

    # Preserve the existing fallback for tool-enabled Gemma 4 requests until we
    # land the full prompt grammar. Cards can declare Gemma 4 behavior, but the
    # runtime must still stay conservative here.
    if (
        task_params is not None
        and task_params.tools
        and profile.prompt_renderer == PromptRendererType.Gemma4
    ):
        profile = profile.model_copy(
            update={"prompt_renderer": PromptRendererType.Tokenizer}
        )

    return profile
