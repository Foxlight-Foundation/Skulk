# pyright: reportPrivateUsage=false
"""Tests for derived model tags exposed by the API."""

from exo.api.main import API
from exo.shared.models.model_cards import (
    ModelCard,
    ModelTask,
    OutputParserType,
    PromptRendererType,
    ReasoningCardConfig,
    ReasoningFormat,
    RuntimeCapabilityCardConfig,
    ToolCallFormat,
    ToolingCardConfig,
)
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory


def test_model_tags_include_vision() -> None:
    """Vision-capable models should expose a `vision` display tag."""
    card = ModelCard(
        model_id=ModelId("google/gemma-4b-it"),
        storage_size=Memory.from_bytes(1024),
        n_layers=1,
        hidden_size=1,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        capabilities=["text", "vision", "thinking"],
        quantization="4bit",
    )

    assert API._model_tags(card) == ["thinking", "vision", "tensor"]


def test_model_list_entry_exposes_declared_and_resolved_capabilities() -> None:
    """Model list entries should expose refined declared and resolved metadata."""
    card = ModelCard(
        model_id=ModelId("mlx-community/gemma-4-26b-a4b-it-4bit"),
        storage_size=Memory.from_bytes(1024),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        capabilities=["text", "vision", "thinking"],
        family="gemma",
        reasoning=ReasoningCardConfig(
            supports_toggle=True,
            format=ReasoningFormat.ChannelDelimited,
        ),
        runtime=RuntimeCapabilityCardConfig(
            prompt_renderer=PromptRendererType.Gemma4,
            output_parser=OutputParserType.Gemma4,
        ),
    )

    entry = API._model_list_entry(card)

    assert entry.reasoning is not None
    assert entry.reasoning.supports_toggle is True
    assert entry.runtime is not None
    assert entry.runtime.prompt_renderer == "gemma4"
    assert entry.resolved_capabilities is not None
    assert entry.resolved_capabilities.supports_thinking_toggle is True
    assert entry.resolved_capabilities.prompt_renderer == "gemma4"


def test_model_list_entry_exposes_gpt_oss_runtime_capabilities() -> None:
    """Model list entries should surface tool/runtime metadata for GPT-OSS cards."""
    card = ModelCard(
        model_id=ModelId("mlx-community/gpt-oss-20b-MXFP4-Q8"),
        storage_size=Memory.from_bytes(1024),
        n_layers=1,
        hidden_size=1,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        capabilities=["text", "thinking"],
        family="gpt-oss",
        tooling=ToolingCardConfig(
            supports_tool_calling=True,
            tool_call_format=ToolCallFormat.GptOss,
        ),
        runtime=RuntimeCapabilityCardConfig(
            output_parser=OutputParserType.GptOss,
        ),
    )

    entry = API._model_list_entry(card)

    assert entry.tooling is not None
    assert entry.tooling.supports_tool_calling is True
    assert entry.resolved_capabilities is not None
    assert entry.resolved_capabilities.supports_tool_calling is True
    assert entry.resolved_capabilities.tool_call_format == "gpt_oss"
    assert entry.resolved_capabilities.output_parser == "gpt_oss"


def test_model_list_entry_keeps_legacy_cards_compatible() -> None:
    """Legacy cards without advanced sections should still serialize cleanly."""
    card = ModelCard(
        model_id=ModelId("mlx-community/Qwen3.5-9B-4bit"),
        storage_size=Memory.from_bytes(1024),
        n_layers=1,
        hidden_size=1,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        capabilities=["text", "thinking"],
    )

    entry = API._model_list_entry(card)

    assert entry.reasoning is None
    assert entry.modalities is None
    assert entry.tooling is None
    assert entry.runtime is None
    assert entry.resolved_capabilities is not None
    assert entry.resolved_capabilities.prompt_renderer == "tokenizer"
    assert entry.resolved_capabilities.output_parser == "generic"


def test_model_list_entry_exposes_deepseek_family_defaults_without_card_extensions() -> None:
    """Resolved capabilities should expose DeepSeek family defaults even without advanced card sections."""
    card = ModelCard(
        model_id=ModelId("mlx-community/DeepSeek-V3.2-4bit"),
        storage_size=Memory.from_bytes(1024),
        n_layers=1,
        hidden_size=1,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        capabilities=["text", "thinking"],
        family="deepseek-v3.2",
    )

    entry = API._model_list_entry(card)

    assert entry.reasoning is None
    assert entry.runtime is None
    assert entry.resolved_capabilities is not None
    assert entry.resolved_capabilities.supports_thinking_toggle is True
    assert entry.resolved_capabilities.supports_tool_calling is True
    assert entry.resolved_capabilities.prompt_renderer == "dsml"
    assert entry.resolved_capabilities.output_parser == "deepseek_v32"
    assert entry.resolved_capabilities.tool_call_format == "dsml"


def test_model_list_entry_honors_coarse_thinking_toggle_capability() -> None:
    """Legacy coarse capability tags should still drive toggle support in the API contract."""
    card = ModelCard(
        model_id=ModelId("mlx-community/Qwen3.5-122B-A10B-4bit"),
        storage_size=Memory.from_bytes(1024),
        n_layers=1,
        hidden_size=1,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        capabilities=["text", "thinking", "thinking_toggle"],
        family="qwen",
    )

    entry = API._model_list_entry(card)

    assert entry.reasoning is None
    assert entry.resolved_capabilities is not None
    assert entry.resolved_capabilities.supports_thinking is True
    assert entry.resolved_capabilities.supports_thinking_toggle is True


def test_model_list_entry_serializes_declared_capabilities_in_snake_case() -> None:
    """API JSON should keep declared capability sections in snake_case."""
    card = ModelCard(
        model_id=ModelId("mlx-community/gemma-4-26b-a4b-it-4bit"),
        storage_size=Memory.from_bytes(1024),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        capabilities=["text", "vision", "thinking"],
        family="gemma",
        reasoning=ReasoningCardConfig(
            supports_toggle=True,
            supports_budget=True,
            format=ReasoningFormat.ChannelDelimited,
        ),
        tooling=ToolingCardConfig(
            supports_tool_calling=True,
            tool_call_format=ToolCallFormat.Gemma4,
        ),
        runtime=RuntimeCapabilityCardConfig(
            prompt_renderer=PromptRendererType.Gemma4,
            output_parser=OutputParserType.Gemma4,
        ),
    )

    payload = API._model_list_entry(card).model_dump(by_alias=True)

    assert payload["reasoning"]["supports_toggle"] is True
    assert payload["reasoning"]["supports_budget"] is True
    assert payload["tooling"]["supports_tool_calling"] is True
    assert payload["tooling"]["tool_call_format"] == "gemma4"
    assert payload["runtime"]["prompt_renderer"] == "gemma4"
