from exo.shared.models.capabilities import (
    ResolvedCapabilityProfile,
    resolve_model_capability_profile,
)
from exo.shared.models.model_cards import (
    BuiltinToolType,
    ModalitiesCardConfig,
    ModelCard,
    ModelId,
    ModelTask,
    OutputParserType,
    PromptRendererType,
    ReasoningCardConfig,
    ReasoningFormat,
    RuntimeCapabilityCardConfig,
    ToolCallFormat,
    ToolingCardConfig,
)
from exo.shared.types.memory import Memory
from exo.shared.types.text_generation import (
    InputMessage,
    TextGenerationTaskParams,
    resolve_reasoning_params,
)


def _base_model_card(model_id: str) -> ModelCard:
    return ModelCard(
        model_id=ModelId(model_id),
        storage_size=Memory.from_mb(100),
        n_layers=10,
        hidden_size=1024,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        family="gemma",
        capabilities=["text", "vision", "thinking"],
    )


def test_resolve_model_capability_profile_uses_extended_model_card_fields() -> None:
    card = _base_model_card("example/gemma-test").model_copy(
        update={
            "reasoning": ReasoningCardConfig(
                supports_toggle=True,
                supports_budget=True,
                format=ReasoningFormat.ChannelDelimited,
                default_effort="high",
                disabled_effort="none",
            ),
            "modalities": ModalitiesCardConfig(
                supports_audio_input=True,
                supports_native_multimodal=True,
            ),
            "tooling": ToolingCardConfig(
                supports_tool_calling=True,
                tool_call_format=ToolCallFormat.Gemma4,
            ),
            "runtime": RuntimeCapabilityCardConfig(
                prompt_renderer=PromptRendererType.Gemma4,
                output_parser=OutputParserType.Gemma4,
            ),
        }
    )

    profile = resolve_model_capability_profile(card.model_id, model_card=card)

    assert profile.supports_thinking_toggle is True
    assert profile.supports_thinking_budget is True
    assert profile.thinking_format == ReasoningFormat.ChannelDelimited
    assert profile.default_reasoning_effort == "high"
    assert profile.supports_audio_input is True
    assert profile.tool_call_format == ToolCallFormat.Gemma4
    assert profile.prompt_renderer == PromptRendererType.Gemma4
    assert profile.output_parser == OutputParserType.Gemma4


def test_resolve_model_capability_profile_keeps_gemma4_tool_fallback() -> None:
    card = _base_model_card("mlx-community/gemma-4-26b-a4b-it-4bit").model_copy(
        update={
            "runtime": RuntimeCapabilityCardConfig(
                prompt_renderer=PromptRendererType.Gemma4,
                output_parser=OutputParserType.Gemma4,
            )
        }
    )
    task_params = TextGenerationTaskParams(
        model=card.model_id,
        input=[InputMessage(role="user", content="hello")],
        chat_template_messages=[{"role": "user", "content": "hello"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup_weather",
                    "description": "Lookup weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )

    profile = resolve_model_capability_profile(
        card.model_id,
        model_card=card,
        task_params=task_params,
    )

    assert profile.supports_tool_calling is True
    assert profile.prompt_renderer == PromptRendererType.Tokenizer
    assert profile.output_parser == OutputParserType.Gemma4


def test_resolve_reasoning_params_uses_profile_defaults() -> None:
    profile = ResolvedCapabilityProfile(
        supports_thinking=True,
        supports_thinking_toggle=True,
        default_reasoning_effort="high",
        disabled_reasoning_effort="none",
    )

    assert resolve_reasoning_params(None, True, profile) == ("high", True)
    assert resolve_reasoning_params(None, False, profile) == ("none", False)


def test_resolve_reasoning_params_treats_none_as_disabled_even_for_custom_profiles() -> None:
    profile = ResolvedCapabilityProfile(
        supports_thinking=True,
        supports_thinking_toggle=True,
        default_reasoning_effort="high",
        disabled_reasoning_effort="minimal",
    )

    assert resolve_reasoning_params("none", None, profile) == ("minimal", False)
    assert resolve_reasoning_params("none", True, profile) == ("minimal", False)


def test_resolve_reasoning_params_ignores_toggle_inputs_for_non_toggleable_profiles() -> None:
    profile = ResolvedCapabilityProfile(
        supports_thinking=True,
        supports_thinking_toggle=False,
        default_reasoning_effort="high",
        disabled_reasoning_effort="minimal",
    )

    assert resolve_reasoning_params(None, False, profile) == (None, None)
    assert resolve_reasoning_params("minimal", None, profile) == ("minimal", None)
    assert resolve_reasoning_params("high", False, profile) == ("high", None)


def test_resolve_model_capability_profile_uses_safe_generic_fallback() -> None:
    card = ModelCard(
        model_id=ModelId("example/plain-text-model"),
        storage_size=Memory.from_mb(100),
        n_layers=10,
        hidden_size=1024,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        family="example",
        capabilities=["text"],
    )

    profile = resolve_model_capability_profile(card.model_id, model_card=card)

    assert profile.family == "example"
    assert profile.supports_thinking is False
    assert profile.supports_thinking_toggle is False
    assert profile.supports_image_input is False
    assert profile.supports_tool_calling is False
    assert profile.prompt_renderer == PromptRendererType.Tokenizer
    assert profile.output_parser == OutputParserType.Generic


def test_resolve_model_capability_profile_infers_family_from_short_model_id() -> None:
    profile = resolve_model_capability_profile(
        ModelId("mlx-community/gemma-4-custom"),
        model_card=None,
    )

    assert profile.family == "gemma"


def test_resolve_model_capability_profile_honors_coarse_thinking_toggle_capability() -> None:
    card = ModelCard(
        model_id=ModelId("mlx-community/Qwen3.5-122B-A10B-4bit"),
        storage_size=Memory.from_mb(100),
        n_layers=10,
        hidden_size=1024,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        family="qwen",
        capabilities=["text", "thinking", "thinking_toggle"],
    )

    profile = resolve_model_capability_profile(card.model_id, model_card=card)

    assert profile.supports_thinking is True
    assert profile.supports_thinking_toggle is True
    assert profile.prompt_renderer == PromptRendererType.Tokenizer
    assert profile.output_parser == OutputParserType.Generic


def test_resolve_model_capability_profile_honors_nemotron_reasoning_metadata() -> None:
    card = ModelCard(
        model_id=ModelId("mlx-community/NVIDIA-Nemotron-Nano-9B-v2-4bits"),
        storage_size=Memory.from_mb(100),
        n_layers=10,
        hidden_size=1024,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        family="nemotron",
        capabilities=["text", "thinking", "thinking_toggle"],
        reasoning=ReasoningCardConfig(
            supports_toggle=True,
            format=ReasoningFormat.TokenDelimited,
            default_effort="medium",
            disabled_effort="none",
        ),
    )

    profile = resolve_model_capability_profile(card.model_id, model_card=card)

    assert profile.supports_thinking is True
    assert profile.supports_thinking_toggle is True
    assert profile.thinking_format == ReasoningFormat.TokenDelimited
    assert profile.default_reasoning_effort == "medium"
    assert profile.disabled_reasoning_effort == "none"


def test_resolve_model_capability_profile_honors_qwen35_reasoning_metadata() -> None:
    card = ModelCard(
        model_id=ModelId("mlx-community/Qwen3.5-9B-4bit"),
        storage_size=Memory.from_mb(100),
        n_layers=10,
        hidden_size=1024,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        family="qwen",
        capabilities=["text", "thinking", "thinking_toggle"],
        reasoning=ReasoningCardConfig(
            supports_toggle=True,
            format=ReasoningFormat.TokenDelimited,
            default_effort="medium",
            disabled_effort="none",
        ),
    )

    profile = resolve_model_capability_profile(card.model_id, model_card=card)

    assert profile.supports_thinking is True
    assert profile.supports_thinking_toggle is True
    assert profile.thinking_format == ReasoningFormat.TokenDelimited
    assert profile.default_reasoning_effort == "medium"
    assert profile.disabled_reasoning_effort == "none"


def test_resolve_model_capability_profile_honors_deepseek_v32_metadata() -> None:
    card = ModelCard(
        model_id=ModelId("mlx-community/DeepSeek-V3.2-4bit"),
        storage_size=Memory.from_mb(100),
        n_layers=10,
        hidden_size=1024,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        family="deepseek",
        capabilities=["text", "thinking", "thinking_toggle"],
        reasoning=ReasoningCardConfig(
            supports_toggle=True,
            format=ReasoningFormat.TokenDelimited,
            default_effort="medium",
            disabled_effort="none",
        ),
        tooling=ToolingCardConfig(
            supports_tool_calling=True,
            tool_call_format=ToolCallFormat.Dsml,
        ),
        runtime=RuntimeCapabilityCardConfig(
            prompt_renderer=PromptRendererType.Dsml,
            output_parser=OutputParserType.DeepseekV32,
        ),
    )

    profile = resolve_model_capability_profile(card.model_id, model_card=card)

    assert profile.supports_thinking is True
    assert profile.supports_thinking_toggle is True
    assert profile.thinking_format == ReasoningFormat.TokenDelimited
    assert profile.supports_tool_calling is True
    assert profile.prompt_renderer == PromptRendererType.Dsml
    assert profile.output_parser == OutputParserType.DeepseekV32
    assert profile.tool_call_format == ToolCallFormat.Dsml


def test_resolve_model_capability_profile_uses_declared_tooling_without_model_id_match() -> None:
    card = ModelCard(
        model_id=ModelId("custom/open-model"),
        storage_size=Memory.from_mb(100),
        n_layers=10,
        hidden_size=1024,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        family="custom",
        capabilities=["text"],
        tooling=ToolingCardConfig(
            supports_tool_calling=True,
            tool_call_format=ToolCallFormat.GptOss,
        ),
        runtime=RuntimeCapabilityCardConfig(
            output_parser=OutputParserType.GptOss,
        ),
    )

    profile = resolve_model_capability_profile(card.model_id, model_card=card)

    assert profile.supports_tool_calling is True
    assert profile.tool_call_format == ToolCallFormat.GptOss
    assert profile.output_parser == OutputParserType.GptOss


def test_resolve_model_capability_profile_uses_gpt_oss_family_defaults() -> None:
    card = ModelCard(
        model_id=ModelId("custom/oss-20b"),
        storage_size=Memory.from_mb(100),
        n_layers=10,
        hidden_size=1024,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        family="gpt-oss",
        capabilities=["text", "thinking"],
    )

    profile = resolve_model_capability_profile(card.model_id, model_card=card)

    assert profile.family == "gpt-oss"
    assert profile.supports_thinking is True
    assert profile.supports_tool_calling is True
    assert profile.tool_call_format == ToolCallFormat.GptOss
    assert profile.output_parser == OutputParserType.GptOss


def test_resolve_model_capability_profile_exposes_builtin_tools() -> None:
    card = ModelCard(
        model_id=ModelId("mlx-community/gpt-oss-20b-MXFP4-Q8"),
        storage_size=Memory.from_mb(100),
        n_layers=10,
        hidden_size=1024,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        family="gpt-oss",
        capabilities=["text", "thinking"],
        tooling=ToolingCardConfig(
            supports_tool_calling=True,
            builtin_tools=[
                BuiltinToolType.WebSearch,
                BuiltinToolType.OpenUrl,
                BuiltinToolType.ExtractPage,
            ],
            tool_call_format=ToolCallFormat.GptOss,
        ),
    )

    profile = resolve_model_capability_profile(card.model_id, model_card=card)

    assert profile.builtin_tools == (
        BuiltinToolType.WebSearch,
        BuiltinToolType.OpenUrl,
        BuiltinToolType.ExtractPage,
    )


def test_resolve_model_capability_profile_uses_deepseek_v32_family_defaults() -> None:
    card = ModelCard(
        model_id=ModelId("custom/deepseek-compatible"),
        storage_size=Memory.from_mb(100),
        n_layers=10,
        hidden_size=1024,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        family="deepseek-v3.2",
        capabilities=["text", "thinking"],
    )

    profile = resolve_model_capability_profile(card.model_id, model_card=card)

    assert profile.supports_thinking is True
    assert profile.supports_thinking_toggle is True
    assert profile.supports_tool_calling is True
    assert profile.prompt_renderer == PromptRendererType.Dsml
    assert profile.output_parser == OutputParserType.DeepseekV32
    assert profile.tool_call_format == ToolCallFormat.Dsml


def test_resolve_model_capability_profile_keeps_native_multimodal_conservative_by_default() -> None:
    card = ModelCard(
        model_id=ModelId("mlx-community/vision-model"),
        storage_size=Memory.from_mb(100),
        n_layers=10,
        hidden_size=1024,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        family="vision",
        capabilities=["text", "vision"],
    )

    profile = resolve_model_capability_profile(card.model_id, model_card=card)

    assert profile.supports_image_input is True
    assert profile.supports_native_multimodal is False


# ---------------------------------------------------------------------------
# is_gemma4_family — consolidated detection predicate
# ---------------------------------------------------------------------------


def _bare_card(model_id: str, *, family: str = "") -> ModelCard:
    """Build a minimal card with no gemma4 hints declared."""
    return ModelCard(
        model_id=ModelId(model_id),
        storage_size=Memory.from_mb(100),
        n_layers=10,
        hidden_size=1024,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        family=family,
        capabilities=["text"],
    )


def test_is_gemma4_family_matches_card_with_gemma_4_in_id() -> None:
    """Gemma 4 cards from resources/ have ``gemma-4`` in the id and must match."""
    from exo.shared.models.capabilities import is_gemma4_family

    card = _bare_card("mlx-community/gemma-4-26b-a4b-it-4bit", family="gemma")
    assert is_gemma4_family(card) is True


def test_is_gemma4_family_matches_id_with_underscore_normalization() -> None:
    """``gemma_4`` and ``gemma-4`` are both valid Gemma 4 markers."""
    from exo.shared.models.capabilities import is_gemma4_family

    assert is_gemma4_family(_bare_card("mlx-community/gemma_4_e4b_it")) is True
    assert is_gemma4_family(_bare_card("mlx-community/gemma4-26b")) is True


def test_is_gemma4_family_matches_card_family_field_explicitly() -> None:
    """If a card doesn't have ``gemma-4`` in the id, family field still detects."""
    from exo.shared.models.capabilities import is_gemma4_family

    card = _bare_card("mlx-community/some-model", family="gemma4")
    assert is_gemma4_family(card) is True


def test_is_gemma4_family_matches_via_vision_model_type() -> None:
    """Vision-declared gemma4 model_type wins even without id/family hints."""
    from exo.shared.models.capabilities import is_gemma4_family
    from exo.shared.models.model_cards import VisionCardConfig

    card = _bare_card("mlx-community/relabelled-model")
    card = card.model_copy(
        update={
            "vision": VisionCardConfig(
                image_token_id=258880,
                model_type="gemma4",
                boi_token_id=255999,
                eoi_token_id=258882,
            )
        }
    )
    assert is_gemma4_family(card) is True


def test_is_gemma4_family_matches_via_runtime_prompt_renderer() -> None:
    """Cards declaring runtime.prompt_renderer = Gemma4 are gemma4."""
    from exo.shared.models.capabilities import is_gemma4_family

    card = _bare_card("mlx-community/relabelled-model")
    card = card.model_copy(
        update={
            "runtime": RuntimeCapabilityCardConfig(prompt_renderer=PromptRendererType.Gemma4)
        }
    )
    assert is_gemma4_family(card) is True


def test_is_gemma4_family_matches_via_runtime_output_parser() -> None:
    from exo.shared.models.capabilities import is_gemma4_family

    card = _bare_card("mlx-community/relabelled-model")
    card = card.model_copy(
        update={
            "runtime": RuntimeCapabilityCardConfig(output_parser=OutputParserType.Gemma4)
        }
    )
    assert is_gemma4_family(card) is True


def test_is_gemma4_family_matches_via_tooling_format() -> None:
    """Cards declaring tooling.tool_call_format = Gemma4 are gemma4."""
    from exo.shared.models.capabilities import is_gemma4_family

    card = _bare_card("mlx-community/relabelled-model")
    card = card.model_copy(
        update={"tooling": ToolingCardConfig(tool_call_format=ToolCallFormat.Gemma4)}
    )
    assert is_gemma4_family(card) is True


def test_is_gemma4_family_rejects_non_gemma_models() -> None:
    """Plain non-Gemma cards must not match."""
    from exo.shared.models.capabilities import is_gemma4_family

    for model_id in (
        "mlx-community/Qwen2.5-7B-Instruct",
        "mlx-community/Llama-3.3-70B",
        "mlx-community/DeepSeek-V3.2",
        "mlx-community/Gemma-3n-E4B-it",
        "mlx-community/Gemma-2-9b-it",
    ):
        card = _bare_card(model_id)
        assert is_gemma4_family(card) is False, (
            f"non-gemma4 card {model_id} should not be detected as gemma4"
        )


def test_is_gemma4_family_handles_none_card_with_id_fallback() -> None:
    """Detection should still work via id alone when no card is supplied."""
    from exo.shared.models.capabilities import is_gemma4_family

    assert (
        is_gemma4_family(model_id=ModelId("mlx-community/gemma-4-26b-a4b-it-4bit"))
        is True
    )
    assert (
        is_gemma4_family(model_id=ModelId("mlx-community/Llama-3.3-70B"))
        is False
    )


def test_is_gemma4_family_handles_no_inputs_at_all() -> None:
    """Defensive: both arguments None returns False without raising."""
    from exo.shared.models.capabilities import is_gemma4_family

    assert is_gemma4_family() is False
    assert is_gemma4_family(card=None, model_id=None) is False


def test_is_gemma4_family_resource_cards_all_match() -> None:
    """Every Gemma 4 model card shipped under resources/ must be detected."""
    import tomllib
    from pathlib import Path

    from exo.shared.models.capabilities import is_gemma4_family

    cards_dir = (
        Path(__file__).resolve().parents[4]
        / "resources"
        / "inference_model_cards"
    )
    gemma4_paths = list(cards_dir.glob("*gemma-4*.toml")) + list(
        cards_dir.glob("*gemma_4*.toml")
    )
    assert gemma4_paths, (
        "expected gemma-4 cards under resources/inference_model_cards/"
    )
    for path in gemma4_paths:
        with path.open("rb") as fp:
            data = tomllib.load(fp)
        card = ModelCard.model_validate(data)
        assert is_gemma4_family(card) is True, f"{path.name} should be gemma4"


def test_is_gemma4_family_resource_cards_other_families_dont_match() -> None:
    """Sanity: non-Gemma-4 cards under resources/ must not detect as gemma4."""
    import tomllib
    from pathlib import Path

    from exo.shared.models.capabilities import is_gemma4_family

    cards_dir = (
        Path(__file__).resolve().parents[4]
        / "resources"
        / "inference_model_cards"
    )
    non_gemma4_paths = [
        path
        for path in cards_dir.glob("*.toml")
        if "gemma-4" not in path.name and "gemma_4" not in path.name
    ]
    if not non_gemma4_paths:
        return  # No control set; skip silently rather than fail
    for path in non_gemma4_paths:
        with path.open("rb") as fp:
            data = tomllib.load(fp)
        card = ModelCard.model_validate(data)
        # Some non-gemma-4 cards may legitimately set tooling.tool_call_format
        # to something other than Gemma4 — those must not be detected as gemma4.
        assert is_gemma4_family(card) is False, (
            f"{path.name} should not match the gemma4 predicate"
        )
