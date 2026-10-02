from typing import cast

from mlx_lm.tokenizer_utils import TokenizerWrapper

from skulk.worker.engines.mlx import vision as vision_module
from skulk.worker.engines.mlx.gemma4_prompt import render_gemma4_prompt


def test_render_gemma4_prompt_appends_empty_thought_channel_when_thinking_disabled():
    prompt = render_gemma4_prompt(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ],
        add_generation_prompt=True,
        enable_thinking=False,
    )

    assert prompt == (
        "<bos>"
        "<|turn>system\n"
        "You are helpful."
        "<turn|>\n"
        "<|turn>user\n"
        "Hello"
        "<turn|>\n"
        "<|turn>model\n"
        "<|channel>thought\n"
        "<channel|>"
    )


def test_render_gemma4_prompt_can_suppress_empty_thought_channel_for_warmup():
    prompt = render_gemma4_prompt(
        [{"role": "user", "content": "Hello"}],
        add_generation_prompt=True,
        enable_thinking=False,
        suppress_empty_thought_channel=True,
    )

    assert prompt.endswith("<|turn>model\n")
    assert "<|channel>thought" not in prompt
    assert "<channel|>" not in prompt


def test_render_gemma4_prompt_keeps_think_marker_when_thinking_enabled():
    prompt = render_gemma4_prompt(
        [{"role": "user", "content": "Hello"}],
        add_generation_prompt=True,
        enable_thinking=True,
    )

    assert prompt.startswith("<bos><|turn>system\n<|think|>\n<turn|>\n")
    assert prompt.endswith("<|turn>model\n")


def test_render_gemma4_prompt_includes_optional_image_labels():
    prompt = render_gemma4_prompt(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "compare"},
                    {"type": "image", "label": "Image 1"},
                ],
            }
        ],
        add_generation_prompt=False,
    )

    assert "compare\nImage 1:\n<|image|>" in prompt


def test_render_gemma4_prompt_unlabeled_image_matches_reference_shape():
    """Unlabeled images sit flush against adjacent text — that's the reference.

    The Gemma 4 chat template is trained with no separator between ``<|image|>``
    and the surrounding text in the unlabeled case (which is the common
    single-image shape). Earlier review feedback suggested inserting a
    newline boundary; doing so breaks ``test_multimodal_prompt_matches_reference_shape``
    because the model was trained on the no-separator form. This test pins
    that expectation so future "tidy up the rendering" attempts don't
    silently regress multimodal instruction fidelity.
    """
    prompt = render_gemma4_prompt(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "before"},
                    {"type": "image"},
                    {"type": "text", "text": "after"},
                ],
            }
        ],
        add_generation_prompt=False,
    )

    # Reference shape: ``before<|image|>after`` with no inserted boundary.
    assert "before<|image|>after" in prompt


class _GemmaPromptTokenizer:
    def decode(self, _token_ids: list[int]) -> str:
        raise AssertionError("Gemma 4 prompt debug test should not decode BOI/EOI")


def test_build_vision_prompt_debug_records_raw_placeholder_offsets():
    built = vision_module._build_vision_prompt_with_debug(  # pyright: ignore[reportPrivateUsage]
        cast(TokenizerWrapper, cast(object, _GemmaPromptTokenizer())),
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "before"},
                    {"type": "image"},
                    {"type": "text", "text": "after"},
                ],
            }
        ],
        [3],
        "<|image|>",
        model_type="gemma4",
    )

    raw_placeholder = built.raw_prompt.index("<|image|>")
    assert built.raw_prompt.count("<|image|>") == 1
    assert built.prompt.count("<|image|>") == 3
    assert built.debug.raw_image_placeholder_positions == [raw_placeholder]
    assert built.debug.tokens_per_image == [3]
    assert built.debug.attrs()["raw_image_placeholder_offsets"] == [
        str(raw_placeholder)
    ]


class _RawTemplateTokenizer:
    """Renders a raw chat template with one image placeholder and decodes markers."""

    def apply_chat_template(
        self, _messages: list[dict[str, object]], **_kwargs: object
    ) -> str:
        return (
            "<bos><start_of_turn>user\n<image_soft_token>What is shown?"
            "<end_of_turn>\n<start_of_turn>model\n"
        )

    def decode(self, token_ids: list[int]) -> str:
        return {1: "<start_of_image>", 2: "<end_of_image>"}[token_ids[0]]


def _image_message() -> list[dict[str, object]]:
    return [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": "What is shown?"}],
        }
    ]


def test_gemma3n_image_block_gets_the_blank_lines_its_processor_adds():
    """Gemma 3n reads small text badly unless the image block is framed like HF's."""
    built = vision_module._build_vision_prompt_with_debug(  # pyright: ignore[reportPrivateUsage]
        cast(TokenizerWrapper, cast(object, _RawTemplateTokenizer())),
        _image_message(),
        [3],
        "<image_soft_token>",
        model_type="gemma3n",
        boi_token_id=1,
        eoi_token_id=2,
    )

    assert built.prompt == (
        "<bos><start_of_turn>user\n\n\n<start_of_image>"
        + "<image_soft_token>" * 3
        + "<end_of_image>\n\nWhat is shown?<end_of_turn>\n<start_of_turn>model\n"
    )


def test_image_block_stays_bare_for_families_without_blank_line_framing():
    """Families whose processor adds no blank lines keep the plain BOI/EOI block."""
    built = vision_module._build_vision_prompt_with_debug(  # pyright: ignore[reportPrivateUsage]
        cast(TokenizerWrapper, cast(object, _RawTemplateTokenizer())),
        _image_message(),
        [3],
        "<image_soft_token>",
        model_type="qwen3_vl",
        boi_token_id=1,
        eoi_token_id=2,
    )

    assert built.prompt == (
        "<bos><start_of_turn>user\n<start_of_image>"
        + "<image_soft_token>" * 3
        + "<end_of_image>What is shown?<end_of_turn>\n<start_of_turn>model\n"
    )
