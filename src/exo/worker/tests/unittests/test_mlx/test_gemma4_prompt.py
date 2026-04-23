from exo.worker.engines.mlx.gemma4_prompt import render_gemma4_prompt


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

    assert prompt.startswith("<bos><|turn>system\n<|think|><turn|>\n")
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

    assert "compare\n\nImage 1:\n<|image|>\n\n" in prompt
