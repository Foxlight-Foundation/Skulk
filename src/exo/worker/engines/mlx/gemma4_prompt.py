"""Gemma 4 prompt rendering helpers.

This module mirrors the Gemma 4 chat structure used by the reference
Hugging Face template and Ollama's dedicated Gemma 4 renderer. We use it
to avoid relying on generic tokenizer chat templating for Gemma 4 because
we need exact control over the multimodal prompt the model sees.
"""

from typing import Any


def strip_gemma4_thinking(text: str) -> str:
    """Remove Gemma 4 thinking blocks from assistant history."""
    result: list[str] = []
    remaining = text
    while True:
        start = remaining.find("<|channel>")
        if start == -1:
            result.append(remaining)
            break
        result.append(remaining[:start])
        end = remaining.find("<channel|>", start)
        if end == -1:
            break
        remaining = remaining[end + len("<channel|>") :]
    return "".join(result).strip()


def _render_gemma4_content(content: Any, role: str) -> str:
    """Render one Gemma 4 message body."""
    if isinstance(content, str):
        return strip_gemma4_thinking(content) if role == "model" else content.strip()

    if not isinstance(content, list):
        return str(content).strip()

    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type", ""))
        if item_type == "text":
            text = str(item.get("text", ""))
            parts.append(strip_gemma4_thinking(text) if role == "model" else text.strip())
        elif item_type == "image":
            parts.append("\n\n<|image|>\n\n")
        elif item_type == "audio":
            parts.append("<|audio|>")
        elif item_type == "video":
            parts.append("\n\n<|video|>\n\n")
    return "".join(parts)


def render_gemma4_prompt(
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
    enable_thinking: bool | None = None,
) -> str:
    """Render a Gemma 4 prompt matching the reference chat template.

    The renderer intentionally stays narrow: it covers the text and
    multimodal message structure used by our current Gemma 4 requests.
    Tool-enabled Gemma 4 requests continue to use the generic template path
    until we port the full declaration/call grammar.
    """
    prompt_parts = ["<bos>"]
    loop_messages = messages

    has_system_message = bool(messages) and str(messages[0].get("role", "")) in {
        "system",
        "developer",
    }
    if has_system_message or enable_thinking:
        prompt_parts.append("<|turn>system\n")
        if enable_thinking:
            prompt_parts.append("<|think|>")
        if has_system_message:
            prompt_parts.append(_render_gemma4_content(messages[0].get("content", ""), "system"))
            loop_messages = messages[1:]
        prompt_parts.append("<turn|>\n")

    for message in loop_messages:
        role = "model" if str(message.get("role", "user")) == "assistant" else str(
            message.get("role", "user")
        )
        prompt_parts.append(f"<|turn>{role}\n")
        prompt_parts.append(_render_gemma4_content(message.get("content", ""), role))
        prompt_parts.append("<turn|>\n")

    if add_generation_prompt:
        # Gemma's published chat template appends an empty thought channel even
        # when thinking is disabled. We intentionally omit that suffix here
        # because it appears to trigger a pipeline warmup wedge in our MLX
        # distributed path, and real traffic succeeds without this synthetic
        # prefix.
        prompt_parts.append("<|turn>model\n")

    return "".join(prompt_parts)
