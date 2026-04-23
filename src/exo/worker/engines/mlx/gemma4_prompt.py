"""Gemma 4 prompt rendering helpers.

This module follows the Gemma 4 chat structure used by the reference Hugging
Face template and Ollama's dedicated Gemma 4 renderer, while preserving one
intentional warmup-only divergence: distributed warmup can suppress the empty
synthetic thought-channel suffix because that shape has wedged our distributed
MLX warmup path. Real inference keeps the reference suffix so generation begins
at the expected visible assistant-content boundary.
"""

from typing import cast

GemmaContentPart = dict[str, object]
GemmaMessage = dict[str, object]


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


def _render_gemma4_content(content: object, role: str) -> str:
    """Render one Gemma 4 message body."""
    if isinstance(content, str):
        return strip_gemma4_thinking(content) if role == "model" else content.strip()

    if not isinstance(content, list):
        return str(content).strip()

    parts: list[str] = []
    for item in cast(list[object], content):
        if not isinstance(item, dict):
            continue
        part = cast(GemmaContentPart, item)
        item_type = str(part.get("type", ""))
        if item_type == "text":
            text = str(part.get("text", ""))
            parts.append(strip_gemma4_thinking(text) if role == "model" else text.strip())
        elif item_type == "image":
            label = str(part.get("label", "")).strip()
            label_prefix = f"\n\n{label}:\n" if label else "\n\n"
            parts.append(f"{label_prefix}<|image|>\n\n")
        elif item_type == "audio":
            parts.append("<|audio|>")
        elif item_type == "video":
            parts.append("\n\n<|video|>\n\n")
    return "".join(parts)


def render_gemma4_prompt(
    messages: list[GemmaMessage],
    *,
    add_generation_prompt: bool,
    enable_thinking: bool | None = None,
    suppress_empty_thought_channel: bool = False,
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
        prompt_parts.append("<|turn>model\n")
        if not enable_thinking and not suppress_empty_thought_channel:
            # Gemma 4 expects the assistant turn to pass through the thought
            # channel boundary before emitting visible answer text. Keep the
            # reference suffix for normal inference so generation does not
            # begin at a raw ``<|channel>`` marker.
            prompt_parts.append("<|channel>thought\n<channel|>")

    return "".join(prompt_parts)
