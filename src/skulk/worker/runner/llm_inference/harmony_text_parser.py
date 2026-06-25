"""Dependency-free streaming parser for the gpt-oss "harmony" response format.

The MLX engine parses gpt-oss output token-by-token with ``openai_harmony``'s
``StreamableParser`` (see ``model_output_parsers.parse_gpt_oss``). The llama.cpp
engine does not expose token ids: ``create_chat_completion`` hands back already
detokenized *string* deltas whose text still contains the literal harmony control
markers (``<|channel|>analysis<|message|>...<|end|><|start|>assistant<|channel|>``
``final<|message|>...``). Without parsing them the raw markers leak into the chat
output and the reasoning channel is never separated from the answer.

This module reparses that marker stream from strings so the llama.cpp runner can
reach parity with MLX: the ``analysis`` channel is flagged as reasoning, the
``final`` (and any other non-analysis) channel is content, and every control
marker is stripped. It is intentionally pure-Python with no MLX / openai_harmony
imports because it runs on non-Mac GPU nodes (e.g. AMD) where neither is present.
"""

from __future__ import annotations

from typing import final

# Harmony control markers. A message body runs from ``<|message|>`` up to the
# next of these; the channel name sits between ``<|channel|>`` and ``<|message|>``.
_CONTROL_TOKENS: tuple[str, ...] = (
    "<|start|>",
    "<|end|>",
    "<|return|>",
    "<|message|>",
    "<|channel|>",
    "<|call|>",
    "<|constrain|>",
)

# Markers that terminate a message body (return us to "between channels").
_BODY_END_TOKENS: frozenset[str] = frozenset(
    {"<|end|>", "<|start|>", "<|return|>", "<|call|>"}
)

_ANALYSIS_CHANNEL = "analysis"


@final
class HarmonyTextParser:
    """Incrementally strip harmony markers and split reasoning from content.

    Feed raw string deltas with :meth:`feed`; it returns a list of
    ``(text, is_thinking)`` emissions (often empty until a marker boundary or
    enough non-marker text accumulates). Call :meth:`flush` once the stream ends
    to drain any held-back tail. ``is_thinking`` is ``True`` only while inside the
    ``analysis`` channel.

    The parser starts in *content* mode (``is_thinking=False``) so that output
    which never emits a channel marker (e.g. a prompt template that already
    opened the ``final`` channel, or a non-harmony model) still passes its text
    through unchanged instead of being swallowed as header noise.
    """

    def __init__(self) -> None:
        self._buffer: str = ""
        # True while between ``<|message|>`` and the next body-terminating marker.
        self._in_body: bool = True
        # Accumulates the channel header text between ``<|channel|>`` and
        # ``<|message|>`` (the channel name, plus any recipient/constrain text).
        self._header: str = ""
        # Current channel name once a body is open; ``None`` means the default
        # (pre-marker) content body.
        self._channel: str | None = None

    @property
    def _is_thinking(self) -> bool:
        return self._channel == _ANALYSIS_CHANNEL

    def feed(self, text: str) -> list[tuple[str, bool]]:
        """Consume a raw delta, returning ``(text, is_thinking)`` emissions."""
        if text:
            self._buffer += text
        emissions: list[tuple[str, bool]] = []
        self._drain_complete_markers(emissions)
        self._drain_safe_text(emissions, final=False)
        return emissions

    def flush(self) -> list[tuple[str, bool]]:
        """Drain any remaining buffered text once the stream has ended."""
        emissions: list[tuple[str, bool]] = []
        self._drain_complete_markers(emissions)
        self._drain_safe_text(emissions, final=True)
        return emissions

    def _drain_complete_markers(self, emissions: list[tuple[str, bool]]) -> None:
        """Process every complete control marker currently in the buffer."""
        while True:
            best_index = -1
            best_token = ""
            for token in _CONTROL_TOKENS:
                index = self._buffer.find(token)
                if index != -1 and (best_index == -1 or index < best_index):
                    best_index = index
                    best_token = token
            if best_index == -1:
                return
            self._consume_text(self._buffer[:best_index], emissions)
            self._apply_marker(best_token)
            self._buffer = self._buffer[best_index + len(best_token) :]

    def _drain_safe_text(
        self, emissions: list[tuple[str, bool]], *, final: bool
    ) -> None:
        """Emit buffered text that cannot be part of a not-yet-complete marker.

        Mid-stream we hold back a trailing partial marker (e.g. ``"<|cha"``) so a
        marker split across deltas is not mistaken for literal text. On ``final``
        there is nothing more coming, so the whole buffer is flushed.
        """
        if not self._buffer:
            return
        if final:
            self._consume_text(self._buffer, emissions)
            self._buffer = ""
            return
        last_open = self._buffer.rfind("<")
        if last_open != -1:
            tail = self._buffer[last_open:]
            if any(token.startswith(tail) for token in _CONTROL_TOKENS):
                self._consume_text(self._buffer[:last_open], emissions)
                self._buffer = tail
                return
        self._consume_text(self._buffer, emissions)
        self._buffer = ""

    def _consume_text(self, text: str, emissions: list[tuple[str, bool]]) -> None:
        """Route non-marker text: emit body content, accumulate header text."""
        if not text:
            return
        if self._in_body:
            emissions.append((text, self._is_thinking))
        else:
            self._header += text

    def _apply_marker(self, token: str) -> None:
        if token == "<|channel|>":
            # Start of a new channel header.
            self._in_body = False
            self._header = ""
        elif token == "<|message|>":
            # Header complete: the channel name is its first whitespace token.
            stripped = self._header.strip()
            self._channel = stripped.split()[0] if stripped else None
            self._header = ""
            self._in_body = True
        elif token in _BODY_END_TOKENS:
            # End of a message body; wait for the next channel.
            self._in_body = False
            self._channel = None
            self._header = ""
        # ``<|constrain|>`` only appears inside a header (e.g. a tool call's
        # ``<|constrain|>json``); drop it and keep accumulating the header.
