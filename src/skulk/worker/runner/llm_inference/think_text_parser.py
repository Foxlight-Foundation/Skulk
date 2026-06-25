"""Dependency-free streaming parser for token-delimited ``<think>`` reasoning.

The MLX engine separates a token-delimited reasoning model's thinking from its
answer at the token level (``model_output_parsers.parse_thinking_models``). The
llama.cpp engine does not expose token ids: ``create_chat_completion`` hands back
already detokenized *string* deltas, so the ``<think>``/``</think>`` markers
arrive as literal text. Unparsed, the whole reasoning block floods the visible
``content`` (the answer is buried after a wall of chain-of-thought) and nothing
lands in ``reasoning_content``.

This module reparses that stream from strings so the llama.cpp runner reaches
parity with MLX: text inside the markers is flagged ``is_thinking=True``, the
markers themselves are stripped, and a marker split across deltas is held back
rather than leaked. It is the ``<think>`` counterpart to
:class:`~skulk.worker.runner.llm_inference.harmony_text_parser.HarmonyTextParser`
(gpt-oss harmony), and is likewise pure-Python with no MLX import because it runs
on non-Mac GPU nodes (e.g. AMD).

``starts_in_thinking`` mirrors the prompt: a reasoning chat template pre-fills the
opening ``<think>`` in the PROMPT, not the generated output, so the stream begins
mid-reasoning and only the closing ``</think>`` appears in the deltas (verified on
Ornith 1.0-35B GGUF: ``has <think>: False``, ``has </think>: True``). Pass
``starts_in_thinking=True`` for those, which is the llama.cpp default for these
models (the runner does not toggle thinking, so the template's reasoning prefix is
always present). An explicit opening ``<think>`` in the stream is still handled
when ``starts_in_thinking=False``.
"""

from __future__ import annotations

from typing import final


@final
class ThinkTextParser:
    """Incrementally split ``<think>...</think>`` reasoning from content.

    Feed raw string deltas with :meth:`feed`; it returns a list of
    ``(text, is_thinking)`` emissions (often empty until a marker boundary or
    enough non-marker text accumulates). Call :meth:`flush` once the stream ends
    to drain any held-back tail. The interface matches ``HarmonyTextParser`` so
    the runner can drive either through the same loop.
    """

    def __init__(
        self,
        *,
        think_start: str = "<think>",
        think_end: str = "</think>",
        starts_in_thinking: bool = True,
    ) -> None:
        self._think_start = think_start
        self._think_end = think_end
        self._buffer: str = ""
        self._is_thinking: bool = starts_in_thinking

    def feed(self, text: str) -> list[tuple[str, bool]]:
        """Consume a raw delta, returning ``(text, is_thinking)`` emissions."""
        if text:
            self._buffer += text
        emissions: list[tuple[str, bool]] = []
        self._drain_markers(emissions)
        self._drain_safe_text(emissions, final=False)
        return emissions

    def flush(self) -> list[tuple[str, bool]]:
        """Drain any remaining buffered text once the stream has ended.

        Text still buffered when the stream ends (e.g. a generation truncated
        mid-thought by a token cap) is emitted in its current mode, so a model
        that never closes its ``<think>`` block yields all-reasoning rather than
        silently dropping it.
        """
        emissions: list[tuple[str, bool]] = []
        self._drain_markers(emissions)
        self._drain_safe_text(emissions, final=True)
        return emissions

    def _drain_markers(self, emissions: list[tuple[str, bool]]) -> None:
        """Toggle on every complete think marker for the current mode.

        While thinking we look only for ``</think>``; while in content we look
        only for ``<think>`` (mirroring ``parse_thinking_models``), so a stray
        unmatched marker is left as text rather than flipping the mode.
        """
        while True:
            marker = self._think_end if self._is_thinking else self._think_start
            index = self._buffer.find(marker)
            if index == -1:
                return
            if index > 0:
                emissions.append((self._buffer[:index], self._is_thinking))
            self._buffer = self._buffer[index + len(marker) :]
            self._is_thinking = not self._is_thinking

    def _drain_safe_text(
        self, emissions: list[tuple[str, bool]], *, final: bool
    ) -> None:
        """Emit buffered text that cannot be part of a not-yet-complete marker.

        Mid-stream we hold back a trailing partial marker (e.g. ``"</thi"``) so a
        marker split across deltas is not mistaken for literal text. On ``final``
        there is nothing more coming, so the whole buffer is flushed.
        """
        if not self._buffer:
            return
        if final:
            emissions.append((self._buffer, self._is_thinking))
            self._buffer = ""
            return
        last_open = self._buffer.rfind("<")
        if last_open != -1:
            tail = self._buffer[last_open:]
            if self._think_start.startswith(tail) or self._think_end.startswith(tail):
                if last_open > 0:
                    emissions.append((self._buffer[:last_open], self._is_thinking))
                self._buffer = tail
                return
        emissions.append((self._buffer, self._is_thinking))
        self._buffer = ""
