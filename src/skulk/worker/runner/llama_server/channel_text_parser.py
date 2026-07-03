"""Streaming parser for Gemma 4's ``<|channel>`` reasoning markers.

Gemma 4 is a reasoning model: it emits its thinking in a channel and then the
answer, as *literal generated text* (the markers are not tokenizer special
tokens, and there is no chat template that strips them). Served via llama-server
with ``--reasoning-format none`` the raw markers therefore arrive in the content
stream:

    <|channel>thought
    ...the model's reasoning...
    <channel|>...the answer...

llama-server's own reasoning parsers don't understand these tokens (``auto`` puts
the *answer* in ``reasoning_content``; ``none`` leaves the raw markers in
``content``), so the served runner parses them itself, the same way the
in-process runner reparses gpt-oss harmony markers (see
``llm_inference/harmony_text_parser.py``). Pure-Python, no MLX, runs on AMD nodes.

Grammar (observed; verified on kite4 against google/gemma-4-31B-it-qat-q4_0-gguf):

- ``<|channel>`` opens a channel header; the channel *name* runs to the next
  newline, then the channel body begins.
- ``<channel|>`` closes the current channel and returns to the default (answer)
  body.
- A channel named ``thought`` / ``analysis`` / ``reasoning`` is flagged
  ``is_thinking``; the default body and any other channel are content.
- All markers (and the header newline) are stripped from the output.
"""

from __future__ import annotations

from typing import Final, final

#: Opens a channel header (channel name follows, up to the next newline).
_OPEN: Final = "<|channel>"
#: Closes the current channel, returning to the default (answer) body.
_CLOSE: Final = "<channel|>"

# Channel names whose body is reasoning rather than answer content.
_THINKING_CHANNELS: frozenset[str] = frozenset({"thought", "analysis", "reasoning"})


@final
class GemmaChannelTextParser:
    """Incrementally strip ``<|channel>`` markers and split reasoning from content.

    Feed raw string deltas with :meth:`feed`; it returns ``(text, is_thinking)``
    emissions (often empty until a marker/header boundary resolves). Call
    :meth:`flush` once the stream ends to drain the tail. Starts in the default
    content body, so a model that never emits a channel marker passes its text
    through unchanged.
    """

    def __init__(self) -> None:
        self._buffer: str = ""
        # While inside an open ``<|channel>`` header (accumulating the name).
        self._in_header: bool = False
        self._header: str = ""
        # Current channel name; ``None`` is the default (answer) body.
        self._channel: str | None = None

    @property
    def _is_thinking(self) -> bool:
        return self._channel in _THINKING_CHANNELS

    def feed(self, text: str) -> list[tuple[str, bool]]:
        """Consume a raw delta, returning ``(text, is_thinking)`` emissions."""
        if text:
            self._buffer += text
        emissions: list[tuple[str, bool]] = []
        self._drain(emissions, final=False)
        return emissions

    def flush(self) -> list[tuple[str, bool]]:
        """Drain any remaining buffered text once the stream has ended."""
        emissions: list[tuple[str, bool]] = []
        self._drain(emissions, final=True)
        return emissions

    def _drain(self, emissions: list[tuple[str, bool]], *, final: bool) -> None:
        # Resolve every complete boundary (a marker, or a header-ending newline)
        # currently in the buffer, then emit text that can't be a partial marker.
        while self._buffer:
            if self._in_header:
                newline = self._buffer.find("\n")
                if newline == -1:
                    break  # header name not yet complete
                self._header += self._buffer[:newline]
                self._channel = self._header.strip() or None
                self._header = ""
                self._in_header = False
                self._buffer = self._buffer[newline + 1 :]
                continue

            open_at = self._buffer.find(_OPEN)
            close_at = self._buffer.find(_CLOSE)
            candidates = [i for i in (open_at, close_at) if i != -1]
            if not candidates:
                break
            index = min(candidates)
            self._emit(self._buffer[:index], emissions)
            if index == open_at and (close_at == -1 or open_at <= close_at):
                self._buffer = self._buffer[index + len(_OPEN) :]
                self._in_header = True
            else:
                self._buffer = self._buffer[index + len(_CLOSE) :]
                self._channel = None

        # Hold back a trailing partial marker mid-stream so a marker split across
        # deltas isn't mistaken for literal text; flush everything on ``final``.
        if not self._buffer or self._in_header:
            if final and self._in_header:
                # Stream ended mid-header: the (unterminated) header name is not
                # output; nothing to emit.
                self._header = ""
                self._buffer = ""
            return
        if final:
            self._emit(self._buffer, emissions)
            self._buffer = ""
            return
        last_open = self._buffer.rfind("<")
        if last_open != -1:
            tail = self._buffer[last_open:]
            if _OPEN.startswith(tail) or _CLOSE.startswith(tail):
                self._emit(self._buffer[:last_open], emissions)
                self._buffer = tail
                return
        self._emit(self._buffer, emissions)
        self._buffer = ""

    def _emit(self, text: str, emissions: list[tuple[str, bool]]) -> None:
        if text:
            emissions.append((text, self._is_thinking))
