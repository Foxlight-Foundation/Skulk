"""Error-surfacing tests for the Claude/Responses/Ollama wire adapters (#276).

Every adapter generator is consumed by FastAPI's ``StreamingResponse``, so the
HTTP status is committed to 200 before the generator runs. A runner-side error
therefore cannot raise a clean 4xx; it must be emitted as a structured error
envelope inside the body and then returned. These tests assert that a context
rejection (the ``context_length_exceeded:`` sentinel) maps to a 400
``invalid_request_error`` envelope, a generic error maps to a 500 envelope, and
neither case leaks a bogus empty successful completion into the stream.
"""

import json
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import cast

import pytest

from skulk.api.adapters.chat_completions import (
    collect_chat_response,
    generate_chat_stream,
)
from skulk.api.adapters.claude import collect_claude_response, generate_claude_stream
from skulk.api.adapters.ollama import (
    collect_ollama_chat_response,
    collect_ollama_generate_response,
)
from skulk.api.adapters.responses import (
    collect_responses_response,
    generate_responses_stream,
)
from skulk.shared.constants import CONTEXT_LENGTH_EXCEEDED_PREFIX
from skulk.shared.types.chunks import (
    ErrorChunk,
    PrefillProgressChunk,
    TokenChunk,
    ToolCallChunk,
)
from skulk.shared.types.common import CommandId, ModelId

_MODEL = ModelId("mlx-community/gemma-4-26b-a4b-it-4bit")
_CONTEXT_MESSAGE = (
    f"{CONTEXT_LENGTH_EXCEEDED_PREFIX} requested 9000 tokens exceeds max_tokens"
)
_GENERIC_MESSAGE = "runner exploded for an unrelated reason"

# Markers that unambiguously indicate a bogus successful-completion leaked into
# the body. These are the *terminal* success shapes each adapter emits when a
# stream finishes cleanly; a real error must never reach them. (The Claude
# streaming preamble emits a message_start with an empty assistant message
# before any chunk is read, so a bare "role":"assistant" is NOT a reliable
# success marker; the message_stop / message_delta tail is.)
_SUCCESS_MARKERS = (
    '"finish_reason":"stop"',  # chat_completions success choice
    '"status":"completed"',  # responses success
    '"done_reason":"stop"',  # ollama success
    '"type":"message_stop"',  # claude streaming success tail
    '"type":"message_delta"',  # claude streaming completion delta
    '"stop_reason":"end_turn"',  # claude non-streaming success
)


async def _error_stream(
    message: str,
) -> AsyncGenerator[
    ErrorChunk | ToolCallChunk | TokenChunk | PrefillProgressChunk, None
]:
    """Yield a single ``ErrorChunk`` carrying ``message`` and nothing else."""
    yield ErrorChunk(model=_MODEL, error_message=message)


def _extract_envelope(output: str) -> dict[str, object]:
    """Find and parse the structured error envelope from adapter body output.

    Adapters frame the envelope differently (raw JSON, SSE ``data:`` line,
    NDJSON line). This locates the JSON object that carries an ``error`` field
    so the assertions stay framing-agnostic.
    """
    for line in output.replace("\n\n", "\n").splitlines():
        candidate = line
        if candidate.startswith("data: "):
            candidate = candidate[len("data: ") :]
        candidate = candidate.strip()
        if not candidate.startswith("{"):
            continue
        try:
            parsed = cast("object", json.loads(candidate))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            envelope = cast("dict[str, object]", parsed)
            if "error" in envelope:
                return envelope
    raise AssertionError(f"no structured error envelope found in body: {output!r}")


def _error_fields(output: str) -> dict[str, object]:
    """Return the ``error`` sub-object of the structured envelope in ``output``."""
    envelope = _extract_envelope(output)
    error = envelope["error"]
    assert isinstance(error, dict), f"error field is not an object: {output!r}"
    return cast("dict[str, object]", error)


def _assert_no_success(output: str) -> None:
    """Assert the body carries no bogus successful-completion markers."""
    for marker in _SUCCESS_MARKERS:
        assert marker not in output, (
            f"error body leaked a successful-completion marker {marker!r}: {output!r}"
        )


async def _drive_chat_completions(message: str, *, stream: bool) -> str:
    if stream:
        gen = generate_chat_stream(CommandId("cmd"), _error_stream(message))
    else:
        gen = collect_chat_response(CommandId("cmd"), _error_stream(message))
    return "".join([chunk async for chunk in gen])


async def _drive_claude(message: str, *, stream: bool) -> str:
    if stream:
        gen = generate_claude_stream(CommandId("cmd"), "m", _error_stream(message))
    else:
        gen = collect_claude_response(CommandId("cmd"), "m", _error_stream(message))
    return "".join([chunk async for chunk in gen])


async def _drive_responses(message: str, *, stream: bool) -> str:
    if stream:
        gen = generate_responses_stream(CommandId("cmd"), "m", _error_stream(message))
    else:
        gen = collect_responses_response(CommandId("cmd"), "m", _error_stream(message))
    return "".join([chunk async for chunk in gen])


async def _drive_chat_completions_stream(message: str) -> str:
    return await _drive_chat_completions(message, stream=True)


async def _drive_chat_completions_collect(message: str) -> str:
    return await _drive_chat_completions(message, stream=False)


async def _drive_claude_stream(message: str) -> str:
    return await _drive_claude(message, stream=True)


async def _drive_claude_collect(message: str) -> str:
    return await _drive_claude(message, stream=False)


async def _drive_responses_stream(message: str) -> str:
    return await _drive_responses(message, stream=True)


async def _drive_responses_collect(message: str) -> str:
    return await _drive_responses(message, stream=False)


async def _drive_ollama_chat(message: str) -> str:
    # Ollama streaming already emits a native done_reason="error" response and
    # returns (no empty success), so only the non-streaming collector changed.
    gen = collect_ollama_chat_response(CommandId("cmd"), _error_stream(message))
    return "".join([chunk async for chunk in gen])


async def _drive_ollama_generate(message: str) -> str:
    gen = collect_ollama_generate_response(CommandId("cmd"), _error_stream(message))
    return "".join([chunk async for chunk in gen])


_Driver = Callable[[str], Awaitable[str]]

# Every adapter arm that funnels an ErrorChunk into the response body. The
# Ollama streaming functions are intentionally absent: they already emit a
# native done_reason="error" response and return, so #276 did not change them.
_DRIVERS: list[tuple[str, _Driver]] = [
    ("chat_completions_stream", _drive_chat_completions_stream),
    ("chat_completions_collect", _drive_chat_completions_collect),
    ("claude_stream", _drive_claude_stream),
    ("claude_collect", _drive_claude_collect),
    ("responses_stream", _drive_responses_stream),
    ("responses_collect", _drive_responses_collect),
    ("ollama_chat_collect", _drive_ollama_chat),
    ("ollama_generate_collect", _drive_ollama_generate),
]


@pytest.mark.anyio
@pytest.mark.parametrize("label,driver", _DRIVERS, ids=[d[0] for d in _DRIVERS])
async def test_context_error_surfaces_400_envelope(
    label: str, driver: _Driver
) -> None:
    """A context rejection becomes a 400 invalid_request_error envelope."""
    output = await driver(_CONTEXT_MESSAGE)

    error = _error_fields(output)
    assert error["type"] == "invalid_request_error", label
    assert error["code"] == 400, label
    assert error["message"] == _CONTEXT_MESSAGE, label

    _assert_no_success(output)


@pytest.mark.anyio
@pytest.mark.parametrize("label,driver", _DRIVERS, ids=[d[0] for d in _DRIVERS])
async def test_generic_error_surfaces_500_envelope(
    label: str, driver: _Driver
) -> None:
    """A non-context error maps to the 500 internal-error envelope."""
    output = await driver(_GENERIC_MESSAGE)

    error = _error_fields(output)
    assert error["type"] == "InternalServerError", label
    assert error["code"] == 500, label
    assert error["message"] == _GENERIC_MESSAGE, label

    _assert_no_success(output)
