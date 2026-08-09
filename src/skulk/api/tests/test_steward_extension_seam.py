"""Chat middleware on the steward's bespoke turn.

The steward answers through ``StewardHarness`` rather than the ordinary
dispatch path, so ``API.chat_completions`` returns before its extension hook.
These tests pin the explicit seam that gives middleware the same two hooks on
a steward turn, with the same never-degrade guarantee.
"""

import asyncio
from typing import TYPE_CHECKING, cast

from skulk.api.main import API
from skulk.api.steward import (
    STEWARD_SYSTEM_PROMPT,
    STEWARD_VIRTUAL_MODEL_ID,
    StewardChatMessage,
)

# Test-helper reuse across sibling suites: the scripted harness and the inert
# extension context are exactly the fixtures this seam needs, and duplicating
# either would let the copies drift from the surfaces they stand in for.
from skulk.api.tests.test_steward_chunk_stream import (
    _ScriptedHarness,  # pyright: ignore[reportPrivateUsage]
)
from skulk.extensions import (
    BaseChatMiddleware,
    ChatResponseSummary,
    ExtensionContext,
    LoadedExtensions,
)
from skulk.extensions.tests.test_extensions import (
    _context,  # pyright: ignore[reportPrivateUsage]
)
from skulk.shared.types.common import ModelId
from skulk.shared.types.text_generation import (
    InputMessage,
    TextGenerationTaskParams,
)

if TYPE_CHECKING:
    from skulk.extensions.types import SkulkExtension

MEMORY_BLOCK = "[cluster memory] the operator's cat is called Wren"


class _InjectingMiddleware(BaseChatMiddleware):
    """Appends a memory block to instructions and records what it observed."""

    def __init__(self) -> None:
        self.seen_input: list[InputMessage] = []
        self.summaries: list[ChatResponseSummary] = []

    async def transform_chat_request(
        self,
        context: ExtensionContext,
        task_params: TextGenerationTaskParams,
    ) -> TextGenerationTaskParams:
        self.seen_input = list(task_params.input)
        return task_params.model_copy(
            update={"instructions": f"{task_params.instructions}\n\n{MEMORY_BLOCK}"}
        )

    async def observe_chat_response(
        self,
        context: ExtensionContext,
        task_params: TextGenerationTaskParams,
        summary: ChatResponseSummary,
    ) -> None:
        self.summaries.append(summary)


class _RaisingMiddleware(BaseChatMiddleware):
    """Both hooks explode; the steward must answer anyway."""

    async def transform_chat_request(
        self,
        context: ExtensionContext,
        task_params: TextGenerationTaskParams,
    ) -> TextGenerationTaskParams:
        raise RuntimeError("boom")

    async def observe_chat_response(
        self,
        context: ExtensionContext,
        task_params: TextGenerationTaskParams,
        summary: ChatResponseSummary,
    ) -> None:
        raise RuntimeError("boom")


class _DroppingMiddleware(BaseChatMiddleware):
    """Leaves the turn with no trailing user message (a broken transform).

    Also edits ``instructions``, so the rejection test can prove the whole
    transform is discarded rather than only its history.
    """

    async def transform_chat_request(
        self,
        context: ExtensionContext,
        task_params: TextGenerationTaskParams,
    ) -> TextGenerationTaskParams:
        return task_params.model_copy(
            update={
                "input": [],
                "instructions": f"{task_params.instructions}\n\n{MEMORY_BLOCK}",
            }
        )


class _StubExtension:
    """Minimal SkulkExtension carrying one middleware."""

    def __init__(self, middleware: BaseChatMiddleware) -> None:
        self._middleware = middleware

    @property
    def name(self) -> str:
        return "stub"

    @property
    def skulk_requires(self) -> str:
        return ">=0"

    def chat_middleware(self) -> BaseChatMiddleware:
        return self._middleware


class _StubApi:
    """Just the two attributes the seam helper reads off ``API``."""

    def __init__(self, middleware: BaseChatMiddleware | None) -> None:
        self._extensions = (
            None
            if middleware is None
            else LoadedExtensions([cast("SkulkExtension", _StubExtension(middleware))])
        )
        self._extension_context = _context()


async def _transform(
    middleware: BaseChatMiddleware | None, history: list[StewardChatMessage]
) -> tuple[list[StewardChatMessage], str, TextGenerationTaskParams]:
    return await API._steward_extension_transform(  # pyright: ignore[reportPrivateUsage]
        cast("API", cast(object, _StubApi(middleware))), history, stream=False
    )


def _history() -> list[StewardChatMessage]:
    return [StewardChatMessage(role="user", content="who lives here?")]


async def test_no_middleware_leaves_the_turn_untouched() -> None:
    history = _history()
    result, prompt, params = await _transform(None, history)
    assert result == history
    assert prompt == STEWARD_SYSTEM_PROMPT
    assert [message.content for message in params.input] == ["who lives here?"]


async def test_middleware_sees_the_real_prompt_and_history() -> None:
    middleware = _InjectingMiddleware()
    _, _, _ = await _transform(middleware, _history())
    assert [message.role for message in middleware.seen_input] == ["user"]
    assert middleware.seen_input[0].content == "who lives here?"


async def test_injected_instructions_become_the_turn_system_prompt() -> None:
    _, prompt, _ = await _transform(_InjectingMiddleware(), _history())
    assert prompt.startswith(STEWARD_SYSTEM_PROMPT)
    assert MEMORY_BLOCK in prompt


async def test_raising_transform_leaves_the_steward_prompt_intact() -> None:
    history = _history()
    result, prompt, _ = await _transform(_RaisingMiddleware(), history)
    assert result == history
    assert prompt == STEWARD_SYSTEM_PROMPT


async def test_transform_that_drops_the_question_is_discarded() -> None:
    """Rejection discards the whole transform, not only its history.

    Keeping the transformed prompt or params would run the turn on one
    conversation while telling observers it was another.
    """
    history = _history()
    result, prompt, params = await _transform(_DroppingMiddleware(), history)
    assert result == history
    assert prompt == STEWARD_SYSTEM_PROMPT
    assert [message.content for message in params.input] == ["who lives here?"]
    assert params.instructions == STEWARD_SYSTEM_PROMPT


class _OverreachingMiddleware(BaseChatMiddleware):
    """Keeps a valid question but also edits channels the harness ignores."""

    async def transform_chat_request(
        self,
        context: ExtensionContext,
        task_params: TextGenerationTaskParams,
    ) -> TextGenerationTaskParams:
        return task_params.model_copy(
            update={
                "model": ModelId("someone/else"),
                "instructions": "",
                "input": [
                    InputMessage(role="system", content="dropped by the filter"),
                    InputMessage(role="user", content=""),
                    *task_params.input,
                ],
            }
        )


async def test_returned_params_describe_the_turn_actually_served() -> None:
    """Observers must never be told about inputs or a model that never ran."""
    history, prompt, params = await _transform(
        _OverreachingMiddleware(), _history()
    )
    # The reserved model always serves, whatever the middleware asked for.
    assert str(params.model) == STEWARD_VIRTUAL_MODEL_ID
    # Empty instructions fall back, and the params say so too.
    assert prompt == STEWARD_SYSTEM_PROMPT
    assert params.instructions == STEWARD_SYSTEM_PROMPT
    # The filtered system and empty-content messages are gone from both.
    assert [message.content for message in history] == ["who lives here?"]
    assert [
        (message.role, message.content) for message in params.input
    ] == [("user", "who lives here?")]


async def test_harness_renders_the_injected_system_prompt() -> None:
    """The override reaches the model as the turn's system message."""
    harness = _ScriptedHarness(turns=[("Wren does.", [])])
    async for _ in harness.run_turn_chunks(
        _history(), system_prompt=f"{STEWARD_SYSTEM_PROMPT}\n\n{MEMORY_BLOCK}"
    ):
        pass
    assert harness.system_prompts
    assert MEMORY_BLOCK in harness.system_prompts[0]


async def test_response_tap_observes_the_final_answer() -> None:
    """The stream tap hands observers the steward's answer, not its trace."""
    middleware = _InjectingMiddleware()
    loaded = LoadedExtensions([cast("SkulkExtension", _StubExtension(middleware))])
    harness = _ScriptedHarness(turns=[("Wren does.", [])])
    params = TextGenerationTaskParams(
        model=ModelId(STEWARD_VIRTUAL_MODEL_ID),
        input=[InputMessage(role="user", content="who?")],
    )
    stream = loaded.tap_chat_stream(
        _context(), params, harness.run_turn_chunks(_history())
    )
    async for _ in stream:
        pass
    # The observer runs as a scheduled task; let it drain.
    await asyncio.sleep(0)
    assert [summary.text for summary in middleware.summaries] == ["Wren does."]
