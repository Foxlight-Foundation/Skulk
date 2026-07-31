"""The steward's chat-completions chunk stream (virtual-model turn).

StewardHarness is deliberately not @final: overriding its generation and
tool collaborators is the loop's unit-test seam.
"""

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from skulk.api.main import API

from skulk.api.steward import StewardChatMessage, StewardHarness
from skulk.api.types.api import ToolCall, ToolCallItem
from skulk.shared.models.model_cards import ModelId
from skulk.shared.types.chunks import ErrorChunk, TokenChunk
from skulk.shared.types.worker.instances import InstanceId


class _ScriptedHarness(StewardHarness):
    """Harness with generation and tools stubbed for loop testing."""

    def __init__(self, turns: list[tuple[str, list[ToolCall]]]) -> None:
        # The API handle is unused once the collaborators are overridden.
        super().__init__(cast("API", cast(object, None)))
        self._turns = turns
        self._cursor = 0
        self.executed: list[str] = []

    def steward_instance(self) -> tuple[InstanceId, str] | None:
        return InstanceId(), "org/steward-model"

    async def execute_tool(self, name: str, arguments: dict[str, object]) -> str:
        self.executed.append(name)
        return '{"ok": true}'

    async def _generate(
        self,
        messages: list[Any],
        model_id: str,
        instance_id: InstanceId,
    ) -> tuple[str, list[ToolCall], str | None]:
        text, calls = self._turns[min(self._cursor, len(self._turns) - 1)]
        self._cursor += 1
        return text, calls, None


def _call(name: str) -> ToolCall:
    return ToolCall(id=f"call-{name}", index=0, function=ToolCallItem(name=name, arguments="{}"))


async def _collect(harness: StewardHarness, question: str) -> list[TokenChunk | ErrorChunk]:
    chunks: list[TokenChunk | ErrorChunk] = []
    async for chunk in harness.run_turn_chunks(
        [StewardChatMessage(role="user", content=question)]
    ):
        assert isinstance(chunk, (TokenChunk, ErrorChunk))
        chunks.append(chunk)
    return chunks


async def test_tool_steps_stream_as_thinking_then_reply_stops() -> None:
    harness = _ScriptedHarness(
        turns=[
            ("", [_call("get_cluster_state")]),
            ("All healthy.", []),
        ]
    )
    chunks = await _collect(harness, "Is the cluster healthy?")
    assert [type(c) for c in chunks] == [TokenChunk, TokenChunk]
    first, last = cast(TokenChunk, chunks[0]), cast(TokenChunk, chunks[1])
    assert first.is_thinking and "get_cluster_state" in first.text
    assert not last.is_thinking
    assert last.text == "All healthy."
    assert last.finish_reason == "stop"
    assert harness.executed == ["get_cluster_state"]


async def test_missing_steward_yields_error_chunk() -> None:
    harness = _ScriptedHarness(turns=[("irrelevant", [])])
    harness.steward_instance = lambda: None
    chunks = await _collect(harness, "hello")
    assert len(chunks) == 1
    assert isinstance(chunks[0], ErrorChunk)


async def test_budget_exhaustion_still_emits_terminal_stop() -> None:
    harness = _ScriptedHarness(turns=[("", [_call("get_cluster_state")])])
    chunks = await _collect(harness, "loop forever")
    last = chunks[-1]
    assert isinstance(last, TokenChunk)
    assert last.finish_reason == "stop"
    thinking = [c for c in chunks if isinstance(c, TokenChunk) and c.is_thinking]
    assert len(thinking) >= 7  # every budgeted step surfaced as trace


async def test_text_markup_tool_calls_are_recovered() -> None:
    harness = _ScriptedHarness(
        turns=[
            ("<tool_call>\n<function=run_doctor>\n</function>\n</tool_call>", []),
            ("Doctor says fine.", []),
        ]
    )
    chunks = await _collect(harness, "run a checkup")
    assert harness.executed == ["run_doctor"]
    last = chunks[-1]
    assert isinstance(last, TokenChunk)
    assert last.text == "Doctor says fine."
    assert last.model == ModelId("skulk/steward")


async def test_abandoned_stream_cancels_active_generation() -> None:
    """Closing the stream mid-turn cancels the in-flight inner generation."""
    cancelled: list[object] = []

    class _Api:
        async def send_task_cancellation(self, command_id: object) -> None:
            cancelled.append(command_id)

    harness = _ScriptedHarness(turns=[("", [_call("get_cluster_state")])])
    harness._api = cast("API", cast(object, _Api()))  # pyright: ignore[reportPrivateUsage]
    harness._active_command_id = cast(Any, "cmd-inner-1")  # pyright: ignore[reportPrivateUsage]

    stream = harness.run_turn_chunks(
        [StewardChatMessage(role="user", content="hi")]
    )
    first = await stream.__anext__()
    assert isinstance(first, TokenChunk)
    await stream.aclose()
    assert cancelled == ["cmd-inner-1"]
