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

    async def _generate_events(
        self,
        messages: list[Any],
        model_id: str,
        instance_id: InstanceId,
    ):
        text, calls = self._turns[min(self._cursor, len(self._turns) - 1)]
        self._cursor += 1
        if text:
            yield ("text", text)
        yield ("result", (text, calls, None))


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
    token_chunks = [c for c in chunks if isinstance(c, TokenChunk)]
    assert token_chunks[0].is_thinking
    assert "get_cluster_state" in token_chunks[0].text
    content = "".join(c.text for c in token_chunks if not c.is_thinking)
    assert content == "All healthy."
    assert token_chunks[-1].finish_reason == "stop"
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
    token_chunks = [c for c in chunks if isinstance(c, TokenChunk)]
    content = "".join(c.text for c in token_chunks if not c.is_thinking)
    assert content == "Doctor says fine."
    assert token_chunks[-1].model == ModelId("skulk/steward")
    # The markup step's text was held back, never streamed as content.
    assert "<function=" not in content


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


async def test_final_answer_streams_live_with_holdback() -> None:
    """Answer pieces stream as they arrive; markup-suspicious tails hold."""
    harness = _ScriptedHarness(turns=[("The cluster looks a<b fine.", [])])
    chunks = await _collect(harness, "status?")
    token_chunks = [c for c in chunks if isinstance(c, TokenChunk)]
    content = "".join(c.text for c in token_chunks if not c.is_thinking)
    assert content == "The cluster looks a<b fine."
    # more than one content chunk proves live emission plus flushed tail
    assert len([c for c in token_chunks if not c.is_thinking]) >= 2


def test_splittable_prefix_gates_only_marker_prefixes() -> None:
    from skulk.api.steward import splittable_prefix

    assert splittable_prefix("plain words") == len("plain words")
    held = splittable_prefix("answer <tool")
    assert "answer <tool"[held:] == "<tool"
    assert splittable_prefix("a<b compare") == len("a<b compare")
    assert splittable_prefix("ends with <") == len("ends with ")


async def test_false_marker_mention_is_not_lost() -> None:
    """An answer that MENTIONS markup syntax keeps its full text."""
    answer = "Tools are invoked with <tool_call> blocks, like this example."
    harness = _ScriptedHarness(turns=[(answer, [])])
    chunks = await _collect(harness, "how do tools work?")
    token_chunks = [c for c in chunks if isinstance(c, TokenChunk)]
    content = "".join(c.text for c in token_chunks if not c.is_thinking)
    assert content == answer


def test_earliest_complete_marker_wins_over_tail_prefix() -> None:
    from skulk.api.steward import splittable_prefix

    assert splittable_prefix("<tool_call>\n<function=") == 0
    text = "answer first <tool_call>\n<function="
    assert splittable_prefix(text) == text.index("<tool_call>")


async def test_pure_malformed_block_never_flushes_as_reply() -> None:
    """A withheld tail that is nothing but markup is a malformed tool
    attempt and must not leak as content."""
    malformed = "<tool_call>\n<function=broken\n</tool_call>"
    harness = _ScriptedHarness(turns=[(malformed, [])])
    chunks = await _collect(harness, "status?")
    token_chunks = [c for c in chunks if isinstance(c, TokenChunk)]
    content = "".join(c.text for c in token_chunks if not c.is_thinking)
    assert "<tool_call>" not in content


async def test_prefix_only_literal_example_survives_in_final_answer() -> None:
    """An answer whose prose all precedes the example block must not be
    truncated: the prose streams live, and the withheld block (markup-only
    on its own) still flushes because the FULL turn contains prose."""
    answer = "The syntax is <tool_call>example</tool_call>"
    harness = _ScriptedHarness(turns=[(answer, [])])
    chunks = await _collect(harness, "how do tools work?")
    token_chunks = [c for c in chunks if isinstance(c, TokenChunk)]
    content = "".join(c.text for c in token_chunks if not c.is_thinking)
    assert content == answer


async def test_complete_literal_example_survives_in_final_answer() -> None:
    """Markup embedded in prose is a literal example inside a real answer
    and must flush intact, complete block included."""
    answer = "The syntax is <tool_call>example</tool_call>, wrapped exactly so."
    harness = _ScriptedHarness(turns=[(answer, [])])
    chunks = await _collect(harness, "how do tools work?")
    token_chunks = [c for c in chunks if isinstance(c, TokenChunk)]
    content = "".join(c.text for c in token_chunks if not c.is_thinking)
    assert content == answer
