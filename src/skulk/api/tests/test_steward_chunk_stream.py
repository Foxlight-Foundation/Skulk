"""The steward's chat-completions chunk stream (virtual-model turn).

StewardHarness is deliberately not @final: overriding its generation and
tool collaborators is the loop's unit-test seam.
"""

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from skulk.api.main import API

from skulk.api.steward import StewardChatMessage, StewardHarness
from skulk.api.types.api import ChatCompletionMessage, ToolCall, ToolCallItem
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
        self.system_prompts: list[str] = []

    def steward_instance(self) -> tuple[InstanceId, str] | None:
        return InstanceId(), "org/steward-model"

    async def execute_tool(self, name: str, arguments: dict[str, object]) -> str:
        self.executed.append(name)
        return '{"ok": true}'

    async def _generate_events(
        self,
        messages: list[ChatCompletionMessage],
        model_id: str,
        instance_id: InstanceId,
    ):
        self.system_prompts.append(str(messages[0].content))
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


async def test_cancel_turn_stops_the_generation_and_the_loop() -> None:
    """Cancelling the advertised outer id ends the whole turn (#830).

    The latch matters as much as the cancellation: without it the
    investigation loop would treat the cancelled step as complete and
    dispatch the next one, leaving the turn running behind the caller's
    back.
    """
    cancelled: list[object] = []

    class _Api:
        async def cancel_local_command(self, command_id: object) -> bool:
            # The shared path closes the local queue; recording it here is
            # the test's proof the turn cancelled through that path.
            cancelled.append(command_id)
            return True

        async def send_task_cancellation(self, command_id: object) -> None:
            raise AssertionError(
                "the fallback must not fire when the local cancel succeeded"
            )

    # A script that would otherwise tool-loop for the full step budget.
    harness = _ScriptedHarness(turns=[("", [_call("get_cluster_state")])])
    harness._api = cast("API", cast(object, _Api()))  # pyright: ignore[reportPrivateUsage]
    harness._active_command_id = cast(Any, "cmd-inner-1")  # pyright: ignore[reportPrivateUsage]

    stream = harness.run_turn_chunks(
        [StewardChatMessage(role="user", content="hi")]
    )
    first = await stream.__anext__()
    assert isinstance(first, TokenChunk)
    generations_before_cancel = len(harness.system_prompts)

    await harness.cancel_turn()
    assert cancelled == ["cmd-inner-1"]

    async for _chunk in stream:
        pass
    # The loop stopped at the latch instead of dispatching further steps.
    assert len(harness.system_prompts) == generations_before_cancel


async def test_cancel_command_routes_a_registered_steward_turn() -> None:
    """POST /v1/cancel/{outer_id} must reach the turn's harness, not 404."""
    from skulk.api.main import API
    from skulk.shared.types.common import CommandId

    cancelled: list[object] = []

    class _Api:
        async def cancel_local_command(self, command_id: object) -> bool:
            # The inner queue is already gone in this scenario; the harness
            # must fall back to the bare worker notification.
            return False

        async def send_task_cancellation(self, command_id: object) -> None:
            cancelled.append(command_id)

    harness = _ScriptedHarness(turns=[("irrelevant", [])])
    harness._api = cast("API", cast(object, _Api()))  # pyright: ignore[reportPrivateUsage]
    harness._active_command_id = cast(Any, "cmd-inner-9")  # pyright: ignore[reportPrivateUsage]

    outer_id = CommandId()
    api = API.__new__(API)
    api._text_generation_queues = {}  # pyright: ignore[reportPrivateUsage]
    api._image_generation_queues = {}  # pyright: ignore[reportPrivateUsage]
    api._embedding_queues = {}  # pyright: ignore[reportPrivateUsage]
    api._audio_speech_queues = {}  # pyright: ignore[reportPrivateUsage]
    api._audio_transcription_queues = {}  # pyright: ignore[reportPrivateUsage]
    api._steward_turn_harnesses = {outer_id: harness}  # pyright: ignore[reportPrivateUsage]

    response = await api.cancel_command(outer_id)

    assert response.command_id == outer_id
    assert cancelled == ["cmd-inner-9"]


async def test_cancellation_racing_dispatch_still_cancels_the_step() -> None:
    """A latch set while dispatch is in flight cancels the fresh command.

    cancel_turn can run between step start and dispatch completion, when no
    inner id exists yet to cancel. The step must then cancel the command it
    just obtained and end the turn instead of streaming one more full
    generation behind an accepted cancellation (#833).
    """
    from types import SimpleNamespace

    from skulk.shared.models.model_cards import ModelCard, ModelTask
    from skulk.shared.types.memory import Memory

    cancelled: list[object] = []
    stream_requests: list[object] = []
    harness_holder: list[StewardHarness] = []

    class _RacingApi:
        async def running_model_card(self, model_id: ModelId) -> ModelCard:
            return ModelCard(
                model_id=ModelId("steward-brain"),
                storage_size=Memory.from_gb(3),
                n_layers=12,
                hidden_size=30,
                supports_tensor=True,
                tasks=[ModelTask.TextGeneration],
            )

        async def dispatch_text_generation(
            self, task_params: object, target_instance_id: object = None
        ) -> object:
            # The race: cancellation lands while dispatch is in flight.
            await harness_holder[0].cancel_turn()
            return SimpleNamespace(command_id="cmd-fresh-inner")

        def text_generation_chunk_stream(
            self, command: object, task_params: object, *, extension_tap: bool = True
        ) -> object:
            stream_requests.append(command)
            raise AssertionError("a cancelled step must not open a stream")

        async def send_task_cancellation(self, command_id: object) -> None:
            cancelled.append(command_id)

    harness = StewardHarness(cast("API", cast(object, _RacingApi())))
    harness_holder.append(harness)
    harness.steward_instance = lambda: (InstanceId(), "steward-brain")

    chunks = [
        chunk
        async for chunk in harness.run_turn_chunks(
            [StewardChatMessage(role="user", content="hi")]
        )
    ]

    assert cancelled == ["cmd-fresh-inner"]
    assert stream_requests == []
    final = chunks[-1]
    assert isinstance(final, TokenChunk)
    assert final.finish_reason == "stop"
