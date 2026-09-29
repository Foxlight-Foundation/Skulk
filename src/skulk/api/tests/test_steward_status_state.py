"""Steward lifecycle state: derivation, canary history, and the 503 preflight."""

import weakref
from collections.abc import AsyncGenerator, AsyncIterator
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
from fastapi import HTTPException

from skulk.api.main import (
    API,
    _steward_canary_failure_command,  # pyright: ignore[reportPrivateUsage] - terminal command boundary under test
)
from skulk.api.steward import (
    STEWARD_NOT_READY_MESSAGES,
    STEWARD_RETRY_AFTER_SECONDS,
    StewardCanaryState,
    StewardChatMessage,
    StewardHarness,
    StewardStatusResponse,
    derive_steward_state,
)
from skulk.api.types.api import ChatCompletionMessage, ChatCompletionRequest
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.memory import Memory
from skulk.shared.types.telemetry import TelemetryView
from skulk.shared.types.worker.downloads import DownloadOngoing, DownloadProgressData
from skulk.shared.types.worker.instances import InstanceId
from skulk.shared.types.worker.shards import PipelineShardMetadata

if TYPE_CHECKING:
    from skulk.api.steward import StewardState


def test_disabled_mode_wins_over_every_other_signal() -> None:
    assert (
        derive_steward_state(
            enabled=False,
            present=True,
            ready=True,
            downloading=True,
            canary_failures=2,
        )
        == "disabled"
    )


def test_ready_with_a_clean_canary_is_ready() -> None:
    assert (
        derive_steward_state(
            enabled=True,
            present=True,
            ready=True,
            downloading=False,
            canary_failures=0,
        )
        == "ready"
    )


def test_one_outstanding_canary_failure_is_degraded() -> None:
    """Degraded is the early warning, not the teardown: one failure shows."""
    assert (
        derive_steward_state(
            enabled=True,
            present=True,
            ready=True,
            downloading=False,
            canary_failures=1,
        )
        == "degraded"
    )


def test_live_download_on_a_placed_steward_is_downloading() -> None:
    assert (
        derive_steward_state(
            enabled=True,
            present=True,
            ready=False,
            downloading=True,
            canary_failures=0,
        )
        == "downloading"
    )


def test_placed_but_loading_is_starting() -> None:
    assert (
        derive_steward_state(
            enabled=True,
            present=True,
            ready=False,
            downloading=False,
            canary_failures=0,
        )
        == "starting"
    )


def test_no_placement_yet_is_starting() -> None:
    """Enabled with nothing placed: the invariant is still establishing it."""
    assert (
        derive_steward_state(
            enabled=True,
            present=False,
            ready=False,
            downloading=False,
            canary_failures=0,
        )
        == "starting"
    )


def test_canary_state_counts_consecutive_failures_per_instance() -> None:
    canary = StewardCanaryState()
    first = InstanceId()
    canary.track(first)
    assert canary.consecutive_failures_for(first) == 0
    assert canary.record_failure() == 1
    assert canary.record_failure() == 2
    assert canary.consecutive_failures_for(first) == 2

    # A count earned by one steward says nothing about another instance.
    assert canary.consecutive_failures_for(InstanceId()) == 0

    canary.clear_failures()
    assert canary.consecutive_failures_for(first) == 0
    assert canary.instance_id == first

    canary.reset()
    assert canary.instance_id is None


def test_tracking_a_new_instance_drops_the_previous_failure_run() -> None:
    canary = StewardCanaryState()
    first = InstanceId()
    canary.track(first)
    _ = canary.record_failure()
    second = InstanceId()
    canary.track(second)
    assert canary.consecutive_failures_for(second) == 0


def test_canary_teardown_records_terminal_steward_failure() -> None:
    """Automatic canary recovery must not look like a clean operator stop."""
    instance_id = InstanceId("steward-instance")

    command = _steward_canary_failure_command(instance_id)

    assert command.instance_id == instance_id
    assert command.error_code == "runner_unresponsive"
    assert "three consecutive health probes" in command.error_message


def _request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=ModelId("skulk/steward"),
        messages=[ChatCompletionMessage(role="user", content="is the fleet ok?")],
        stream=True,
    )


def _stub_api(status: StewardStatusResponse) -> API:
    """An API stand-in exposing only what the preflight path touches."""
    return cast(
        "API",
        cast(
            object,
            SimpleNamespace(
                _intelligent_fabric_enabled=lambda: status.enabled,
                _steward_status=lambda: status,
            ),
        ),
    )


def _status(state: "StewardState", *, enabled: bool = True) -> StewardStatusResponse:
    return StewardStatusResponse(
        enabled=enabled,
        present=state in ("downloading", "ready", "degraded"),
        ready=state in ("ready", "degraded"),
        steward_model="org/steward-brain" if state != "disabled" else None,
        instance_id="inst-1" if state != "disabled" else None,
        state=state,
    )


@pytest.mark.parametrize("state", ["starting", "downloading"])
async def test_not_ready_steward_answers_503_with_the_status_payload(
    state: "StewardState",
) -> None:
    """A steward that cannot answer is a 503 contract, not a 200 error chunk."""
    status = _status(state)
    with pytest.raises(HTTPException) as raised:
        await API._steward_chat_completions(  # pyright: ignore[reportPrivateUsage]
            _stub_api(status), _request()
        )
    error = raised.value
    assert error.status_code == 503
    assert isinstance(error.detail, dict)
    detail = cast("dict[str, object]", error.detail)
    assert detail["state"] == state
    assert detail["ready"] is False
    assert detail["steward_model"] == "org/steward-brain"
    assert detail["message"] == STEWARD_NOT_READY_MESSAGES[state]
    assert error.headers is not None
    assert error.headers["Retry-After"] == str(STEWARD_RETRY_AFTER_SECONDS)


async def test_disabled_mode_still_answers_404_not_503() -> None:
    """The disabled contract predates this preflight and does not change."""
    with pytest.raises(HTTPException) as raised:
        await API._steward_chat_completions(  # pyright: ignore[reportPrivateUsage]
            _stub_api(_status("disabled", enabled=False)), _request()
        )
    assert raised.value.status_code == 404


async def test_a_forced_tool_choice_is_a_400_on_the_steward_surface() -> None:
    """The reserved id branches before the ordinary tool_choice resolution.

    The steward accepts no client tools, so a forced function choice can
    never be honored; it gets the same 400 the documented boundary gives
    rather than a steward answer that silently ignored the choice.
    """
    payload = ChatCompletionRequest(
        model=ModelId("skulk/steward"),
        messages=[ChatCompletionMessage(role="user", content="is the fleet ok?")],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
        stream=True,
    )
    with pytest.raises(HTTPException) as raised:
        await API._steward_chat_completions(  # pyright: ignore[reportPrivateUsage]
            _stub_api(_status("ready")), payload
        )
    assert raised.value.status_code == 400
    assert "get_weather" in str(raised.value.detail)


async def test_malformed_conversation_is_still_a_400_before_the_503() -> None:
    """A client error stays a client error even while the steward is starting."""
    payload = ChatCompletionRequest(
        model=ModelId("skulk/steward"),
        messages=[ChatCompletionMessage(role="assistant", content="unprompted")],
        stream=True,
    )
    with pytest.raises(HTTPException) as raised:
        await API._steward_chat_completions(  # pyright: ignore[reportPrivateUsage]
            _stub_api(_status("starting")), payload
        )
    assert raised.value.status_code == 400


def _download_record(model_id: str, node_id: NodeId) -> DownloadOngoing:
    card = ModelCard(
        model_id=ModelId(model_id),
        storage_size=Memory.from_gb(20),
        n_layers=40,
        hidden_size=2048,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )
    return DownloadOngoing(
        node_id=node_id,
        shard_metadata=PipelineShardMetadata(
            model_card=card,
            device_rank=0,
            world_size=1,
            start_layer=0,
            end_layer=40,
            n_layers=40,
        ),
        download_progress=DownloadProgressData(
            total=Memory.from_gb(20),
            downloaded=Memory.from_gb(4),
            downloaded_this_session=Memory.from_gb(4),
            completed_files=1,
            total_files=3,
            speed=1.0,
            eta_ms=1000,
            files={},
        ),
    )


def _downloads_api(records: list[DownloadOngoing]) -> API:
    """An API stand-in exposing only the download lookup's collaborators."""
    node_id = NodeId()
    return cast(
        "API",
        cast(
            object,
            SimpleNamespace(
                _telemetry_view=TelemetryView(),
                state=SimpleNamespace(downloads={node_id: records}),
            ),
        ),
    )


def test_live_download_for_the_steward_model_is_detected() -> None:
    """The downloading state hinges on this lookup matching by model id."""
    api = _downloads_api([_download_record("org/steward-brain", NodeId())])
    assert (
        API._steward_model_is_downloading(  # pyright: ignore[reportPrivateUsage]
            api, "org/steward-brain"
        )
        is True
    )
    assert (
        API._steward_model_is_downloading(  # pyright: ignore[reportPrivateUsage]
            api, "org/some-other-model"
        )
        is False
    )


def test_no_download_records_is_not_downloading() -> None:
    assert (
        API._steward_model_is_downloading(  # pyright: ignore[reportPrivateUsage]
            _downloads_api([]), "org/steward-brain"
        )
        is False
    )


class _ReadyTurnApi:
    """An API stand-in that lets one ready steward turn reach its response."""

    def __init__(self) -> None:
        self._extensions = None
        self._steward_turns: weakref.WeakValueDictionary[CommandId, StewardHarness] = (
            weakref.WeakValueDictionary()
        )
        # No placement: the turn answers with its in-stream error chunk, which
        # is enough to drive the response from first byte to last.
        self.state = SimpleNamespace(instances={})

    @property
    def turns(self) -> "weakref.WeakValueDictionary[CommandId, StewardHarness]":
        return self._steward_turns

    def _intelligent_fabric_enabled(self) -> bool:
        return True

    def _steward_status(self) -> StewardStatusResponse:
        return _status("ready")

    async def _steward_extension_transform(
        self, history: list[StewardChatMessage], *, stream: bool
    ) -> tuple[list[StewardChatMessage], str, None]:
        return history, "steward prompt", None

    def _release_steward_turn_after(
        self, command_id: CommandId, response_stream: AsyncIterator[str]
    ) -> AsyncGenerator[str, None]:
        return API._release_steward_turn_after(  # pyright: ignore[reportPrivateUsage]
            cast("API", cast(object, self)), command_id, response_stream
        )


async def test_a_turn_is_cancellable_by_its_advertised_id_until_it_ends() -> None:
    """The id the stream advertises is registered for exactly the stream's life."""
    api = _ReadyTurnApi()
    response = await API._steward_chat_completions(  # pyright: ignore[reportPrivateUsage]
        cast("API", cast(object, api)), _request()
    )

    advertised: str | None = None
    registered_while_streaming = False
    async for item in response.body_iterator:
        assert isinstance(item, str)
        if item.startswith(": command_id "):
            advertised = item.removeprefix(": command_id ").strip()
            registered_while_streaming = advertised in {
                str(command_id) for command_id in api.turns
            }

    assert advertised is not None
    assert registered_while_streaming
    assert len(api.turns) == 0
