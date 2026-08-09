"""Steward lifecycle state: derivation, canary history, and the 503 preflight."""

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
from fastapi import HTTPException

from skulk.api.main import API
from skulk.api.steward import (
    STEWARD_NOT_READY_MESSAGES,
    STEWARD_RETRY_AFTER_SECONDS,
    StewardCanaryState,
    StewardStatusResponse,
    derive_steward_state,
)
from skulk.api.types.api import ChatCompletionMessage, ChatCompletionRequest
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.common import NodeId
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
