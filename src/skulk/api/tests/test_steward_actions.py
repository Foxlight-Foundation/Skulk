"""Approval-gated intelligent-fabric basic-action contracts."""

import json
from datetime import datetime, timedelta, timezone
from typing import cast

import anyio
import pytest

from skulk.api.main import API
from skulk.api.steward import StewardHarness, steward_tool_definitions
from skulk.master.main import Master
from skulk.shared.apply import apply
from skulk.shared.tests.conftest import get_pipeline_shard_metadata
from skulk.shared.types.commands import (
    CancelDownload,
    DecideStewardAction,
    ForwarderCommand,
    ForwarderDownloadCommand,
    ProposeStewardAction,
)
from skulk.shared.types.common import ModelId, NodeId, SessionId, SystemId
from skulk.shared.types.events import (
    Event,
    GlobalForwarderEvent,
    IndexedEvent,
    LocalForwarderEvent,
    StateSnapshotHydrated,
    StewardActionProposalChanged,
)
from skulk.shared.types.state import State
from skulk.shared.types.state_sync import StateSyncMessage
from skulk.shared.types.steward_actions import (
    StewardActionProposal,
    StewardCancelDownloadAction,
)
from skulk.shared.types.telemetry import NodeTelemetry, TelemetryView
from skulk.shared.types.worker.downloads import DownloadPending
from skulk.utils.channels import channel


def _cancel_proposal() -> StewardActionProposal:
    now = datetime.now(tz=timezone.utc)
    return StewardActionProposal(
        action=StewardCancelDownloadAction(
            node_id=NodeId("worker"),
            node_name="Worker",
            model_id=ModelId("org/model"),
        ),
        rationale="The transfer is stalled.",
        evidence=("No progress for ten minutes.",),
        expected_effect="Stop the active transfer without deleting stored data.",
        created_at=now,
        expires_at=now + timedelta(minutes=10),
    )


def test_action_tools_only_create_proposals() -> None:
    """The model receives proposal verbs, never direct mutating verbs."""
    names = {
        cast("dict[str, object]", item["function"])["name"]
        for item in steward_tool_definitions()
    }

    assert {
        "propose_place_model",
        "propose_stop_model",
        "propose_restart_model",
        "propose_cancel_download",
    } <= names
    assert "place_model" not in names
    assert "stop_model" not in names
    assert "restart_model" not in names


@pytest.mark.asyncio
async def test_harness_proposal_is_inert_until_separate_decision() -> None:
    """Creating a proposal submits only proposal state and reports no execution."""
    submitted: list[StewardActionProposal] = []

    class _FakeAPI:
        async def submit_steward_action_proposal(
            self, proposal: StewardActionProposal
        ) -> None:
            submitted.append(proposal)

    harness = StewardHarness(cast("API", cast("object", _FakeAPI())))
    result = cast(
        "dict[str, object]",
        json.loads(
            await harness._propose_action(  # pyright: ignore[reportPrivateUsage]
                _cancel_proposal().action,
                {
                    "rationale": "The transfer is stalled.",
                    "evidence": ["No progress for ten minutes."],
                    "expected_effect": "Stop the transfer.",
                },
            )
        ),
    )

    assert len(submitted) == 1
    assert submitted[0].status == "pending"
    assert result["approvalRequired"] is True
    assert result["note"] == "No cluster action has executed."


def test_proposal_event_round_trips_through_replicated_state() -> None:
    """Proposal audit survives JSON snapshots and event replay."""
    proposal = _cancel_proposal()
    state = apply(
        State(),
        IndexedEvent(
            idx=0,
            event=StewardActionProposalChanged(proposal=proposal),
        ),
    )

    restored = State.model_validate_json(state.model_dump_json(by_alias=True))

    assert restored.steward_action_proposals[proposal.proposal_id] == proposal


@pytest.mark.asyncio
async def test_master_approves_a_proposal_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Back-to-back approvals cannot dispatch the same action twice."""
    proposal = _cancel_proposal()
    node_id = NodeId("master")
    session_id = SessionId(master_node_id=node_id, election_clock=0)
    global_sender, global_receiver = channel[GlobalForwarderEvent]()
    command_sender, command_receiver = channel[ForwarderCommand]()
    _, local_event_receiver = channel[LocalForwarderEvent]()
    _, state_sync_receiver = channel[StateSyncMessage]()
    state_sync_sender, _ = channel[StateSyncMessage]()
    download_sender, download_receiver = channel[ForwarderDownloadCommand]()
    event_sender, event_receiver = channel[Event]()
    telemetry_view = TelemetryView()
    master = Master(
        node_id,
        session_id,
        event_sender=event_sender,
        global_event_sender=global_sender,
        local_event_receiver=local_event_receiver,
        command_receiver=command_receiver,
        state_sync_receiver=state_sync_receiver,
        state_sync_sender=state_sync_sender,
        download_command_sender=download_sender,
        initial_state=State(
            steward_action_proposals={proposal.proposal_id: proposal}
        ),
        telemetry_view=telemetry_view,
    )

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(master.run)
        # Initial state is indexed before commands so failover carries pending
        # proposals into the new master's serialized decision view.
        seed = await global_receiver.receive()
        assert isinstance(seed.event, StateSnapshotHydrated)
        telemetry_view.apply(
            NodeTelemetry(
                node_id=NodeId("worker"),
                info=DownloadPending(
                    node_id=NodeId("worker"),
                    shard_metadata=get_pipeline_shard_metadata(
                        ModelId("org/model"), device_rank=0
                    ),
                ),
            )
        )
        decision = DecideStewardAction(
            proposal_id=proposal.proposal_id,
            approved=True,
            decided_by="trusted_fabric_operator",
        )
        await command_sender.send(
            ForwarderCommand(origin=SystemId("api"), command=decision)
        )
        changed = await event_receiver.receive()
        assert isinstance(changed, StewardActionProposalChanged)
        assert changed.proposal.status == "dispatched", changed.proposal.outcome
        dispatched = await download_receiver.receive()
        assert isinstance(dispatched.command, CancelDownload)
        assert dispatched.command.model_id == ModelId("org/model")

        await command_sender.send(
            ForwarderCommand(
                origin=SystemId("api"),
                command=DecideStewardAction(
                    proposal_id=proposal.proposal_id,
                    approved=True,
                    decided_by="trusted_fabric_operator",
                ),
            )
        )
        with anyio.move_on_after(0.1) as duplicate_wait:
            await download_receiver.receive()
        assert duplicate_wait.cancel_called

        blocked = _cancel_proposal()
        await command_sender.send(
            ForwarderCommand(
                origin=SystemId("api"),
                command=ProposeStewardAction(proposal=blocked),
            )
        )
        blocked_pending = await event_receiver.receive()
        assert isinstance(blocked_pending, StewardActionProposalChanged)
        monkeypatch.setenv("SKULK_FABRIC_CAPABILITIES_DISABLE", "1")
        await command_sender.send(
            ForwarderCommand(
                origin=SystemId("api"),
                command=DecideStewardAction(
                    proposal_id=blocked.proposal_id,
                    approved=True,
                    decided_by="trusted_fabric_operator",
                ),
            )
        )
        blocked_result = await event_receiver.receive()
        assert isinstance(blocked_result, StewardActionProposalChanged)
        assert blocked_result.proposal.status == "failed"
        assert "kill switch" in (blocked_result.proposal.outcome or "")
        with anyio.move_on_after(0.1) as blocked_wait:
            await download_receiver.receive()
        assert blocked_wait.cancel_called
        monkeypatch.delenv("SKULK_FABRIC_CAPABILITIES_DISABLE")

        expiring = _cancel_proposal()
        await command_sender.send(
            ForwarderCommand(
                origin=SystemId("api"),
                command=ProposeStewardAction(proposal=expiring),
            )
        )
        proposed = await event_receiver.receive()
        assert isinstance(proposed, StewardActionProposalChanged)
        assert proposed.proposal.status == "pending"
        await master._expire_steward_action_proposals(  # pyright: ignore[reportPrivateUsage]
            expiring.expires_at + timedelta(seconds=1)
        )
        expired = await event_receiver.receive()
        assert isinstance(expired, StewardActionProposalChanged)
        assert expired.proposal.status == "expired"
        assert expired.proposal.decided_by == "fabric_expiry"
        task_group.cancel_scope.cancel()
