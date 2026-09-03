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
from skulk.shared.models.model_cards import ModelCard, ModelTask
from skulk.shared.tests.conftest import get_pipeline_shard_metadata
from skulk.shared.types.commands import (
    CancelDownload,
    DecideStewardAction,
    ForwarderCommand,
    ForwarderDownloadCommand,
    PlaceInstance,
    ProposeStewardAction,
)
from skulk.shared.types.common import CommandId, ModelId, NodeId, SessionId, SystemId
from skulk.shared.types.events import (
    Event,
    GlobalForwarderEvent,
    IndexedEvent,
    InstanceCreated,
    InstanceDeleted,
    LocalForwarderEvent,
    StateSnapshotHydrated,
    StewardActionProposalChanged,
)
from skulk.shared.types.memory import Memory
from skulk.shared.types.state import State
from skulk.shared.types.state_sync import StateSyncMessage
from skulk.shared.types.steward_actions import (
    StewardActionProposal,
    StewardCancelDownloadAction,
    StewardPlaceModelAction,
    StewardRestartInstanceAction,
    StewardStopInstanceAction,
)
from skulk.shared.types.telemetry import NodeTelemetry, TelemetryView
from skulk.shared.types.worker.downloads import DownloadPending
from skulk.shared.types.worker.instances import (
    InstanceId,
    InstanceMeta,
    MlxRingInstance,
    ShardAssignments,
)
from skulk.shared.types.worker.runners import RunnerId
from skulk.shared.types.worker.shards import PipelineShardMetadata, Sharding
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


def _ordinary_instance() -> MlxRingInstance:
    """Return one minimal ordinary placement for restart lifecycle tests."""
    node_id = NodeId("worker")
    runner_id = RunnerId("runner")
    card = ModelCard(
        model_id=ModelId("org/restart-model"),
        storage_size=Memory.from_gb(8),
        n_layers=4,
        hidden_size=8,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )
    return MlxRingInstance(
        instance_id=InstanceId("original-instance"),
        shard_assignments=ShardAssignments(
            model_id=card.model_id,
            runner_to_shard={
                runner_id: PipelineShardMetadata(
                    model_card=card,
                    device_rank=0,
                    world_size=1,
                    start_layer=0,
                    end_layer=4,
                    n_layers=4,
                )
            },
            node_to_runner={node_id: runner_id},
        ),
        hosts_by_node={node_id: []},
        ephemeral_port=52415,
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
async def test_restart_waits_for_teardown_before_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restart dispatches deletion first and resumes from approved audit state."""
    instance = _ordinary_instance()
    now = datetime.now(tz=timezone.utc)
    proposal = StewardActionProposal(
        action=StewardRestartInstanceAction(instance=instance),
        rationale="The runner is degraded.",
        evidence=("Three consecutive probes failed.",),
        expected_effect="Replace the ordinary model instance.",
        created_at=now,
        expires_at=now + timedelta(minutes=10),
    )
    event_sender, event_receiver = channel[Event]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    master = object.__new__(Master)
    master.state = State(instances={instance.instance_id: instance})
    master._ordered_steward_proposals = {  # pyright: ignore[reportPrivateUsage]
        proposal.proposal_id: proposal
    }
    master._steward_restart_teardown_issued = set()  # pyright: ignore[reportPrivateUsage]
    master._steward_dispatched_effect_issued = set()  # pyright: ignore[reportPrivateUsage]
    master._recently_freed_bytes = {}  # pyright: ignore[reportPrivateUsage]
    master._telemetry_view = TelemetryView()  # pyright: ignore[reportPrivateUsage]
    master._model_trust_approvals = set()  # pyright: ignore[reportPrivateUsage]
    master._system_id = SystemId("master")  # pyright: ignore[reportPrivateUsage]
    master.event_sender = event_sender
    master.download_command_sender = download_sender

    events, command_id, status = (
        await master._execute_approved_steward_action(  # pyright: ignore[reportPrivateUsage]
            proposal
        )
    )

    assert status == "approved"
    assert command_id
    assert len(events) == 1
    assert isinstance(events[0], InstanceDeleted)
    assert proposal.proposal_id in master._steward_restart_teardown_issued  # pyright: ignore[reportPrivateUsage]

    approved = proposal.model_copy(
        update={
            "status": "approved",
            "decided_at": now,
            "decided_by": "trusted_fabric_operator",
            "command_id": command_id,
        }
    )
    master.state = State()
    master._ordered_steward_proposals = {  # pyright: ignore[reportPrivateUsage]
        proposal.proposal_id: approved
    }
    replacement = instance.model_copy(
        update={"instance_id": InstanceId("replacement-instance")}
    )

    def place_after_release(
        _command: object, current_instances: object
    ) -> dict[InstanceId, MlxRingInstance]:
        assert current_instances == {}
        return {replacement.instance_id: replacement}

    monkeypatch.setattr(master, "_place_for_steward_action", place_after_release)
    await master._resume_approved_steward_restarts(  # pyright: ignore[reportPrivateUsage]
        now + timedelta(seconds=30)
    )

    changed = await event_receiver.receive()
    created = await event_receiver.receive()
    assert isinstance(changed, StewardActionProposalChanged)
    assert changed.proposal.status == "dispatched"
    assert isinstance(created, InstanceCreated)
    assert created.instance.instance_id == replacement.instance_id


@pytest.mark.asyncio
async def test_approved_restart_reissues_teardown_once_after_master_failover() -> None:
    """A promoted master resumes an approval whose delete was not replicated."""
    instance = _ordinary_instance()
    now = datetime.now(tz=timezone.utc)
    proposal = StewardActionProposal(
        action=StewardRestartInstanceAction(instance=instance),
        rationale="The runner is degraded.",
        evidence=("Three consecutive probes failed.",),
        expected_effect="Replace the ordinary model instance.",
        created_at=now - timedelta(minutes=1),
        expires_at=now + timedelta(minutes=9),
        status="approved",
        decided_at=now - timedelta(seconds=30),
        decided_by="trusted_fabric_operator",
    )
    event_sender, event_receiver = channel[Event]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    master = object.__new__(Master)
    master.state = State(instances={instance.instance_id: instance})
    master._ordered_steward_proposals = {  # pyright: ignore[reportPrivateUsage]
        proposal.proposal_id: proposal
    }
    master._steward_restart_teardown_issued = set()  # pyright: ignore[reportPrivateUsage]
    master._steward_dispatched_effect_issued = set()  # pyright: ignore[reportPrivateUsage]
    master._recently_freed_bytes = {}  # pyright: ignore[reportPrivateUsage]
    master._telemetry_view = TelemetryView()  # pyright: ignore[reportPrivateUsage]
    master._model_trust_approvals = set()  # pyright: ignore[reportPrivateUsage]
    master._system_id = SystemId("promoted-master")  # pyright: ignore[reportPrivateUsage]
    master.event_sender = event_sender
    master.download_command_sender = download_sender

    await master._resume_approved_steward_restarts(now)  # pyright: ignore[reportPrivateUsage]

    deleted = await event_receiver.receive()
    assert isinstance(deleted, InstanceDeleted)
    assert deleted.instance_id == instance.instance_id
    assert proposal.proposal_id in master._steward_restart_teardown_issued  # pyright: ignore[reportPrivateUsage]

    await master._resume_approved_steward_restarts(  # pyright: ignore[reportPrivateUsage]
        now + timedelta(seconds=1)
    )
    with anyio.move_on_after(0.01) as receive_scope:
        await event_receiver.receive()
    assert receive_scope.cancel_called


@pytest.mark.asyncio
async def test_dispatched_restart_reissues_exact_replacement_after_failover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A promoted master recovers the action-event dispatch window."""
    original = _ordinary_instance()
    replacement_id = InstanceId("replacement-command")
    now = datetime.now(tz=timezone.utc)
    proposal = StewardActionProposal(
        action=StewardRestartInstanceAction(instance=original),
        rationale="The runner is degraded.",
        evidence=("Three consecutive probes failed.",),
        expected_effect="Replace the ordinary model instance.",
        created_at=now - timedelta(minutes=1),
        expires_at=now + timedelta(minutes=9),
        status="dispatched",
        decided_at=now - timedelta(seconds=30),
        decided_by="trusted_fabric_operator",
        command_id=CommandId(str(replacement_id)),
    )
    replacement = original.model_copy(update={"instance_id": replacement_id})
    event_sender, event_receiver = channel[Event]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    master = object.__new__(Master)
    master.state = State()
    master._ordered_steward_proposals = {  # pyright: ignore[reportPrivateUsage]
        proposal.proposal_id: proposal
    }
    master._steward_restart_teardown_issued = set()  # pyright: ignore[reportPrivateUsage]
    master._steward_dispatched_effect_issued = set()  # pyright: ignore[reportPrivateUsage]
    master._recently_freed_bytes = {}  # pyright: ignore[reportPrivateUsage]
    master._telemetry_view = TelemetryView()  # pyright: ignore[reportPrivateUsage]
    master._model_trust_approvals = set()  # pyright: ignore[reportPrivateUsage]
    master._system_id = SystemId("promoted-master")  # pyright: ignore[reportPrivateUsage]
    master.event_sender = event_sender
    master.download_command_sender = download_sender

    def place_exact_command(
        command: PlaceInstance, current_instances: object
    ) -> dict[InstanceId, MlxRingInstance]:
        assert command.command_id == proposal.command_id
        assert current_instances == {}
        return {replacement.instance_id: replacement}

    monkeypatch.setattr(master, "_place_for_steward_action", place_exact_command)
    await master._reconcile_dispatched_steward_actions(now)  # pyright: ignore[reportPrivateUsage]

    created = await event_receiver.receive()
    assert isinstance(created, InstanceCreated)
    assert created.instance.instance_id == replacement_id
    assert proposal.proposal_id in master._steward_dispatched_effect_issued  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_dispatched_place_and_stop_reissue_after_master_failover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A promoted master recovers missing place and stop transition events."""
    original = _ordinary_instance()
    card = next(iter(original.shard_assignments.runner_to_shard.values())).model_card
    now = datetime.now(tz=timezone.utc)
    place_command_id = CommandId("place-command")
    place_proposal = StewardActionProposal(
        action=StewardPlaceModelAction(
            model_card=card,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.MlxRing,
            min_nodes=1,
        ),
        rationale="Capacity is required.",
        evidence=("The requested model has no active instance.",),
        expected_effect="Place the requested model.",
        created_at=now - timedelta(minutes=1),
        expires_at=now + timedelta(minutes=9),
        status="dispatched",
        decided_at=now - timedelta(seconds=30),
        decided_by="trusted_fabric_operator",
        command_id=place_command_id,
    )
    stop_proposal = StewardActionProposal(
        action=StewardStopInstanceAction(
            instance_id=original.instance_id,
            model_id=card.model_id,
        ),
        rationale="The instance is no longer required.",
        evidence=("No active workload requires the instance.",),
        expected_effect="Stop the ordinary model instance.",
        created_at=now - timedelta(minutes=1),
        expires_at=now + timedelta(minutes=9),
        status="dispatched",
        decided_at=now - timedelta(seconds=30),
        decided_by="trusted_fabric_operator",
        command_id=CommandId("stop-command"),
    )
    replacement = original.model_copy(
        update={"instance_id": InstanceId(str(place_command_id))}
    )
    event_sender, event_receiver = channel[Event]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    master = object.__new__(Master)
    master.state = State(instances={original.instance_id: original})
    master._ordered_steward_proposals = {  # pyright: ignore[reportPrivateUsage]
        place_proposal.proposal_id: place_proposal,
        stop_proposal.proposal_id: stop_proposal,
    }
    master._steward_restart_teardown_issued = set()  # pyright: ignore[reportPrivateUsage]
    master._steward_dispatched_effect_issued = set()  # pyright: ignore[reportPrivateUsage]
    master._recently_freed_bytes = {}  # pyright: ignore[reportPrivateUsage]
    master._telemetry_view = TelemetryView()  # pyright: ignore[reportPrivateUsage]
    master._model_trust_approvals = set()  # pyright: ignore[reportPrivateUsage]
    master._system_id = SystemId("promoted-master")  # pyright: ignore[reportPrivateUsage]
    master.event_sender = event_sender
    master.download_command_sender = download_sender

    def place_exact_command(
        command: PlaceInstance,
        current_instances: object,
    ) -> dict[InstanceId, MlxRingInstance]:
        assert command.command_id == place_command_id
        assert current_instances == {original.instance_id: original}
        return {
            original.instance_id: original,
            replacement.instance_id: replacement,
        }

    monkeypatch.setattr(master, "_place_for_steward_action", place_exact_command)
    await master._reconcile_dispatched_steward_actions(now)  # pyright: ignore[reportPrivateUsage]

    created = await event_receiver.receive()
    deleted = await event_receiver.receive()
    assert isinstance(created, InstanceCreated)
    assert created.instance.instance_id == replacement.instance_id
    assert isinstance(deleted, InstanceDeleted)
    assert deleted.instance_id == original.instance_id


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
        expiry_events = master._expire_steward_action_proposals(  # pyright: ignore[reportPrivateUsage]
            expiring.expires_at + timedelta(seconds=1)
        )
        assert len(expiry_events) == 1
        expired = expiry_events[0]
        assert isinstance(expired, StewardActionProposalChanged)
        assert expired.proposal.status == "expired"
        assert expired.proposal.decided_by == "fabric_expiry"
        task_group.cancel_scope.cancel()
