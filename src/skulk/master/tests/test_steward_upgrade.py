# pyright: reportPrivateUsage=false, reportAttributeAccessIssue=false
"""Controlled best-brain convergence for the intelligent-fabric steward."""

from collections.abc import Mapping

import pytest

from skulk.master.main import Master
from skulk.routing.router import get_node_id_keypair
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.commands import (
    ForwarderCommand,
    ForwarderDownloadCommand,
    StartDownload,
)
from skulk.shared.types.common import NodeId, SessionId
from skulk.shared.types.events import (
    Event,
    GlobalForwarderEvent,
    InstanceDeleted,
    LocalForwarderEvent,
)
from skulk.shared.types.memory import Memory
from skulk.shared.types.profiling import MemoryUsage
from skulk.shared.types.state import State
from skulk.shared.types.state_sync import StateSyncMessage
from skulk.shared.types.worker.downloads import DownloadCompleted
from skulk.shared.types.worker.instances import (
    Instance,
    InstanceId,
    MlxRingInstance,
    ShardAssignments,
)
from skulk.shared.types.worker.runners import RunnerId, RunnerReady
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.utils.channels import Receiver, channel


def _master() -> tuple[Master, Receiver[ForwarderDownloadCommand], Receiver[Event]]:
    node_id = NodeId(get_node_id_keypair().to_node_id())
    session_id = SessionId(master_node_id=node_id, election_clock=0)
    global_sender, _ = channel[GlobalForwarderEvent]()
    _, command_receiver = channel[ForwarderCommand]()
    _, local_event_receiver = channel[LocalForwarderEvent]()
    state_sync_sender, state_sync_receiver = channel[StateSyncMessage]()
    download_sender, download_receiver = channel[ForwarderDownloadCommand]()
    event_sender, event_receiver = channel[Event]()
    return (
        Master(
            node_id,
            session_id,
            event_sender=event_sender,
            global_event_sender=global_sender,
            local_event_receiver=local_event_receiver,
            command_receiver=command_receiver,
            state_sync_receiver=state_sync_receiver,
            state_sync_sender=state_sync_sender,
            download_command_sender=download_sender,
        ),
        download_receiver,
        event_receiver,
    )


def _instance(
    node_id: NodeId, model_id: str
) -> tuple[MlxRingInstance, PipelineShardMetadata]:
    card = ModelCard(
        model_id=ModelId(model_id),
        storage_size=Memory.from_gb(1),
        n_layers=4,
        hidden_size=16,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )
    runner_id = RunnerId()
    shard = PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=4,
        n_layers=4,
    )
    return (
        MlxRingInstance(
            instance_id=InstanceId(),
            shard_assignments=ShardAssignments(
                model_id=card.model_id,
                runner_to_shard={runner_id: shard},
                node_to_runner={node_id: runner_id},
            ),
            hosts_by_node={node_id: []},
            ephemeral_port=12345,
            system_role="steward",
        ),
        shard,
    )


def _memory(available_gb: float) -> MemoryUsage:
    return MemoryUsage(
        ram_total=Memory.from_gb(16),
        ram_available=Memory.from_gb(available_gb),
        swap_total=Memory(),
        swap_available=Memory(),
    )


@pytest.mark.asyncio
async def test_upgrade_stages_then_waits_for_idle_before_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A better brain never creates a standby or interrupts active work."""
    master, download_receiver, event_receiver = _master()
    old, _ = _instance(master.node_id, "org/current")
    candidate, candidate_shard = _instance(master.node_id, "org/better")
    old_runner = next(iter(old.shard_assignments.node_to_runner.values()))
    master.state = State(
        instances={old.instance_id: old},
        runners={old_runner: RunnerReady()},
    )

    async def place_candidate(
        _model_ref: str,
        current_instances: Mapping[InstanceId, Instance],
        node_memory: Mapping[NodeId, MemoryUsage] | None = None,
    ) -> dict[InstanceId, Instance]:
        assert old.instance_id not in current_instances
        assert node_memory is not None
        assert node_memory[master.node_id].ram_available.in_gb > 2
        return {**current_instances, candidate.instance_id: candidate}

    monkeypatch.setattr(master, "_place_steward_model", place_candidate)
    master._telemetry_view.node_memory[master.node_id] = _memory(2)
    now = [0.0]
    monkeypatch.setattr("skulk.master.main.time.monotonic", lambda: now[0])

    preference = ("org/better", "org/current")
    await master._maintain_steward_upgrade(old.instance_id, preference)
    assert download_receiver.collect() == []

    now[0] = 301.0
    await master._maintain_steward_upgrade(old.instance_id, preference)
    downloads = download_receiver.collect()
    assert len(downloads) == 1
    assert isinstance(downloads[0].command, StartDownload)
    assert str(downloads[0].command.shard_metadata.model_card.model_id) == "org/better"
    assert old.instance_id in master.state.instances

    master.state = master.state.model_copy(
        update={
            "downloads": {
                master.node_id: [
                    DownloadCompleted(
                        node_id=master.node_id,
                        shard_metadata=candidate_shard,
                        total=Memory.from_gb(1),
                    )
                ]
            }
        }
    )
    now[0] = 302.0
    await master._maintain_steward_upgrade(old.instance_id, preference)
    assert event_receiver.collect() == []

    now[0] = 333.0
    await master._maintain_steward_upgrade(old.instance_id, preference)
    events = event_receiver.collect()
    assert any(
        isinstance(event, InstanceDeleted) and event.instance_id == old.instance_id
        for event in events
    )
    assert not any(event.__class__.__name__ == "InstanceCreated" for event in events)


def test_completed_steward_download_requires_exact_shard_metadata() -> None:
    """A stale completion for another shard cannot authorize replacement."""
    master, _download_receiver, _event_receiver = _master()
    candidate, candidate_shard = _instance(master.node_id, "org/better")
    _other, other_shard = _instance(master.node_id, "org/better")
    other_shard = other_shard.model_copy(update={"end_layer": 3})
    master.state = State(
        downloads={
            master.node_id: [
                DownloadCompleted(
                    node_id=master.node_id,
                    shard_metadata=other_shard,
                    total=Memory.from_gb(1),
                )
            ]
        }
    )

    assert candidate.shard_assignments.model_id == other_shard.model_card.model_id
    assert not master._steward_model_download_completed(master.node_id, candidate_shard)


def test_replacement_admission_credits_only_outgoing_steward_memory() -> None:
    """Prestaging eligibility can account for memory released at replacement."""
    master, _download_receiver, _event_receiver = _master()
    old, _shard = _instance(master.node_id, "org/current")
    other_node = NodeId(get_node_id_keypair().to_node_id())
    master._telemetry_view.node_memory = {
        master.node_id: _memory(2),
        other_node: _memory(4),
    }

    replacement = master._steward_replacement_memory(old)

    assert replacement[master.node_id].ram_available.in_gb > 2
    assert replacement[master.node_id].ram_available.in_gb <= 16
    assert replacement[other_node].ram_available.in_gb == 4
