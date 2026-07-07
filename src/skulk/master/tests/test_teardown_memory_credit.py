# pyright: reportPrivateUsage=false
"""Recently-freed memory credit for placement admission (#314).

After a teardown, gossiped node memory lags the freed capacity, so the master
credits a just-deleted instance's per-node footprint back to the placement
fit-check inputs for a short grace window. This keeps back-to-back placements
from being spuriously refused on stale availability; the credit expires so a
genuine shortfall reasserts, and the worker's live pre-load guard (#383) is the
OOM backstop.
"""

import pytest

from skulk.master.main import Master
from skulk.routing.router import get_node_id_keypair
from skulk.shared.models.memory_estimate import (
    estimate_shard_footprint,
    shard_fraction_of_model,
)
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.commands import (
    ForwarderCommand,
    ForwarderDownloadCommand,
)
from skulk.shared.types.common import NodeId, SessionId
from skulk.shared.types.events import (
    Event,
    GlobalForwarderEvent,
    LocalForwarderEvent,
)
from skulk.shared.types.memory import Memory
from skulk.shared.types.profiling import MemoryUsage
from skulk.shared.types.state_sync import StateSyncMessage
from skulk.shared.types.worker.instances import (
    InstanceId,
    MlxRingInstance,
    ShardAssignments,
)
from skulk.shared.types.worker.runners import RunnerId
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.utils.channels import channel


def _make_master() -> Master:
    node_id = NodeId(get_node_id_keypair().to_node_id())
    session_id = SessionId(master_node_id=node_id, election_clock=0)
    ge_sender, _ = channel[GlobalForwarderEvent]()
    _, co_receiver = channel[ForwarderCommand]()
    _, le_receiver = channel[LocalForwarderEvent]()
    state_sync_sender, state_sync_receiver = channel[StateSyncMessage]()
    fcds, _ = channel[ForwarderDownloadCommand]()
    ev_send, _ = channel[Event]()
    return Master(
        node_id,
        session_id,
        event_sender=ev_send,
        global_event_sender=ge_sender,
        local_event_receiver=le_receiver,
        command_receiver=co_receiver,
        state_sync_receiver=state_sync_receiver,
        state_sync_sender=state_sync_sender,
        download_command_sender=fcds,
    )


def _instance(node_id: NodeId) -> tuple[MlxRingInstance, ModelCard]:
    card = ModelCard(
        model_id=ModelId("org/m"),
        storage_size=Memory.from_gb(8.0),
        n_layers=32,
        hidden_size=4096,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )
    runner_id = RunnerId()
    shard = PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=32,
        n_layers=32,
    )
    instance = MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=card.model_id,
            runner_to_shard={runner_id: shard},
            node_to_runner={node_id: runner_id},
        ),
        hosts_by_node={node_id: []},
        ephemeral_port=12345,
    )
    return instance, card


def _mem(gb: float) -> MemoryUsage:
    return MemoryUsage(
        ram_total=Memory.from_gb(64.0),
        ram_available=Memory.from_gb(gb),
        swap_total=Memory(),
        swap_available=Memory(),
    )


def test_no_recent_free_leaves_memory_unchanged() -> None:
    master = _make_master()
    node_id = NodeId(get_node_id_keypair().to_node_id())
    master._telemetry_view.node_memory[node_id] = _mem(4.0)
    memory, _vram = master._placement_memory_inputs()
    assert memory[node_id].ram_available.in_gb == 4.0


def test_freed_instance_credits_its_footprint() -> None:
    master = _make_master()
    node_id = NodeId(get_node_id_keypair().to_node_id())
    instance, card = _instance(node_id)
    # Gossip still shows the loaded memory (deflated availability).
    master._telemetry_view.node_memory[node_id] = _mem(2.0)

    master._record_freed_instance(instance)
    memory, _vram = master._placement_memory_inputs()

    fraction = shard_fraction_of_model(
        next(iter(instance.shard_assignments.runner_to_shard.values()))
    )
    assert fraction is not None
    footprint = estimate_shard_footprint(card, fraction)
    expected = Memory.from_gb(2.0).in_bytes + footprint.in_bytes
    assert memory[node_id].ram_available.in_bytes == expected
    # ram_total is never credited (context-ceiling math reads it).
    assert memory[node_id].ram_total.in_gb == 64.0


def test_credit_expires_after_grace(monkeypatch: pytest.MonkeyPatch) -> None:
    import skulk.master.main as master_main

    master = _make_master()
    node_id = NodeId(get_node_id_keypair().to_node_id())
    instance, _card = _instance(node_id)
    master._telemetry_view.node_memory[node_id] = _mem(2.0)

    base = master_main.time.monotonic()
    master._record_freed_instance(instance)
    # Jump past the grace window: the credit must expire and be pruned.
    monkeypatch.setattr(
        master_main.time,
        "monotonic",
        lambda: base + master_main.RECENTLY_FREED_MEMORY_GRACE_SECONDS + 1.0,
    )
    memory, _vram = master._placement_memory_inputs()
    assert memory[node_id].ram_available.in_gb == 2.0
    assert node_id not in master._recently_freed_bytes
