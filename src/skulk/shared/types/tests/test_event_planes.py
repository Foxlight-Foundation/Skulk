"""Durable control-event classification coverage."""

from datetime import datetime, timezone

from skulk.shared.models.model_cards import ModelId
from skulk.shared.types.chunks import AudioInputChunk
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.events import (
    InputChunkReceived,
    NodeGatheredInfo,
    TestEvent,
    TracesCollected,
    is_persistable_control_event,
)
from skulk.shared.types.memory import Memory
from skulk.shared.types.profiling import MemoryUsage, NetworkInterfaceInfo
from skulk.shared.types.tasks import TaskId
from skulk.utils.info_gatherer.info_gatherer import NodeNetworkInterfaces


def test_event_census_separates_decisions_from_payloads_and_telemetry() -> None:
    """Only durable control facts may cross the master's persistence gate."""

    now = datetime.now(tz=timezone.utc).isoformat()
    node_id = NodeId("node")
    assert is_persistable_control_event(TestEvent())
    assert is_persistable_control_event(
        NodeGatheredInfo(
            node_id=node_id,
            when=now,
            info=NodeNetworkInterfaces(
                ifaces=[NetworkInterfaceInfo(name="en0", ip_address="127.0.0.1")]
            ),
        )
    )
    assert not is_persistable_control_event(
        NodeGatheredInfo(
            node_id=node_id,
            when=now,
            info=MemoryUsage(
                ram_total=Memory.from_mb(16),
                ram_available=Memory.from_mb(8),
                swap_total=Memory(),
                swap_available=Memory(),
            ),
        )
    )
    assert not is_persistable_control_event(
        InputChunkReceived(
            command_id=CommandId("speech"),
            chunk=AudioInputChunk(
                model=ModelId("org/stt"),
                command_id=CommandId("speech"),
                data="YXVkaW8=",
                chunk_index=0,
                total_chunks=1,
                audio_sha256="0" * 64,
            ),
        )
    )
    assert not is_persistable_control_event(
        TracesCollected(task_id=TaskId("task"), rank=0, traces=[])
    )
