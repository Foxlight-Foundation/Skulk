# pyright: reportPrivateUsage=false
"""Owner-addressed trace data transport coverage."""

from pathlib import Path

import anyio
import pytest

import skulk.api.main as api_main
from skulk.api.main import API
from skulk.routing.trace_data import TraceDataPacket
from skulk.shared.election import ElectionMessage
from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.common import NodeId
from skulk.shared.types.events import IndexedEvent, TraceEventData
from skulk.shared.types.tasks import TaskId
from skulk.utils.channels import channel


@pytest.mark.asyncio
async def test_api_merges_all_trace_ranks_without_control_events(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The owning API persists a trace only after every expected rank arrives."""

    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    _, event_receiver = channel[IndexedEvent]()
    _, election_receiver = channel[ElectionMessage]()
    trace_sender, trace_receiver = channel[TraceDataPacket](2)
    monkeypatch.setattr(api_main, "SKULK_TRACING_CACHE_DIR", tmp_path)
    api = API(
        NodeId("api-owner"),
        port=52415,
        event_receiver=event_receiver,
        command_sender=command_sender,
        download_command_sender=download_sender,
        election_receiver=election_receiver,
        trace_data_receiver=trace_receiver,
        enable_event_log=False,
        mount_dashboard=False,
    )
    task_id = TaskId("trace-task")
    trace_zero = TraceEventData(
        name="rank-zero",
        start_us=1,
        duration_us=2,
        rank=0,
        category="decode",
    )
    trace_one = TraceEventData(
        name="rank-one",
        start_us=3,
        duration_us=4,
        rank=1,
        category="decode",
    )

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(api._apply_trace_data)
        await trace_sender.send(
            TraceDataPacket(
                owner_node=api.node_id,
                source_node=NodeId("worker-one"),
                task_id=task_id,
                rank=1,
                expected_ranks=(0, 1),
                traces=(trace_one,),
            )
        )
        await anyio.sleep(0)
        assert not (tmp_path / f"trace_{task_id}.json").exists()
        await trace_sender.send(
            TraceDataPacket(
                owner_node=api.node_id,
                source_node=NodeId("worker-zero"),
                task_id=task_id,
                rank=0,
                expected_ranks=(0, 1),
                traces=(trace_zero,),
            )
        )
        with anyio.fail_after(1):
            while not (tmp_path / f"trace_{task_id}.json").exists():
                await anyio.sleep(0)
        task_group.cancel_scope.cancel()

    assert api._pending_trace_data == {}
