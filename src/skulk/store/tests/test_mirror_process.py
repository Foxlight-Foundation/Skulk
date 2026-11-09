"""Real observation pipes retain completion through leader exit and group drain."""

import asyncio
import os
import sys
import time
from pathlib import Path

import psutil
import pytest

from skulk.store.mirror_process import MirrorProcess
from skulk.store.mirror_worker import MirrorWorkerEvent
from skulk.store.preservation_io import PreservationIO


async def ready(path: Path) -> None:
    """Bound cold interpreter startup separately from the active pipe assertions."""
    async with asyncio.timeout(120):
        while not path.exists():
            await asyncio.sleep(0.01)


@pytest.mark.parametrize("stall_loop", [False, True])
async def test_completed_worker_keeps_final_observation_after_read_poll_expires(
    tmp_path: Path, stall_loop: bool
) -> None:
    """Leader exit never discards the real success line queued during a loop stall."""
    marker = tmp_path / "ready"
    program = (
        "import sys; from pathlib import Path; "
        "Path(sys.argv[1]).touch(); sys.stdin.readline(); "
        "print('{\"event\":\"result\"}',flush=True)"
    )
    storage = PreservationIO()
    owned = await MirrorProcess.start(
        (sys.executable, "-c", program, str(marker)), None, storage
    )
    observations: list[MirrorWorkerEvent] = []
    reading: asyncio.Task[None] | None = None
    try:
        await ready(marker)
        reading = asyncio.create_task(owned.observations(observations.append))
        await asyncio.sleep(0.01)
        assert owned.process.stdin is not None
        owned.process.stdin.write("finish\n")
        owned.process.stdin.flush()
        if stall_loop:
            # A real completed child and OS pipe outlive this delayed poll;
            # mocking alive() or feeding a StreamReader misses that race.
            time.sleep(2.2)
        await asyncio.wait_for(reading, 10)
        assert [event.event for event in observations] == ["result"]
        await owned.stop()
        assert owned.process.returncode == 0
    finally:
        if reading is not None and not reading.done():
            reading.cancel()
            await asyncio.gather(reading, return_exceptions=True)
        await owned.stop()
        await storage.close()


async def test_exited_leader_drains_inherited_writer_before_observation_returns(
    tmp_path: Path,
) -> None:
    """An orphan holding stdout cannot hide completion or leave the group alive."""
    marker = tmp_path / "writer.pid"
    program = (
        "import subprocess,sys; from pathlib import Path; "
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']); "
        "Path(sys.argv[1]).write_text(str(child.pid)); "
        "print('{\"event\":\"result\"}',flush=True)"
    )
    storage = PreservationIO()
    owned = await MirrorProcess.start(
        (sys.executable, "-c", program, str(marker)), None, storage
    )
    observations: list[MirrorWorkerEvent] = []
    try:
        await ready(marker)
        writer = psutil.Process(int(marker.read_text()))
        await asyncio.wait_for(owned.observations(observations.append), 10)
        assert [event.event for event in observations] == ["result"]
        assert not writer.is_running() or writer.status() == psutil.STATUS_ZOMBIE
        assert owned.process.returncode == 0
    finally:
        await owned.stop()
        await storage.close()


async def test_cancelled_observer_still_drains_live_worker(tmp_path: Path) -> None:
    """The final-output repair must retain explicit cancellation and ownership."""
    marker = tmp_path / "ready"
    program = (
        "import signal,sys,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM,lambda *_:sys.exit(0)); "
        "Path(sys.argv[1]).touch(); sys.stdin.readline(); time.sleep(30)"
    )
    storage = PreservationIO()
    owned = await MirrorProcess.start(
        (sys.executable, "-c", program, str(marker)), None, storage
    )
    reading: asyncio.Task[None] | None = None
    try:
        await ready(marker)
        reading = asyncio.create_task(owned.observations(lambda _: None))
        await asyncio.sleep(0)
        reading.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reading
        stopping = asyncio.create_task(owned.stop())
        await asyncio.sleep(0)
        stopping.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(stopping, 15)
        assert owned.process.returncode is not None
        with pytest.raises(ProcessLookupError):
            os.kill(owned.process.pid, 0)
    finally:
        if reading is not None and not reading.done():
            reading.cancel()
            await asyncio.gather(reading, return_exceptions=True)
        await owned.stop()
        await storage.close()
