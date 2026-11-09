"""Real spawn and descendant exit preserve source-reader ownership."""

import asyncio
import multiprocessing
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from skulk.utils.runtime_files import RuntimeLock
from skulk.worker.runner.demucs import source_fence as fence_module
from skulk.worker.runner.demucs.resident import spawn_resident
from skulk.worker.runner.demucs.source_fence import (
    SourceReadFence,
    acquire_source_read_fence,
)


def test_real_native_spawn_creates_private_files_without_changing_parent_umask(
    tmp_path: Path,
) -> None:
    """Actual child-created output modes meet the addressed producer's file contract."""
    executable = tmp_path / "native-test"
    executable.write_text(
        f"#!{sys.executable}\n"
        + """from pathlib import Path
import os,sys
configuration=Path(sys.argv[sys.argv.index('--serve')+1])
configuration.with_suffix('.output').write_text('owned native output')
"""
    )
    executable.chmod(0o700)
    owner = RuntimeLock(tmp_path / "lifetime", name="attachment.lock")
    configuration = tmp_path / "configuration.json"
    before = tmp_path / "parent-before"
    before.write_text("parent")
    try:
        process = spawn_resident(executable, configuration, owner.descriptor, {})
        process.communicate(timeout=5)
        assert process.returncode == 0
        assert configuration.with_suffix(".output").stat().st_mode & 0o777 == 0o600
        after = tmp_path / "parent-after"
        after.write_text("parent")
        assert after.stat().st_mode & 0o777 == before.stat().st_mode & 0o777
    finally:
        owner.close()


def _native_reader(fence: SourceReadFence, ready: Path) -> None:
    """Start an owned descendant that inherits the same reader-lock description."""
    native = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        pass_fds=(fence.descriptor,),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    ready.write_text(str(native.pid))
    time.sleep(30)


def test_spawned_native_reader_keeps_fence_after_worker_and_runner_death(
    tmp_path: Path,
) -> None:
    """A replacement owner cannot reclaim files while an orphan still reads them."""
    root = tmp_path / "lifetime"
    owner = RuntimeLock(root, name="attachment.lock")
    fence = SourceReadFence(owner.descriptor)
    ready = tmp_path / "ready"
    runner = multiprocessing.get_context("spawn").Process(
        target=_native_reader, args=(fence, ready)
    )
    native_pid: int | None = None
    try:
        runner.start()
        deadline = time.monotonic() + 15
        while not ready.exists() and runner.is_alive() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists(), f"spawned reader did not start: {runner.exitcode}"
        native_pid = int(ready.read_text())
        owner.close()
        fence.close()
        fence.close()
        with pytest.raises(BlockingIOError):
            RuntimeLock(root, name="attachment.lock")
        runner.kill()
        runner.join(5)
        assert not runner.is_alive()
        # SIGKILL closes the runner's duplicate but cannot release the native
        # descendant's descriptor. No timer or remembered process state suffices.
        with pytest.raises(BlockingIOError):
            RuntimeLock(root, name="attachment.lock")
        os.kill(native_pid, signal.SIGKILL)
        native_pid = None
        deadline = time.monotonic() + 5
        while True:
            try:
                recovered = RuntimeLock(root, name="attachment.lock")
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.02)
        recovered.close()
    finally:
        if native_pid is not None:
            os.kill(native_pid, signal.SIGKILL)
        if runner.is_alive():
            runner.kill()
        if runner.pid is not None:
            runner.join(5)
        fence.close()
        owner.close()


def test_closed_fence_cannot_be_serialized(tmp_path: Path) -> None:
    """A closed lifetime cannot become a fresh reader through stale spawn state."""
    owner = RuntimeLock(tmp_path / "lifetime", name="attachment.lock")
    fence = SourceReadFence(owner.descriptor)
    fence.close()
    with pytest.raises(RuntimeError, match="closed"):
        fence.__reduce__()
    owner.close()


async def test_async_acquisition_waits_for_actual_prior_reader_exit(
    tmp_path: Path,
) -> None:
    """A busy native fence delays acquisition without blocking other async work."""
    root = tmp_path / "reader"
    previous = RuntimeLock(root, name="attachment.lock")
    pending = asyncio.create_task(acquire_source_read_fence(root))
    try:
        await asyncio.sleep(0.1)
        assert not pending.done()
        previous.close()
        fence = await asyncio.wait_for(pending, 3)
        try:
            with pytest.raises(BlockingIOError):
                RuntimeLock(root, name="attachment.lock")
        finally:
            fence.close()
        RuntimeLock(root, name="attachment.lock").close()
    finally:
        previous.close()
        if not pending.done():
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending


async def test_busy_reader_has_finite_acquisition_budget(tmp_path: Path) -> None:
    """An orphaned reader cannot hold a preparation coroutine indefinitely."""
    root = tmp_path / "reader"
    previous = RuntimeLock(root, name="attachment.lock")
    try:
        with pytest.raises(TimeoutError):
            await acquire_source_read_fence(root, timeout_seconds=0.05)
    finally:
        previous.close()
    RuntimeLock(root, name="attachment.lock").close()


async def test_cancellation_joins_lock_effect_and_releases_acquired_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A late thread acquisition cannot leak a lock after its caller cancels."""
    started, release = threading.Event(), threading.Event()
    root = tmp_path / "reader"

    def delayed_lock(path: Path, *, name: str) -> RuntimeLock:
        owner = RuntimeLock(path, name=name)
        started.set()
        assert release.wait(3)
        return owner

    monkeypatch.setattr(fence_module, "RuntimeLock", delayed_lock)
    pending = asyncio.create_task(acquire_source_read_fence(root))
    try:
        assert await asyncio.to_thread(started.wait, 3)
        pending.cancel()
        await asyncio.sleep(0)
        assert not pending.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
        RuntimeLock(root, name="attachment.lock").close()
    finally:
        release.set()
        if not pending.done():
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
