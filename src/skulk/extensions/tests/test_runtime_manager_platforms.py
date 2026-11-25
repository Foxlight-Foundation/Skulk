"""The plugin manager starts, answers and stops on every platform Skulk runs on."""

import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from skulk.extensions import runtime_manager
from skulk.extensions.runtime_attachment import HostSettings
from skulk.extensions.runtime_manager import (
    InventoryRequest,
    RuntimeManager,
    manager_request,
)
from skulk.utils.runtime_files import (
    RuntimeLock,
    has_root_authority,
    private_directory,
    write_private,
)


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A provisioned, owner-only service root, as local setup leaves it."""
    # Test runners on some platforms run as an administrator; the rule that
    # refuses that has its own tests.
    monkeypatch.setattr(runtime_manager, "has_root_authority", lambda: False)
    path = tmp_path / "service"
    private_directory(path)
    write_private(
        path / "host.json",
        HostSettings(transport_node_id="node-a", profile_id="a" * 32)
        .model_dump_json()
        .encode(),
    )
    return path


async def test_a_manager_serves_its_inventory_and_stops(root: Path) -> None:
    manager = RuntimeManager(root)
    await manager.start()
    try:
        assert manager.boot is not None
        await manager.boot
        assert manager.errors == {}
        for _ in range(3):
            reply = await manager_request(root, InventoryRequest())
            result = reply["result"]
            assert isinstance(result, dict)
            assert result["installations"] == []
    finally:
        await manager.close()

    with pytest.raises(OSError):
        await manager_request(root, InventoryRequest())


async def test_one_root_has_one_manager(root: Path) -> None:
    manager = RuntimeManager(root)
    await manager.start()
    try:
        second = RuntimeManager(root)
        with pytest.raises(BlockingIOError):
            await second.start()
        # The refused one did not disturb the first.
        assert "result" in await manager_request(root, InventoryRequest())
    finally:
        await manager.close()

    again = RuntimeManager(root)
    await again.start()
    try:
        assert "result" in await manager_request(root, InventoryRequest())
    finally:
        await again.close()


def test_a_manager_refuses_root_authority(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime_manager, "has_root_authority", lambda: True)
    with pytest.raises(ValueError, match="without root"):
        RuntimeManager(root)


@pytest.mark.skipif(has_root_authority(), reason="the manager refuses root authority")
def test_the_manager_runs_as_a_program_and_stops_when_asked(root: Path) -> None:
    """``runtime_manager serve`` answers, and a stop signal ends it in order."""
    if sys.platform == "win32":
        import ctypes

        if not ctypes.WinDLL("kernel32").GetConsoleCP():
            pytest.skip("no console to send Ctrl-Break through")
        request_to_stop, new_group = (
            signal.CTRL_BREAK_EVENT,
            subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    else:
        request_to_stop, new_group = signal.SIGTERM, 0
    program = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "skulk.extensions.runtime_manager",
            "serve",
            "--root",
            str(root),
        ],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        creationflags=new_group,
    )
    try:
        deadline = time.monotonic() + 60
        while True:
            assert program.poll() is None, "the manager ended early"
            try:
                reply = asyncio.run(manager_request(root, InventoryRequest()))
            except OSError:
                assert time.monotonic() < deadline, "the manager never answered"
                time.sleep(0.1)
                continue
            assert "result" in reply
            break
        os.kill(program.pid, request_to_stop)
        assert program.wait(60) == 0
    finally:
        program.kill()
        program.wait()
    # It let go of everything: another manager could start here.
    RuntimeLock(root, "manager.lock").close()


async def test_a_manager_names_the_copy_it_was_started_from_whatever_it_selects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After selecting its successor, the old manager still answers as itself.

    A reload leaves the old manager answering for a moment with the new copy
    already selected. What it reports is the copy it runs, taken once when it
    was constructed, so nobody waiting for the successor takes it for one.
    """
    from skulk.extensions.runtime_manager import ReloadRuntimeRequest
    from skulk.extensions.tests.test_managed_services import manager_fixture
    from skulk.extensions.tests.test_service_snapshot import staged

    running = staged(tmp_path)
    successor = staged(tmp_path, name="b" * 32)
    assert successor.manifest_sha256 != running.manifest_sha256
    monkeypatch.setattr(
        sys, "prefix", str(tmp_path / "core-runtimes" / running.generation / "runtime")
    )
    manager = manager_fixture(tmp_path, monkeypatch)
    manager.request_stop = lambda: None
    as_started = {
        "generation": running.generation,
        "manifest_sha256": running.manifest_sha256,
    }
    await manager.start()
    try:
        before = await manager_request(tmp_path, InventoryRequest())
        assert before == {
            "result": {
                "installations": [],
                "reload_runtime": True,
                "runtime": as_started,
            }
        }
        selected = await manager_request(
            tmp_path,
            ReloadRuntimeRequest(
                generation=successor.generation,
                manifest_sha256=successor.manifest_sha256,
            ),
        )
        assert selected == {
            "result": {"generation": successor.generation, "restarting": True}
        }
        assert await manager_request(tmp_path, InventoryRequest()) == before
    finally:
        await manager.close()
