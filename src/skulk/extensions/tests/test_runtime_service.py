"""Actual isolated generic owner launch, trust refusal and bounded process lifetime."""

import asyncio
import dataclasses
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import psutil
import pytest

from skulk.extensions.runtime_files import RuntimeLock, read_private, write_private
from skulk.extensions.runtime_integrity import generation_directory
from skulk.extensions.runtime_selection import RuntimeSelector
from skulk.extensions.runtime_service import (
    RuntimeService,
    RuntimeServiceStatus,
    lifetime_arguments,
)
from skulk.extensions.tests.test_runtime_install import artifacts

# How the stand-in owner waits for the end of its launcher: at the place the
# launcher named on its command line.
OWNER_WAITS = (
    "(sys.stdin.buffer if a.lifetime_stdin else os.fdopen(a.lifetime_fd,'rb')).read()"
)
# A stand-in for a plugin's owner, in one source for every platform. It takes
# the supervisor lock as a real owner does, says it has started, and ends when
# its launcher's end of the lifetime pipe closes. It runs in the runtime's
# isolated interpreter, so it takes the lock with the standard library alone:
# on Windows that is the one byte, far past the end of the file, that
# ``RuntimeLock`` locks there.
OWNER_SOURCE = f"""import argparse,os,sys,time
p=argparse.ArgumentParser()
p.add_argument('--root');p.add_argument('--lifetime-fd',type=int)
p.add_argument('--lifetime-stdin',action='store_true')
a=p.parse_args()
f=os.open(os.path.join(a.root,'supervisor.lock'),os.O_CREAT|os.O_RDWR,0o600)
if sys.platform=='win32':
    import ctypes,msvcrt
    class Region(ctypes.Structure):
        _fields_=[('a',ctypes.c_void_p),('b',ctypes.c_void_p),('low',ctypes.c_ulong),('high',ctypes.c_ulong),('e',ctypes.c_void_p)]
    region=Region(None,None,0xFFFFFFFF,0x7FFFFFFF,None)
    held=ctypes.WinDLL('kernel32').LockFileEx(ctypes.c_void_p(msvcrt.get_osfhandle(f)),3,0,1,0,ctypes.byref(region))
    assert held,'another owner holds the supervisor lock'
else:
    import fcntl,signal
    fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,lambda *args:None)
print('sensitive fixture output',flush=True)
{OWNER_WAITS}
os.close(f)
"""
# What ending a whole tree leaves as the owner's exit code.
KILLED = 1 if sys.platform == "win32" else -9


async def installed(
    root: Path, monkeypatch: pytest.MonkeyPatch, source: str = OWNER_SOURCE
) -> RuntimeSelector:
    """Stage a real signed fixture in an empty offline runtime and select it."""
    metadata, trust, host = artifacts(root / "source", owner_source=source)
    monkeypatch.setattr("skulk.extensions.runtime_install.measure_host", lambda: host)
    selector = RuntimeSelector(root / "installed")
    write_private(
        selector.root / "publisher-trust.json", trust.model_dump_json().encode()
    )
    staged = await selector.installer.stage(metadata, root / "source")
    await selector.activate(staged.runtime_digest, expected_revision=0)
    return selector


def status(root: Path) -> RuntimeServiceStatus:
    """Read the protected observation without exposing any child output."""
    return RuntimeServiceStatus.model_validate_json(
        read_private(root / "service-status.json")
    )


async def running(service: RuntimeService) -> None:
    """Wait for actual fixture output, proving entrypoint and ownership acquisition."""
    async with asyncio.timeout(10):
        while service.output_bytes == 0:
            await asyncio.sleep(0.01)
    assert status(service.root).state == "running"


async def test_service_owns_fixed_runtime_and_closes_lifetime_pipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Output is discarded, duplicate launch is fenced and pipe EOF ends the owner."""
    selector = await installed(tmp_path, monkeypatch)
    stopped = asyncio.Event()
    service = RuntimeService(selector.root)
    task = asyncio.create_task(service.serve(stopped))
    try:
        await running(service)
        with pytest.raises(BlockingIOError):
            await RuntimeService(selector.root).serve(asyncio.Event())
        with pytest.raises(BlockingIOError):
            RuntimeLock(selector.root, "supervisor.lock")
        assert b"sensitive" not in read_private(selector.root / "service-status.json")
    finally:
        stopped.set()
        async with asyncio.timeout(10):
            assert await task
    observation = status(selector.root)
    assert observation.state == "stopped" and observation.active_digest is None
    assert observation.output_bytes > 0
    assert service.process is not None and service.process.returncode == 0
    RuntimeLock(selector.root, "service.lock").close()
    RuntimeLock(selector.root, "supervisor.lock").close()


async def test_service_cancellation_reaps_owned_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancellation cannot return before the fixed owner has relinquished ownership."""
    selector = await installed(tmp_path, monkeypatch)
    service = RuntimeService(selector.root)
    task = asyncio.create_task(service.serve(asyncio.Event()))
    await running(service)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        async with asyncio.timeout(10):
            await task
    assert service.process is not None and service.process.returncode == 0
    RuntimeLock(selector.root, "supervisor.lock").close()
    RuntimeLock(selector.root, "service.lock").close()


@pytest.mark.parametrize("stop_before_release", [False, True])
async def test_start_waits_for_metadata_fence_without_failed_lifetime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stop_before_release: bool,
) -> None:
    """Brief discovery ownership cannot spend the owner's restart crash budget."""
    selector = await installed(tmp_path, monkeypatch)
    inspection = RuntimeLock(selector.installer.installer)
    stopped = asyncio.Event()
    service = RuntimeService(selector.root)
    task = asyncio.create_task(service.serve(stopped))
    try:
        async with asyncio.timeout(2):
            while not (selector.root / "service-status.json").exists():
                await asyncio.sleep(0.001)
        await asyncio.sleep(0.05)
        assert not task.done(), "discovery contention became a failed owner lifetime"
        assert service.process is None
        assert status(selector.root).state == "verifying"
        if stop_before_release:
            stopped.set()
        inspection.close()
        if stop_before_release:
            async with asyncio.timeout(5):
                assert await task
            assert service.process is None
        else:
            await running(service)
    finally:
        inspection.close()
        stopped.set()
        async with asyncio.timeout(5):
            await task
    assert status(selector.root).state == "stopped"
    assert status(selector.root).error_code is None
    RuntimeLock(selector.root, "service.lock").close()
    RuntimeLock(selector.root, "supervisor.lock").close()


async def test_service_refuses_tampered_runtime_and_disabled_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No private interpreter executes after integrity failure; disable stays usable."""
    selector = await installed(tmp_path, monkeypatch)
    selection = selector.current()
    assert selection is not None
    generation = generation_directory(selector.root, selection.runtime_digest)
    (generation / "runtime/unsealed.py").write_bytes(b"raise Exception('never run')")
    service = RuntimeService(selector.root)
    assert not await service.serve(asyncio.Event())
    assert service.process is None
    assert status(selector.root).error_code == "verification_failed"
    selector.disable(expected_revision=1)
    assert await RuntimeService(selector.root).serve(asyncio.Event())
    assert status(selector.root).state == "stopped"


async def test_owner_exit_is_not_replayed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even a zero exit is unexpected without a stop request and ends this lifetime."""
    selector = await installed(tmp_path, monkeypatch, "print('exit fixture')\n")
    service = RuntimeService(selector.root)
    assert not await service.serve(asyncio.Event())
    assert status(selector.root).error_code == "owner_exited"
    assert service.process is not None and service.process.returncode == 0


async def test_running_service_revalidates_trust_and_stops_revoked_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An observed running process is withdrawn when current release trust changes."""
    from skulk.extensions.runtime_artifacts import RuntimeTrust

    selector = await installed(tmp_path, monkeypatch)
    monkeypatch.setattr("skulk.extensions.runtime_service._CHECK_SECONDS", 0.05)
    service = RuntimeService(selector.root)
    task = asyncio.create_task(service.serve(asyncio.Event()))
    try:
        await running(service)
        selection = selector.current()
        assert selection is not None
        trust = RuntimeTrust.model_validate_json(
            read_private(selector.root / "publisher-trust.json")
        )
        revoked = trust.model_copy(
            update={"revision": 2, "revoked_artifacts": (selection.runtime_digest,)}
        )
        write_private(
            selector.root / "publisher-trust.json", revoked.model_dump_json().encode()
        )
        async with asyncio.timeout(10):
            assert not await task
        assert status(selector.root).error_code == "verification_failed"
        assert service.process is not None and service.process.returncode == 0
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_owner_ignoring_shutdown_is_killed_within_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unresponsive owner cannot hold cancellation or service ownership forever."""
    selector = await installed(
        tmp_path,
        monkeypatch,
        OWNER_SOURCE.replace(OWNER_WAITS, "time.sleep(60)"),
    )
    monkeypatch.setattr("skulk.extensions.runtime_service._SHUTDOWN_SECONDS", 0.05)
    service = RuntimeService(selector.root)
    stopped = asyncio.Event()
    task = asyncio.create_task(service.serve(stopped))
    try:
        await running(service)
    finally:
        stopped.set()
        async with asyncio.timeout(5):
            assert await task
    assert service.process is not None and service.process.returncode == KILLED
    RuntimeLock(selector.root, "supervisor.lock").close()


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Windows ends what an owner left when the owner is released",
)
async def test_an_owner_that_exits_as_the_bound_runs_out_does_not_leave_its_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bound ends the whole tree even when its leader went a moment before.

    The owner ends at once when asked, leaving a child that holds the
    supervisor fence. The launcher's wait for the owner is made to miss that
    exit, as it does when the exit and the bound fall together.
    """
    source = OWNER_SOURCE.replace(
        "print('sensitive fixture output',flush=True)",
        "import subprocess\n"
        "subprocess.Popen([sys.executable,'-I','-c','import time;time.sleep(120)'],"
        "pass_fds=(f,))\n"
        "print('sensitive fixture output',flush=True)",
    )
    selector = await installed(tmp_path, monkeypatch, source)
    monkeypatch.setattr("skulk.extensions.runtime_service._SHUTDOWN_SECONDS", 0.5)
    service = RuntimeService(selector.root)
    stopped = asyncio.Event()
    task = asyncio.create_task(service.serve(stopped))
    try:
        await running(service)
        assert service.process is not None
        owner = service.process
        wait, missed = owner.wait, False

        async def misses_the_exit() -> int:
            nonlocal missed
            if missed:
                return await wait()
            missed = True
            await wait()
            await asyncio.sleep(3600)
            raise AssertionError("the bound never ran out")

        monkeypatch.setattr(owner, "wait", misses_the_exit)
    finally:
        stopped.set()
        async with asyncio.timeout(15):
            assert await task
    assert missed and service.process.returncode == 0
    assert status(selector.root).state == "stopped"
    RuntimeLock(selector.root, "supervisor.lock").close()


def test_the_owner_is_told_where_its_end_of_the_lifetime_pipe_is() -> None:
    """A descriptor number where descriptors are inherited; standard input otherwise."""
    assert lifetime_arguments(7) == ("--lifetime-fd", "7")
    assert lifetime_arguments(None) == ("--lifetime-stdin",)


async def test_an_owner_starts_and_stops_again_and_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each lifetime ends in order and leaves nothing that stops the next."""
    selector = await installed(tmp_path, monkeypatch)
    for _ in range(3):
        stopped = asyncio.Event()
        service = RuntimeService(selector.root)
        task = asyncio.create_task(service.serve(stopped))
        try:
            await running(service)
        finally:
            stopped.set()
            async with asyncio.timeout(10):
                assert await task
        assert service.process is not None and service.process.returncode == 0
        assert status(selector.root).state == "stopped"
    RuntimeLock(selector.root, "service.lock").close()
    RuntimeLock(selector.root, "supervisor.lock").close()


async def test_an_owner_built_without_the_lifetime_argument_ends_the_lifetime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An owner that does not know where its launcher put the pipe never runs on."""
    selector = await installed(
        tmp_path,
        monkeypatch,
        "import argparse,time\n"
        "p=argparse.ArgumentParser();p.add_argument('--root')\n"
        "p.parse_args()\n"
        "time.sleep(60)\n",
    )
    service = RuntimeService(selector.root)
    async with asyncio.timeout(30):
        assert not await service.serve(asyncio.Event())
    assert status(selector.root).error_code == "owner_exited"
    assert service.process is not None and service.process.returncode == 2
    RuntimeLock(selector.root, "service.lock").close()


# A launcher in a process of its own, so that it can be ended without warning.
# It serves the installation it is given until then.
_LAUNCHER = (
    "import asyncio, json, sys\n"
    "from pathlib import Path\n"
    "from skulk.extensions import runtime_install\n"
    "from skulk.extensions.runtime_artifacts import QualifiedHost\n"
    "from skulk.extensions.runtime_service import RuntimeService\n"
    "host = QualifiedHost(**json.loads(sys.argv[2]))\n"
    "runtime_install.measure_host = lambda: host\n"
    "asyncio.run(RuntimeService(Path(sys.argv[1])).serve(asyncio.Event()))\n"
)


def _gone(identifier: int) -> bool:
    try:
        return psutil.Process(identifier).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


async def test_a_launcher_that_dies_leaves_an_owner_that_shuts_itself_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The owner is asked, not ended: it puts its things away with no launcher alive.

    Until it has gone it keeps the service fence, so no second launcher can
    start an owner beside it. Afterwards the next launcher starts cleanly.
    """
    source = OWNER_SOURCE.replace(
        "print('sensitive fixture output',flush=True)",
        "open(os.path.join(a.root,'owner-started'),'w').write(str(os.getpid()))\n"
        "print('sensitive fixture output',flush=True)",
    ).replace(
        "os.close(f)",
        # Shutting down takes a moment, and is seen to finish.
        "time.sleep(1.5)\n"
        "open(os.path.join(a.root,'owner-put-away'),'w').write('done')\n"
        "os.close(f)",
    )
    metadata, trust, host = artifacts(tmp_path / "source", owner_source=source)
    monkeypatch.setattr("skulk.extensions.runtime_install.measure_host", lambda: host)
    selector = RuntimeSelector(tmp_path / "installed")
    write_private(
        selector.root / "publisher-trust.json", trust.model_dump_json().encode()
    )
    staged = await selector.installer.stage(metadata, tmp_path / "source")
    await selector.activate(staged.runtime_digest, expected_revision=0)
    started = selector.root / "owner-started"
    put_away = selector.root / "owner-put-away"

    launcher = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _LAUNCHER,
            str(selector.root),
            json.dumps(dataclasses.asdict(host)),
        ],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    owner = 0
    try:
        async with asyncio.timeout(60):
            while not started.exists() or not started.read_text():
                assert launcher.poll() is None, "the launcher ended early"
                await asyncio.sleep(0.05)
        owner = int(started.read_text())
        assert status(selector.root).state == "running"

        launcher.kill()
        launcher.wait()
        ended = time.monotonic()
        # Still running, still fenced, and not yet done putting away.
        with pytest.raises(BlockingIOError):
            RuntimeLock(selector.root, "service.lock")
        assert not _gone(owner) and not put_away.exists()
        assert time.monotonic() - ended < 1.0, "too slow to have seen the owner"

        async with asyncio.timeout(30):
            while not _gone(owner):
                await asyncio.sleep(0.05)
        assert put_away.read_text() == "done"
    finally:
        launcher.kill()
        launcher.wait()
        if owner and not _gone(owner):
            psutil.Process(owner).kill()

    # Nothing of the lost lifetime stands in the way of the next one.
    async with asyncio.timeout(10):
        while True:
            try:
                RuntimeLock(selector.root, "service.lock").close()
                break
            except BlockingIOError:
                await asyncio.sleep(0.05)
    put_away.unlink()
    stopped = asyncio.Event()
    service = RuntimeService(selector.root)
    task = asyncio.create_task(service.serve(stopped))
    try:
        await running(service)
    finally:
        stopped.set()
        async with asyncio.timeout(15):
            assert await task
    assert put_away.read_text() == "done"
    assert status(selector.root).state == "stopped"


def test_owner_bootstrap_follows_the_bundle_shape(tmp_path: Path) -> None:
    """A shim in the archive keeps the old launch; none means the declared launcher."""
    from skulk.extensions.runtime_service import owner_bootstrap

    with_shim = tmp_path / "shim.pyz"
    with zipfile.ZipFile(with_shim, "w") as archive:
        archive.writestr("__owner__.py", "pass\n")
    without = tmp_path / "bare.pyz"
    with zipfile.ZipFile(without, "w") as archive:
        archive.writestr("__main__.py", "pass\n")
    assert "run_module('__owner__'" in owner_bootstrap(with_shim)
    launcher = owner_bootstrap(without)
    assert "entry_points(group='skulk.capability_runtime')" in launcher
    assert "e.name=='owner'" in launcher
    assert "run_module" not in launcher
