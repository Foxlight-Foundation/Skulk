"""An owned child's whole tree can be stopped, and the child never outlives the lock."""

import ast
import asyncio
import contextlib
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import IO, cast

import psutil
import pytest

import skulk.extensions
from skulk.utils.owned_process import isolated_environment, start_owned
from skulk.utils.runtime_files import RuntimeLock, private_directory

# What only the process seam may do: start a child in its own session, hand it
# descriptors, or signal a whole process group.
_DIRECT_PROCESS_CONTROL = frozenset(
    {"killpg", "pass_fds", "preexec_fn", "setsid", "start_new_session"}
)
# Plugin-hosting modules that still start or stop child processes themselves,
# so cannot run on Windows. Remove a module when it is ported; add none.
_NOT_YET_PORTED = frozenset({"runtime_service.py", "service_snapshot.py"})

# Appends one byte at a time, unbuffered, to the file it is given, for far
# longer than any test runs: a file that has stopped growing means the writer
# has stopped running.
_WRITER = (
    "import os, sys, time\n"
    "log = os.open(sys.argv[1], os.O_WRONLY | os.O_CREAT | os.O_APPEND)\n"
    "while True:\n"
    "    os.write(log, b'x')\n"
    "    time.sleep(0.001)\n"
)
# A child that starts a grandchild, reports the grandchild's process
# identifier, and then both write in the working directory.
_TREE = (
    "import subprocess, sys\n"
    f"writer = {_WRITER!r}\n"
    "grandchild = subprocess.Popen("
    "[sys.executable, '-I', '-c', writer, 'grandchild.log'])\n"
    "print(grandchild.pid, flush=True)\n"
    "sys.argv = ['-c', 'child.log']\n"
    "exec(writer)\n"
)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """An owner-only directory, as the installer's working directory is."""
    path = tmp_path / "root"
    private_directory(path)
    return path


def _running(identifier: int) -> bool:
    try:
        return psutil.Process(identifier).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _wait_until_gone(identifiers: list[int], seconds: float = 10) -> None:
    deadline = time.monotonic() + seconds
    while any(_running(identifier) for identifier in identifiers):
        assert time.monotonic() < deadline, f"still running: {identifiers}"
        time.sleep(0.05)


def _written(root: Path) -> tuple[int, int]:
    """How much the child and the grandchild have each written so far."""
    return (
        (root / "child.log").stat().st_size,
        (root / "grandchild.log").stat().st_size,
    )


def _wait_until_both_write(root: Path, seconds: float = 10) -> None:
    deadline = time.monotonic() + seconds
    while True:
        with contextlib.suppress(FileNotFoundError):
            if all(_written(root)):
                return
        assert time.monotonic() < deadline, "the tree never started writing"
        time.sleep(0.02)


def _assert_writing_has_stopped(root: Path) -> None:
    before = _written(root)
    time.sleep(0.5)
    assert _written(root) == before


async def test_a_child_runs_with_its_output_captured(root: Path) -> None:
    lock = RuntimeLock(root, "installer.lock")
    try:
        child = await start_owned(
            (sys.executable, "-I", "-c", "print('ready'); raise SystemExit(3)"),
            cwd=root,
            lock=lock,
        )
        try:
            assert (await child.stdout.read()).split() == [b"ready"]
            assert child.returncode is None or child.returncode == 3
            assert await child.wait() == 3
            assert child.returncode == 3
        finally:
            await child.close()
    finally:
        lock.close()


async def test_a_child_starts_in_the_given_directory_without_the_callers_environment(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SKULK_OWNED_PROCESS_SECRET", "must-not-leak")
    lock = RuntimeLock(root, "installer.lock")
    try:
        child = await start_owned(
            (
                sys.executable,
                "-I",
                "-c",
                "import os; print(os.getcwd()); print('\\n'.join(sorted(os.environ)))",
            ),
            cwd=root,
            lock=lock,
        )
        try:
            output = (await child.stdout.read()).decode()
            assert await child.wait() == 0
        finally:
            await child.close()
    finally:
        lock.close()
    directory, *names = output.splitlines()
    assert Path(directory).resolve() == root.resolve()
    # Windows reports environment names in upper case.
    seen = {name.upper() for name in names}
    assert "SKULK_OWNED_PROCESS_SECRET" not in seen
    # The operating system may add names of its own; none of ours is missing.
    assert {name.upper() for name in isolated_environment()} <= seen


async def test_a_program_that_cannot_be_started_is_an_error(root: Path) -> None:
    lock = RuntimeLock(root, "installer.lock")
    try:
        with pytest.raises(OSError):
            await start_owned((str(root / "no-such-program"),), cwd=root, lock=lock)
    finally:
        lock.close()


async def test_the_lock_is_not_free_while_a_child_still_runs(root: Path) -> None:
    """Letting go of the lock does not free it: the child holds it too.

    This is what an abrupt end of the owner looks like to anyone else. POSIX
    gives the child the lock's descriptor; Windows gives it a file the lock
    checks.
    """
    lock = RuntimeLock(root, "installer.lock")
    child = await start_owned(
        (sys.executable, "-I", "-c", "import time; time.sleep(120)"),
        cwd=root,
        lock=lock,
    )
    try:
        lock.close()
        with pytest.raises(BlockingIOError):
            RuntimeLock(root, "installer.lock")
        child.kill()
        await asyncio.wait_for(child.wait(), 10)
    finally:
        await child.close()
    RuntimeLock(root, "installer.lock").close()


async def test_stopping_a_child_stops_everything_it_started(root: Path) -> None:
    lock = RuntimeLock(root, "installer.lock")
    try:
        child = await start_owned(
            (sys.executable, "-I", "-c", _TREE), cwd=root, lock=lock
        )
        try:
            grandchild = int(await child.stdout.readline())
            assert _running(child.pid) and _running(grandchild)
            await asyncio.to_thread(_wait_until_both_write, root)
            child.kill()
            await asyncio.wait_for(child.wait(), 10)
            # With every holder gone the output ends, as a reader relies on.
            assert await asyncio.wait_for(child.stdout.read(), 10) == b""
            await asyncio.to_thread(_wait_until_gone, [child.pid, grandchild])
            await asyncio.to_thread(_assert_writing_has_stopped, root)
            # Stopping what has already stopped is quiet.
            child.kill()
        finally:
            await child.close()
        # Nothing of the tree still has its files open: on Windows a file
        # another process holds open cannot be removed.
        (root / "child.log").unlink()
        (root / "grandchild.log").unlink()
    finally:
        lock.close()


async def test_closing_ends_what_a_finished_child_left_running_on_windows(
    root: Path,
) -> None:
    """A descendant that outlives the first child is still part of the tree."""
    if sys.platform != "win32":
        pytest.skip("on POSIX such a descendant keeps holding the lock instead")
    leaves_one_behind = (
        "import subprocess, sys\n"
        f"writer = {_WRITER!r}\n"
        "print(subprocess.Popen("
        "[sys.executable, '-I', '-c', writer, 'grandchild.log'],"
        " stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).pid, flush=True)\n"
    )
    lock = RuntimeLock(root, "installer.lock")
    try:
        child = await start_owned(
            (sys.executable, "-I", "-c", leaves_one_behind), cwd=root, lock=lock
        )
        try:
            left_behind = int(await child.stdout.readline())
            assert await asyncio.wait_for(child.wait(), 10) == 0
            assert _running(left_behind)
            # Let it open its file and start writing before it is ended.
            log = root / "grandchild.log"
            deadline = time.monotonic() + 10
            while not (log.exists() and log.stat().st_size):
                assert time.monotonic() < deadline
                await asyncio.sleep(0.02)
        finally:
            await child.close()
        # Closing returns only once the descendant has ended and let go of
        # its file, which Windows would otherwise refuse to remove. (Windows
        # may go on listing an ended process for a moment, so that is not
        # what is checked first.)
        (root / "grandchild.log").unlink()
        await asyncio.to_thread(_wait_until_gone, [left_behind])
    finally:
        lock.close()


async def test_a_temporary_directory_with_any_character_in_its_name_is_passed_on(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A character outside the Basic Multilingual Plane takes two units on Windows."""
    if sys.platform != "win32":
        pytest.skip("only the Windows child is given a temporary directory")
    unusual = root / "temp-\U00020000-\U0001f98a"
    unusual.mkdir()
    monkeypatch.setattr("tempfile.tempdir", str(unusual))
    lock = RuntimeLock(root, "installer.lock")
    try:
        child = await start_owned(
            (
                sys.executable,
                "-I",
                "-c",
                "import os, sys; sys.stdout.buffer.write(os.environ['TEMP'].encode())",
            ),
            cwd=root,
            lock=lock,
        )
        try:
            assert (await child.stdout.read()).decode() == str(unusual)
            assert await child.wait() == 0
        finally:
            await child.close()
    finally:
        lock.close()


def test_a_tree_never_writes_once_another_process_holds_the_lock(root: Path) -> None:
    """The owner dies without warning while a competitor waits for its lock.

    Both platforms keep the lock from the competitor for as long as the
    orphaned child runs. Windows ends the orphans with the owner; on POSIX
    the test has to. From the moment the competitor has the lock, no member
    of this tree writes again.
    """
    owner_script = (
        "import asyncio, os, sys\n"
        "from pathlib import Path\n"
        "from skulk.utils.owned_process import start_owned\n"
        "from skulk.utils.runtime_files import RuntimeLock\n"
        "async def main():\n"
        "    root = Path(sys.argv[1])\n"
        "    lock = RuntimeLock(root, 'installer.lock')\n"
        "    child = await start_owned("
        "(sys.executable, '-I', '-c', sys.argv[2]), cwd=root, lock=lock)\n"
        "    grandchild = int(await child.stdout.readline())\n"
        "    print(child.pid, grandchild, flush=True)\n"
        "    sys.stdin.read()\n"
        "    os._exit(0)\n"
        "asyncio.run(main())\n"
    )
    owner = subprocess.Popen(
        [sys.executable, "-c", owner_script, str(root), _TREE],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    child = grandchild = 0
    try:
        assert owner.stdout is not None and owner.stdin is not None
        reported = cast(IO[bytes], owner.stdout).readline()
        child, grandchild = (int(value) for value in reported.split())
        _wait_until_both_write(root)
        with pytest.raises(BlockingIOError):
            RuntimeLock(root, "installer.lock")

        # The owner goes, with the competitor already trying for the lock.
        owner.stdin.close()
        orphans_ended_by_hand = False
        held_until = time.monotonic() + 1
        deadline = time.monotonic() + 15
        while True:
            try:
                competitor = RuntimeLock(root, "installer.lock")
                break
            except BlockingIOError:
                assert time.monotonic() < deadline
                if (
                    sys.platform != "win32"
                    and not orphans_ended_by_hand
                    and time.monotonic() >= held_until
                ):
                    # Held this whole time by orphans that are still writing.
                    assert _running(child) and _running(grandchild)
                    os.killpg(child, 9)
                    orphans_ended_by_hand = True
        try:
            before = _written(root)
            assert orphans_ended_by_hand == (sys.platform != "win32")
            time.sleep(0.5)
            assert _written(root) == before
            _wait_until_gone([child, grandchild])
        finally:
            competitor.close()
    finally:
        owner.kill()
        owner.wait()
        for identifier in (child, grandchild):
            with contextlib.suppress(psutil.NoSuchProcess):
                if identifier:
                    psutil.Process(identifier).kill()


def test_only_the_seam_starts_or_stops_a_process_tree() -> None:
    """Plugin-hosting code starts owned children through the seam alone.

    Sessions, inherited descriptors and process-group signals exist only on
    POSIX, so a module that uses them directly cannot run on Windows.
    """
    package_root = Path(skulk.extensions.__file__).parent
    direct: set[str] = set()
    for source in sorted(package_root.glob("*.py")):
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Attribute) and node.attr in _DIRECT_PROCESS_CONTROL
            ) or (
                isinstance(node, ast.keyword) and node.arg in _DIRECT_PROCESS_CONTROL
            ):
                direct.add(source.name)

    assert direct == _NOT_YET_PORTED
