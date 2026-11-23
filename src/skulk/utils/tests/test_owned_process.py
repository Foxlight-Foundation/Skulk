"""An owned child's whole tree can be stopped, and the child never outlives the lock."""

import ast
import asyncio
import contextlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import cast

import psutil
import pytest

import skulk.extensions
from skulk.utils.owned_process import hand_over, isolated_environment, start_owned
from skulk.utils.runtime_files import RuntimeLock, private_directory, read_private

# What only the process seam may do: start a child in its own session, hand it
# descriptors, signal a whole process group, or replace this process.
_DIRECT_PROCESS_CONTROL = frozenset(
    {
        "execl",
        "execle",
        "execlp",
        "execlpe",
        "execv",
        "execve",
        "execvp",
        "execvpe",
        "killpg",
        "pass_fds",
        "preexec_fn",
        "setsid",
        "start_new_session",
    }
)
# Plugin-hosting modules that still start, stop or replace processes
# themselves, so cannot run on Windows. Remove a module when it is ported; add
# none.
_NOT_YET_PORTED = frozenset(
    {"runtime_service.py", "service_bootstrap.py", "service_snapshot.py"}
)

# Takes the lock, then hands the terminal over to the program it is given.
_HANDS_OVER = (
    "import os, sys\n"
    "from pathlib import Path\n"
    "from skulk.utils.owned_process import hand_over\n"
    "from skulk.utils.runtime_files import RuntimeLock\n"
    "lock = RuntimeLock(Path(sys.argv[1]), 'installer.lock')\n"
    "os.set_inheritable(lock.descriptor, True)\n"
    "hand_over([sys.executable, '-I', '-c', sys.argv[2], *sys.argv[3:]], lock=lock.path)\n"
)
# Says it is ready, echoes one line of input on each output stream, and exits
# with a code of its own.
_CONVERSES = (
    "import sys\n"
    "print('ready', sys.argv[1], flush=True)\n"
    "line = sys.stdin.readline().strip()\n"
    "print('out:' + line, flush=True)\n"
    "print('err:' + line, file=sys.stderr, flush=True)\n"
    "raise SystemExit(7)\n"
)

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


async def test_the_children_fence_is_owner_only_state_on_windows(root: Path) -> None:
    """The file a child holds is protected like the lock it stands in for."""
    if sys.platform != "win32":
        pytest.skip("POSIX passes the lock's own descriptor to the child")
    fence = root / "installer.lock.children"
    lock = RuntimeLock(root, "installer.lock")
    try:
        child = await start_owned(
            (sys.executable, "-I", "-c", "pass"), cwd=root, lock=lock
        )
        try:
            assert await child.wait() == 0
        finally:
            await child.close()
        # Created through the file seam: it passes the owner-only read.
        assert read_private(fence) == b""

        # One made by other means is refused, as any runtime state would be.
        fence.unlink()
        fence.write_bytes(b"")
        with pytest.raises(ValueError, match="owner-only"):
            await start_owned((sys.executable, "-I", "-c", "pass"), cwd=root, lock=lock)

        # So is a link standing in its place.
        fence.unlink()
        elsewhere = root / "elsewhere"
        elsewhere.write_bytes(b"")
        try:
            fence.symlink_to(elsewhere)
        except OSError:
            return  # this account cannot create symbolic links
        with pytest.raises((ValueError, OSError)):
            await start_owned((sys.executable, "-I", "-c", "pass"), cwd=root, lock=lock)
        assert elsewhere.read_bytes() == b""
    finally:
        lock.close()


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
            # Stopping what has already stopped is quiet, and on POSIX signals
            # nobody: the child's identifier is no longer this tree's.
            child.kill()
        finally:
            await child.close()
        # Nothing of the tree still has its files open: on Windows a file
        # another process holds open cannot be removed.
        (root / "child.log").unlink()
        (root / "grandchild.log").unlink()
    finally:
        lock.close()


async def test_a_child_that_was_waited_for_is_never_signalled_on_posix(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Its identifier may by then belong to an unrelated process group."""
    if sys.platform == "win32":
        pytest.skip("Windows addresses the tree by its job, which is never reused")
    lock = RuntimeLock(root, "installer.lock")
    try:
        child = await start_owned(
            (sys.executable, "-I", "-c", "pass"), cwd=root, lock=lock
        )
        try:
            assert await child.wait() == 0
            signalled: list[int] = []

            def record(group: int, sent: int) -> None:
                del sent
                signalled.append(group)

            monkeypatch.setattr(os, "killpg", record)
            child.kill()
            assert signalled == []
        finally:
            await child.close()
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
        reported = owner.stdout.readline()
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


def _hand_over_in_a_process(
    root: Path, program: str, *arguments: str
) -> "subprocess.Popen[bytes]":
    """Start a process that takes the lock in ``root`` and hands over to ``program``."""
    return subprocess.Popen(
        [sys.executable, "-c", _HANDS_OVER, str(root), program, *arguments],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )


def test_handing_over_gives_the_program_this_terminal_and_the_lock(root: Path) -> None:
    """Input, both outputs and the exit code pass straight through, under the lock."""
    process = _hand_over_in_a_process(root, _CONVERSES, "an argument")
    try:
        assert process.stdin is not None and process.stdout is not None
        assert process.stdout.readline().split() == [b"ready", b"an", b"argument"]
        # The program is running and waiting for input: the lock is still held.
        with pytest.raises(BlockingIOError):
            RuntimeLock(root, "installer.lock")
        output, errors = process.communicate(b"typed by a person\n", timeout=30)
        assert output.split(b":", 1)[1].strip() == b"typed by a person"
        assert errors.strip() == b"err:typed by a person"
        assert process.returncode == 7
    finally:
        process.kill()
        process.wait()
    RuntimeLock(root, "installer.lock").close()


def test_a_program_handed_over_to_gets_only_the_fixed_environment(root: Path) -> None:
    program = (
        "import json, os, sys\nprint(json.dumps(sorted(os.environ)), flush=True)\n"
    )
    process = _hand_over_in_a_process(root, program)
    try:
        output, errors = process.communicate(timeout=30)
        assert process.returncode == 0, errors.decode()
    finally:
        process.kill()
        process.wait()
    expected = {name.upper() for name in isolated_environment()}
    names = {name.upper() for name in cast(list[str], json.loads(output))}
    # An interpreter may add names of its own at startup; nothing of the
    # caller's, such as the search path this test started it with, gets in.
    assert expected <= names
    assert "PYTHONPATH" not in names and "HOME" not in names and "USERNAME" not in names


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX replaces the process")
def test_handing_over_on_posix_is_one_process_replacement(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replaced: list[tuple[str, tuple[str, ...], dict[str, str]]] = []

    def record(
        path: str, arguments: tuple[str, ...], environment: dict[str, str]
    ) -> None:
        replaced.append((path, arguments, environment))
        raise SystemExit(0)

    monkeypatch.setattr(os, "execve", record)
    with pytest.raises(SystemExit):
        hand_over(("/usr/bin/true", "one", "two"), lock=root / "installer.lock")
    assert replaced == [
        ("/usr/bin/true", ("/usr/bin/true", "one", "two"), isolated_environment())
    ]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows waits for the program")
def test_handing_over_on_windows_ends_what_the_program_left_running(root: Path) -> None:
    """Nothing the program started outlives the lock its owner then releases."""
    program = (
        "import os, subprocess, sys, time\n"
        "left = subprocess.Popen([sys.executable, '-I', '-c', sys.argv[2], sys.argv[1]])\n"
        "while not (os.path.exists(sys.argv[1]) and os.path.getsize(sys.argv[1])):\n"
        "    time.sleep(0.01)\n"
        "print(left.pid, flush=True)\n"
    )
    process = _hand_over_in_a_process(root, program, str(root / "child.log"), _WRITER)
    try:
        output, errors = process.communicate(timeout=60)
        assert process.returncode == 0, errors.decode()
        left_running = int(output)
    finally:
        process.kill()
        process.wait()
    _wait_until_gone([left_running])
    before = (root / "child.log").stat().st_size
    time.sleep(0.3)
    assert (root / "child.log").stat().st_size == before
    RuntimeLock(root, "installer.lock").close()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows exit codes are 32 bits")
def test_a_windows_exit_code_above_the_signed_range_passes_through(root: Path) -> None:
    """A program ended by Ctrl-C reports 0xC000013A; so must its owner."""
    program = "import ctypes\nctypes.windll.kernel32.ExitProcess(0xC000013A)\n"
    process = _hand_over_in_a_process(root, program)
    try:
        process.communicate(timeout=30)
        assert process.returncode == 0xC000013A
    finally:
        process.kill()
        process.wait()


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
