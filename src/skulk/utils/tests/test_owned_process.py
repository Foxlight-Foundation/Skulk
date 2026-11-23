"""An owned child's whole tree can be stopped, and the child never outlives the lock."""

import ast
import asyncio
import contextlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import cast

import psutil
import pytest

import skulk.extensions
from skulk.utils.owned_process import (
    hand_over,
    isolated_environment,
    start_owned,
    start_supervised,
)
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
_NOT_YET_PORTED = frozenset({"service_bootstrap.py", "service_snapshot.py"})

# Takes the lock, then hands the terminal over to the program it is given.
_TAKES_THE_LOCK = (
    "import os, sys\n"
    "from pathlib import Path\n"
    "from skulk.utils.owned_process import hand_over\n"
    "from skulk.utils.runtime_files import RuntimeLock\n"
    "lock = RuntimeLock(Path(sys.argv[1]), 'installer.lock')\n"
    "os.set_inheritable(lock.descriptor, True)\n"
)
_THEN_HANDS_OVER = "hand_over([sys.executable, '-I', '-c', sys.argv[2], *sys.argv[3:]], lock=lock.path)\n"
_HANDS_OVER = _TAKES_THE_LOCK + _THEN_HANDS_OVER
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


# A long-running program that shuts itself down when asked. It says it is
# ready, waits for the end of its lifetime pipe at the place it is told, takes
# as long over putting things away as it is told, leaves a file to show it
# finished, and exits with a code of its own.
_SUPERVISED = (
    "import os, sys, time\n"
    "end, pause = sys.argv[1], float(sys.argv[2])\n"
    "print('ready', os.getpid(), flush=True)\n"
    "(sys.stdin.buffer if end == 'stdin' else os.fdopen(int(end), 'rb')).read()\n"
    "time.sleep(pause)\n"
    "open('put-away', 'w').close()\n"
    "raise SystemExit(4)\n"
)


def _supervised(
    source: str, pause: float = 0.0
) -> Callable[[int | None], tuple[str, ...]]:
    """The command line of a supervised program, built from where its pipe is."""

    def arguments(descriptor: int | None) -> tuple[str, ...]:
        # Standard input on Windows, an inherited descriptor everywhere else.
        assert (descriptor is None) == (sys.platform == "win32")
        end = "stdin" if descriptor is None else str(descriptor)
        return (sys.executable, "-I", "-c", source, end, str(pause))

    return arguments


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


# Starts a writer that outlives it, reports the writer's process identifier
# and ends at once.
_LEAVES_ONE_BEHIND = (
    "import subprocess, sys\n"
    f"writer = {_WRITER!r}\n"
    "print(subprocess.Popen("
    "[sys.executable, '-I', '-c', writer, 'grandchild.log'],"
    " stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).pid, flush=True)\n"
)


async def test_a_stranger_with_a_waited_childs_identifier_is_never_signalled_on_posix(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once the tree is gone its identifier may belong to an unrelated process."""
    if sys.platform == "win32":
        pytest.skip("Windows addresses the tree by its job, which is never reused")
    lock = RuntimeLock(root, "installer.lock")
    try:
        child = await start_owned(
            (sys.executable, "-I", "-c", "pass"), cwd=root, lock=lock
        )
        try:
            assert await child.wait() == 0
            # Nothing is left and nobody has the identifier: quiet.
            child.kill()

            signalled: list[int] = []

            def record(group: int, sent: int) -> None:
                del sent
                signalled.append(group)

            def alive(identifier: int, sent: int) -> None:
                """A live process, some stranger's, now has the identifier."""
                assert (identifier, sent) == (child.pid, 0)

            monkeypatch.setattr(os, "killpg", record)
            monkeypatch.setattr(os, "kill", alive)
            child.kill()
            assert signalled == []

            def another_accounts(identifier: int, sent: int) -> None:
                raise PermissionError(identifier, sent)

            monkeypatch.setattr(os, "kill", another_accounts)
            child.kill()
            assert signalled == []
        finally:
            await child.close()
    finally:
        lock.close()


async def test_stopping_a_waited_child_ends_what_it_left_running_on_posix(
    root: Path,
) -> None:
    """The group outlives its leader; it is still this tree, and still stoppable."""
    if sys.platform == "win32":
        pytest.skip("Windows ends such a descendant when the child is closed")
    lock = RuntimeLock(root, "installer.lock")
    left_behind = 0
    try:
        child = await start_owned(
            (sys.executable, "-I", "-c", _LEAVES_ONE_BEHIND), cwd=root, lock=lock
        )
        try:
            left_behind = int(await child.stdout.readline())
            assert await asyncio.wait_for(child.wait(), 10) == 0
            assert child.returncode == 0 and _running(left_behind)
            child.kill()
            await asyncio.to_thread(_wait_until_gone, [left_behind])
        finally:
            await child.close()
    finally:
        lock.close()
        with contextlib.suppress(psutil.NoSuchProcess):
            if left_behind:
                psutil.Process(left_behind).kill()


async def test_closing_ends_what_a_finished_child_left_running_on_windows(
    root: Path,
) -> None:
    """A descendant that outlives the first child is still part of the tree."""
    if sys.platform != "win32":
        pytest.skip("on POSIX such a descendant keeps holding the lock instead")
    lock = RuntimeLock(root, "installer.lock")
    try:
        child = await start_owned(
            (sys.executable, "-I", "-c", _LEAVES_ONE_BEHIND), cwd=root, lock=lock
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


@pytest.mark.skipif(sys.platform != "win32", reason="Windows copies the streams")
@pytest.mark.parametrize("closed", [1, 2])
def test_a_closed_output_stream_on_windows_reaches_the_program_as_the_null_device(
    root: Path, closed: int
) -> None:
    """A copy of standard input must not land in the closed stream's place.

    A copied descriptor takes the lowest free number, which is the closed
    stream's own, so the stream would then look open and the program would be
    given a second, read-only standard input to write its output to.
    """
    other = 3 - closed
    program = (
        "import os, stat, sys\n"
        f"device = stat.S_ISCHR(os.fstat({closed}).st_mode)\n"
        "try:\n"
        f"    os.write({closed}, b'discarded')\n"
        "    written = True\n"
        "except OSError:\n"
        "    written = False\n"
        "line = sys.stdin.readline().strip()\n"
        f"os.write({other}, f'{{device}} {{written}} {{line}}'.encode())\n"
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            # Closed last, so that nothing this process opens takes its place.
            _TAKES_THE_LOCK + f"os.close({closed})\n" + _THEN_HANDS_OVER,
            str(root),
            program,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    try:
        streams = process.communicate(b"typed\n", timeout=30)
        assert streams[other - 1] == b"True True typed"
        assert streams[closed - 1] == b""
        assert process.returncode == 0
    finally:
        process.kill()
        process.wait()


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


async def test_a_supervised_child_shuts_itself_down_when_released(root: Path) -> None:
    lock = RuntimeLock(root, "service.lock")
    try:
        child = await start_supervised(_supervised(_SUPERVISED), cwd=root, lock=lock)
        try:
            assert (await child.stdout.readline()).split()[0] == b"ready"
            # It is waiting, not ending by itself.
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(asyncio.shield(child.wait()), 0.3)
            assert not (root / "put-away").exists()
            child.release()
            assert await asyncio.wait_for(child.wait(), 10) == 4
            assert (root / "put-away").exists()
            # Releasing what is released is quiet.
            child.release()
        finally:
            await child.close()
    finally:
        lock.close()
    RuntimeLock(root, "service.lock").close()


async def test_no_other_program_inherits_the_end_that_asks_for_shutdown(
    root: Path,
) -> None:
    """A bystander started later must not keep the child's lifetime pipe open."""
    lock = RuntimeLock(root, "service.lock")
    try:
        child = await start_supervised(_supervised(_SUPERVISED), cwd=root, lock=lock)
        bystander = subprocess.Popen(
            [sys.executable, "-I", "-c", "import time; time.sleep(120)"],
            close_fds=False,
        )
        try:
            assert (await child.stdout.readline()).split()[0] == b"ready"
            child.release()
            assert await asyncio.wait_for(child.wait(), 10) == 4
            assert bystander.poll() is None
        finally:
            bystander.kill()
            bystander.wait()
            await child.close()
    finally:
        lock.close()


async def test_a_supervised_child_that_does_not_go_is_ended_with_its_tree(
    root: Path,
) -> None:
    lock = RuntimeLock(root, "service.lock")
    try:
        # A tree that never reads its lifetime pipe.
        child = await start_supervised(
            lambda _: (sys.executable, "-I", "-c", _TREE), cwd=root, lock=lock
        )
        try:
            grandchild = int(await child.stdout.readline())
            await asyncio.to_thread(_wait_until_both_write, root)
            child.release()
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(asyncio.shield(child.wait()), 0.5)
            child.kill()
            await asyncio.wait_for(child.wait(), 10)
            await asyncio.to_thread(_wait_until_gone, [child.pid, grandchild])
            await asyncio.to_thread(_assert_writing_has_stopped, root)
        finally:
            await child.close()
        (root / "child.log").unlink()
        (root / "grandchild.log").unlink()
    finally:
        lock.close()


def test_a_supervised_child_is_asked_not_ended_when_its_parent_dies(root: Path) -> None:
    """The parent dies without warning; the child reads end of input and finishes.

    The system does not end it, on either platform, and nobody can take the
    lock until it has gone.
    """
    parent_script = (
        "import asyncio, os, sys\n"
        "from pathlib import Path\n"
        "from skulk.utils.owned_process import start_supervised\n"
        "from skulk.utils.runtime_files import RuntimeLock\n"
        "async def main():\n"
        "    root = Path(sys.argv[1])\n"
        "    lock = RuntimeLock(root, 'service.lock')\n"
        "    child = await start_supervised(lambda end: (sys.executable, '-I', '-c',"
        " sys.argv[2], 'stdin' if end is None else str(end), '1.5'),"
        " cwd=root, lock=lock)\n"
        "    print((await child.stdout.readline()).split()[1].decode(), flush=True)\n"
        "    sys.stdin.read()\n"
        "    os._exit(0)\n"
        "asyncio.run(main())\n"
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", parent_script, str(root), _SUPERVISED],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    child = 0
    try:
        assert parent.stdout is not None and parent.stdin is not None
        child = int(parent.stdout.readline())
        assert _running(child)
        with pytest.raises(BlockingIOError):
            RuntimeLock(root, "service.lock")

        parent.stdin.close()
        assert parent.wait(10) == 0
        # The parent is gone; the child is putting things away and is fenced.
        assert _running(child) and not (root / "put-away").exists()
        with pytest.raises(BlockingIOError):
            RuntimeLock(root, "service.lock")

        _wait_until_gone([child])
        assert (root / "put-away").exists()
        deadline = time.monotonic() + 10
        while True:
            try:
                RuntimeLock(root, "service.lock").close()
                break
            except BlockingIOError:
                assert time.monotonic() < deadline
                time.sleep(0.05)
    finally:
        parent.kill()
        parent.wait()
        with contextlib.suppress(psutil.NoSuchProcess):
            if child:
                psutil.Process(child).kill()


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
