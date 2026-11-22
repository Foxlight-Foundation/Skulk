"""An owned child's whole tree can be stopped, and never outlives its owner's lock."""

import ast
import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

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

# A child that starts a grandchild, reports both process identifiers, and
# then both wait far longer than any test runs.
_TREE = (
    "import subprocess, sys, time\n"
    "grandchild = subprocess.Popen("
    "[sys.executable, '-I', '-c', 'import time; time.sleep(120)'])\n"
    "print(grandchild.pid, flush=True)\n"
    "time.sleep(120)\n"
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


async def test_a_child_runs_with_its_output_captured(root: Path) -> None:
    lock = RuntimeLock(root, "installer.lock")
    try:
        child = await start_owned(
            (sys.executable, "-I", "-c", "print('ready'); raise SystemExit(3)"),
            cwd=root,
            lock=lock,
        )
        try:
            assert child.process.stdout is not None
            assert (await child.process.stdout.read()).split() == [b"ready"]
            assert await child.process.wait() == 3
        finally:
            child.close()
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
            assert child.process.stdout is not None
            output = (await child.process.stdout.read()).decode()
            assert await child.process.wait() == 0
        finally:
            child.close()
    finally:
        lock.close()
    directory, *names = output.splitlines()
    assert Path(directory).resolve() == root.resolve()
    # Windows reports environment names in upper case.
    seen = {name.upper() for name in names}
    assert "SKULK_OWNED_PROCESS_SECRET" not in seen
    # The operating system may add names of its own; none of ours is missing.
    assert {name.upper() for name in isolated_environment()} <= seen


async def test_stopping_a_child_stops_everything_it_started(root: Path) -> None:
    lock = RuntimeLock(root, "installer.lock")
    try:
        child = await start_owned(
            (sys.executable, "-I", "-c", _TREE), cwd=root, lock=lock
        )
        try:
            assert child.process.stdout is not None
            grandchild = int(await child.process.stdout.readline())
            assert _running(child.process.pid) and _running(grandchild)
            child.kill()
            await asyncio.wait_for(child.process.wait(), 10)
            # With every holder gone the output ends, as a reader relies on.
            assert await asyncio.wait_for(child.process.stdout.read(), 10) == b""
            _wait_until_gone([child.process.pid, grandchild])
            # Stopping what has already stopped is quiet.
            child.kill()
        finally:
            child.close()
    finally:
        lock.close()


def test_a_tree_never_outlives_its_owners_lock(root: Path) -> None:
    """The owner dies without warning; nobody else may work beside its orphans.

    POSIX keeps the lock held by the orphans until they exit. Windows ends the
    orphans with the owner. Either way the lock is never free while a member
    of the old tree is still running.
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
        "    grandchild = int(await child.process.stdout.readline())\n"
        "    print(child.process.pid, grandchild, flush=True)\n"
        "    os._exit(0)\n"
        "asyncio.run(main())\n"
    )
    owner = subprocess.run(
        [sys.executable, "-c", owner_script, str(root), _TREE],
        stdout=subprocess.PIPE,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        timeout=60,
        check=True,
    )
    child, grandchild = (int(value) for value in owner.stdout.split())
    try:
        if sys.platform == "win32":
            _wait_until_gone([child, grandchild])
        else:
            assert _running(child) and _running(grandchild)
            with pytest.raises(BlockingIOError):
                RuntimeLock(root, "installer.lock")
            os.killpg(child, 9)
            _wait_until_gone([child, grandchild])
        deadline = time.monotonic() + 10
        while True:
            try:
                RuntimeLock(root, "installer.lock").close()
                break
            except BlockingIOError:
                assert time.monotonic() < deadline
                time.sleep(0.05)
    finally:
        for identifier in (child, grandchild):
            try:
                psutil.Process(identifier).kill()
            except psutil.NoSuchProcess:
                pass


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
