"""Owner-only runtime files behave the same on every platform Skulk runs on."""

import ast
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

import skulk.extensions
from skulk.utils.runtime_files import (
    RuntimeLock,
    acquire_runtime_lock,
    is_private_directory,
    open_private,
    private_directory,
    read_private,
    remove_private,
    sync_directory,
    write_private,
)

windows_only = pytest.mark.skipif(
    sys.platform != "win32", reason="examines a Windows access list"
)
posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="examines POSIX permission bits"
)

# What only the file seam may do: ask who owns a file, or open one without
# following a link, by calling the operating system directly.
_DIRECT_FILE_CUSTODY = frozenset(
    {
        "O_DIRECTORY",
        "O_NOFOLLOW",
        "chown",
        "fchmod",
        "fchown",
        "getegid",
        "geteuid",
        "getgid",
        "getuid",
        "st_gid",
        "st_uid",
    }
)
# Plugin-hosting modules not yet moved onto the seam, so not yet able to run on
# Windows. Remove a module from this list when it is ported; add none.
_NOT_YET_PORTED = frozenset(
    {
        "local_setup.py",
        "runtime_integrity.py",
        "service_bootstrap.py",
        "service_registration.py",
        "service_setup.py",
        "service_snapshot.py",
    }
)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A directory created owner-only, as runtime state always is."""
    path = tmp_path / "root"
    private_directory(path)
    return path


def _link_or_skip(link: Path, target: Path, *, directory: bool) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except OSError:
        pytest.skip("this account cannot create symbolic links")


def test_a_private_directory_is_created_with_its_missing_parents(
    tmp_path: Path,
) -> None:
    path = tmp_path / "a" / "b" / "state"
    private_directory(path)
    assert path.is_dir()
    # Validating it again changes nothing and accepts it.
    private_directory(path)


def test_a_link_to_a_directory_is_not_a_private_directory(
    tmp_path: Path, root: Path
) -> None:
    link = tmp_path / "link"
    _link_or_skip(link, root, directory=True)
    with pytest.raises(ValueError, match="owner-only"):
        private_directory(link)


def test_a_file_is_not_a_private_directory(root: Path) -> None:
    write_private(root / "state.json", b"{}")
    with pytest.raises(OSError):
        private_directory(root / "state.json")


def test_a_private_file_round_trips_and_is_replaced_atomically(root: Path) -> None:
    path = root / "state.json"
    write_private(path, b"first")
    assert read_private(path) == b"first"
    write_private(path, b"second, and longer")
    assert read_private(path) == b"second, and longer"
    # No temporary file is left beside it.
    assert [entry.name for entry in root.iterdir()] == ["state.json"]


def _open_as_a_reader_does(path: Path) -> int:
    """Open a file the way this module's own readers hold it."""
    if sys.platform == "win32":
        from foxlight_model_preservation import safefs

        return safefs.open_path(path, os.O_RDONLY)
    return os.open(path, os.O_RDONLY)


def test_a_private_file_is_replaced_while_a_reader_holds_it(root: Path) -> None:
    path = root / "state.json"
    write_private(path, b"first")
    descriptor = _open_as_a_reader_does(path)
    try:
        write_private(path, b"second")
        # The reader keeps the contents it opened.
        assert os.read(descriptor, 16) == b"first"
    finally:
        os.close(descriptor)
    assert read_private(path) == b"second"


@windows_only
def test_a_program_that_forbids_replacement_blocks_it_and_leaves_no_debris(
    root: Path,
) -> None:
    """Recorded limit: Windows lets a program open a file so that it cannot be replaced.

    Python's own ``open`` does. The write waits briefly, then fails, and the
    existing file is untouched.
    """
    path = root / "state.json"
    write_private(path, b"first")
    with path.open("rb"), pytest.raises(PermissionError):
        write_private(path, b"second")
    assert read_private(path) == b"first"
    assert [entry.name for entry in root.iterdir()] == ["state.json"]


def test_writing_creates_the_missing_parent_as_private(root: Path) -> None:
    path = root / "nested" / "state.json"
    write_private(path, b"x")
    assert read_private(path) == b"x"
    private_directory(path.parent)


def test_reading_is_bounded(root: Path) -> None:
    path = root / "state.json"
    write_private(path, b"x" * 100)
    assert read_private(path, limit=100) == b"x" * 100
    with pytest.raises(ValueError, match="exceeds bound"):
        read_private(path, limit=99)


def test_reading_refuses_a_link(tmp_path: Path, root: Path) -> None:
    write_private(root / "real.json", b"secret")
    link = root / "link.json"
    _link_or_skip(link, root / "real.json", directory=False)
    with pytest.raises(OSError):
        read_private(link)


def test_reading_a_missing_file_raises(root: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_private(root / "absent.json")


def test_removing_is_quiet_when_the_file_is_absent(root: Path) -> None:
    path = root / "state.json"
    write_private(path, b"x")
    remove_private(path)
    assert not path.exists()
    remove_private(path)


def test_a_lock_excludes_a_second_holder_until_it_is_closed(root: Path) -> None:
    first = RuntimeLock(root, "manager.lock")
    try:
        with pytest.raises(BlockingIOError):
            RuntimeLock(root, "manager.lock")
        # A different lock in the same directory is independent.
        RuntimeLock(root, "installer.lock").close()
    finally:
        first.close()
    RuntimeLock(root, "manager.lock").close()
    first.close()


def test_a_lock_is_released_when_its_process_dies(root: Path) -> None:
    script = (
        "import sys, time\n"
        "from pathlib import Path\n"
        "from skulk.utils.runtime_files import RuntimeLock\n"
        "RuntimeLock(Path(sys.argv[1]), 'manager.lock')\n"
        "print('held', flush=True)\n"
        "time.sleep(60)\n"
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", script, str(root)],
        stdout=subprocess.PIPE,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == b"held"
        with pytest.raises(BlockingIOError):
            RuntimeLock(root, "manager.lock")
    finally:
        holder.kill()
        holder.wait()
    # The operating system drops the lock as it tears the process down, which
    # on Windows can finish a moment after the process is reported gone.
    deadline = time.monotonic() + 10
    while True:
        try:
            RuntimeLock(root, "manager.lock").close()
            break
        except BlockingIOError:
            assert time.monotonic() < deadline, "the dead holder's lock was kept"
            time.sleep(0.05)


async def test_waiting_for_a_lock_gives_up_after_the_bound(root: Path) -> None:
    held = RuntimeLock(root, "installer.lock")
    try:
        with pytest.raises(BlockingIOError):
            await acquire_runtime_lock(root, wait_seconds=0.2, poll_seconds=0.02)
    finally:
        held.close()
    (await acquire_runtime_lock(root, wait_seconds=0.2)).close()


def test_an_unknown_lock_name_is_refused(root: Path) -> None:
    with pytest.raises(ValueError, match="unknown runtime lock"):
        RuntimeLock(root, "anything.lock")


@windows_only
def test_a_directory_made_by_other_means_is_not_owner_only(tmp_path: Path) -> None:
    """It inherits its access list, so someone other than the account may be on it."""
    ordinary = tmp_path / "ordinary"
    ordinary.mkdir()
    with pytest.raises(ValueError, match="owner-only"):
        private_directory(ordinary)
    with pytest.raises(ValueError, match="owner-only"):
        RuntimeLock(ordinary, "manager.lock")


@windows_only
def test_a_file_made_by_other_means_is_not_owner_only(root: Path) -> None:
    """Even inside a private directory: its access list is inherited, not its own."""
    path = root / "dropped.json"
    path.write_bytes(b"{}")
    with pytest.raises(ValueError, match="owner-only"):
        read_private(path)
    # Writing it through the seam replaces it with an owner-only file.
    write_private(path, b"ours")
    assert read_private(path) == b"ours"


def test_a_directory_made_through_the_seam_is_private(root: Path) -> None:
    assert is_private_directory(root)


def test_a_link_or_a_file_is_not_a_private_directory(
    tmp_path: Path, root: Path
) -> None:
    write_private(root / "state.json", b"{}")
    assert not is_private_directory(root / "state.json")
    link = tmp_path / "link"
    _link_or_skip(link, root, directory=True)
    assert not is_private_directory(link)


def test_a_missing_directory_cannot_be_examined(root: Path) -> None:
    with pytest.raises(FileNotFoundError):
        is_private_directory(root / "absent")


@posix_only
def test_a_directory_others_can_enter_is_not_private(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o750)
    assert not is_private_directory(shared)
    shared.chmod(0o700)
    assert is_private_directory(shared)


@windows_only
def test_a_directory_with_an_inherited_access_list_is_not_private(
    tmp_path: Path,
) -> None:
    ordinary = tmp_path / "ordinary"
    ordinary.mkdir()
    assert not is_private_directory(ordinary)


def test_a_file_opened_for_creation_is_private_and_exclusive(root: Path) -> None:
    path = root / "artifact.partial"
    descriptor = open_private(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "wb") as target:
        target.write(b"line one\nline two\n")
    with pytest.raises(FileExistsError):
        open_private(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    # The bytes are exactly what was written, and the file passes the private read.
    assert read_private(path) == b"line one\nline two\n"


def test_an_existing_private_file_opens_and_one_that_is_absent_is_created(
    root: Path,
) -> None:
    path = root / "operations.sqlite3"
    os.close(open_private(path, os.O_RDWR | os.O_CREAT))
    assert path.is_file()
    write_private(root / "state.json", b"kept")
    descriptor = open_private(root / "state.json", os.O_RDONLY)
    with os.fdopen(descriptor, "rb") as source:
        assert source.read() == b"kept"


def test_opening_refuses_a_link_and_a_missing_file(tmp_path: Path, root: Path) -> None:
    with pytest.raises(FileNotFoundError):
        open_private(root / "absent.json", os.O_RDONLY)
    write_private(root / "state.json", b"{}")
    link = tmp_path / "link.json"
    _link_or_skip(link, root / "state.json", directory=False)
    with pytest.raises(OSError):
        open_private(link, os.O_RDONLY)


def test_opening_refuses_a_directory(root: Path) -> None:
    with pytest.raises((ValueError, OSError)):
        open_private(root, os.O_RDONLY)


@posix_only
def test_opening_refuses_a_file_others_can_read(root: Path) -> None:
    path = root / "loose.json"
    path.write_bytes(b"{}")
    path.chmod(0o640)
    with pytest.raises(ValueError, match="owner-only"):
        open_private(path, os.O_RDONLY)


@windows_only
def test_opening_refuses_a_file_with_an_inherited_access_list(root: Path) -> None:
    path = root / "dropped.json"
    path.write_bytes(b"{}")
    with pytest.raises(ValueError, match="owner-only"):
        open_private(path, os.O_RDONLY)


def test_a_directory_syncs_and_a_missing_one_does_not(root: Path) -> None:
    write_private(root / "state.json", b"{}")
    sync_directory(root)
    with pytest.raises(OSError):
        sync_directory(root / "absent")


def test_only_the_seam_asks_who_owns_a_file_or_opens_one_link_safely() -> None:
    """Plugin-hosting code reaches protected files through the seam alone.

    A direct ownership comparison or link-safe open is POSIX-only, so a module
    that makes one cannot run on Windows, and each copy is one more place the
    rule for "private to this account" could drift.
    """
    package_root = Path(skulk.extensions.__file__).parent
    direct: set[str] = set()
    for source in sorted(package_root.glob("*.py")):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        if any(
            isinstance(node, ast.Attribute) and node.attr in _DIRECT_FILE_CUSTODY
            for node in ast.walk(tree)
        ):
            direct.add(source.name)

    assert direct == _NOT_YET_PORTED
