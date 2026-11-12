"""Real pre-write journaling and drained native music result cleanup."""

import json
import os
import stat
from pathlib import Path

import pytest
from pydantic import JsonValue, TypeAdapter

from skulk.shared.types.common import CommandId
from skulk.shared.types.native_music import NativeMusicDirectoryIdentity
from skulk.utils.runtime_files import private_directory
from skulk.worker.runner.ace_step.result_custody import (
    music_result_directory,
    prepare_result_directory,
    reclaim_music_result,
    reclaim_result_directory,
)

_JSON = TypeAdapter[dict[str, JsonValue]](dict[str, JsonValue])


def parents(root: Path) -> NativeMusicDirectoryIdentity:
    """Create a real exclusive fixture lifetime without a native process."""
    private_directory(root)
    workspace = root / "workspace"
    workspace.mkdir(mode=0o700)
    requests = workspace / "requests"
    requests.mkdir(mode=0o700)
    facts = requests.stat()
    return NativeMusicDirectoryIdentity(device=facts.st_dev, inode=facts.st_ino)


def write(path: Path, data: bytes = b"partial native output") -> None:
    """Create the same owner-only ordinary files the native child may write."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)


def outputs(directory: Path) -> tuple[Path, ...]:
    """Create bounded partial outputs; failed inference need not produce valid WAV."""
    result = tuple(
        directory / name for name in ("conditioning.wav", "result.wav", "result.json")
    )
    for path in result:
        write(path)
    return result


def receipt(directory: Path) -> Path:
    """Locate the sibling journal, outside all native-owned output contents."""
    return directory.with_name("." + directory.name + ".owner.json")


def test_result_is_exclusively_created_and_journaled_before_native_write(
    tmp_path: Path,
) -> None:
    """The parent can send inode truth only after durable custody succeeds."""
    parent = parents(tmp_path)
    directory = music_result_directory(tmp_path, CommandId("take-1"))
    identity = prepare_result_directory(directory, parent)
    body = _JSON.validate_json(receipt(directory).read_bytes())
    assert body == {
        "version": 1,
        "directory": "take-1",
        "identity": identity.model_dump(),
    }
    assert (identity.device, identity.inode) == (
        directory.stat().st_dev,
        directory.stat().st_ino,
    )
    assert directory.stat().st_mode & 0o777 == 0o700
    assert receipt(directory).stat().st_mode & 0o777 == 0o600
    assert not tuple(directory.iterdir())
    outputs(directory)
    assert reclaim_result_directory(directory)
    assert not directory.exists() and not receipt(directory).exists()
    assert reclaim_result_directory(directory)


def test_absent_complete_lifetime_is_already_reclaimed(tmp_path: Path) -> None:
    """Worker retry can settle a result after the entire empty workspace is gone."""
    assert reclaim_music_result(tmp_path, CommandId("take-1"))


@pytest.mark.parametrize(
    "identifier", ["../outside", "a/b", ".", "..", "x\n", "a" * 161]
)
def test_command_identity_cannot_select_an_arbitrary_path(
    tmp_path: Path, identifier: str
) -> None:
    """Even private recovery metadata cannot turn opaque IDs into path traversal."""
    with pytest.raises(ValueError, match="command identity"):
        music_result_directory(tmp_path, CommandId(identifier))
    assert not reclaim_music_result(tmp_path, CommandId(identifier))


def test_changed_parent_prevents_even_creating_a_result(tmp_path: Path) -> None:
    """A stale model configuration cannot allocate beneath a replacement parent."""
    parent = parents(tmp_path)
    requests = tmp_path / "workspace/requests"
    requests.rename(requests.with_name("retained-requests"))
    requests.mkdir(mode=0o700)
    directory = music_result_directory(tmp_path, CommandId("take-1"))
    with pytest.raises(ValueError, match="parent changed"):
        prepare_result_directory(directory, parent)
    assert not directory.exists() and not receipt(directory).exists()


def test_existing_result_requires_explicit_recovery_before_reuse(
    tmp_path: Path,
) -> None:
    """An empty previous attempt must not be silently repurposed by mkdir reuse."""
    parent = parents(tmp_path)
    directory = music_result_directory(tmp_path, CommandId("take-1"))
    directory.mkdir(mode=0o700)
    with pytest.raises(FileExistsError):
        prepare_result_directory(directory, parent)
    assert reclaim_result_directory(directory)
    prepare_result_directory(directory, parent)
    assert reclaim_result_directory(directory)


def test_unresolved_receipt_is_retained_until_explicit_recovery(tmp_path: Path) -> None:
    """A previous journal cannot be overwritten even if its output is absent."""
    parent = parents(tmp_path)
    directory = music_result_directory(tmp_path, CommandId("take-1"))
    prepare_result_directory(directory, parent)
    directory.rmdir()
    original = receipt(directory).read_bytes()
    with pytest.raises(ValueError, match="unresolved"):
        prepare_result_directory(directory, parent)
    assert receipt(directory).read_bytes() == original
    assert reclaim_result_directory(directory)
    assert not receipt(directory).exists()


def test_receipt_sync_failure_never_returns_admission_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failed durable metadata rolls back its empty result without native work."""
    parent = parents(tmp_path)
    directory = music_result_directory(tmp_path, CommandId("take-1"))
    sync = os.fsync

    def refuse_file(descriptor: int) -> None:
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("injected ownership sync failure")
        sync(descriptor)

    monkeypatch.setattr(os, "fsync", refuse_file)
    with pytest.raises(OSError, match="ownership sync failure"):
        prepare_result_directory(directory, parent)
    monkeypatch.undo()
    assert not directory.exists() and not receipt(directory).exists()
    assert reclaim_result_directory(directory)


def test_replaced_inode_preserves_foreign_and_original_results(tmp_path: Path) -> None:
    """A familiar pathname does not grant ownership of its new contents."""
    parent = parents(tmp_path)
    directory = music_result_directory(tmp_path, CommandId("take-1"))
    prepare_result_directory(directory, parent)
    original = outputs(directory)
    retained = directory.with_name("retained-original")
    directory.rename(retained)
    directory.mkdir(mode=0o700)
    foreign = directory / "result.wav"
    write(foreign, b"must survive")
    assert not reclaim_result_directory(directory)
    assert foreign.read_bytes() == b"must survive"
    assert all((retained / path.name).exists() for path in original)
    assert receipt(directory).exists()


def test_foreign_name_preserves_all_owned_files_before_any_delete(
    tmp_path: Path,
) -> None:
    """The directory is validated as a whole before partial-output removal starts."""
    parent = parents(tmp_path)
    directory = music_result_directory(tmp_path, CommandId("take-1"))
    prepare_result_directory(directory, parent)
    owned = outputs(directory)
    foreign = directory / "operator-note.txt"
    write(foreign, b"must survive")
    assert not reclaim_result_directory(directory)
    assert all(path.exists() for path in owned)
    assert foreign.read_bytes() == b"must survive"


@pytest.mark.parametrize("journal", ["absent", "malformed", "another-request"])
def test_unproven_nonempty_result_is_preserved(tmp_path: Path, journal: str) -> None:
    """Known filenames cannot replace missing, corrupt or mismatched ownership."""
    parent = parents(tmp_path)
    directory = music_result_directory(tmp_path, CommandId("take-1"))
    prepare_result_directory(directory, parent)
    owned = outputs(directory)
    if journal == "absent":
        receipt(directory).unlink()
    elif journal == "malformed":
        receipt(directory).write_bytes(b"{")
    else:
        body = _JSON.validate_json(receipt(directory).read_bytes())
        body["directory"] = "another-request"
        receipt(directory).write_text(json.dumps(body))
    assert not reclaim_result_directory(directory)
    assert all(path.exists() for path in owned)


def test_over_limit_partial_output_retains_ownership(tmp_path: Path) -> None:
    """An unexpected output extent is preserved rather than treated as normal work."""
    parent = parents(tmp_path)
    directory = music_result_directory(tmp_path, CommandId("take-1"))
    prepare_result_directory(directory, parent)
    path = directory / "result.wav"
    write(path)
    with path.open("r+b") as stream:
        stream.truncate(64 * 1024**2 + 1)
    assert not reclaim_result_directory(directory)
    assert path.stat().st_size == 64 * 1024**2 + 1
    assert receipt(directory).exists()


@pytest.mark.parametrize("link", ["symbolic", "hard"])
def test_linked_output_cannot_grant_custody_of_another_file(
    tmp_path: Path, link: str
) -> None:
    """Both path aliases and multiple-inode owners preserve creator bytes."""
    parent = parents(tmp_path)
    directory = music_result_directory(tmp_path, CommandId("take-1"))
    prepare_result_directory(directory, parent)
    original = tmp_path / "original.wav"
    write(original, b"creator original")
    target = directory / "result.wav"
    if link == "symbolic":
        target.symlink_to(original)
    else:
        os.link(original, target)
    assert not reclaim_result_directory(directory)
    assert target.exists() and original.read_bytes() == b"creator original"


def test_receipt_fifo_is_refused_without_blocking(tmp_path: Path) -> None:
    """A foreign named pipe cannot stall the Worker while it reads a journal."""
    parents(tmp_path)
    directory = music_result_directory(tmp_path, CommandId("take-1"))
    directory.mkdir(mode=0o700)
    os.mkfifo(receipt(directory), 0o600)
    assert not reclaim_result_directory(directory)
    assert directory.is_dir()


def test_failed_unlink_retains_receipt_and_can_be_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An incomplete cleanup returns no permission to release its source capacity."""
    parent = parents(tmp_path)
    directory = music_result_directory(tmp_path, CommandId("take-1"))
    prepare_result_directory(directory, parent)
    outputs(directory)
    unlink = os.unlink

    def refuse_result(
        path: str | os.PathLike[str], *, dir_fd: int | None = None
    ) -> None:
        if path == "result.wav":
            raise PermissionError("injected unlink failure")
        unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(os, "unlink", refuse_result)
    assert not reclaim_result_directory(directory)
    assert receipt(directory).is_file() and (directory / "result.wav").exists()
    monkeypatch.undo()
    assert reclaim_result_directory(directory)
    assert not directory.exists() and not receipt(directory).exists()
