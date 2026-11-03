"""Actual reader fences, receipt-bound output inodes and source quota recovery."""

from pathlib import Path

import pytest

from skulk.utils.runtime_files import RuntimeLock, private_directory, write_private
from skulk.worker.runner.demucs.model import separation_result_directory
from skulk.worker.runner.demucs.result_custody import remember_result_directory
from skulk.worker.tests.test_separation_custody import KEY, store
from skulk.worker.tests.test_separation_sources import retain, task


def owned_result(root: Path, command: str, *, receipt: bool = True) -> Path:
    """Create exactly the directory/files that a native interruption may leave."""
    private_directory(root / "readers" / KEY)
    directory = separation_result_directory(
        root / "readers" / KEY, task(command).command_id
    )
    directory.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory.mkdir(mode=0o700)
    if receipt:
        remember_result_directory(directory)
    return directory


def test_replacement_recovers_owned_partial_results_only_after_reader_drain(
    tmp_path: Path,
) -> None:
    """A stale result cannot accumulate after each restart or erase a live source."""
    first = store(tmp_path)
    admitted = task("native-interrupted")
    retain(first, admitted)
    reader = RuntimeLock(tmp_path / "readers" / KEY, name="attachment.lock")
    directory = owned_result(tmp_path, "native-interrupted")
    write_private(directory / "drums.wav", b"partial native wav")
    write_private(directory / "result.json", b"partial native metadata")
    original = tmp_path / "inputs/native-interrupted/original.wav"
    before = original.read_bytes()
    first.detach()
    second = store(tmp_path)
    try:
        assert second.unresolved_source_count() == 1 and second.recover_stale() == 0
        assert original.read_bytes() == before and directory.is_dir()
        reader.close()
        assert second.recover_stale() == 1
        assert not original.exists() and not directory.exists()
        assert not directory.with_name("." + directory.name + ".owner.json").exists()
        assert second.unresolved_source_count() == 0
        for number in range(4):
            retain(second, task(f"recovered-{number}"))
    finally:
        reader.close()
        second.close()


@pytest.mark.parametrize("changed", ["inode", "symlink", "foreign"])
def test_changed_or_foreign_result_keeps_original_and_capacity(
    tmp_path: Path, changed: str
) -> None:
    """A receipt is not authority over a replacement directory or arbitrary files."""
    first = store(tmp_path)
    retain(first, task("preserved"))
    directory = owned_result(tmp_path, "preserved")
    if changed == "inode":
        directory.rename(directory.with_name(directory.name + "-original"))
        directory.mkdir(mode=0o700)
        write_private(directory / "drums.wav", b"foreign replacement")
    elif changed == "symlink":
        target = tmp_path / "keep"
        target.write_bytes(b"foreign original")
        (directory / "drums.wav").symlink_to(target)
    else:
        write_private(directory / "unrelated", b"foreign content")
    first.detach()
    second = store(tmp_path)
    try:
        assert second.unresolved_source_count() == 1 and second.recover_stale() == 0
        assert (
            tmp_path / "inputs/preserved/original.wav"
        ).is_file() and directory.is_dir()
        for number in range(3):
            retain(second, task(f"new-{number}"))
        with pytest.raises(ValueError, match="capacity"):
            second.admit(task("fifth"))
    finally:
        second.close()


def test_crash_before_receipt_cleans_only_empty_private_result_directory(
    tmp_path: Path,
) -> None:
    """Native work was not sent before its receipt; an empty mkdir gap is recoverable."""
    first = store(tmp_path)
    retain(first, task("mkdir-gap"))
    directory = owned_result(tmp_path, "mkdir-gap", receipt=False)
    first.detach()
    second = store(tmp_path)
    try:
        assert second.unresolved_source_count() == 0
        assert not directory.exists() and not (tmp_path / "inputs/mkdir-gap").exists()
    finally:
        second.close()
