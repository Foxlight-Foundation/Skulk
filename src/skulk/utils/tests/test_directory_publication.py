"""Real filesystem publication refuses every pre-existing destination."""

import errno
import os
from pathlib import Path

import pytest

from skulk.utils.directory_publication import publish_directory


@pytest.mark.parametrize("entry", ["absent", "empty", "populated", "file", "symlink"])
def test_publication_never_replaces_existing_entries(tmp_path: Path, entry: str) -> None:
    source = tmp_path / "stage"
    source.mkdir()
    (source / "complete").write_bytes(b"verified bytes")
    destination = tmp_path / "final"
    if entry in {"empty", "populated"}:
        destination.mkdir()
        if entry == "populated":
            (destination / "operator-file").write_bytes(b"retain me")
    elif entry == "file":
        destination.write_bytes(b"retain me")
    elif entry == "symlink":
        destination.symlink_to(source, target_is_directory=True)
    original = destination.lstat() if entry != "absent" else None
    parent = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        if entry == "absent":
            inode = source.stat().st_ino
            publish_directory(source.name, parent, destination.name, parent)
            assert destination.stat().st_ino == inode
            assert not source.exists()
            assert (destination / "complete").read_bytes() == b"verified bytes"
        else:
            with pytest.raises(FileExistsError):
                publish_directory(source.name, parent, destination.name, parent)
            assert original and destination.lstat().st_ino == original.st_ino
            assert (source / "complete").read_bytes() == b"verified bytes"
            if entry == "populated":
                assert (destination / "operator-file").read_bytes() == b"retain me"
    finally:
        os.close(parent)


def test_unsupported_publication_retains_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import skulk.utils.directory_publication as publication

    source = tmp_path / "stage"
    source.mkdir()
    monkeypatch.setattr(publication.sys, "platform", "unsupported")
    parent = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(OSError) as refused:
            publish_directory(source.name, parent, "final", parent)
        assert refused.value.errno == errno.ENOTSUP
        assert source.is_dir() and not (tmp_path / "final").exists()
    finally:
        os.close(parent)


def test_replaced_parent_path_cannot_substitute_the_verified_stage(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    stage = work / "stage"
    stage.mkdir()
    (stage / "weights").write_bytes(b"verified original")
    source_parent = os.open(work, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    parent = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        work.rename(tmp_path / "original-work")
        stage.mkdir(parents=True)
        (stage / "weights").write_bytes(b"unverified replacement")
        publish_directory("stage", source_parent, "final", parent)
        assert (tmp_path / "final" / "weights").read_bytes() == b"verified original"
        assert (stage / "weights").read_bytes() == b"unverified replacement"
    finally:
        os.close(parent)
        os.close(source_parent)
