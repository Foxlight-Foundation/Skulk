"""Abrupt worker death, real reader locks and ownership-checked original cleanup."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from skulk.worker.runner.demucs.model import separation_result_directory
from skulk.worker.separation_custody import (
    SOURCE_OWNERSHIP_FILE,
    SeparationSourceCustody,
)
from skulk.worker.separation_sources import SeparationSourceStore
from skulk.worker.tests.test_separation_sources import retain, task

KEY = "b" * 64


def crashed_owner(
    root: Path, *, reader: bool = False, stage: str = "complete"
) -> int | None:
    """Leave actual source/journal files through os._exit, with an optional reader."""
    program = """
import os,subprocess,sys
from pathlib import Path
from skulk.utils.runtime_files import RuntimeLock
from skulk.worker.separation_custody import SeparationSourceCustody
from skulk.worker.separation_sources import SeparationSourceStore
from skulk.worker.tests.test_separation_sources import retain,task
root=Path(sys.argv[1]);key='b'*64
custody=SeparationSourceCustody(root/'inputs',root/'readers',lambda _:key)
store=SeparationSourceStore(root/'inputs',custody=custody)
if sys.argv[3]=='complete':
 retain(store,task('prior'))
else:
 custody.claim(task('prior'))
 if sys.argv[3]=='mkdir': (root/'inputs/prior').mkdir(mode=0o700)
if sys.argv[2]=='reader':
 fence=RuntimeLock(root/'readers'/key,name='attachment.lock')
 child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],pass_fds=(fence.descriptor,),stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 (root/'reader.pid').write_text(str(child.pid))
os._exit(0)
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            program,
            str(root),
            "reader" if reader else "none",
            stage,
        ],
        check=True,
        timeout=15,
    )
    return int((root / "reader.pid").read_text()) if reader else None


def store(root: Path) -> SeparationSourceStore:
    """Reopen the same real paths with the production bounded custody journal."""
    return SeparationSourceStore(
        root / "inputs",
        custody=SeparationSourceCustody(
            root / "inputs", root / "readers", lambda _: KEY
        ),
    )


def test_detach_and_unreconciled_native_output_preserve_the_original_slot(tmp_path: Path) -> None:
    """Restart cannot forget an output set or delete an original on parent exit."""
    first = store(tmp_path)
    admitted = task('retained-result')
    retain(first, admitted)
    original = tmp_path / 'inputs' / str(admitted.command_id) / 'original.wav'
    before = original.read_bytes()
    (tmp_path / "readers" / KEY).mkdir(mode=0o700, parents=True)
    result = separation_result_directory(tmp_path/'readers'/KEY, admitted.command_id)
    result.mkdir(mode=0o700, parents=True)
    unknown = result / 'preserve'
    unknown.write_text('not yet reconciled')
    first.detach()
    assert original.read_bytes() == before
    second = store(tmp_path)
    try:
        assert second.unresolved_source_count() == 1
        assert second.recover_stale() == 0
        assert original.read_bytes() == before and unknown.read_text() == 'not yet reconciled'
        for number in range(3):
            retain(second, task(f'new-{number}'))
        with pytest.raises(ValueError, match='capacity'):
            second.admit(task('would-be-fifth'))
        # Only this test owns these fixture files. Production must prove result
        # ownership before resolving this state; ordinary source cleanup cannot.
        unknown.unlink()
        result.rmdir()
        assert second.recover_stale() == 1 and not original.exists()
        retain(second, task('fourth'))
    finally:
        second.close()


def test_replacement_worker_waits_for_actual_native_reader_before_reclaiming(
    tmp_path: Path,
) -> None:
    """Parent death is insufficient; an inherited reader fence still protects input."""
    native = crashed_owner(tmp_path, reader=True)
    assert native is not None
    owner = store(tmp_path)
    original = tmp_path / "inputs/prior/original.wav"
    before = original.read_bytes()
    try:
        assert owner.unresolved_source_count() == 1 and owner.recover_stale() == 0
        assert original.read_bytes() == before
        for index in range(3):
            retain(owner, task(f"current-{index}"))
        with pytest.raises(ValueError, match="capacity"):
            owner.admit(task("fifth"))
        os.kill(native, signal.SIGKILL)
        native = None
        deadline = time.monotonic() + 5
        while owner.recover_stale() == 0:
            assert time.monotonic() < deadline
            time.sleep(0.02)
        assert not original.exists() and owner.unresolved_source_count() == 0
        assert all(
            (tmp_path / f"inputs/current-{index}/original.wav").is_file()
            for index in range(3)
        )
        retain(owner, task("fourth"))
    finally:
        owner.close()
        if native is not None:
            os.kill(native, signal.SIGKILL)


@pytest.mark.parametrize("stage", ["complete", "intent", "mkdir"])
def test_completed_original_and_empty_crash_gaps_are_recovered(
    tmp_path: Path, stage: str
) -> None:
    """The intent precedes mkdir; no source bytes precede its directory receipt."""
    crashed_owner(tmp_path, stage=stage)
    reopened = store(tmp_path)
    try:
        assert (
            reopened.unresolved_source_count() == 0
            and not (tmp_path / "inputs/prior").exists()
        )
        for index in range(4):
            retain(reopened, task(f"new-{index}"))
        with pytest.raises(ValueError, match="capacity"):
            reopened.admit(task("overflow"))
    finally:
        reopened.close()


@pytest.mark.parametrize(
    "change", ["unmarked", "symlink", "changed_inode", "intent_with_content"]
)
def test_foreign_or_ambiguous_content_is_preserved_and_charged(
    tmp_path: Path, change: str
) -> None:
    """A journal never authorizes deleting unrelated entries or following links."""
    crashed_owner(
        tmp_path, stage="intent" if change == "intent_with_content" else "complete"
    )
    directory = tmp_path / "inputs/prior"
    outside = tmp_path / "creator-original.wav"
    outside.write_bytes(b"creator original remains unchanged")
    if change == "unmarked":
        (tmp_path / "inputs" / SOURCE_OWNERSHIP_FILE).unlink()
    elif change == "symlink":
        (directory / "original.wav").unlink()
        (directory / "original.wav").symlink_to(outside)
    elif change == "changed_inode":
        directory.rename(tmp_path / "inputs/retained")
        directory.mkdir(mode=0o700)
        (directory / "original.wav").write_bytes(b"foreign copy")
    else:
        directory.mkdir(mode=0o700)
        (directory / "original.wav").write_bytes(
            b"unknown content in unreceipted directory"
        )
    reopened = store(tmp_path)
    try:
        assert reopened.unresolved_source_count() >= 1 and reopened.recover_stale() == 0
        assert (directory / "original.wav").exists()
    finally:
        reopened.close()
    assert (
        directory / "original.wav"
    ).exists() and outside.read_bytes() == b"creator original remains unchanged"


def test_invalid_ownership_metadata_refuses_and_releases_attempted_root_fence(
    tmp_path: Path,
) -> None:
    """Corrupt metadata is not ignored or interpreted as permission to remove a path."""
    crashed_owner(tmp_path)
    journal = tmp_path / "inputs" / SOURCE_OWNERSHIP_FILE
    valid = journal.read_bytes()
    journal.write_bytes(
        b'{"version":1,"records":[{"command_id":"../outside","reader_key":"'
        + KEY.encode()
        + b'"}]}'
    )
    with pytest.raises(ValueError):
        store(tmp_path)
    assert (tmp_path / "inputs/prior/original.wav").is_file()
    journal.write_bytes(valid)
    reopened = store(tmp_path)
    reopened.close()
