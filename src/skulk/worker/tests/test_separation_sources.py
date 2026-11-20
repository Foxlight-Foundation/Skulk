"""Actual private WAV assembly, source identity and bounded copy ownership."""

import io
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import pytest

from skulk.shared.types.audio_separation import (
    AudioSeparationTaskParams,
    source_from_wav,
)
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.tasks import AudioSeparation, TaskStatus
from skulk.shared.types.worker.instances import InstanceId
from skulk.worker.separation_sources import SeparationSourceStore


def source() -> bytes:
    """A real one-second PCM16 mono original, not a PCM24 generated stem."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setparams((1, 2, 8000, 8000, "NONE", "not compressed"))
        audio.writeframes(bytes(16000))
    return buffer.getvalue()


def task(identifier: str = "source-1", *, chunks: int = 2) -> AudioSeparation:
    """Represent authoritative task facts after master placement, with no paths."""
    return AudioSeparation(
        command_id=CommandId(identifier), owner_node=NodeId("owner"),
        instance_id=InstanceId("mounted"),
        task_params=AudioSeparationTaskParams(
            model="fixture/separator", source=source_from_wav(source()),
            total_input_chunks=chunks,
        ),
    )


def retain(store: SeparationSourceStore, admitted: AudioSeparation) -> Path:
    """Send a complete original through the real shared assembler."""
    store.admit(admitted)
    data = source()
    store.append(admitted.command_id, 2, data[8000:])
    assert store.complete(admitted.command_id) is None
    store.append(admitted.command_id, 1, data[:8000])
    retained = store.complete(admitted.command_id)
    assert retained is not None
    return retained


def test_reordered_original_is_private_verified_and_idempotently_admitted(tmp_path: Path) -> None:
    """Only complete original facts unlock the fixed private resident input path."""
    store = SeparationSourceStore(tmp_path / "inputs")
    admitted = task()
    store.admit(admitted)
    store.append(admitted.command_id, 2, source()[8000:])
    store.admit(admitted)
    store.admit(admitted.model_copy(update={"task_status": TaskStatus.Running}))
    assert store.complete(admitted.command_id) is None
    store.append(admitted.command_id, 1, source()[:8000])
    retained = store.complete(admitted.command_id)
    assert retained is not None
    assert retained == tmp_path / "inputs/source-1/original.wav"
    assert retained.read_bytes() == source() and retained.stat().st_mode & 0o077 == 0
    assert store.ready_path(admitted.command_id) == store.complete(admitted.command_id) == retained
    store.close()
    assert not retained.exists()


@pytest.mark.parametrize("status", [TaskStatus.Complete, TaskStatus.Cancelled, TaskStatus.Failed, TaskStatus.TimedOut])
def test_terminal_task_cannot_admit_new_source(tmp_path: Path, status: TaskStatus) -> None:
    """A replayed terminal task cannot allocate a new private source copy."""
    store = SeparationSourceStore(tmp_path / "inputs")
    with pytest.raises(ValueError, match="terminal"):
        store.admit(task().model_copy(update={"task_status": status}))
    assert not (store.root / "source-1").exists()
    store.close()


def test_unknown_frames_and_changed_task_identity_do_not_replace_progress(tmp_path: Path) -> None:
    """Raw packets need an admitted task; a changed owner cannot steal its source."""
    store = SeparationSourceStore(tmp_path / "inputs")
    admitted = task()
    with pytest.raises(ValueError, match="receiving"):
        store.append(admitted.command_id, 1, source())
    with pytest.raises(ValueError, match="admitted"):
        store.complete(admitted.command_id)
    store.admit(admitted)
    store.append(admitted.command_id, 1, source()[:8000])
    for changed in (
        admitted.model_copy(update={"owner_node": NodeId("other")}),
        task(),
        admitted.model_copy(update={"instance_id": InstanceId("other")}),
    ):
        with pytest.raises(ValueError, match="changed"):
            store.admit(changed)
    store.append(admitted.command_id, 2, source()[8000:])
    assert store.complete(admitted.command_id) is not None
    store.close()


def test_four_verified_inputs_still_hold_capacity_until_owner_releases(tmp_path: Path) -> None:
    """Finishing transfer does not free the memory/disk slot used by its model."""
    store = SeparationSourceStore(tmp_path / "inputs")
    originals = [retain(store, task(f"job-{index}")) for index in range(4)]
    with pytest.raises(ValueError, match="capacity"):
        store.admit(task("fifth"))
    store.delete(CommandId("job-0"))
    retain(store, task("fifth"))
    assert all(path.is_file() for path in originals[1:])
    with pytest.raises(ValueError, match="capacity"):
        store.admit(task("sixth"))
    store.close()


def test_corrupt_original_is_removed_without_deleting_other_sources(tmp_path: Path) -> None:
    """A bad digest cannot acknowledge bytes or erase a different retained input."""
    store = SeparationSourceStore(tmp_path / "inputs")
    first = retain(store, task("good"))
    admitted = task("bad", chunks=1)
    store.admit(admitted)
    store.append(admitted.command_id, 1, source()[:-1] + b"x")
    with pytest.raises(ValueError, match="SHA-256"):
        store.complete(admitted.command_id)
    assert store.ready_path(admitted.command_id) is None
    assert not (store.root / "bad").exists() and first.read_bytes() == source()
    store.close()


def test_correct_bytes_with_wrong_declared_geometry_fail_before_ack(tmp_path: Path) -> None:
    """Matching encoded hash does not permit forged source sample geometry."""
    store = SeparationSourceStore(tmp_path / "inputs")
    admitted = task("geometry", chunks=1)
    admitted = admitted.model_copy(update={"task_params": admitted.task_params.model_copy(update={
        "source": admitted.task_params.source.model_copy(update={"frames": 7999}),
    })})
    store.admit(admitted)
    store.append(admitted.command_id, 1, source())
    with pytest.raises(ValueError, match="WAV facts"):
        store.complete(admitted.command_id)
    assert not (store.root / "geometry").exists()
    store.close()


def test_source_lifetime_excludes_overlapping_owner_and_preserves_user_file(tmp_path: Path) -> None:
    """Only internal source copies belong to cleanup; lifetime fences can reopen."""
    original = tmp_path / "creator-original.wav"
    original.write_bytes(source())
    root = tmp_path / "inputs"
    store = SeparationSourceStore(root)
    with pytest.raises(BlockingIOError):
        SeparationSourceStore(root)
    retain(store, task())
    store.close()
    store.close()
    assert original.read_bytes() == source()
    reopened = SeparationSourceStore(root)
    assert reopened.ready_path(CommandId("source-1")) is None
    with pytest.raises(RuntimeError, match="closed"):
        store.admit(task("late"))
    reopened.close()


def test_stale_or_foreign_copy_is_not_overwritten(tmp_path: Path) -> None:
    """A new owner cannot silently overwrite a preexisting unowned directory."""
    store = SeparationSourceStore(tmp_path / "inputs")
    directory = store.root / "source-1"
    directory.mkdir(mode=0o700)
    existing = directory / "original.wav"
    existing.write_bytes(source())
    with pytest.raises(ValueError, match="stale-copy"):
        store.admit(task())
    store.close()
    assert existing.read_bytes() == source()


def test_failed_open_has_no_retained_writer_or_capacity_slot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A permission error after opening staging closes the actual assembly handle."""
    store = SeparationSourceStore(tmp_path / "inputs")
    actual = Path.chmod

    def refuse(path: Path, mode: int, *, follow_symlinks: bool = True) -> None:
        if path.name == "output.wav.part":
            raise OSError("injected source permission failure")
        actual(path, mode, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "chmod", refuse)
    with pytest.raises(OSError, match="permission failure"):
        store.admit(task())
    assert not (store.root / "source-1").exists()
    monkeypatch.setattr(Path, "chmod", actual)
    retain(store, task())
    store.close()


def test_failed_delete_retains_capacity_and_owner_for_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed file drain cannot free the bound or release the source lifetime."""
    store = SeparationSourceStore(tmp_path / "inputs")
    retained = [retain(store, task(f"owned-{index}")) for index in range(4)]
    remove = shutil.rmtree

    def refuse(path: Path, ignore_errors: bool = False) -> None:
        pass  # Model the shared assembler's ignore_errors behavior on a locked file.

    monkeypatch.setattr(shutil, "rmtree", refuse)
    with pytest.raises(OSError, match="cleanup is incomplete"):
        store.delete(CommandId("owned-0"))
    with pytest.raises(ValueError, match="capacity"):
        store.admit(task("fifth"))
    with pytest.raises(OSError, match="cleanup is incomplete"):
        store.close()
    with pytest.raises(BlockingIOError):
        SeparationSourceStore(store.root)
    assert all(path.exists() for path in retained)
    monkeypatch.setattr(shutil, "rmtree", remove)
    store.close()
    assert not any(path.exists() for path in retained)


def test_abrupt_worker_exit_cannot_reset_source_capacity(tmp_path: Path) -> None:
    """Four actual originals survive owner death and still charge the next owner."""
    root = tmp_path / "inputs"
    script = """
import os
from pathlib import Path
import sys
from skulk.worker.separation_sources import SeparationSourceStore
from skulk.worker.tests.test_separation_sources import retain, task
store = SeparationSourceStore(Path(sys.argv[1]))
for index in range(4):
    retain(store, task(f'crashed-{index}'))
os._exit(0)
"""
    subprocess.run([sys.executable, "-c", script, str(root)], check=True, timeout=60)
    for _ in range(2):
        reopened = SeparationSourceStore(root)
        try:
            assert reopened.unresolved_source_count() == 4
            with pytest.raises(ValueError, match="capacity"):
                reopened.admit(task("next-worker"))
            reopened.delete(CommandId("crashed-0"))  # No current ownership proof.
            assert len(tuple(root.glob("*/original.wav"))) == 4
        finally:
            reopened.close()
        assert len(tuple(root.glob("*/original.wav"))) == 4


def test_unknown_entries_consume_slots_without_being_followed_or_deleted(tmp_path: Path) -> None:
    """Preserve foreign files/symlinks while refusing an unbounded restart leak."""
    root = tmp_path / "inputs"
    root.mkdir(mode=0o700)
    outside = tmp_path / "foreign.wav"
    outside.write_bytes(source())
    (root / "foreign-link").symlink_to(outside)
    (root / "foreign-directory").mkdir()
    (root / "foreign-file").write_bytes(b"preserve")
    reopened = SeparationSourceStore(root)
    assert reopened.unresolved_source_count() == 3
    retained = retain(reopened, task("owned"))
    with pytest.raises(ValueError, match="capacity"):
        reopened.admit(task("overflow"))
    reopened.close()
    assert not retained.exists() and outside.read_bytes() == source()
    assert (root / "foreign-link").is_symlink()
    assert (root / "foreign-file").read_bytes() == b"preserve"
    assert (root / "foreign-directory").is_dir()


def test_overfull_unknown_root_refuses_boundedly_and_releases_new_fence(tmp_path: Path) -> None:
    """A root already over policy cannot become empty capacity through a scan."""
    root = tmp_path / "inputs"
    root.mkdir(mode=0o700)
    for index in range(5):
        (root / f"unknown-{index}").mkdir()
    with pytest.raises(ValueError, match="ownership-checked"):
        SeparationSourceStore(root)
    (root / "unknown-4").rmdir()
    reopened = SeparationSourceStore(root)
    assert reopened.unresolved_source_count() == 4
    reopened.close()
