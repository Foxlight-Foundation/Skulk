"""Actual WAV bytes, durable receipts and restart/cancel boundaries for edits."""

import os
import shutil
import time
from pathlib import Path

import pytest

from skulk.api.music_edit_jobs import MusicEditJob, MusicEditJobRegistry
from skulk.api.music_edit_media import MusicEditMediaStore
from skulk.shared.tests.music_edit_fixtures import music_edit_fixture
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.music_edits import MusicEditingOutput
from skulk.shared.types.native_music import NativeMusicWav


def retain(
    store: MusicEditMediaStore, identifier: CommandId,
    expected: NativeMusicWav, data: bytes,
) -> None:
    """Exercise real reversed-chunk assembly and independently measured custody."""
    store.open_assembly(identifier, total_bytes=len(data), total_chunks=2)
    store.append(identifier, 2, data[1000:])
    assert not store.assembly_complete(identifier)
    store.open_assembly(identifier, total_bytes=len(data), total_chunks=2)
    store.append(identifier, 1, data[:1000])
    assert store.assembly_complete(identifier)
    receipt = store.commit(identifier, expected, total_chunks=2)
    assert receipt.facts == expected and receipt.file_path.read_bytes() == data
    assert receipt.file_path.stat().st_mode & 0o077 == 0


def completed(job: MusicEditJob, output: MusicEditingOutput) -> MusicEditJob:
    """Validate a fully acknowledged durable record for startup adoption tests."""
    return MusicEditJob.model_validate(job.model_dump() | {
        "producer_node": NodeId("producer"), "status": "completed",
        "candidate_output": output, "render_finished": True,
        "verified_output": True, "return_acknowledged": True,
        "completed_at": int(time.time()), "expires_at": int(time.time()) + 3600,
    })


def test_verified_wav_stays_private_until_durable_render_and_return_receipt(tmp_path: Path) -> None:
    """A file alone, manifest alone or early render finish never grants a download."""
    job, output, _, data = music_edit_fixture()
    store = MusicEditMediaStore(tmp_path / "media")
    jobs = MusicEditJobRegistry(tmp_path / "jobs.json")
    try:
        jobs.create(job)
        jobs.bind_worker(job.id, NodeId("producer"))
        jobs.record_manifest(job.id, NodeId("producer"), output)
        retain(store, job.id, output.result.output, data)
        assert store.content(job) is None
        receipt_job = jobs.record_verified_output(
            job.id, NodeId("producer"), output, expires_at=int(time.time()) + 3600,
        )
        assert receipt_job is not None and store.content(receipt_job) is None
        jobs.finish_render(job.id, NodeId("producer"))
        waiting = jobs.settle(job.id)
        assert waiting is not None and store.content(waiting) is None
        jobs.acknowledge_return(job.id, NodeId("producer"))
        ready = jobs.settle(job.id)
        assert ready is not None and ready.status == "completed"
        content = store.content(ready)
        assert content is not None and content.file_path.read_bytes() == data
        assert store.reserved_bytes() == len(data)
    finally:
        store.close()
        jobs.close()


@pytest.mark.parametrize("damage", ["none", "bytes", "mode", "hardlink", "missing"])
def test_restart_rechecks_complete_bytes_before_adoption(tmp_path: Path, damage: str) -> None:
    """A prior completed record cannot bless changed, public or multiply linked data."""
    job, output, _, data = music_edit_fixture()
    ready = completed(job, output)
    root = tmp_path / "media"
    store = MusicEditMediaStore(root)
    retain(store, job.id, output.result.output, data)
    stored = store.content(ready)
    assert stored is not None
    path = stored.file_path
    store.close()
    if damage == "bytes":
        changed = bytearray(data)
        changed[-1] ^= 1
        path.write_bytes(changed)
    elif damage == "mode":
        path.chmod(0o644)
    elif damage == "hardlink":
        os.link(path, tmp_path / "retained-link.wav")
    elif damage == "missing":
        path.unlink()
    reopened = MusicEditMediaStore(root)
    try:
        assert reopened.content(ready) is None
        assert reopened.adopt_complete(ready) is (damage == "none")
        assert (reopened.content(ready) is not None) is (damage == "none")
    finally:
        reopened.close()


@pytest.mark.parametrize("field,value", [
    ("peak", 0.5), ("frames", 96000), ("size_bytes", 288050), ("sha256", "f" * 64),
])
def test_native_manifest_cannot_substitute_for_measured_wav(
    tmp_path: Path, field: str, value: int | float | str,
) -> None:
    """Every encoded result fact, including late peak, is independently checked."""
    job, output, original, data = music_edit_fixture()
    creator = tmp_path / "original.wav"
    creator.write_bytes(original)
    store = MusicEditMediaStore(tmp_path / "media")
    try:
        store.open_assembly(job.id, total_bytes=len(data), total_chunks=1)
        store.append(job.id, 1, data)
        with pytest.raises(ValueError):
            store.commit(job.id, output.result.output.model_copy(update={field: value}), total_chunks=1)
        assert store.reserved_bytes() == 0 and creator.read_bytes() == original
        assert not (store.root / "wav" / str(job.id)).exists()
    finally:
        store.close()


def test_byte_budget_preserves_retained_result_and_failed_delete_keeps_charge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed cleanup cannot admit another result into already promised space."""
    job, output, _, data = music_edit_fixture()
    store = MusicEditMediaStore(tmp_path / "media", max_total_bytes=len(data))

    def retain_files(_path: Path, *, ignore_errors: bool = False) -> None:
        """Model a filesystem refusing the assembler's best-effort tree deletion."""

    try:
        retain(store, job.id, output.result.output, data)
        monkeypatch.setattr(shutil, "rmtree", retain_files)
        with pytest.raises(OSError, match="cleanup is incomplete"):
            store.delete(job.id)
        assert store.reserved_bytes() == len(data)
        with pytest.raises(ValueError, match="byte capacity"):
            store.open_assembly(CommandId("next"), total_bytes=len(data), total_chunks=1)
        monkeypatch.undo()
        store.delete(job.id)
        assert store.reserved_bytes() == 0
        store.open_assembly(CommandId("next"), total_bytes=len(data), total_chunks=1)
    finally:
        monkeypatch.undo()
        store.close()


def test_fsync_failure_exposes_no_result_and_preserves_other_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hash-valid WAV without a durable receipt must not survive as completed."""
    first, first_output, _, first_data = music_edit_fixture("first")
    second, second_output, _, second_data = music_edit_fixture("second")
    store = MusicEditMediaStore(tmp_path / "media")
    try:
        retain(store, first.id, first_output.result.output, first_data)
        store.open_assembly(second.id, total_bytes=len(second_data), total_chunks=1)
        store.append(second.id, 1, second_data)

        def refuse(_descriptor: int) -> None:
            raise OSError("Injected result sync failure")

        monkeypatch.setattr(os, "fsync", refuse)
        with pytest.raises(OSError, match="sync failure"):
            store.commit(second.id, second_output.result.output, total_chunks=1)
        monkeypatch.undo()
        assert store.reserved_bytes() == len(first_data)
        assert store.content(completed(second, second_output)) is None
        assert store.content(completed(first, first_output)) is not None
    finally:
        monkeypatch.undo()
        store.close()


def test_expiry_failed_job_and_owner_fence_never_expose_bytes(tmp_path: Path) -> None:
    """Completed-looking files need current completed unexpired metadata and owner."""
    job, output, _, data = music_edit_fixture()
    store = MusicEditMediaStore(tmp_path / "media")
    try:
        with pytest.raises(BlockingIOError):
            MusicEditMediaStore(store.root)
        retain(store, job.id, output.result.output, data)
        assert store.content(job) is None
        ready = completed(job, output)
        expired = ready.model_copy(update={"expires_at": int(time.time()) - 1})
        assert store.content(expired) is None
        assert store.content(ready) is not None
        store.delete(job.id)
        assert store.content(ready) is None
    finally:
        store.close()
