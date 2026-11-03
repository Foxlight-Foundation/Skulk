"""Real bounded WAV custody, complete-set exposure and process-reopen verification."""

import hashlib
import io
import os
import time
import wave
from pathlib import Path

import pytest

from skulk.api.separation_jobs import SeparationJob
from skulk.api.separation_media import SeparationMediaStore
from skulk.api.video_store import VideoStore
from skulk.shared.types.audio_separation import (
    SEPARATION_STEMS,
    AudioSeparationOutput,
    SeparatedStem,
    SeparationSource,
    SeparationStemName,
)
from skulk.shared.types.common import CommandId, ModelId, NodeId


def wav(value: int = 1000, *, rate: int = 44100) -> bytes:
    """Make a one-second PCM24 stereo file with one exact nonzero sample."""
    data = bytearray(rate * 6)
    data[:3] = value.to_bytes(3, "little", signed=True)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setparams((2, 3, rate, rate, "NONE", "not compressed"))
        audio.writeframes(data)
    return buffer.getvalue()


def queued() -> SeparationJob:
    """Create source metadata only; no inference or upload execution is claimed."""
    return SeparationJob(
        id=CommandId("native-result"),
        model=ModelId("adefossez/HTDemucs"),
        created_at=int(time.time()) - 2,
        producer_node=NodeId("producer"),
        source=SeparationSource(
            sha256="e" * 64,
            size_bytes=16044,
            frames=8000,
            sample_rate=8000,
            channels=1,
            encoding="pcm",
            bit_depth=16,
        ),
    )


def completed(facts: tuple[SeparatedStem, ...]) -> SeparationJob:
    """Validate the durable complete-set contract used by a public content read."""
    original = queued()
    return SeparationJob(
        id=original.id,
        model=original.model,
        created_at=original.created_at,
        producer_node=original.producer_node,
        source=original.source,
        status="completed",
        completed_at=int(time.time()),
        expires_at=int(time.time()) + 3600,
        candidate_output=AudioSeparationOutput(
            source=original.source,
            stems=facts,
            common_gain=1.0,
            pre_gain_peak=max(item.peak for item in facts),
        ),
        render_finished=True,
        verified_stems=frozenset(SEPARATION_STEMS),
    )


def retain(
    store: SeparationMediaStore, name: SeparationStemName, value: int = 1000
) -> SeparatedStem:
    """Send actual bytes through the assembler and independent closed-file check."""
    data = wav(value)
    identifier = queued().id
    store.open_assembly(identifier, name, total_bytes=len(data), total_chunks=1)
    store.append(identifier, name, 1, data)
    assert store.assembly_complete(identifier, name)
    receipt = store.commit(
        identifier, name, sha256=hashlib.sha256(data).hexdigest(), total_chunks=1
    )
    assert receipt.file_path.read_bytes() == data
    return receipt.facts


def test_partial_files_stay_private_even_after_all_transfers_finish(
    tmp_path: Path,
) -> None:
    """Only durable completed metadata plus four exact files grants content."""
    store = SeparationMediaStore(tmp_path / "media")
    facts: list[SeparatedStem] = []
    try:
        for index, name in enumerate(SEPARATION_STEMS, start=1):
            facts.append(retain(store, name, 1000 * index))
            assert store.content(queued(), "drums") is None
        job = completed(tuple(facts))
        for expected in facts:
            artifact = store.content(job, expected.name)
            assert artifact is not None and artifact.facts == expected
        assert store.reserved_bytes() == sum(item.size_bytes for item in facts)
    finally:
        store.close()


def test_identical_repeated_open_preserves_received_chunks(tmp_path: Path) -> None:
    """Transport duplication cannot discard already written output."""
    store = SeparationMediaStore(tmp_path / "media")
    data = wav()
    identifier = queued().id
    try:
        store.open_assembly(identifier, "drums", total_bytes=len(data), total_chunks=2)
        store.append(identifier, "drums", 1, data[:1000])
        store.open_assembly(identifier, "drums", total_bytes=len(data), total_chunks=2)
        store.append(identifier, "drums", 2, data[1000:])
        receipt = store.commit(
            identifier, "drums", sha256=hashlib.sha256(data).hexdigest(), total_chunks=2
        )
        assert receipt.file_path.read_bytes() == data
    finally:
        store.close()


def test_last_invalid_wav_discards_only_owned_result_set(tmp_path: Path) -> None:
    """A valid transport hash cannot make a non-WAV or partial set public."""
    original = tmp_path / "original.wav"
    original.write_bytes(wav())
    before = original.read_bytes()
    store = SeparationMediaStore(tmp_path / "media")
    try:
        for name in SEPARATION_STEMS[:3]:
            retain(store, name)
        bad = b"x" * 128
        store.open_assembly(queued().id, "vocals", total_bytes=len(bad), total_chunks=1)
        store.append(queued().id, "vocals", 1, bad)
        with pytest.raises(ValueError):
            store.commit(
                queued().id,
                "vocals",
                sha256=hashlib.sha256(bad).hexdigest(),
                total_chunks=1,
            )
        assert store.reserved_bytes() == 0
        assert not tuple((tmp_path / "media").glob("*/native-result/*"))
        assert original.read_bytes() == before
    finally:
        store.close()


@pytest.mark.parametrize("damage", [False, True])
def test_restart_adopts_every_exact_file_or_exposes_none(
    tmp_path: Path, *, damage: bool
) -> None:
    """Size/digest/geometry/peak are independently checked after owner restart."""
    store = SeparationMediaStore(tmp_path / "media")
    facts = tuple(
        retain(store, name, index * 1000)
        for index, name in enumerate(SEPARATION_STEMS, start=1)
    )
    job = completed(facts)
    artifact = store.content(job, "vocals")
    assert artifact is not None
    store.close()
    if damage:
        data = bytearray(artifact.file_path.read_bytes())
        data[-1] ^= 1
        artifact.file_path.write_bytes(data)
    reopened = SeparationMediaStore(tmp_path / "media")
    try:
        assert reopened.content(job, "drums") is None
        assert reopened.adopt_complete(job) is (not damage)
        for name in SEPARATION_STEMS:
            assert (reopened.content(job, name) is not None) is (not damage)
    finally:
        reopened.close()


def test_media_owner_fence_and_active_transfer_bounds(tmp_path: Path) -> None:
    """A second owner cannot race files, and four jobs bound pending disk promises."""
    store = SeparationMediaStore(tmp_path / "media")
    try:
        with pytest.raises(BlockingIOError):
            SeparationMediaStore(tmp_path / "media")
        for number in range(4):
            store.open_assembly(
                CommandId(f"job-{number}"), "drums", total_bytes=100, total_chunks=1
            )
        with pytest.raises(ValueError, match="active-job"):
            store.open_assembly(
                CommandId("overflow"), "bass", total_bytes=100, total_chunks=1
            )
    finally:
        store.close()
    reopened = SeparationMediaStore(tmp_path / "media")
    assert reopened.purge_unknown(frozenset()) == 4
    reopened.close()


def test_total_budget_cannot_evict_previous_closed_stems(tmp_path: Path) -> None:
    """Capacity refusal preserves earlier complete copies while metadata settles."""
    size = len(wav())
    store = SeparationMediaStore(tmp_path / "media", max_total_bytes=size * 2)
    try:
        first = retain(store, "drums")
        second = retain(store, "bass")
        with pytest.raises(ValueError, match="byte capacity"):
            store.open_assembly(queued().id, "other", total_bytes=size, total_chunks=1)
        assert store.reserved_bytes() == first.size_bytes + second.size_bytes
        assert len(tuple((tmp_path / "media").glob("*/native-result/output.wav"))) == 2
    finally:
        store.close()


@pytest.mark.parametrize("second", ["bass", "other", "vocals"])
def test_roles_share_unwritten_filesystem_promises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, second: SeparationStemName
) -> None:
    """One remaining file's free space cannot be promised again to another role."""
    size = len(wav())

    def free(_self: VideoStore) -> int:
        return 1024**3 + size

    monkeypatch.setattr(VideoStore, "_free_disk_bytes", free)
    store = SeparationMediaStore(tmp_path / "media")
    try:
        store.open_assembly(queued().id, "drums", total_bytes=size, total_chunks=1)
        # These same unchanged filesystem bytes are visible from every child.
        with pytest.raises(ValueError, match="filesystem headroom"):
            store.open_assembly(queued().id, second, total_bytes=size, total_chunks=1)
        assert store.reserved_bytes() == size
    finally:
        store.close()


def test_written_bytes_are_not_reserved_twice_across_roles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Actual disk consumption plus remaining promises has one consistent budget."""
    data = wav()
    available = [1024**3 + 3 * len(data)]

    def free(_self: VideoStore) -> int:
        return available[0]

    monkeypatch.setattr(VideoStore, "_free_disk_bytes", free)
    store = SeparationMediaStore(tmp_path / "media")
    try:
        for name in ("drums", "bass"):
            store.open_assembly(queued().id, name, total_bytes=len(data), total_chunks=1)
        store.append(queued().id, "drums", 1, data)
        available[0] -= len(data)  # The filesystem now accounts for this write.
        store.open_assembly(queued().id, "other", total_bytes=len(data), total_chunks=1)
        with pytest.raises(ValueError, match="filesystem headroom"):
            store.open_assembly(queued().id, "vocals", total_bytes=len(data), total_chunks=1)
        assert store.reserved_bytes() == 3 * len(data)
    finally:
        store.close()


def test_reordered_chunks_still_reserve_their_unwritten_disk_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RAM-stashed media has not consumed disk space and still needs its promise."""
    data = wav()

    def free(_self: VideoStore) -> int:
        return 1024**3 + 2 * len(data)

    monkeypatch.setattr(VideoStore, "_free_disk_bytes", free)
    store = SeparationMediaStore(tmp_path / "media")
    try:
        store.open_assembly(queued().id, "drums", total_bytes=len(data), total_chunks=2)
        store.append(queued().id, "drums", 2, data[1000:])
        store.open_assembly(queued().id, "bass", total_bytes=len(data), total_chunks=1)
        with pytest.raises(ValueError, match="filesystem headroom"):
            store.open_assembly(queued().id, "other", total_bytes=100, total_chunks=1)
        assert store.reserved_bytes() == 2 * len(data)
    finally:
        store.close()


def test_receipts_sync_file_job_role_and_new_root_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completed result cannot outlive any unsynced parent directory entry."""
    root = tmp_path / "new-parent" / "media"
    real_sync = os.fsync
    synced: list[tuple[int, int]] = []

    def observe(descriptor: int) -> None:
        real_sync(descriptor)
        info = os.fstat(descriptor)
        synced.append((info.st_dev, info.st_ino))

    def identity(path: Path) -> tuple[int, int]:
        info = path.stat()
        return info.st_dev, info.st_ino

    monkeypatch.setattr(os, "fsync", observe)
    store = SeparationMediaStore(root)
    try:
        for directory in (tmp_path, root.parent, root, *(root / name for name in SEPARATION_STEMS)):
            assert identity(directory) in synced
        synced.clear()
        retain(store, "drums")
        job_directory = root / "drums" / queued().id
        # Ordering matters: file bytes, its directory entry, then that
        # directory's own link must settle before the receipt is returned.
        assert synced[-3:] == [
            identity(job_directory / "output.wav"),
            identity(job_directory),
            identity(job_directory.parent),
        ]
    finally:
        store.close()


def test_failed_role_sync_cannot_return_a_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A durability error removes only the failed job's private API copies."""
    root = tmp_path / "media"
    store = SeparationMediaStore(root)
    real_sync = os.fsync
    role = (root / "drums").stat()

    def refuse(descriptor: int) -> None:
        info = os.fstat(descriptor)
        if (info.st_dev, info.st_ino) == (role.st_dev, role.st_ino):
            raise OSError("injected role sync failure")
        real_sync(descriptor)

    monkeypatch.setattr(os, "fsync", refuse)
    try:
        with pytest.raises(OSError, match="role sync failure"):
            retain(store, "drums")
        assert store.reserved_bytes() == 0
        assert not tuple(root.glob("*/native-result/output.wav"))
    finally:
        store.close()


def test_failed_new_root_sync_releases_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failed storage initialization cannot retain a lock or claim durability."""
    root = tmp_path / "new-parent" / "media"
    real_sync = os.fsync
    parent = tmp_path.stat()

    def refuse(descriptor: int) -> None:
        info = os.fstat(descriptor)
        if (info.st_dev, info.st_ino) == (parent.st_dev, parent.st_ino):
            raise OSError("injected root sync failure")
        real_sync(descriptor)

    monkeypatch.setattr(os, "fsync", refuse)
    with pytest.raises(OSError, match="root sync failure"):
        SeparationMediaStore(root)
    assert not root.exists() and not root.parent.exists()
    retried = False

    def observe_retry(descriptor: int) -> None:
        nonlocal retried
        info = os.fstat(descriptor)
        if (info.st_dev, info.st_ino) == (parent.st_dev, parent.st_ino):
            retried = True
        real_sync(descriptor)

    monkeypatch.setattr(os, "fsync", observe_retry)
    reopened = SeparationMediaStore(root)
    assert retried
    retain(reopened, "drums")
    reopened.close()


@pytest.mark.parametrize("release", ["complete", "delete"])
def test_committed_partial_sets_hold_active_slots_until_complete_or_deleted(
    tmp_path: Path, release: str
) -> None:
    """Five partial jobs cannot hide their active slots by committing only drums."""
    store = SeparationMediaStore(tmp_path / "media")
    data = wav()
    digest = hashlib.sha256(data).hexdigest()

    def commit(identifier: CommandId, role: SeparationStemName) -> None:
        store.open_assembly(identifier, role, total_bytes=len(data), total_chunks=1)
        store.append(identifier, role, 1, data)
        store.commit(identifier, role, sha256=digest, total_chunks=1)

    try:
        for number in range(4):
            commit(CommandId(f"partial-{number}"), "drums")
        with pytest.raises(ValueError, match="active-job"):
            commit(CommandId("fifth"), "drums")
        first = CommandId("partial-0")
        if release == "complete":
            for role in SEPARATION_STEMS[1:]:
                commit(first, role)
        else:
            store.delete(first)
        commit(CommandId("fifth"), "drums")
        # Settling one job never frees another partial set's slot or its bytes.
        with pytest.raises(ValueError, match="active-job"):
            commit(CommandId("sixth"), "bass")
        assert len(tuple((store.root / "drums").glob("partial-*/output.wav"))) == (4 if release == "complete" else 3)
    finally:
        store.close()
