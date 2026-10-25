"""Prepared content fences, interrupted explicit removal and no implicit cleanup."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
from foxlight_model_preservation.operations import OperationCancelled, OperationControl
from foxlight_model_preservation.storage import open_file

from skulk.store.mirror_kits import KIT_FILENAME, MirrorKitStore, seal_kit
from skulk.store.preservation_journal import PreservationHostError, PreservationJournal

JOB = "1" * 64
MANIFEST = "a" * 64
CONTENT = b"synthetic prepared bytes; filesystem fixture only\n" * 100


def prepare(store: MirrorKitStore):
    store.prepare(JOB, MANIFEST)
    descriptor = store.target(JOB)
    try:
        with open_file(descriptor, KIT_FILENAME, write=True) as output:
            output.write(CONTENT)
            output.flush()
            os.fsync(output.fileno())
        seal = seal_kit(descriptor, MANIFEST, OperationControl())
    finally:
        os.close(descriptor)
    public = store.record(JOB, seal)
    assert public.sha256 == hashlib.sha256(CONTENT).hexdigest()
    return seal


@pytest.mark.parametrize("damage", ["content", "file", "generation", "root"])
def test_changed_prepared_identity_refuses_without_new_hash(
    tmp_path: Path, damage: str
) -> None:
    profile = tmp_path / "owner"
    journal = PreservationJournal(profile)
    store = MirrorKitStore(profile, journal)
    try:
        seal = prepare(store)
        generation = store.path / JOB
        path = generation / KIT_FILENAME
        if damage == "content":
            path.chmod(0o600)
            path.write_bytes(b"x" * len(CONTENT))
            path.chmod(0o400)
            os.utime(
                path,
                ns=(
                    seal.fingerprint.modified_ns,
                    seal.fingerprint.modified_ns + 2_000_000_000,
                ),
            )
        elif damage == "file":
            path.rename(generation / "original")
            path.write_bytes(CONTENT)
            path.chmod(0o400)
        elif damage == "generation":
            generation.rename(store.path / "previous")
            generation.mkdir(mode=0o700)
            (generation / KIT_FILENAME).write_bytes(CONTENT)
        else:
            store.path.rename(profile / "previous-root")
            store.path.mkdir(mode=0o700)
        with pytest.raises((OSError, PreservationHostError)):
            store.open(JOB)
        assert journal.setting("mirror-kit:" + JOB) is not None
    finally:
        journal.close()


def test_unknown_entries_survive_failed_removal_and_only_explicit_retry_finishes(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "owner"
    journal = PreservationJournal(profile)
    store = MirrorKitStore(profile, journal)
    prepare(store)
    unexpected = store.path / JOB / "unselected-private.pem"
    unexpected.write_bytes(b"must not be adopted or deleted")
    try:
        with pytest.raises(PreservationHostError, match="mirror_kit_directory_changed"):
            store.remove(JOB, JOB)
        quarantine = store.path / (".removed-" + JOB)
        assert (quarantine / KIT_FILENAME).read_bytes() == CONTENT
        assert (
            quarantine / unexpected.name
        ).read_bytes() == b"must not be adopted or deleted"
        with pytest.raises(PreservationHostError):
            store.open(JOB)
    finally:
        journal.close()
    # Reopening the real journal/owner never resumes physical deletion.
    journal = PreservationJournal(profile)
    try:
        reopened = MirrorKitStore(profile, journal)
        assert (quarantine / KIT_FILENAME).read_bytes() == CONTENT
        (quarantine / unexpected.name).unlink()
        result = reopened.remove(JOB, JOB)
        assert result is not None and result.state == "removed"
        assert not quarantine.exists()
        assert reopened.remove(JOB, JOB) == result
    finally:
        journal.close()


def test_symlink_never_exposes_or_deletes_outside_file(tmp_path: Path) -> None:
    profile = tmp_path / "owner"
    journal = PreservationJournal(profile)
    store = MirrorKitStore(profile, journal)
    try:
        prepare(store)
        secret = tmp_path / "outside-secret"
        secret.write_bytes(b"unrelated private bytes")
        path = store.path / JOB / KIT_FILENAME
        path.unlink()
        path.symlink_to(secret)
        with pytest.raises((OSError, PreservationHostError)):
            store.open(JOB)
        with pytest.raises((OSError, PreservationHostError)):
            store.remove(JOB, JOB)
        assert secret.read_bytes() == b"unrelated private bytes"
    finally:
        journal.close()


def test_cancellation_unsealed_output_and_interrupted_empty_enrollment(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "owner"
    journal = PreservationJournal(profile)
    store = MirrorKitStore(profile, journal)
    try:
        store.prepare(JOB, MANIFEST)
        descriptor = store.target(JOB)
        try:
            with open_file(descriptor, KIT_FILENAME, write=True) as output:
                output.write(CONTENT)
            cancelled = OperationControl()
            cancelled.cancel()
            with pytest.raises(OperationCancelled):
                seal_kit(descriptor, MANIFEST, cancelled)
        finally:
            os.close(descriptor)
        with pytest.raises(PreservationHostError, match="not_ready"):
            store.open(JOB)
        assert store.remove(JOB, JOB) is None
        assert not (store.path / JOB).exists()
        # Simulate a crash after durable intent but before directory enrollment.
        second = "2" * 64
        journal.set_setting("mirror-kit:" + second, {"manifest_id": MANIFEST})
        (store.path / second).mkdir(mode=0o700)
        assert store.remove(second, second) is None
        assert not (store.path / second).exists()
        assert store.remove("3" * 64, "3" * 64) is None
    finally:
        journal.close()


def test_confirmations_and_immutable_seal_do_not_replace_or_delete(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "owner"
    journal = PreservationJournal(profile)
    store = MirrorKitStore(profile, journal)
    try:
        seal = prepare(store)
        with pytest.raises(PreservationHostError):
            store.record(JOB, seal.model_copy(update={"sha256": "0" * 64}))
        with pytest.raises(PreservationHostError):
            store.remove(JOB, "0" * 64)
        with pytest.raises(PreservationHostError):
            store.remove("../unknown", "../unknown")
        descriptor, observed = store.open(JOB)
        try:
            assert observed == seal and os.read(descriptor, len(CONTENT)) == CONTENT
        finally:
            os.close(descriptor)
    finally:
        journal.close()
