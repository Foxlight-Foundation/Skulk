"""Actual enrolled-folder custody, interruption and Core-independent recovery."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from foxlight_model_preservation.kits import export_kit
from foxlight_model_preservation.signing import load_trust
from foxlight_model_preservation.storage import verify_store

import skulk.store.preservation_host as host_module
from skulk.store.installed_cards import read_installed_card
from skulk.store.model_store import ModelStore
from skulk.store.preservation_contracts import (
    FolderConfirmation,
    FolderReviewRequest,
    PreservationPlanRequest,
    PreservationStart,
)
from skulk.store.preservation_host import PreservationHost
from skulk.store.preservation_journal import PreservationHostError
from skulk.store.tests.test_library_recovery import wait_job
from skulk.store.tests.test_preservation_export import make_preservation_installation


def owner(root: Path) -> PreservationHost:
    """Build an actual host with complete synthetic installed companion records."""
    installed = make_preservation_installation(root, "assistant")
    return PreservationHost(
        ModelStore(installed.primary.parent), root / "core", custom_cards=root / "cards"
    )


async def test_folder_survives_original_core_and_restores_complete_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SKULK_OFFLINE", "true")
    original = tmp_path / "original"
    host = owner(original)
    archive = tmp_path / "mounted-storage" / "models"
    archive.parent.mkdir()
    try:
        review = await host.review_folder(
            FolderReviewRequest(path=str(archive), label="My NAS")
        )
        assert review.creates_folder and review.free_bytes > 0
        assert not archive.exists() and not (host.root / "custody").exists()
        location = await host.setup_folder(
            FolderConfirmation(review_id=review.review_id, confirmed=True)
        )
        assert location.reachable and location.label == "My NAS"
        assert (
            await host.setup_folder(
                FolderConfirmation(review_id=review.review_id, confirmed=True)
            )
        ).location_id == location.location_id
        page = await host.page()
        assert page.default_location_id == location.location_id
        model = page.models[0]
        plan = await host.plan(
            PreservationPlanRequest(
                kind="preserve",
                location_id=location.location_id,
                model_id=model.model_id,
                generation_id=model.generation_id,
            )
        )
        kept = host.start(
            PreservationStart(plan_id=plan.plan_id, operation_id="nas-keep")
        )
        await wait_job(host, kept.job_id)
        kept = host.journal.job(kept.job_id)
        assert kept.state == "succeeded" and kept.copy_id, kept
        copy = host.journal.copy(kept.copy_id)
        fingerprint = location.publisher_key_id
    finally:
        await host.close()
    reopened = PreservationHost(
        ModelStore(original / "installed"),
        original / "core",
        custom_cards=original / "cards",
    )
    try:
        assert (await reopened.page()).default_location_id == location.location_id
        detached = archive.with_name("disconnected")
        archive.rename(detached)
        page = await reopened.page()
        assert page.default_location_id == location.location_id
        assert not next(
            item for item in page.locations if item.location_id == location.location_id
        ).reachable
        assert next(
            item for item in page.locations if item.location_id == "device"
        ).reachable
        with pytest.raises(PreservationHostError, match="archive_unavailable"):
            await reopened.default_location(location.location_id)
        assert not archive.exists()
        detached.rename(archive)
    finally:
        await reopened.close()
    # Only the archive and its ordinary public verification snapshot survive.
    shutil.rmtree(original)
    trust = load_trust((archive / "archive-trust.json").read_bytes())
    assert trust.keys[0].key_id == fingerprint
    assert not list(archive.rglob("*.pem"))
    assert len(verify_store(archive, copy.manifest_id, trust).members) == 3
    kit = tmp_path / "retained.foxkit"
    export_kit(archive, copy.manifest_id, trust, kit)
    fresh_store = tmp_path / "fresh-models"
    fresh_store.mkdir()
    fresh = PreservationHost(
        ModelStore(fresh_store),
        tmp_path / "fresh-core",
        custom_cards=tmp_path / "fresh-cards",
    )
    try:
        upload = str(fresh.uploads.create(kit.stat().st_size)["upload_id"])
        fresh.uploads.append(upload, 0, kit.read_bytes())
        plan = await fresh.plan(
            PreservationPlanRequest(
                kind="recover", location_id="kit", upload_id=upload, historical=True
            )
        )
        job = fresh.start(
            PreservationStart(
                plan_id=plan.plan_id,
                operation_id="recover-nas",
                accept_owner_capture=True,
            )
        )
        await wait_job(fresh, job.job_id)
        assert fresh.journal.job(job.job_id).state == "succeeded", fresh.journal.job(
            job.job_id
        )
        for alias in ("example/base", "example/companion"):
            path = fresh.store.get_store_path(alias)
            assert path is not None
            record = read_installed_card(path)
            assert (
                record
                and record.preservation
                and record.preservation.manifest_id == copy.manifest_id
            )
    finally:
        await fresh.close()


async def test_interrupted_snapshot_is_unavailable_until_exact_enrollment_repaired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = owner(tmp_path / "source")
    archive = tmp_path / "external"
    try:
        review = await host.review_folder(
            FolderReviewRequest(path=str(archive), label="Backup")
        )

        def failed_write(*args: object, **kwargs: object) -> None:
            raise OSError("simulated trust snapshot interruption")

        with monkeypatch.context() as fault:
            fault.setattr(host_module, "atomic_bytes", failed_write)
            with pytest.raises(OSError):
                await host.setup_folder(
                    FolderConfirmation(review_id=review.review_id, confirmed=True)
                )
        page = await host.page()
        external = next(item for item in page.locations if item.location_id != "device")
        assert not external.reachable and page.default_location_id == "device"
        assert archive.is_dir() and not tuple(archive.iterdir())
        review = await host.review_folder(
            FolderReviewRequest(path=str(archive), label="Backup")
        )
        repaired = await host.setup_folder(
            FolderConfirmation(review_id=review.review_id, confirmed=True)
        )
        assert repaired.location_id == external.location_id and repaired.reachable
        assert (await host.page()).default_location_id == repaired.location_id
    finally:
        await host.close()


async def test_changed_mount_parent_and_expired_review_never_create_backup(
    tmp_path: Path,
) -> None:
    host = owner(tmp_path / "source")
    parent = tmp_path / "mount"
    parent.mkdir()
    archive = parent / "backup"
    try:
        review = await host.review_folder(
            FolderReviewRequest(path=str(archive), label="NAS")
        )
        parent.rename(tmp_path / "old-mount")
        parent.mkdir()
        with pytest.raises(PreservationHostError, match="storage_review_changed"):
            await host.setup_folder(
                FolderConfirmation(review_id=review.review_id, confirmed=True)
            )
        assert not archive.exists()
        saved = host.journal.setting("folder-review")
        assert isinstance(saved, dict) and isinstance(saved["view"], dict)
        saved["view"]["expires_at"] = 0
        host.journal.set_setting("folder-review", saved)
        with pytest.raises(PreservationHostError, match="storage_review_expired"):
            await host.setup_folder(
                FolderConfirmation(review_id=review.review_id, confirmed=True)
            )
        assert not archive.exists()
    finally:
        await host.close()


@pytest.mark.parametrize(
    "kind",
    ["nonempty", "symlink", "permissions", "core", "store", "relative", "control"],
)
async def test_unsafe_folder_review_has_no_archive_or_key_effect(
    tmp_path: Path, kind: str
) -> None:
    host = owner(tmp_path / "source")
    archive = tmp_path / "selected"
    archive.mkdir(mode=0o700)
    if kind == "nonempty":
        (archive / "existing").write_bytes(b"keep")
    elif kind == "symlink":
        link = tmp_path / "link"
        link.symlink_to(archive, target_is_directory=True)
        archive = link
    elif kind == "permissions":
        archive.chmod(0o755)
    elif kind == "core":
        archive = host.root / "archive"
    elif kind == "store":
        archive = host.store.store_path / "archive"
    path = (
        "relative/backup"
        if kind == "relative"
        else str(archive) + "\n"
        if kind == "control"
        else str(archive)
    )
    try:
        with pytest.raises((OSError, PreservationHostError)):
            await host.review_folder(FolderReviewRequest(path=path, label="Backup"))
        assert not (host.root / "custody").exists()
        assert not (await host.page()).locations
        if kind == "nonempty":
            assert (archive / "existing").read_bytes() == b"keep"
    finally:
        await host.close()


async def test_existing_root_replacement_and_unconfirmed_setup_refuse(
    tmp_path: Path,
) -> None:
    host = owner(tmp_path / "source")
    archive = tmp_path / "empty"
    archive.mkdir(mode=0o700)
    try:
        review = await host.review_folder(
            FolderReviewRequest(path=str(archive), label="Private")
        )
        with pytest.raises(
            PreservationHostError, match="storage_confirmation_required"
        ):
            await host.setup_folder(
                FolderConfirmation(review_id=review.review_id, confirmed=False)
            )
        assert not (host.root / "custody").exists()
        archive.rename(tmp_path / "original-empty")
        archive.mkdir(mode=0o700)
        with pytest.raises(PreservationHostError, match="storage_review_changed"):
            await host.setup_folder(
                FolderConfirmation(review_id=review.review_id, confirmed=True)
            )
        assert not tuple(archive.iterdir())
        assert (await host.page()).default_location_id == "device"
    finally:
        await host.close()
