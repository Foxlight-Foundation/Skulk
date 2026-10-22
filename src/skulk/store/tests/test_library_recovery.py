"""Real-worker disaster recovery through the ordinary operator Library API."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from skulk.api.preservation import create_preservation_router
from skulk.store.installed_cards import (
    discover_installed_cards,
    read_installed_card,
    verify_installed_card,
)
from skulk.store.model_store import ModelStore
from skulk.store.preservation_contracts import (
    KitUploadStatus,
    PreservationJob,
    PreservationPlan,
    PreservationPlanRequest,
    PreservationStart,
)
from skulk.store.preservation_host import PreservationHost
from skulk.store.tests.test_preservation_export import make_preservation_installation


async def wait_job(host: PreservationHost, job_id: str) -> None:
    async with asyncio.timeout(20):
        while host.journal.job(job_id).state in {"queued", "running", "cancelling"}:
            await asyncio.sleep(0.02)


@pytest.fixture(autouse=True)
def isolate_recovery_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep actual receiver cache registration local to each isolated test."""
    from skulk.shared.models import model_cards

    for name in (
        "_card_cache",
        "_installed_card_cache",
        "_installed_card_mutation_versions",
        "_installed_current_registry_ids",
    ):
        monkeypatch.setattr(model_cards, name, {})
    monkeypatch.setattr(model_cards, "_installed_card_cache_version", 0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "role", ["assistant", "served_draft", "mtp_sidecar", "vision_weights", "projector"]
)
async def test_kit_restores_complete_closure_after_original_loss(
    tmp_path: Path, role: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typing import cast

    from skulk.store.tests.test_preservation_export import Role

    original = tmp_path / "original-core"
    original.mkdir()
    installed = make_preservation_installation(original, cast(Role, role))
    source = PreservationHost(
        ModelStore(installed.primary.parent),
        original / "private",
        custom_cards=original / "cards",
    )
    await source.setup_device()
    model = (await source.page()).models[0]
    plan = await source.plan(
        PreservationPlanRequest(
            kind="preserve",
            location_id="device",
            model_id=model.model_id,
            generation_id=model.generation_id,
        )
    )
    assert plan.archive_format == "pmf-v0.1"
    saved = source.start(PreservationStart(plan_id=plan.plan_id, operation_id="save"))
    await wait_job(source, saved.job_id)
    saved = source.journal.job(saved.job_id)
    assert saved.state == "succeeded", saved
    plan = await source.plan(
        PreservationPlanRequest(
            kind="kit_export", location_id="device", copy_id=saved.copy_id
        )
    )
    kit_job = source.start(PreservationStart(plan_id=plan.plan_id, operation_id="kit"))
    await wait_job(source, kit_job.job_id)
    kit_job = source.journal.job(kit_job.job_id)
    assert kit_job.state == "succeeded", kit_job
    kit_bytes = source.kit_path(kit_job.job_id).read_bytes()
    from foxlight_model_preservation.kits import PMF_MAGIC

    assert kit_bytes.startswith(PMF_MAGIC)
    await source.close()
    shutil.rmtree(original)

    receiver = tmp_path / "receiver"
    receiver.mkdir()
    store_path = receiver / "store"
    store_path.mkdir()
    host = PreservationHost(
        ModelStore(store_path), receiver / "private", custom_cards=receiver / "cards"
    )

    async def provider() -> PreservationHost:
        return host

    app = FastAPI()
    app.include_router(create_preservation_router(provider, lambda request: None))
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://local"
        ) as client:
            upload_reply = await client.post(
                "/preservation/kit-uploads", json={"size_bytes": len(kit_bytes)}
            )
            upload = KitUploadStatus.model_validate_json(upload_reply.content)
            uid = upload.upload_id
            for offset in range(0, len(kit_bytes), 4096):
                reply = await client.put(
                    "/preservation/kit-uploads/" + uid,
                    params={"offset": offset},
                    content=kit_bytes[offset : offset + 4096],
                    headers={"Content-Type": "application/octet-stream"},
                )
                assert reply.status_code == 200, reply.text
            reply = await client.post(
                "/preservation/plans",
                json={
                    "kind": "recover",
                    "location_id": "kit",
                    "upload_id": uid,
                    "historical": True,
                },
            )
            assert reply.status_code == 200, reply.text
            review = PreservationPlan.model_validate_json(reply.content)
            assert review.archive_format == "pmf-v0.1"
            assert review.requires_owner_acceptance is True
            assert review.requires_code_permission is (
                role in {"vision_weights", "projector"}
            )
            assert review.publication == "owner-only"
            assert len(review.publisher_key_ids) == 1
            assert review.members == (
                ("example/base",)
                if role == "projector"
                else ("example/base", "example/companion")
            )
            refused = await client.post(
                "/preservation/jobs",
                json={"plan_id": review.plan_id, "operation_id": "restore-refused"},
            )
            assert refused.status_code == 409
            request = {
                "plan_id": review.plan_id,
                "operation_id": "restore",
                "accept_owner_capture": True,
                "allow_repository_code": role in {"vision_weights", "projector"},
            }
            started = PreservationJob.model_validate_json(
                (await client.post("/preservation/jobs", json=request)).content
            )
            job_id = started.job_id
            await wait_job(host, job_id)
            job = host.journal.job(job_id)
            assert job.state == "succeeded", job
            assert (
                PreservationJob.model_validate_json(
                    (await client.post("/preservation/jobs", json=request)).content
                ).job_id
                == job_id
            )
        for alias in review.members:
            path = host.store.get_store_path(alias)
            assert path is not None
            record = read_installed_card(path)
            assert record is not None and verify_installed_card(
                path, record, verify_hashes=True
            )
            assert (
                record.preservation
                and record.preservation.manifest_id == kit_job.expected_manifest_id
            )
        generation = store_path / ".recoveries" / job_id
        assert (generation / "preservation" / "envelope.json").is_file()
        assert (generation / "preservation" / "owner-trust.json").is_file()
        assert len(discover_installed_cards([store_path])) == len(review.members)
        assert not list(generation.rglob("*.pem"))
        import skulk.shared.constants as constants
        from skulk.download.download_utils import (
            build_companion_model_path,
            build_sidecar_path,
            installed_artifact_in_path,
            model_companions_present_on_disk,
        )
        from skulk.shared.models.model_cards import ModelId

        monkeypatch.setattr(constants, "SKULK_MODELS_PATH", (store_path,))
        primary = host.store.get_entry("example/base")
        assert primary is not None and primary.installed_card is not None
        recovered_card = primary.installed_card.model_card
        assert installed_artifact_in_path(recovered_card) == host.store.get_store_path(
            "example/base"
        )
        assert model_companions_present_on_disk(recovered_card)
        if role in {"assistant", "vision_weights"}:
            assert build_companion_model_path(
                ModelId("example/companion"), "b" * 40
            ) == host.store.get_store_path("example/companion")
        if role == "mtp_sidecar":
            assert (
                build_sidecar_path(
                    ModelId("example/companion"), "mtp.safetensors", "b" * 40
                )
                == store_path
                / ".recoveries"
                / job_id
                / "model-store/example--companion/mtp.safetensors"
            )
        assert (receiver / "cards" / "example--base.toml").is_file()
    finally:
        await host.close()
    restarted = ModelStore(store_path)
    assert len(restarted.list_models()) == len(review.members)


@pytest.mark.asyncio
async def test_upload_resume_limits_and_core_restart(tmp_path: Path) -> None:
    store = tmp_path / "store"
    store.mkdir()
    host = PreservationHost(ModelStore(store), tmp_path / "private")
    upload = host.uploads.create(10)
    uid = str(upload["upload_id"])
    assert host.uploads.append(uid, 0, b"1234")["offset"] == 4
    assert host.uploads.append(uid, 0, b"1234")["offset"] == 4
    with pytest.raises(ValueError, match="replay_changed"):
        host.uploads.append(uid, 0, b"4321")
    with pytest.raises(ValueError, match="offset_changed"):
        host.uploads.append(uid, 7, b"1")
    with pytest.raises(ValueError):
        host.uploads.append(uid, 4, b"1234567")
    await host.close()
    host = PreservationHost(ModelStore(store), tmp_path / "private")
    try:
        assert host.uploads.status(uid)["offset"] == 4
        assert host.uploads.append(uid, 4, b"567890")["complete"] is True
        with pytest.raises(ValueError):
            await host.plan(
                PreservationPlanRequest(
                    kind="recover", location_id="kit", upload_id=uid
                )
            )
        assert not host.store.list_models()
    finally:
        await host.close()


@pytest.mark.asyncio
async def test_removed_closure_consent_and_crash_reconciliation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from foxlight_model_preservation.kits import export_kit
    from foxlight_model_preservation.signing import load_key

    from skulk.store.model_store import read_reconciliation_tombstones
    from skulk.store.preservation_export import preserve_installed
    from skulk.store.preservation_journal import PreservationHostError

    installed = make_preservation_installation(tmp_path, "assistant")
    identity = preserve_installed(
        installed.primary, installed.archive, (load_key(installed.key, None),)
    )
    kit = tmp_path / "complete.foxkit"
    export_kit(installed.archive, identity, installed.trust, kit)
    store_path = tmp_path / "receiver-store"
    shutil.copytree(installed.primary.parent, store_path)
    store = ModelStore(store_path)
    assert await store.delete_model_serialized("example/base")
    host = PreservationHost(
        store, tmp_path / "receiver-private", custom_cards=tmp_path / "receiver-cards"
    )
    value = host.uploads.create(kit.stat().st_size)
    upload_id = str(value["upload_id"])
    host.uploads.append(upload_id, 0, kit.read_bytes())
    selection = PreservationPlanRequest(
        kind="recover", location_id="kit", upload_id=upload_id, historical=True
    )
    try:
        plan = await host.plan(selection)
        assert plan.deleted_aliases == ("example/base",)
        assert plan.existing_companions == ("example/companion",)
        with pytest.raises(
            PreservationHostError, match="removed_models_need_confirmation"
        ):
            host.start(
                PreservationStart(
                    plan_id=plan.plan_id,
                    operation_id="not-approved",
                    accept_owner_capture=True,
                )
            )

        def crash_after_index() -> None:
            raise OSError("simulated interruption after the whole index commit")

        monkeypatch.setattr(store, "_reconcile_recovery_deletions", crash_after_index)
        job = host.start(
            PreservationStart(
                plan_id=plan.plan_id,
                operation_id="approved",
                accept_owner_capture=True,
                restore_deleted=True,
            )
        )
        await wait_job(host, job.job_id)
        assert host.journal.job(job.job_id).state == "interrupted"
        assert len(store.list_models()) == 2
        assert read_reconciliation_tombstones(store_path) == frozenset({"example/base"})
        assert not (tmp_path / "receiver-cards/example--base.toml").exists()
    finally:
        await host.close()
    restarted = ModelStore(store_path)
    assert not read_reconciliation_tombstones(store_path)
    assert len(restarted.list_models()) == 2
    host = PreservationHost(
        restarted,
        tmp_path / "receiver-private",
        custom_cards=tmp_path / "receiver-cards",
    )
    try:
        # This finishes an already committed exact closure after fresh review;
        # it does not replay work or resurrect a later deletion automatically.
        review = await host.plan(selection)
        assert not review.deleted_aliases
        retried = host.start(
            PreservationStart(
                plan_id=review.plan_id,
                operation_id="finish-admission",
                accept_owner_capture=True,
            )
        )
        await wait_job(host, retried.job_id)
        assert host.journal.job(retried.job_id).state == "succeeded"
        assert (tmp_path / "receiver-cards/example--base.toml").is_file()
    finally:
        await host.close()
    assert await restarted.delete_model_serialized("example/base")
    again = ModelStore(store_path)
    assert "example/base" in read_reconciliation_tombstones(store_path)
    assert again.get_store_path("example/base") is None
