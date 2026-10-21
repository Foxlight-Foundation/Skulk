"""Publication-boundary recovery cleanup and explicit fresh-consent retries."""

from __future__ import annotations

import shutil
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import httpx
import pytest
from fastapi import FastAPI
from foxlight_model_preservation.kits import export_kit
from foxlight_model_preservation.signing import load_key

from skulk.api.preservation import create_preservation_router
from skulk.store.model_store import ModelStore, StoreModelEntry, StoreRegistryIndex
from skulk.store.preservation_contracts import (
    PreservationPlanRequest,
    PreservationStart,
)
from skulk.store.preservation_export import preserve_installed
from skulk.store.preservation_host import PreservationHost
from skulk.store.preservation_journal import PreservationHostError
from skulk.store.tests.test_library_recovery import wait_job
from skulk.store.tests.test_preservation_export import make_preservation_installation


@dataclass
class InterruptedRecovery:
    host: PreservationHost
    job_id: str
    selection: PreservationPlanRequest
    request_bytes: bytes


@pytest.fixture
async def interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[InterruptedRecovery]:
    from skulk.shared.models import model_cards

    for name in (
        "_card_cache",
        "_installed_card_cache",
        "_installed_card_mutation_versions",
        "_installed_current_registry_ids",
    ):
        monkeypatch.setattr(model_cards, name, {})
    monkeypatch.setattr(model_cards, "_installed_card_cache_version", 0)
    original = tmp_path / "original"
    original.mkdir()
    installed = make_preservation_installation(original, "assistant")
    manifest = preserve_installed(
        installed.primary, installed.archive, (load_key(installed.key, None),)
    )
    kit = tmp_path / "saved.foxkit"
    export_kit(installed.archive, manifest, installed.trust, kit)
    root = tmp_path / "receiver"
    root.mkdir()
    host = PreservationHost(
        ModelStore(root), tmp_path / "private", custom_cards=tmp_path / "cards"
    )
    upload_id = str(host.uploads.create(kit.stat().st_size)["upload_id"])
    host.uploads.append(upload_id, 0, kit.read_bytes())
    selection = PreservationPlanRequest(
        kind="recover", location_id="kit", upload_id=upload_id, historical=True
    )
    plan = await host.plan(selection)
    request_bytes = b""

    def fail_commit(
        entries: dict[str, StoreModelEntry], *, durable: bool = False
    ) -> None:
        nonlocal request_bytes
        if not (root / ".recoveries").exists():
            (root / "registry.json").write_text(
                StoreRegistryIndex(entries=entries).model_dump_json()
            )
            return
        request_bytes = next(
            (root / ".recovery-work").glob("*/request.json")
        ).read_bytes()
        raise OSError("simulated failure before index replacement")

    with monkeypatch.context() as fault:
        fault.setattr(host.store, "_write_registry", fail_commit)
        job = host.start(
            PreservationStart(
                plan_id=plan.plan_id, operation_id="first", accept_owner_capture=True
            )
        )
        await wait_job(host, job.job_id)
    assert host.journal.job(job.job_id).state == "interrupted"
    assert not host.store.list_models()
    assert request_bytes
    yield InterruptedRecovery(host, job.job_id, selection, request_bytes)
    await host.close()


async def test_explicit_cleanup_unlocks_capacity_and_requires_new_consent(
    interrupted: InterruptedRecovery, monkeypatch: pytest.MonkeyPatch
) -> None:
    host, job_id = interrupted.host, interrupted.job_id
    generation = host.store.store_path / ".recoveries" / job_id
    usage = shutil.disk_usage(host.store.store_path)

    class Capacity(NamedTuple):
        total: int
        used: int
        free: int

    def available(_: str | Path) -> Capacity:
        return Capacity(
            usage.total, usage.used, 0 if generation.exists() else usage.free
        )

    with monkeypatch.context() as capacity:
        capacity.setattr(shutil, "disk_usage", available)
        with pytest.raises(PreservationHostError, match="insufficient_space"):
            await host.plan(interrupted.selection)
        result = await host.reclaim_recovery(job_id)
        assert result.generation_removed and not result.indexed_generation_retained
        review = await host.plan(interrupted.selection)
    assert not generation.exists()
    with pytest.raises(PreservationHostError, match="owner_capture"):
        host.start(PreservationStart(plan_id=review.plan_id, operation_id="no-consent"))
    job = host.start(
        PreservationStart(
            plan_id=review.plan_id, operation_id="retry", accept_owner_capture=True
        )
    )
    await wait_job(host, job.job_id)
    assert host.journal.job(job.job_id).state == "succeeded"
    assert len(host.store.list_models()) == 2
    assert not (await host.reclaim_recovery(job_id)).generation_removed
    assert len(host.store.list_models()) == 2


@pytest.mark.parametrize(
    "index", [None, b"{", b"[]", b'{"schema_version":99,"entries":{}}']
)
async def test_unreadable_index_refuses_destructive_cleanup(
    interrupted: InterruptedRecovery, index: bytes | None
) -> None:
    host = interrupted.host
    path = host.store.store_path / ".recoveries" / interrupted.job_id
    registry = host.store.store_path / "registry.json"
    if index is None:
        registry.unlink()
    else:
        registry.write_bytes(index)
    with pytest.raises(ValueError):
        await host.reclaim_recovery(interrupted.job_id)
    assert (path / "preservation/envelope.json").is_file()


async def test_failed_deletion_resumes_from_local_inode_receipt_after_restart(
    interrupted: InterruptedRecovery, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = interrupted.host
    generation = host.store.store_path / ".recoveries" / interrupted.job_id
    work = host.store.store_path / ".recovery-work" / interrupted.job_id
    work.mkdir(mode=0o700)
    (work / "request.json").write_bytes(interrupted.request_bytes)
    (work / "incomplete-object").write_bytes(b"incomplete worker bytes")
    original_rmtree = shutil.rmtree

    class FailedDeletion:
        # Preserve the capability exposed by the real POSIX implementation.
        avoids_symlink_attacks = True

        def __call__(self, path: Path) -> None:
            (path / "preservation/envelope.json").unlink()
            raise OSError("simulated crash during temporary-file deletion")

    with monkeypatch.context() as fault:
        fault.setattr(shutil, "rmtree", FailedDeletion())
        with pytest.raises(OSError):
            await host.reclaim_recovery(interrupted.job_id)
    assert not generation.exists()
    trash = (
        host.store.store_path / ".recovery-trash" / (interrupted.job_id + "-generation")
    )
    assert trash.exists()
    await host.close()
    restarted = PreservationHost(host.store, host.root, custom_cards=host.custom_cards)
    try:
        result = await restarted.reclaim_recovery(interrupted.job_id)
        assert result.generation_removed and result.temporary_removed
        assert not trash.exists() and not work.exists()
        assert not (
            await restarted.reclaim_recovery(interrupted.job_id)
        ).generation_removed
    finally:
        await restarted.close()
    assert shutil.rmtree is original_rmtree


async def test_indexed_generation_is_retained_and_api_is_operator_gated(
    interrupted: InterruptedRecovery,
) -> None:
    from skulk.store.preservation_live import recovered_index

    host, job_id = interrupted.host, interrupted.job_id
    generation = host.store.store_path / ".recoveries" / job_id
    index = recovered_index(generation)
    (host.store.store_path / "registry.json").write_text(
        StoreRegistryIndex(
            entries={
                alias: entry.model_copy(
                    update={
                        "store_path": str(
                            Path(".recoveries")
                            / job_id
                            / "model-store"
                            / entry.store_path
                        )
                    }
                )
                for alias, entry in index.entries.items()
            }
        ).model_dump_json()
    )
    result = await host.reclaim_recovery(job_id)
    assert result.indexed_generation_retained and not result.generation_removed
    assert generation.is_dir() and len(host.store.list_models()) == 2

    async def provider() -> PreservationHost:
        return host

    app = FastAPI()
    app.include_router(create_preservation_router(provider, lambda _: None))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://local"
    ) as client:
        reply = await client.post(f"/preservation/jobs/{job_id}/reclaim")
        assert reply.status_code == 200
        assert reply.json()["indexed_generation_retained"] is True


@pytest.mark.parametrize("part", ["generation", "parent", "metadata", "index"])
async def test_cleanup_refuses_links_without_touching_other_storage(
    interrupted: InterruptedRecovery, tmp_path: Path, part: str
) -> None:
    host = interrupted.host
    generation = host.store.store_path / ".recoveries" / interrupted.job_id
    if part == "generation":
        selected = generation
    elif part == "parent":
        selected = generation.parent
    elif part == "metadata":
        selected = generation / "preservation/envelope.json"
    else:
        selected = host.store.store_path / "registry.json"
        selected.write_text('{"schema_version":1,"entries":{}}')
    outside = tmp_path / ("external-" + part)
    selected.rename(outside)
    selected.symlink_to(outside, target_is_directory=outside.is_dir())
    with pytest.raises((ValueError, OSError)):
        await host.reclaim_recovery(interrupted.job_id)
    assert outside.exists()
    if outside.is_dir():
        assert list(outside.iterdir())


async def test_cleanup_waits_for_live_store_lease(
    interrupted: InterruptedRecovery,
) -> None:
    import asyncio

    host = interrupted.host
    async with host.store.preservation_lease():
        cleanup = asyncio.create_task(host.reclaim_recovery(interrupted.job_id))
        await asyncio.sleep(0.05)
        assert not cleanup.done()
        assert (host.store.store_path / ".recoveries" / interrupted.job_id).is_dir()
    assert (await cleanup).generation_removed


async def test_disconnected_cleanup_keeps_lease_until_filesystem_work_stops(
    interrupted: InterruptedRecovery, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio
    import threading

    from foxlight_model_preservation.models import TrustProfile

    from skulk.store import preservation_reclaim
    from skulk.store.preservation_contracts import RecoveryCleanup
    from skulk.store.preservation_journal import PreservationJournal

    original = preservation_reclaim.reclaim_recovery_files
    entered, release = threading.Event(), threading.Event()

    def paused(
        store: ModelStore,
        journal: PreservationJournal,
        job_id: str,
        manifest_id: str,
        trust: TrustProfile,
    ) -> RecoveryCleanup:
        entered.set()
        if not release.wait(10):
            raise TimeoutError("test maintenance was not released")
        return original(store, journal, job_id, manifest_id, trust)

    monkeypatch.setattr(preservation_reclaim, "reclaim_recovery_files", paused)
    host = interrupted.host
    cleanup = asyncio.create_task(host.reclaim_recovery(interrupted.job_id))
    assert await asyncio.to_thread(entered.wait, 10)
    cleanup.cancel()
    await asyncio.sleep(0)
    assert not cleanup.done()
    cleanup.cancel()
    await asyncio.sleep(0)
    assert not cleanup.done()
    acquired = asyncio.Event()

    async def mutation() -> None:
        async with host.store.preservation_lease():
            acquired.set()

    other = asyncio.create_task(mutation())
    try:
        await asyncio.sleep(0.05)
        assert not acquired.is_set()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await cleanup
    await other
    assert acquired.is_set()
    assert host.journal.job(interrupted.job_id).state == "interrupted"
