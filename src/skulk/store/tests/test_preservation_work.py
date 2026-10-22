"""Stopped-work cleanup retains signed copies and survives interrupted deletion."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from threading import Event

import httpx
import pytest
from fastapi import FastAPI, HTTPException, Request
from foxlight_model_preservation.storage import verify_store

from skulk.api.preservation import create_preservation_router
from skulk.store import preservation_host, preservation_work
from skulk.store.model_store import ModelStore
from skulk.store.preservation_contracts import (
    PreservationJob,
    PreservationPlanRequest,
    PreservationStart,
    TemporaryWork,
)
from skulk.store.preservation_host import PreservationHost
from skulk.store.preservation_journal import PreservationHostError, PreservationJournal
from skulk.store.tests.test_preservation_sharing import (
    empty_host,
    import_copy,
    public_kit,
    ready,
    settled,
)


@dataclass
class Leftover:
    """A real sharing worker whose normal staging deletion was interrupted."""

    host: PreservationHost
    job_id: str
    work: Path
    request: bytes
    copy_id: str


@pytest.fixture
async def leftover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[Leftover]:
    publisher = tmp_path / "publisher"
    publisher.mkdir()
    content, _, _ = public_kit(publisher)
    host = empty_host(tmp_path)
    copy = await import_copy(host, content)
    plan = await host.plan(
        PreservationPlanRequest(
            kind="http_share", location_id=copy.location_id, copy_id=copy.copy_id
        )
    )
    job = host.start(
        PreservationStart(
            plan_id=plan.plan_id,
            operation_id="interrupted-cleanup",
            public_participation=True,
        )
    )
    await ready(host, job.job_id)
    work = host.root / "work" / job.job_id
    request = (work / "request.json").read_bytes()
    (work / "torrent").mkdir()
    (work / "torrent" / "weights.bin").write_bytes(
        b"kept only as temporary staging" * 100
    )
    original = preservation_host.shutil.rmtree

    def interrupted(path: str | Path, *, dir_fd: int | None = None) -> None:
        if Path(path) == work:
            raise OSError("simulated loss during final staging cleanup")
        original(path, dir_fd=dir_fd)

    with monkeypatch.context() as fault:
        fault.setattr(preservation_host.shutil, "rmtree", interrupted)
        await host.cancel(job.job_id)
        await settled(host, job.job_id)
        while True:
            try:
                await host.temporary_work(job.job_id)
                break
            except PreservationHostError as error:
                assert error.code == "temporary_job_not_stopped"
                await asyncio.sleep(0.01)

    assert work.is_dir() and not (work / "request.json").exists()
    try:
        yield Leftover(host, job.job_id, work, request, copy.copy_id)
    finally:
        await host.close()


async def test_inspect_and_clear_only_known_work_keeps_public_copy(
    leftover: Leftover,
) -> None:
    host = leftover.host
    copy = host.journal.copy(leftover.copy_id)
    location = host.root / "public-copies"
    kept = next(location.iterdir())
    before = {
        p.relative_to(kept): p.read_bytes() for p in kept.rglob("*") if p.is_file()
    }
    (host.root / "torrents").mkdir(exist_ok=True)
    artifact = host.root / "torrents" / (leftover.job_id + ".torrent")
    artifact.write_bytes(b"kept metainfo sentinel")
    page = await host.page()
    assert leftover.job_id in page.temporary_work_jobs
    report = await host.temporary_work(leftover.job_id)
    assert (
        report.found
        and report.logical_bytes > 0
        and report.files == 1
        and report.complete
        and not report.removed
    )
    result = await host.temporary_work(leftover.job_id, remove=True)
    assert result.removed and not leftover.work.exists()
    assert artifact.read_bytes() == b"kept metainfo sentinel"
    assert before == {
        p.relative_to(kept): p.read_bytes() for p in kept.rglob("*") if p.is_file()
    }
    verify_store(kept, copy.manifest_id, host.public_trust(copy.copy_id))
    assert not (await host.temporary_work(leftover.job_id, remove=True)).found
    assert leftover.job_id not in (await host.page()).temporary_work_jobs
    assert not host.store.list_models()


async def test_changed_inode_and_symlink_parent_refuse(leftover: Leftover) -> None:
    host = leftover.host
    original = leftover.work.with_name("retained-original")
    leftover.work.rename(original)
    leftover.work.mkdir()
    sentinel = leftover.work / "operator-data"
    sentinel.write_bytes(b"keep")
    with pytest.raises(PreservationHostError, match="temporary_work_identity_changed"):
        await host.temporary_work(leftover.job_id, remove=True)
    assert sentinel.read_bytes() == b"keep"
    moved = host.root / "work-held"
    (host.root / "work").rename(moved)
    (host.root / "work").symlink_to(moved, target_is_directory=True)
    with pytest.raises((OSError, PreservationHostError)):
        await host.temporary_work(leftover.job_id, remove=True)
    assert (moved / leftover.job_id / "operator-data").read_bytes() == b"keep"


async def test_legacy_request_is_needed_before_receipt(leftover: Leftover) -> None:
    host = leftover.host
    key = "temporary-work:" + leftover.job_id
    host.journal.set_setting(key, None)
    with pytest.raises(FileNotFoundError):
        await host.temporary_work(leftover.job_id, remove=True)
    (leftover.work / "request.json").write_bytes(leftover.request)
    assert (await host.temporary_work(leftover.job_id)).found
    assert host.journal.setting(key) is None  # Reads enroll nothing.
    assert (await host.temporary_work(leftover.job_id, remove=True)).removed
    assert host.journal.setting(key) is not None


async def test_partial_measure_and_interrupted_quarantine_retry(
    leftover: Leftover, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = leftover.host
    original = preservation_work.shutil.rmtree

    class InterruptedDelete:
        avoids_symlink_attacks = True

        def __call__(self, path: str, *, dir_fd: int | None = None) -> None:
            os.unlink(path + "/torrent/weights.bin", dir_fd=dir_fd)
            raise OSError("simulated interrupted deletion")

    with monkeypatch.context() as fault:
        fault.setattr(preservation_work, "MAX_MEASURE_ENTRIES", 0)
        report = await host.temporary_work(leftover.job_id)
        assert not report.complete
        fault.setattr(preservation_work.shutil, "rmtree", InterruptedDelete())
        with pytest.raises(OSError, match="interrupted deletion"):
            await host.temporary_work(leftover.job_id, remove=True)
    assert not leftover.work.exists()
    assert (host.root / "work-trash" / leftover.job_id).is_dir()
    await host.close()
    restarted = PreservationHost(ModelStore(host.store.store_path), host.root)
    try:
        assert (await restarted.temporary_work(leftover.job_id, remove=True)).removed
        assert restarted.journal.copy(leftover.copy_id)
    finally:
        await restarted.close()
    assert preservation_work.shutil.rmtree is original


async def test_active_jobs_and_unauthorized_requests_refuse(leftover: Leftover) -> None:
    host = leftover.host
    copy = host.journal.copy(leftover.copy_id)
    plan = await host.plan(
        PreservationPlanRequest(
            kind="http_share", location_id=copy.location_id, copy_id=copy.copy_id
        )
    )
    job = host.start(
        PreservationStart(
            plan_id=plan.plan_id, operation_id="active", public_participation=True
        )
    )
    await ready(host, job.job_id)
    try:
        with pytest.raises(PreservationHostError, match="temporary_job_not_stopped"):
            await host.temporary_work(job.job_id, remove=True)
        app = FastAPI()

        async def provider() -> PreservationHost:
            return host

        def authorize(request: Request) -> None:
            if request.headers.get("authorization") != "Bearer fixture":
                raise HTTPException(403)

        app.include_router(create_preservation_router(provider, authorize))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://localhost"
        ) as client:
            for suffix, method in [
                ("temporary-files", "GET"),
                ("reclaim-work", "POST"),
            ]:
                response = await client.request(
                    method, f"/preservation/jobs/{leftover.job_id}/{suffix}"
                )
                assert response.status_code == 403
            report = await client.get(
                f"/preservation/jobs/{leftover.job_id}/temporary-files",
                headers={"Authorization": "Bearer fixture"},
            )
            assert report.status_code == 200 and report.json()["found"]
        assert (await host.temporary_work(leftover.job_id, remove=True)).removed
        assert host.journal.job(job.job_id).sharing is not None
        assert host.journal.job(job.job_id).state == "running"
    finally:
        await host.cancel(job.job_id)


async def test_disconnected_cleanup_keeps_host_ownership(
    leftover: Leftover, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = leftover.host
    entered, release = Event(), Event()
    original = preservation_host.manage_temporary_work

    def paused(
        root: Path, journal: PreservationJournal, job: PreservationJob, *, remove: bool
    ) -> TemporaryWork:
        entered.set()
        assert release.wait(5)
        return original(root, journal, job, remove=remove)

    monkeypatch.setattr(preservation_host, "manage_temporary_work", paused)
    cleanup = asyncio.create_task(host.temporary_work(leftover.job_id, remove=True))
    while not entered.is_set():
        await asyncio.sleep(0.01)
    cleanup.cancel()
    await asyncio.sleep(0.05)
    pending_read = asyncio.create_task(host.temporary_work(leftover.job_id))
    await asyncio.sleep(0.05)
    assert not cleanup.done() and not pending_read.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await cleanup
    assert not (await pending_read).found
    assert not leftover.work.exists()


@pytest.mark.parametrize("quarantined", [False, True])
async def test_history_retirement_keeps_work_until_explicit_cleanup(
    leftover: Leftover, quarantined: bool
) -> None:
    host = leftover.host
    original = host.journal.job(leftover.job_id)
    receipt_key = "temporary-work:" + original.job_id
    if quarantined:
        trash = host.root / "work-trash"
        trash.mkdir(mode=0o700)
        leftover.work.rename(trash / original.job_id)
    else:
        # Legacy work also needs its row retained before custody enrollment.
        host.journal.set_setting(receipt_key, None)
        (leftover.work / "request.json").write_bytes(leftover.request)

    def complete(index: int) -> None:
        job, created = host.journal.reserve(
            PreservationStart(
                plan_id=original.plan_id,
                operation_id=f"history-{index}",
                public_participation=True,
            )
        )
        assert created
        host.journal.update_job(
            job.model_copy(
                update={
                    "state": "failed",
                    "stage": "failed",
                    "updated_at": original.updated_at + index + 1,
                }
            )
        )

    for index in range(260):
        complete(index)
    assert host.journal.job(original.job_id) == original
    assert original in host.journal.jobs(256)
    assert (await host.temporary_work(original.job_id)).found
    assert (await host.temporary_work(original.job_id, remove=True)).removed
    complete(260)
    with pytest.raises(PreservationHostError, match="job_not_found"):
        host.journal.job(original.job_id)
    assert host.journal.setting(receipt_key) is None
    with pytest.raises(PreservationHostError, match="operation_retired"):
        host.journal.reserve(
            PreservationStart(
                plan_id=original.plan_id,
                operation_id=original.operation_id,
                owner_id=original.owner_id,
                public_participation=True,
            )
        )
    assert host.journal.copy(leftover.copy_id)
