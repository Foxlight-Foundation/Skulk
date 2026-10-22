"""Stalled mounts leave the event loop usable without abandoning owned effects."""

from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path

import pytest

import skulk.store.preservation_host as host_module
from skulk.store.preservation_contracts import FolderConfirmation, FolderReviewRequest
from skulk.store.preservation_host import PreservationHost
from skulk.store.preservation_io import PreservationIO
from skulk.store.preservation_journal import PreservationHostError
from skulk.store.tests.test_preservation_folders import owner


async def begun(event: threading.Event) -> None:
    """Wait for a real worker to enter its simulated stalled filesystem call."""
    async with asyncio.timeout(2):
        while not event.is_set():
            await asyncio.sleep(0.01)


async def test_stalled_probe_is_bounded_and_close_retains_journal_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = owner(tmp_path / "source")
    archive = tmp_path / "archive"
    review = await host.review_folder(
        FolderReviewRequest(path=str(archive), label="NAS")
    )
    location = await host.setup_folder(
        FolderConfirmation(review_id=review.review_id, confirmed=True)
    )
    inode = archive.stat().st_ino
    started, release = threading.Event(), threading.Event()
    original = host_module.shutil.disk_usage
    calls = 0
    closing: asyncio.Task[None] | None = None

    def delayed(path: str | int) -> tuple[int, int, int]:
        nonlocal calls
        if isinstance(path, int) and os.fstat(path).st_ino == inode:
            calls += 1
            started.set()
            release.wait(10)
        return original(path)

    monkeypatch.setattr(host_module.shutil, "disk_usage", delayed)
    try:
        page_task = asyncio.create_task(host.page())
        await begun(started)
        await asyncio.wait_for(asyncio.sleep(0.01), 0.5)
        page = await asyncio.wait_for(page_task, 2)
        assert page.default_location_id == location.location_id
        assert not next(
            item for item in page.locations if item.location_id == location.location_id
        ).reachable
        assert next(
            item for item in page.locations if item.location_id == "device"
        ).reachable
        for _ in range(3):
            later = await asyncio.wait_for(host.page(), 2)
            assert next(
                item for item in later.locations if item.location_id == "device"
            ).reachable
        assert calls == 1  # Repeated reads reuse the one outstanding NAS probe.
        closing = asyncio.create_task(host.close())
        await asyncio.sleep(0.03)
        assert not closing.done()
        with pytest.raises(
            PreservationHostError, match="preservation_host_already_running"
        ):
            PreservationHost(host.store, host.root)
        release.set()
        await asyncio.wait_for(closing, 2)
    finally:
        release.set()
        if closing is not None:
            await closing
        await host.close()


async def test_cancelled_setup_retains_ownership_until_actual_write_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = owner(tmp_path / "source")
    archive = tmp_path / "archive"
    review = await host.review_folder(
        FolderReviewRequest(path=str(archive), label="NAS")
    )
    request = FolderConfirmation(review_id=review.review_id, confirmed=True)
    started, release = threading.Event(), threading.Event()
    original = host_module.atomic_bytes

    def delayed(descriptor: int, path: str, content: bytes) -> None:
        started.set()
        release.wait(10)
        original(descriptor, path, content)

    monkeypatch.setattr(host_module, "atomic_bytes", delayed)
    task = asyncio.create_task(host.setup_folder(request))
    later: asyncio.Task[host_module.FolderReview] | None = None
    try:
        await begun(started)
        task.cancel()
        later = asyncio.create_task(
            host.review_folder(FolderReviewRequest(path=str(archive), label="NAS"))
        )
        await asyncio.sleep(0.03)
        assert not task.done() and not later.done()
        with pytest.raises(
            PreservationHostError, match="preservation_host_already_running"
        ):
            PreservationHost(host.store, host.root)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await later
        # The original response can be lost, but its exact committed enrollment
        # remains observable and no filesystem effect continues ownerless.
        page = await host.page()
        assert len(page.locations) == 2 and not page.jobs
        assert all(location.reachable for location in page.locations)
        assert (archive / "archive-trust.json").is_file()
    finally:
        release.set()
        await asyncio.gather(task, *([later] if later else []), return_exceptions=True)
        await host.close()


async def test_cancelled_open_disposes_the_actual_descriptor(tmp_path: Path) -> None:
    storage = PreservationIO()
    started, release = threading.Event(), threading.Event()
    opened: list[int] = []

    def operation() -> int:
        descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
        opened.append(descriptor)
        started.set()
        release.wait(10)
        return descriptor

    task = asyncio.create_task(storage.run(operation, discard=os.close))
    try:
        await begun(started)
        task.cancel()
        await asyncio.sleep(0.03)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(OSError):
            os.fstat(opened[0])
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await storage.close()


async def test_shutdown_keeps_pool_until_a_borrowed_directory_is_closed(
    tmp_path: Path,
) -> None:
    storage = PreservationIO()
    closing: asyncio.Task[None] | None = None
    try:
        async with storage.directory(
            lambda: os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
        ) as descriptor:
            closing = asyncio.create_task(storage.close())
            await asyncio.sleep(0.03)
            assert not closing.done()
            assert os.fstat(descriptor).st_ino == tmp_path.stat().st_ino
        await asyncio.wait_for(closing, 2)
        with pytest.raises(OSError):
            os.fstat(descriptor)
    finally:
        if closing is not None:
            await closing
        await storage.close()


async def test_multiple_stale_probes_leave_effect_and_healthy_probe_capacity() -> None:
    storage = PreservationIO()
    started, release = threading.Event(), threading.Event()
    calls = 0

    def stale() -> str:
        nonlocal calls
        calls += 1
        started.set()
        release.wait(10)
        return "late"

    try:
        assert (
            await storage.probe("local", lambda: "verified-local", "unknown")
            == "verified-local"
        )
        assert await storage.probe("nas-one", stale, "unknown") == "unknown"
        await begun(started)
        assert await storage.probe("nas-two", stale, "unknown") == "unknown"
        assert calls == 2
        assert (
            await storage.probe("local", lambda: "fresh-local", "unknown")
            == "fresh-local"
        )
        assert (
            await asyncio.wait_for(storage.run(lambda: "real-operation-finished"), 1)
            == "real-operation-finished"
        )
    finally:
        release.set()
        await storage.close()


async def test_cancelled_directory_cleanup_keeps_lease_while_waiting_for_slots(
    tmp_path: Path,
) -> None:
    storage = PreservationIO()
    entered, leave = asyncio.Event(), asyncio.Event()
    started_one, started_two, release = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    descriptors: list[int] = []

    async def borrowed() -> None:
        async with storage.directory(
            lambda: os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
        ) as descriptor:
            descriptors.append(descriptor)
            entered.set()
            await leave.wait()

    def occupied(started: threading.Event) -> None:
        started.set()
        release.wait(10)

    task = asyncio.create_task(borrowed())
    await asyncio.wait_for(entered.wait(), 2)
    first = asyncio.create_task(storage.run(lambda: occupied(started_one)))
    second = asyncio.create_task(storage.run(lambda: occupied(started_two)))
    closing: asyncio.Task[None] | None = None
    try:
        await begun(started_one)
        await begun(started_two)
        leave.set()
        await asyncio.sleep(0.03)
        task.cancel()
        closing = asyncio.create_task(storage.close())
        await asyncio.sleep(0.03)
        assert not task.done() and not closing.done()
        assert os.fstat(descriptors[0]).st_ino == tmp_path.stat().st_ino
        release.set()
        await asyncio.gather(first, second)
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(closing, 2)
        with pytest.raises(OSError):
            os.fstat(descriptors[0])
    finally:
        leave.set()
        release.set()
        await asyncio.gather(task, first, second, return_exceptions=True)
        if closing is not None:
            await closing
        await storage.close()
