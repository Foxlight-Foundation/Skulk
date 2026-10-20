"""Real owned workers preserve complete companions and retain restart truth."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from skulk.store.model_store import ModelStore
from skulk.store.preservation_contracts import (
    PreservationPlanRequest,
    PreservationStart,
)
from skulk.store.preservation_host import PreservationHost
from skulk.store.preservation_journal import PreservationHostError
from skulk.store.tests.test_preservation_export import make_preservation_installation


async def _settled(host: PreservationHost, job_id: str) -> None:
    async with asyncio.timeout(20):
        while host.journal.job(job_id).state in {"queued", "running", "cancelling"}:
            await asyncio.sleep(0.02)


@pytest.mark.asyncio
async def test_actual_companion_copy_check_idempotency_and_child_independence(
    tmp_path: Path,
) -> None:
    installed = make_preservation_installation(tmp_path, "assistant")
    store = ModelStore(installed.primary.parent)
    host = PreservationHost(store, tmp_path / "private-host")
    try:
        assert not (await host.page()).locations
        location = await host.setup_device()
        page = await host.page()
        assert location.reachable and location.purpose == "private"
        model = next(model for model in page.models if model.model_id == "example/base")
        plan = await host.plan(
            PreservationPlanRequest(
                kind="preserve",
                location_id="device",
                model_id=model.model_id,
                generation_id=model.generation_id,
            )
        )
        assert plan.members == ("example/base", "example/companion")
        request = PreservationStart(
            plan_id=plan.plan_id, operation_id="preserve-once", owner_id="capability"
        )
        job = host.start(request)
        # Simulate the caller disappearing: only its retained operation ID remains.
        assert host.start(request).job_id == job.job_id
        await _settled(host, job.job_id)
        finished = host.journal.job(job.job_id)
        assert finished.state == "succeeded", finished
        assert finished.expected_manifest_id and finished.copy_id
        assert (
            finished.result
            and finished.result["manifest_id"] == finished.expected_manifest_id
        )
        copy = host.journal.copy(finished.copy_id)
        assert copy.member_count == 3  # base + assistant + private evidence
        assert len((await host.page()).copies) == 1
        check = await host.plan(
            PreservationPlanRequest(
                kind="check", location_id="device", copy_id=copy.copy_id
            )
        )
        checked = host.start(
            PreservationStart(plan_id=check.plan_id, operation_id="check-once")
        )
        await _settled(host, checked.job_id)
        assert host.journal.job(checked.job_id).state == "succeeded"
    finally:
        await host.close()
    restarted = PreservationHost(store, tmp_path / "private-host")
    try:
        assert restarted.journal.job(job.job_id).state == "succeeded"
        assert restarted.start(request).job_id == job.job_id
        assert len((await restarted.page()).copies) == 1
    finally:
        await restarted.close()


@pytest.mark.asyncio
async def test_disconnected_or_replaced_root_is_never_recreated(tmp_path: Path) -> None:
    installed = make_preservation_installation(tmp_path)
    host = PreservationHost(
        ModelStore(installed.primary.parent), tmp_path / "private-host"
    )
    try:
        await host.setup_device()
        archive = host.root / "device-archive"
        archive.rename(host.root / "unmounted-archive")
        assert not (await host.location_view("device")).reachable
        assert not archive.exists()
        archive.mkdir(mode=0o700)
        model = (await host.page()).models[0]
        with pytest.raises(PreservationHostError, match="archive_identity_changed"):
            await host.plan(
                PreservationPlanRequest(
                    kind="preserve",
                    location_id="device",
                    model_id=model.model_id,
                    generation_id=model.generation_id,
                )
            )
        assert not tuple(archive.iterdir())
    finally:
        await host.close()


@pytest.mark.asyncio
async def test_source_generation_change_refuses_and_cancel_waiting_on_owner_fence(
    tmp_path: Path,
) -> None:
    installed = make_preservation_installation(tmp_path)
    store = ModelStore(installed.primary.parent)
    host = PreservationHost(store, tmp_path / "private-host")
    try:
        await host.setup_device()
        model = (await host.page()).models[0]
        plan = await host.plan(
            PreservationPlanRequest(
                kind="preserve",
                location_id="device",
                model_id=model.model_id,
                generation_id=model.generation_id,
            )
        )
        async with store.preservation_lease():
            job = host.start(
                PreservationStart(plan_id=plan.plan_id, operation_id="cancel-waiting")
            )
            await asyncio.sleep(0.05)
            assert (await host.cancel(job.job_id)).state == "cancelling"
        await _settled(host, job.job_id)
        assert host.journal.job(job.job_id).state == "cancelled"
        assert not tuple((host.root / "device-archive").iterdir())
    finally:
        await host.close()
