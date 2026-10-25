"""Actual operator API, copy evidence, persistent ownership and explicit deletion."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from foxlight_model_preservation.kits import export_kit
from foxlight_model_preservation.mirror_state import MirrorLedger
from foxlight_model_preservation.storage import manifest_path, object_path, verify_store

from skulk.api.main import API
from skulk.api.preservation import create_preservation_router
from skulk.store.mirror_contracts import (
    MirrorJob,
    MirrorJobsPage,
    MirrorPage,
    MirrorRequest,
)
from skulk.store.preservation_host import PreservationHost
from skulk.store.tests.test_mirror_worker import archive, configuration
from skulk.store.tests.test_preservation_sharing import empty_host, import_copy


async def settled(host: PreservationHost, identity: str) -> MirrorJob:
    """Wait for the owned real child, without simulating a worker receipt."""
    async with asyncio.timeout(30):
        while True:
            job = host.mirrors.job(identity)
            if job.state not in {"queued", "running", "cancelling"}:
                return job
            await asyncio.sleep(0.025)


async def ready(host: PreservationHost, identity: str) -> MirrorJob:
    """Wait for actual controller ownership/listening, not historical metadata."""
    async with asyncio.timeout(30):
        while True:
            job = host.mirrors.job(identity)
            assert job.state in {"queued", "running"}, job
            if job.activity and job.activity.running and job.activity.http_port:
                return job
            await asyncio.sleep(0.025)


@pytest.mark.asyncio
async def test_mirror_operator_journey_preserves_companions_and_only_deletes_selected_copy(
    tmp_path: Path,
) -> None:
    host = empty_host(tmp_path)

    async def provider() -> PreservationHost:
        return host

    app = FastAPI()
    app.include_router(
        create_preservation_router(provider, API.require_preservation_operator)
    )
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 1000))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://localhost"
    ) as client:

        async def submit(decision: MirrorRequest) -> MirrorJob:
            response = await client.post(
                "/preservation/mirror/jobs", json=decision.model_dump(mode="json")
            )
            assert response.status_code == 202, response.text
            return MirrorJob.model_validate_json(response.text)

        async def page() -> MirrorPage:
            response = await client.get("/preservation/mirror")
            assert response.status_code == 200, response.text
            assert (
                str(tmp_path) not in response.text
                and "capture.pem" not in response.text
            )
            return MirrorPage.model_validate_json(response.text)

        try:
            assert not (await page()).configured
            config = MirrorRequest(
                owner_id="library",
                operation_id="configure",
                kind="configure",
                configuration=configuration(),
            )
            configured = await settled(host, (await submit(config)).job_id)
            assert configured.state == "succeeded", configured
            observed = await page()
            assert (
                observed.configured
                and observed.paused
                and observed.controller_job_id is None
            )
            assert not MirrorLedger(host.root / "managed-mirror").running()
            original, identity, trust = archive(tmp_path)
            expected_wrapper = (original / manifest_path(identity)).read_bytes()
            kit = tmp_path / "complete.foxkit"
            export_kit(original, identity, trust, kit)
            copy = await import_copy(host, kit.read_bytes())
            retain = MirrorRequest(
                owner_id="library",
                operation_id="retain",
                kind="retain",
                policy_id=observed.policy_id,
                copy_id=copy.copy_id,
                manifest_id=identity,
                confirm_redistribution=True,
            )
            retained = await settled(host, (await submit(retain)).job_id)
            assert retained.state == "succeeded", retained
            assert (await submit(retain)).job_id == retained.job_id
            observed = await page()
            assert (
                observed.model_count == 1 and observed.models[0].manifest_id == identity
            )
            # Remove the upstream files and make the original Core archive
            # unavailable. The mirror must need neither source for its own run.
            shutil.rmtree(original)
            shutil.rmtree(tmp_path / "model")
            shutil.rmtree(tmp_path / "tokenizer")
            value = host.journal.setting("location:" + copy.location_id)
            assert isinstance(value, dict) and isinstance(value["path"], str)
            source_path = Path(value["path"])
            moved = source_path.with_name(source_path.name + "-offline")
            source_path.rename(moved)
            source_path.mkdir(mode=0o700)
            run = MirrorRequest(
                owner_id="library",
                operation_id="run",
                kind="run",
                policy_id=observed.policy_id,
                public_participation=True,
                seconds=60,
            )
            running = await ready(host, (await submit(run)).job_id)
            assert (
                running.activity is not None and running.activity.http_port is not None
            )
            assert (await page()).controller_job_id == running.job_id
            manifest = verify_store(host.root / "managed-mirror/store", identity, trust)
            assert len(manifest.members) == 2
            async with httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{running.activity.http_port}",
                trust_env=False,
            ) as recipient:
                async with asyncio.timeout(15):
                    while True:
                        response = await recipient.get("/" + manifest_path(identity))
                        if response.status_code == 200:
                            break
                        await asyncio.sleep(0.025)
                assert response.content == expected_wrapper
                for member in manifest.members:
                    assert member.retained_card == {
                        "future": {"opaque": [1, True, "unchanged"]}
                    }
                    for file in member.files:
                        content = await recipient.get("/" + object_path(file.sha256))
                        assert (
                            content.content
                            == (
                                host.root
                                / "managed-mirror/store"
                                / object_path(file.sha256)
                            ).read_bytes()
                        )
            stopped = await client.delete("/preservation/mirror/jobs/" + running.job_id)
            assert stopped.status_code == 200, stopped.text
            assert stopped.json()["state"] == "cancelled"
            assert (await page()).activity is None and (await page()).paused
            assert not MirrorLedger(host.root / "managed-mirror").running()
            assert (
                await submit(run)
            ).state == "cancelled"  # Permanent replay never starts again.
            second = await ready(
                host,
                (await submit(run.model_copy(update={"operation_id": "again"}))).job_id,
            )
            await client.delete("/preservation/mirror/jobs/" + running.job_id)
            assert (await page()).controller_job_id == second.job_id
            assert MirrorLedger(
                host.root / "managed-mirror"
            ).running()  # Old cancel is harmless.
            await client.delete("/preservation/mirror/jobs/" + second.job_id)
            for action in ("pin", "block"):
                decision = MirrorRequest.model_validate_json(
                    json.dumps(
                        {
                            "owner_id": "library",
                            "operation_id": action,
                            "kind": "model",
                            "policy_id": (await page()).policy_id,
                            "manifest_id": identity,
                            "action": action,
                        }
                    )
                )
                assert (
                    await settled(host, (await submit(decision)).job_id)
                ).state == "succeeded"
            prune = MirrorRequest(
                owner_id="library",
                operation_id="pinned-delete",
                kind="prune",
                policy_id=(await page()).policy_id,
                manifest_id=identity,
                confirm_delete_manifest=identity,
            )
            assert (await settled(host, (await submit(prune)).job_id)).state == "failed"
            assert (
                verify_store(host.root / "managed-mirror/store", identity, trust)
                == manifest
            )
            unpin = MirrorRequest(
                owner_id="library",
                operation_id="unpin",
                kind="model",
                policy_id=(await page()).policy_id,
                manifest_id=identity,
                action="unpin",
            )
            assert (
                await settled(host, (await submit(unpin)).job_id)
            ).state == "succeeded"
            deleted = await settled(
                host,
                (
                    await submit(prune.model_copy(update={"operation_id": "delete"}))
                ).job_id,
            )
            assert deleted.state == "succeeded", deleted
            assert (await page()).models[0].state == "removed"
            assert (moved / manifest_path(identity)).read_bytes() == expected_wrapper
            assert not (
                host.root / "managed-mirror/store" / manifest_path(identity)
            ).exists()
            history = await client.get("/preservation/mirror/jobs")
            assert history.status_code == 200 and any(
                job.kind == "run"
                for job in MirrorJobsPage.model_validate_json(history.text).jobs
            )
        finally:
            await host.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/preservation/mirror", "/preservation/mirror/jobs"])
async def test_mirror_reads_authorize_before_host_or_worker_creation(path: str) -> None:
    async def provider() -> PreservationHost:
        raise AssertionError("unauthorized requests must not construct the owner")

    app = FastAPI()
    app.include_router(
        create_preservation_router(provider, API.require_preservation_operator)
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=("8.8.8.8", 1000)),
        base_url="http://localhost",
    ) as client:
        assert (await client.get(path)).status_code == 403
