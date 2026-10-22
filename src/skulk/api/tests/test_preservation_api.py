"""Operator origin boundary and real durable host API behavior."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from skulk.api.main import API
from skulk.api.preservation import create_preservation_router
from skulk.store.model_store import ModelStore
from skulk.store.preservation_contracts import (
    KitUploadStatus,
    PreservationJob,
    PreservationPage,
    PreservationPlan,
)
from skulk.store.preservation_host import PreservationHost
from skulk.store.tests.test_preservation_export import make_preservation_installation


@pytest.mark.asyncio
async def test_operator_api_guided_setup_plan_and_reopened_job(tmp_path: Path) -> None:
    installed = make_preservation_installation(tmp_path, "assistant")
    host = PreservationHost(
        ModelStore(installed.primary.parent), tmp_path / "private-host"
    )

    async def provider() -> PreservationHost:
        return host

    app = FastAPI()
    app.include_router(
        create_preservation_router(provider, API.require_preservation_operator)
    )
    try:
        transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 1000))
        async with httpx.AsyncClient(
            transport=transport, base_url="http://localhost"
        ) as client:
            assert (await client.get("/preservation/library")).status_code == 200
            assert (
                await client.post("/preservation/setup/device", json={})
            ).status_code == 200
            page = PreservationPage.model_validate_json(
                (await client.get("/preservation/library")).text
            )
            model = page.models[0]
            planned = await client.post(
                "/preservation/plans",
                json={
                    "kind": "preserve",
                    "location_id": "device",
                    "model_id": model.model_id,
                    "generation_id": model.generation_id,
                },
            )
            assert planned.status_code == 200
            submitted = await client.post(
                "/preservation/jobs",
                json={
                    "plan_id": PreservationPlan.model_validate_json(
                        planned.text
                    ).plan_id,
                    "operation_id": "browser-submit",
                },
            )
            assert submitted.status_code == 202
            job_id = PreservationJob.model_validate_json(submitted.text).job_id
            assert (
                "capture.pem" not in submitted.text
                and str(tmp_path) not in submitted.text
            )
        # New page/client has no in-memory job runner or ownership to recover.
        async with httpx.AsyncClient(
            transport=transport, base_url="http://localhost"
        ) as reopened:
            async with asyncio.timeout(20):
                while True:
                    observed = await reopened.get("/preservation/jobs/" + job_id)
                    if observed.json()["state"] not in {
                        "queued",
                        "running",
                        "cancelling",
                    }:
                        break
                    await asyncio.sleep(0.02)
            assert observed.json()["state"] == "succeeded", observed.text
            assert (
                str(tmp_path) not in (await reopened.get("/preservation/library")).text
            )
    finally:
        await host.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers,peer",
    [
        ({}, "8.8.8.8"),
        ({"origin": "https://untrusted.example"}, "127.0.0.1"),
        ({"x-forwarded-for": "127.0.0.1"}, "127.0.0.1"),
    ],
)
async def test_private_inventory_and_setup_reject_untrusted_requests(
    tmp_path: Path, headers: dict[str, str], peer: str
) -> None:
    requested = False

    async def provider() -> PreservationHost:
        nonlocal requested
        requested = True
        raise AssertionError("must authorize before opening the private host")

    app = FastAPI()
    app.include_router(
        create_preservation_router(provider, API.require_preservation_operator)
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=(peer, 1000)),
        base_url="http://localhost",
        headers=headers,
    ) as client:
        assert (await client.get("/preservation/library")).status_code == 403
        assert (await client.post("/preservation/setup/device")).status_code == 403
        assert (await client.post("/preservation/setup/publisher")).status_code == 403
        for path in (
            "copies/example/publication",
            "plans/example/public-manifest",
            "kit-uploads/example/invitation",
            "jobs/example/invitation",
        ):
            assert (await client.get("/preservation/" + path)).status_code == 403
        assert (
            await client.post(
                "/preservation/public-copies/example/invitation",
                json={"mirrors": ["https://example.invalid"]},
            )
        ).status_code == 403
        assert (
            await client.post(
                "/preservation/jobs/example/resume",
                json={"plan_id": "a" * 64, "operation_id": "resume"},
            )
        ).status_code == 403
    assert not requested


@pytest.mark.asyncio
async def test_public_invitation_metadata_is_offline_bounded_and_operator_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import socket

    from foxlight_model_preservation.invitations import read_invitation

    from skulk.store.tests.test_preservation_sharing import (
        empty_host,
        import_copy,
        public_kit,
    )

    source = tmp_path / "publisher"
    source.mkdir()
    content, identity, _ = public_kit(source)
    owner = empty_host(tmp_path)
    try:
        copy = await import_copy(owner, content)

        async def provider() -> PreservationHost:
            return owner

        app = FastAPI()
        app.include_router(
            create_preservation_router(provider, API.require_preservation_operator)
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 1000)),
            base_url="http://localhost",
        ) as client:

            def forbidden(*args: object, **kwargs: object) -> None:
                pytest.fail("metadata handoff made a network request")

            monkeypatch.setattr(socket, "getaddrinfo", forbidden)
            response = await client.post(
                f"/preservation/public-copies/{copy.copy_id}/invitation",
                json={"mirrors": ["https://unavailable.invalid/models"]},
            )
            assert response.status_code == 200
            invitation = read_invitation(response.content)
            assert invitation.manifest_id == identity
            assert str(tmp_path).encode() not in response.content
            bad = await client.post(
                f"/preservation/public-copies/{copy.copy_id}/invitation",
                json={"mirrors": ["https://user:secret@unavailable.invalid/models"]},
            )
            assert bad.status_code == 409 and "secret" not in bad.text
            oversized = await client.post(
                "/preservation/kit-uploads",
                json={"purpose": "invitation", "size_bytes": 8 * 1024**2 + 1},
            )
            assert oversized.status_code == 409
            declared = await client.post(
                "/preservation/kit-uploads",
                json={"purpose": "invitation", "size_bytes": len(response.content)},
            )
            uid = KitUploadStatus.model_validate_json(declared.content).upload_id
            assert (
                declared.status_code == 200
                and declared.json()["purpose"] == "invitation"
            )
            assert (
                await client.put(
                    f"/preservation/kit-uploads/{uid}?offset=0",
                    content=response.content,
                    headers={"Content-Type": "application/octet-stream"},
                )
            ).status_code == 200
            preview = await client.get(f"/preservation/kit-uploads/{uid}/invitation")
            assert (
                preview.status_code == 200 and preview.json()["manifest_id"] == identity
            )
            wrong = await client.post(
                "/preservation/plans",
                json={
                    "kind": "receive",
                    "location_id": "invitation",
                    "upload_id": uid,
                    "accepted_key_ids": ["ed25519:" + "0" * 64],
                },
            )
            assert wrong.status_code == 409
            assert (
                len((await owner.page()).jobs) == 1
            )  # Only the original public kit import.
            assert not owner.store.list_models()
    finally:
        await owner.close()
