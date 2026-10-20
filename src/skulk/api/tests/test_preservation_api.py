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
    assert not requested
