"""Durable offline completion checks reuse actual owned workers and original evidence."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import httpx
import pytest
from foxlight_model_preservation.kits import import_kit
from foxlight_model_preservation.mirror_models import MirrorState
from foxlight_model_preservation.mirror_state import MirrorLedger
from foxlight_model_preservation.models import TrustProfile
from foxlight_model_preservation.storage import manifest_path, object_path, verify_store
from pydantic import TypeAdapter, ValidationError

from skulk.api.tests.test_mirror_recovery_api import app, retained
from skulk.api.tests.test_mirrors_api import ready, settled
from skulk.store.mirror_contracts import MirrorJob, MirrorRequest
from skulk.store.preservation_host import PreservationHost
from skulk.store.preservation_journal import PreservationHostError


async def pending(
    root: Path, *, published: bool = True
) -> tuple[PreservationHost, str, TrustProfile, bytes]:
    """Keep original retained intent/bytes with an interrupted completion receipt."""
    owner, identity, trust, wrapper = await retained(root)
    ledger = MirrorLedger(owner.mirrors.path)

    def interrupted(state: MirrorState) -> MirrorState:
        model = state.models[identity].model_copy(
            update={
                "state": "pending",
                "verified_at": None,
                "error": "storage_error",
                "pinned": True,
                "blocked": True,
            }
        )
        return state.model_copy(update={"models": {identity: model}})

    ledger.update(interrupted)
    if not published:
        (ledger.path / "store" / manifest_path(identity)).unlink()
    return owner, identity, trust, wrapper


@pytest.mark.asyncio
@pytest.mark.parametrize("published", [False, True])
async def test_pending_copy_recovers_after_input_and_client_loss_then_kit_restore(
    tmp_path: Path, published: bool
) -> None:
    owner, identity, trust, wrapper = await pending(tmp_path, published=published)
    current = owner
    try:
        ledger = MirrorLedger(owner.mirrors.path)
        page = await owner.mirrors.page()
        assert page.policy_id is not None
        granted = await owner.mirrors.start(
            MirrorRequest(
                owner_id="library",
                operation_id="grant",
                kind="grant_resume",
                policy_id=page.policy_id,
                confirm_resume=True,
                resume_expires=int(time.time()) + 3600,
            )
        )
        assert (await settled(owner, granted.job_id)).state == "succeeded"
        assert owner.journal.setting("mirror-approved-resume") is not None
        before = ledger.read()
        # Only the owned byte worker is denied sockets. Parent ASGI/event-loop
        # machinery remains real; recovery cannot silently contact a supplier.
        original_command = owner.mirrors.command
        offline_worker = """
import socket,sys
from skulk.store import preservation_worker
def refuse(*args,**kwargs): raise AssertionError('offline worker attempted networking')
socket.socket=refuse
sys.argv=['preservation-worker',sys.argv[1]]
raise SystemExit(preservation_worker.main())
"""

        def offline_command(request: Path) -> tuple[str, ...]:
            return sys.executable, "-c", offline_worker, str(request)

        owner.mirrors.command = offline_command
        decision = MirrorRequest(
            owner_id="library",
            operation_id="recover-original",
            kind="recover",
            policy_id=page.policy_id,
            manifest_id=identity,
        )
        transport = httpx.ASGITransport(app=app(owner), client=("127.0.0.1", 1000))
        async with httpx.AsyncClient(
            transport=transport, base_url="http://localhost"
        ) as client:
            response = await client.post(
                "/preservation/mirror/jobs", json=decision.model_dump(mode="json")
            )
            assert response.status_code == 202, response.text
            submitted = MirrorJob.model_validate_json(response.text)
            assert submitted.kind == "recover" and submitted.kit is None
        # Closing the submitting client cannot cancel an accepted host job.
        recovered = await settled(owner, submitted.job_id)
        assert recovered.state == "succeeded", recovered
        assert recovered.bytes_hashed > 0 and recovered.bytes_received == 0
        assert recovered.bytes_copied == 0 and recovered.activity is None
        assert owner.journal.setting("mirror-approved-resume") is None
        assert owner.journal.setting("mirror-running-intent") is None
        assert owner.journal.setting("mirror-stopped") is True
        after = ledger.read()
        model = after.models[identity]
        assert model.state == "retained" and model.blocked and model.pinned
        assert model.grant == before.models[identity].grant and model.error is None
        assert after.restart is None
        assert (after.ingress_used, after.egress_day_used, after.egress_month_used) == (
            before.ingress_used,
            before.egress_day_used,
            before.egress_month_used,
        )
        assert (ledger.path / "store" / manifest_path(identity)).read_bytes() == wrapper
        assert (await owner.mirrors.start(decision)).job_id == submitted.job_id
        owner.mirrors.command = original_command
        await owner.close()
        current = PreservationHost(owner.store, owner.root)
        assert (await current.mirrors.page()).controller_job_id is None
        assert current.mirrors.job(submitted.job_id).state == "succeeded"
        original_state = (ledger.path / "mirror.json").read_bytes()
        assert (await current.mirrors.start(decision)).job_id == submitted.job_id
        assert (ledger.path / "mirror.json").read_bytes() == original_state
        with pytest.raises(PreservationHostError, match="request_changed"):
            await current.mirrors.start(
                decision.model_copy(update={"manifest_id": "f" * 64})
            )
        kit_job = await current.mirrors.start(
            MirrorRequest(
                owner_id="library",
                operation_id="save-kit",
                kind="prepare_kit",
                policy_id=(await current.mirrors.page()).policy_id,
                manifest_id=identity,
            )
        )
        prepared = await settled(current, kit_job.job_id)
        assert prepared.state == "succeeded" and prepared.kit is not None
        transport = httpx.ASGITransport(app=app(current), client=("127.0.0.1", 1000))
        async with httpx.AsyncClient(
            transport=transport, base_url="http://localhost"
        ) as client:
            data = await client.get("/preservation/mirror/kits/" + prepared.job_id)
            assert data.status_code == 200, data.text
            saved = tmp_path / "recipient.foxkit"
            saved.write_bytes(data.content)
        imported = tmp_path / "independent-recipient"
        import_kit(saved, imported, expected_manifest_id=identity, trust=trust)
        manifest = verify_store(imported, identity, trust)
        assert len(manifest.members) == 2
        assert all(
            member.retained_card == {"future": {"opaque": [1, True, "unchanged"]}}
            for member in manifest.members
        )
        assert (imported / manifest_path(identity)).read_bytes() == wrapper
    finally:
        await current.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["missing", "corrupt", "changed_wrapper", "removed"])
async def test_bad_pending_generation_refuses_without_fetch_or_mutation(
    tmp_path: Path, damage: str
) -> None:
    owner, identity, trust, _wrapper = await pending(tmp_path)
    try:
        ledger = MirrorLedger(owner.mirrors.path)
        manifest = verify_store(ledger.path / "store", identity, trust)
        if damage in {"missing", "corrupt"}:
            path = (
                ledger.path
                / "store"
                / object_path(manifest.members[-1].files[0].sha256)
            )
            if damage == "missing":
                path.unlink()
            else:
                path.write_bytes(b"changed fixture")
        elif damage == "changed_wrapper":
            path = ledger.path / "store" / manifest_path(identity)
            path.write_bytes(path.read_bytes() + b" ")
        else:
            ledger.update(
                lambda state: state.model_copy(
                    update={
                        "models": {
                            identity: state.models[identity].model_copy(
                                update={"state": "removed"}
                            )
                        }
                    }
                )
            )
        before = (ledger.path / "mirror.json").read_bytes()
        job = await owner.mirrors.start(
            MirrorRequest(
                owner_id="library",
                operation_id="bad",
                kind="recover",
                policy_id=(await owner.mirrors.page()).policy_id,
                manifest_id=identity,
            )
        )
        result = await settled(owner, job.job_id)
        assert result.state == "failed" and result.error_code is not None
        assert result.bytes_received == 0 and result.activity is None
        assert (ledger.path / "mirror.json").read_bytes() == before
        assert not list(owner.mirrors.requests.iterdir())
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_active_controller_and_stale_policy_cannot_be_recovered(
    tmp_path: Path,
) -> None:
    owner, identity, _trust, _wrapper = await retained(tmp_path)
    try:
        page = await owner.mirrors.page()
        running = await owner.mirrors.start(
            MirrorRequest(
                owner_id="library",
                operation_id="serve",
                kind="run",
                policy_id=page.policy_id,
                public_participation=True,
                seconds=60,
            )
        )
        await ready(owner, running.job_id)
        decision = MirrorRequest(
            owner_id="library",
            operation_id="busy",
            kind="recover",
            policy_id=page.policy_id,
            manifest_id=identity,
        )
        busy = await owner.mirrors.start(decision)
        assert busy.state == "failed" and busy.error_code == "mirror_busy_stop_first"
        assert owner.mirrors.job(running.job_id).activity is not None
        await owner.mirrors.cancel(running.job_id)
        wrong = await owner.mirrors.start(
            decision.model_copy(update={"operation_id": "stale", "policy_id": "f" * 64})
        )
        refused = await settled(owner, wrong.job_id)
        assert (
            refused.state == "failed" and refused.error_code == "mirror_policy_changed"
        )
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["stop", "close"])
async def test_owned_recovery_stop_and_core_loss_leave_pending_without_replay(
    tmp_path: Path, stop: str
) -> None:
    owner, identity, _trust, _wrapper = await pending(tmp_path, published=False)
    current = owner
    marker = tmp_path / "checking"
    # The fixed test wrapper pauses inside the real verifier, where its actual
    # control checkpoints observe the parent's cancel/EOF fence.
    script = """
import socket,sys,time
from pathlib import Path
from skulk.store import preservation_worker
from foxlight_model_preservation import mirror_recovery
original=mirror_recovery.verify_object
marker=Path(sys.argv[2])
def slow(root,file,*,control=None):
    marker.write_text('checking')
    while True:
        if control: control.checkpoint()
        time.sleep(0.01)
mirror_recovery.verify_object=slow
sys.argv=['preservation-worker',sys.argv[1]]
raise SystemExit(preservation_worker.main())
"""

    def slow_command(request: Path) -> tuple[str, ...]:
        return sys.executable, "-c", script, str(request), str(marker)

    owner.mirrors.command = slow_command
    decision = MirrorRequest(
        owner_id="library",
        operation_id="interrupted",
        kind="recover",
        policy_id=(await owner.mirrors.page()).policy_id,
        manifest_id=identity,
    )
    try:
        submitted = await owner.mirrors.start(decision)
        async with asyncio.timeout(30):
            while not marker.exists():
                observed = owner.mirrors.job(submitted.job_id)
                assert observed.state in {"queued", "running"}, observed
                await asyncio.sleep(0.025)
        ledger = MirrorLedger(owner.mirrors.path)
        if stop == "stop":
            stopped = await owner.mirrors.cancel(submitted.job_id)
            assert stopped.state == "cancelled"
            assert owner.mirrors._processes == {}  # pyright: ignore[reportPrivateUsage]
        await owner.close()
        assert ledger.read().models[identity].state == "pending"
        assert not (ledger.path / "store" / manifest_path(identity)).exists()
        current = PreservationHost(owner.store, owner.root)
        previous = current.mirrors.job(submitted.job_id)
        assert previous.state in {"cancelled", "interrupted"}
        assert (await current.mirrors.start(decision)).job_id == submitted.job_id
        assert (await current.mirrors.page()).controller_job_id is None
        explicit = decision.model_copy(update={"operation_id": "explicit-new-check"})
        result = await settled(current, (await current.mirrors.start(explicit)).job_id)
        assert result.state == "succeeded", result
    finally:
        await current.close()


@pytest.mark.parametrize(
    "extra",
    [
        {"manifest_id": None},
        {"copy_id": "unexpected"},
        {"confirm_reintake": True},
        {"public_participation": True},
        {"confirm_resume": True, "resume_expires": 1},
        {"confirm_delete_manifest": "a" * 64},
        {"action": "retry"},
    ],
)
def test_recovery_contract_rejects_unrelated_authority(
    extra: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        MirrorRequest.model_validate_json(
            json.dumps(
                {
                    "owner_id": "library",
                    "operation_id": "recover",
                    "kind": "recover",
                    "policy_id": "a" * 64,
                    "manifest_id": "b" * 64,
                    **extra,
                }
            )
        )


def test_openapi_exposes_new_kind_for_old_host_feature_negotiation() -> None:
    # Schema construction needs no owner instance or preservation initialization.
    from fastapi import FastAPI

    from skulk.api.main import API
    from skulk.api.preservation import create_preservation_router

    async def unavailable() -> PreservationHost:
        raise AssertionError("OpenAPI unexpectedly initialized preservation")

    application = FastAPI()
    application.include_router(
        create_preservation_router(unavailable, API.require_preservation_operator)
    )
    for name in ("MirrorRequest", "MirrorJob"):
        choices = TypeAdapter(tuple[str, ...]).validate_json(
            json.dumps(
                application.openapi()["components"]["schemas"][name]["properties"][
                    "kind"
                ]["enum"]
            )
        )
        assert "recover" in choices
