"""Real owned public workers preserve opaque cards and stop network participation."""

from __future__ import annotations

import asyncio
import json
import shutil
import socket
from pathlib import Path
from typing import Literal

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from foxlight_model_preservation.kits import export_kit
from foxlight_model_preservation.models import Recipe, TrustProfile
from foxlight_model_preservation.signing import public_record
from foxlight_model_preservation.storage import manifest_path, pack, verify_store
from foxlight_model_preservation.torrent_binding import torrent_engine

from skulk.store.model_store import ModelStore
from skulk.store.preservation_contracts import (
    PreservationCopy,
    PreservationJob,
    PreservationPlanRequest,
    PreservationStart,
)
from skulk.store.preservation_host import PreservationHost
from skulk.store.preservation_journal import PreservationHostError


def public_kit(
    root: Path, audience: Literal["public", "private"] = "public"
) -> tuple[bytes, str, TrustProfile]:
    """Create synthetic signed bytes with a deliberately future runtime card."""
    source = root / "source"
    source.mkdir()
    (source / "weights.bin").write_bytes(b"synthetic community bytes\x00" * 400)
    (source / "LICENSE").write_text("Synthetic fixture; no upstream weights.\n")
    recipe = Recipe.model_validate_json(
        json.dumps(
            {
                "schema_version": "foxlight.model-preservation.v1",
                "name": "Future community model",
                "root_member": "primary",
                "workloads": ["fixture"],
                "runtime_requirements": ["future-reader"],
                "members": [
                    {
                        "member_id": "primary",
                        "role": "primary",
                        "format": "test-only",
                        "origin_repository": "community/future",
                        "origin_revision": "a" * 40,
                        "artifact_identity": "fixture:future",
                        "dependencies": [],
                        "retained_card": {
                            "card_profile": "future.model-card.v999",
                            "future_property": {"nested": [1, 2, 3]},
                        },
                        "intake": {
                            "audience": audience,
                            "license_id": "test-only",
                            "terms_reference": "LICENSE",
                            "decision": "synthetic public fixture",
                            "notice_paths": ["LICENSE"],
                        },
                        "files": [{"path": "weights.bin"}, {"path": "LICENSE"}],
                    }
                ],
            }
        )
    )
    key = Ed25519PrivateKey.generate()
    trust = TrustProfile(
        schema_version="foxlight.model-preservation.trust.v1",
        keys=(public_record(key),),
        threshold=1,
        as_of="2026-10-21T00:00:00Z",
    )
    archive = root / "archive"
    identity = pack(recipe, {"primary": source}, archive, (key,))
    destination = root / "fixture.foxkit"
    export_kit(archive, identity, trust, destination)
    return destination.read_bytes(), identity, trust


async def settled(host: PreservationHost, job_id: str) -> PreservationJob:
    async with asyncio.timeout(30):
        while host.journal.job(job_id).state in {"queued", "running", "cancelling"}:
            await asyncio.sleep(0.025)
    return host.journal.job(job_id)


async def ready(host: PreservationHost, job_id: str) -> PreservationJob:
    try:
        async with asyncio.timeout(30):
            while True:
                job = host.journal.job(job_id)
                if job.sharing is not None and job.sharing.ready:
                    return job
                assert job.state in {"queued", "running"}, job
                await asyncio.sleep(0.025)
    except TimeoutError:
        pytest.fail(
            "Sharing readiness deadline: " + host.journal.job(job_id).model_dump_json()
        )


async def import_copy(
    host: PreservationHost, content: bytes, operation_id: str = "public-import"
) -> PreservationCopy:
    uploaded = host.uploads.create(len(content))
    uid = str(uploaded["upload_id"])
    host.uploads.append(uid, 0, content)
    plan = await host.plan(
        PreservationPlanRequest(
            kind="mirror_import", location_id="public-kit", upload_id=uid
        )
    )
    assert (
        plan.requires_owner_acceptance
        and plan.publication == "public-publisher-declaration"
    )
    request = PreservationStart(
        plan_id=plan.plan_id, operation_id=operation_id, accept_owner_capture=True
    )
    job = host.start(request)
    assert host.start(request).job_id == job.job_id
    result = await settled(host, job.job_id)
    assert result.state == "succeeded", result
    assert result.copy_id is not None
    return host.journal.copy(result.copy_id)


def empty_host(root: Path) -> PreservationHost:
    store = root / "models"
    store.mkdir()
    return PreservationHost(ModelStore(store), root / "private")


@pytest.mark.asyncio
async def test_public_import_keeps_future_card_and_source_independence(
    tmp_path: Path,
) -> None:
    publisher = tmp_path / "publisher"
    publisher.mkdir()
    content, identity, trust = public_kit(publisher)
    host = empty_host(tmp_path)
    try:
        copy = await import_copy(host, content)
        shutil.rmtree(publisher)
        page = await host.page()
        assert not page.copies and not page.models and len(page.public_copies) == 1
        assert copy.manifest_id == identity and copy.publisher_key_ids == tuple(
            key.key_id for key in trust.keys
        )
        assert host.public_trust(copy.copy_id) == trust
        path = next((host.root / "public-copies").iterdir())
        manifest = verify_store(path, identity, trust)
        assert manifest.members[0].retained_card == {
            "card_profile": "future.model-card.v999",
            "future_property": {"nested": [1, 2, 3]},
        }
        assert not host.store.list_models()
    finally:
        await host.close()
    restarted = PreservationHost(ModelStore(tmp_path / "models"), tmp_path / "private")
    try:
        assert len((await restarted.page()).public_copies) == 1
        assert all(job.state == "succeeded" for job in restarted.journal.jobs())
    finally:
        await restarted.close()


@pytest.mark.asyncio
async def test_private_bundle_refuses_before_worker_or_listener(tmp_path: Path) -> None:
    publisher = tmp_path / "publisher"
    publisher.mkdir()
    content, _, _ = public_kit(publisher, "private")
    host = empty_host(tmp_path)
    try:
        upload = host.uploads.create(len(content))
        uid = str(upload["upload_id"])
        host.uploads.append(uid, 0, content)
        with pytest.raises(PreservationHostError, match="public_bundle_required"):
            await host.plan(
                PreservationPlanRequest(
                    kind="mirror_import", location_id="public-kit", upload_id=uid
                )
            )
        assert not host.journal.jobs()
        assert not (host.root / "public-copies").exists()
    finally:
        await host.close()


@pytest.mark.asyncio
async def test_http_explicit_consent_real_fetch_accounting_and_stop(
    tmp_path: Path,
) -> None:
    publisher = tmp_path / "publisher"
    publisher.mkdir()
    content, identity, trust = public_kit(publisher)
    host = empty_host(tmp_path)
    try:
        copy = await import_copy(host, content)
        plan = await host.plan(
            PreservationPlanRequest(
                kind="http_share",
                location_id=copy.location_id,
                copy_id=copy.copy_id,
                duration_seconds=60,
                upload_limit_bytes=100000,
            )
        )
        with pytest.raises(
            PreservationHostError, match="public_participation_required"
        ):
            host.start(PreservationStart(plan_id=plan.plan_id, operation_id="share"))
        request = PreservationStart(
            plan_id=plan.plan_id, operation_id="share", public_participation=True
        )
        job = host.start(request)
        live = await ready(host, job.job_id)
        assert live.sharing is not None
        port = live.sharing.listen_port
        # A long serving session occupies neither the byte queue nor live-store lease.
        async with host.store.preservation_lease():
            duplicate = await import_copy(host, content, "import-while-serving")
            assert duplicate.copy_id == copy.copy_id
        async with httpx.AsyncClient(trust_env=False) as client:
            response = await client.get(
                f"http://127.0.0.1:{port}/" + manifest_path(identity)
            )
            assert response.status_code == 200
            encoded_size = len(response.content)
        async with asyncio.timeout(3):
            while (
                host.journal.job(job.job_id).sharing or live.sharing
            ).charged_bytes < encoded_size:
                await asyncio.sleep(0.025)
        assert host.start(request).job_id == job.job_id
        await host.cancel(job.job_id)
        stopped = await settled(host, job.job_id)
        assert stopped.state == "cancelled", stopped
        assert stopped.sharing is not None and not stopped.sharing.ready
        assert (
            stopped.sharing.listen_port == 0
            and stopped.sharing.charged_bytes == encoded_size
        )
        with socket.socket() as probe:
            probe.settimeout(1)
            assert probe.connect_ex(("127.0.0.1", port)) != 0
        assert host.public_trust(copy.copy_id) == trust and not host.store.list_models()
        assert len((await host.page()).public_copies) == 1
    finally:
        await host.close()


@pytest.mark.asyncio
async def test_torrent_real_peer_and_retained_metadata_after_stop(
    tmp_path: Path,
) -> None:
    pytest.importorskip("libtorrent")
    publisher = tmp_path / "publisher"
    publisher.mkdir()
    content, identity, trust = public_kit(publisher)
    host = empty_host(tmp_path)
    try:
        copy = await import_copy(host, content)
        plan = await host.plan(
            PreservationPlanRequest(
                kind="torrent_share",
                location_id=copy.location_id,
                copy_id=copy.copy_id,
                duration_seconds=60,
            )
        )
        job = host.start(
            PreservationStart(
                plan_id=plan.plan_id, operation_id="torrent", public_participation=True
            )
        )
        live = await ready(host, job.job_id)
        assert live.sharing is not None
        metadata = host.torrent_path(job.job_id).read_bytes()
        engine = torrent_engine()
        receiver = engine.session(
            {
                "listen_interfaces": "127.0.0.1:0",
                "enable_dht": False,
                "enable_lsd": False,
                "enable_upnp": False,
                "enable_natpmp": False,
                "enable_incoming_utp": False,
                "enable_outgoing_utp": False,
                "alert_mask": 0,
            }
        )
        destination = tmp_path / "received"
        destination.mkdir()
        handle = receiver.add_torrent(
            {
                "ti": engine.torrent_info(engine.bdecode(metadata)),
                "save_path": str(destination),
                "flags": 0,
            }
        )
        try:
            handle.connect_peer(("127.0.0.1", live.sharing.listen_port))
            async with asyncio.timeout(20):
                while not handle.status().is_seeding:
                    await asyncio.sleep(0.05)
            restored = verify_store(destination / identity, identity, trust)
            retained = restored.members[0].retained_card
            assert (
                retained is not None
                and retained["card_profile"] == "future.model-card.v999"
            )
        finally:
            receiver.remove_torrent(handle)
            receiver.pause()
            del handle
            del receiver
        await host.cancel(job.job_id)
        stopped = await settled(host, job.job_id)
        assert (
            stopped.state == "cancelled"
            and stopped.sharing is not None
            and not stopped.sharing.ready
        )
        assert host.torrent_path(job.job_id).read_bytes() == metadata
        assert not host.store.list_models()
    finally:
        await host.close()


@pytest.mark.asyncio
async def test_corrupt_public_objects_never_enroll_or_serve(tmp_path: Path) -> None:
    publisher = tmp_path / "publisher"
    publisher.mkdir()
    content, _, _ = public_kit(publisher)
    assert b"synthetic community bytes" in content
    original_content = content
    tampered = bytearray(content)
    tampered[content.index(b"synthetic community bytes")] ^= 1
    content = bytes(tampered)
    host = empty_host(tmp_path)
    try:
        upload = host.uploads.create(len(content))
        uid = str(upload["upload_id"])
        host.uploads.append(uid, 0, content)
        plan = await host.plan(
            PreservationPlanRequest(
                kind="mirror_import", location_id="public-kit", upload_id=uid
            )
        )
        job = host.start(
            PreservationStart(
                plan_id=plan.plan_id, operation_id="tampered", accept_owner_capture=True
            )
        )
        failed = await settled(host, job.job_id)
        assert failed.state == "failed" and failed.expected_manifest_id is None
        assert not (await host.page()).public_copies
        assert not host.store.list_models()
        # Fresh explicit intake is still available after a rejected payload.
        restored = await import_copy(host, original_content, "valid-after-rejection")
        assert restored.audience == "public"
    finally:
        await host.close()


@pytest.mark.asyncio
async def test_http_duration_stops_and_restart_never_replays(tmp_path: Path) -> None:
    publisher = tmp_path / "publisher"
    publisher.mkdir()
    content, _, _ = public_kit(publisher)
    host = empty_host(tmp_path)
    copy = await import_copy(host, content)
    plan = await host.plan(
        PreservationPlanRequest(
            kind="http_share",
            location_id=copy.location_id,
            copy_id=copy.copy_id,
            duration_seconds=1,
        )
    )
    job = host.start(
        PreservationStart(
            plan_id=plan.plan_id, operation_id="duration", public_participation=True
        )
    )
    live = await ready(host, job.job_id)
    assert live.sharing is not None
    port = live.sharing.listen_port
    stopped = await settled(host, job.job_id)
    assert (
        stopped.state == "succeeded"
        and stopped.sharing is not None
        and not stopped.sharing.ready
    )
    assert stopped.sharing_policy == plan.sharing
    await host.close()
    restarted = PreservationHost(ModelStore(tmp_path / "models"), tmp_path / "private")
    try:
        same = restarted.start(
            PreservationStart(
                plan_id=plan.plan_id, operation_id="duration", public_participation=True
            )
        )
        assert same.job_id == job.job_id and same.state == "succeeded"
        assert (
            not (await restarted.page()).jobs[0].sharing
            or not same.sharing
            or not same.sharing.ready
        )
        with socket.socket() as probe:
            probe.settimeout(1)
            assert probe.connect_ex(("127.0.0.1", port)) != 0
    finally:
        await restarted.close()


@pytest.mark.asyncio
async def test_operator_artifacts_are_gated_and_metainfo_tampering_refuses(
    tmp_path: Path,
) -> None:
    from fastapi import FastAPI, HTTPException, Request

    from skulk.api.preservation import create_preservation_router

    pytest.importorskip("libtorrent")
    publisher = tmp_path / "publisher"
    publisher.mkdir()
    content, identity, trust = public_kit(publisher)
    host = empty_host(tmp_path)
    copy = await import_copy(host, content)
    plan = await host.plan(
        PreservationPlanRequest(
            kind="torrent_share",
            location_id=copy.location_id,
            copy_id=copy.copy_id,
            duration_seconds=60,
        )
    )
    job = host.start(
        PreservationStart(
            plan_id=plan.plan_id, operation_id="artifact", public_participation=True
        )
    )
    await ready(host, job.job_id)
    allowed = False

    async def provider() -> PreservationHost:
        return host

    def authorize(request: Request) -> None:
        if not allowed:
            raise HTTPException(403)

    app = FastAPI()
    app.include_router(create_preservation_router(provider, authorize))
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://operator"
        ) as client:
            route = "/preservation/torrents/" + job.job_id
            assert (await client.get(route)).status_code == 403
            allowed = True
            response = await client.get(route)
            assert (
                response.status_code == 200
                and response.content == host.torrent_path(job.job_id).read_bytes()
            )
            public_keys = await client.get(
                "/preservation/public-copies/" + copy.copy_id + "/trust"
            )
            assert TrustProfile.model_validate_json(public_keys.content) == trust
            assert identity in copy.manifest_id
            path = host.torrent_path(job.job_id)
            path.write_bytes(b"changed metadata")
            assert (await client.get(route)).status_code == 409
    finally:
        await host.close()


@pytest.mark.asyncio
async def test_damaged_public_copy_fails_before_serving_without_uncertain_publication(
    tmp_path: Path,
) -> None:
    publisher = tmp_path / "publisher"
    publisher.mkdir()
    content, identity, trust = public_kit(publisher)
    host = empty_host(tmp_path)
    try:
        copy = await import_copy(host, content)
        archive = next((host.root / "public-copies").iterdir())
        manifest = verify_store(archive, identity, trust)
        selected = manifest.members[0].files[0]
        (archive / "objects" / "sha256" / selected.sha256).unlink()
        plan = await host.plan(
            PreservationPlanRequest(
                kind="http_share", location_id=copy.location_id, copy_id=copy.copy_id
            )
        )
        job = host.start(
            PreservationStart(
                plan_id=plan.plan_id, operation_id="damaged", public_participation=True
            )
        )
        stopped = await settled(host, job.job_id)
        # A known source identity is not an unconfirmed model-store publication.
        assert stopped.state == "failed" and stopped.expected_manifest_id == identity
        assert stopped.sharing is None or not stopped.sharing.ready
    finally:
        await host.close()


@pytest.mark.asyncio
async def test_public_check_and_kit_export_retain_public_identity(
    tmp_path: Path,
) -> None:
    from foxlight_model_preservation.kits import inspect_kit

    publisher = tmp_path / "publisher"
    publisher.mkdir()
    content, identity, trust = public_kit(publisher)
    host = empty_host(tmp_path)
    try:
        copy = await import_copy(host, content)
        for kind in ("check", "kit_export"):
            plan = await host.plan(
                PreservationPlanRequest(
                    kind=kind, location_id=copy.location_id, copy_id=copy.copy_id
                )
            )
            job = host.start(PreservationStart(plan_id=plan.plan_id, operation_id=kind))
            result = await settled(host, job.job_id)
            assert result.state == "succeeded", result
            assert host.journal.copy(copy.copy_id).audience == "public"
            if kind == "kit_export":
                exported = inspect_kit(host.kit_path(job.job_id))
                assert exported.manifest_id == identity and exported.trust == trust
                assert all(
                    member.intake.audience == "public"
                    for member in exported.manifest.members
                )
        assert not host.store.list_models()
    finally:
        await host.close()


def test_http_cancel_during_initial_check_closes_without_accepting_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from foxlight_model_preservation.operations import (
        OperationCancelled,
        OperationControl,
    )
    from foxlight_model_preservation.seed import SeedServer

    from skulk.store import preservation_sharing
    from skulk.store.preservation_contracts import SharingPolicy

    publisher = tmp_path / "publisher"
    publisher.mkdir()
    _, identity, trust = public_kit(publisher)
    monkeypatch.chdir(publisher / "archive")
    control = OperationControl()
    servers: list[SeedServer] = []

    def cancelled_server(
        address: tuple[str, int],
        archive: Path,
        selected: tuple[str, ...],
        keys: TrustProfile,
        *,
        public: bool,
        max_upload_bytes: int,
        concurrency: int,
    ) -> SeedServer:
        server = SeedServer(
            address,
            archive,
            selected,
            keys,
            public=public,
            max_upload_bytes=max_upload_bytes,
            concurrency=concurrency,
        )
        servers.append(server)
        control.cancel()
        return server

    monkeypatch.setattr(preservation_sharing, "SeedServer", cancelled_server)
    with pytest.raises(OperationCancelled):
        preservation_sharing.serve_public_http(
            identity,
            trust,
            SharingPolicy(
                public_network=False,
                listen_port=0,
                duration_seconds=60,
                upload_limit_bytes=100000,
                upload_rate_bytes=1024,
            ),
            control,
            lambda observation: pytest.fail("cancelled server accepted requests"),
        )
    assert len(servers) == 1 and servers[0].socket.fileno() == -1
