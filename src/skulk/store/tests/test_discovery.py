"""Real source/API/worker boundaries without registry, inference or private SDK."""

from __future__ import annotations

import asyncio
import base64
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Literal, cast

import httpx
import psutil
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi import FastAPI
from foxlight_model_preservation.discovery_models import (
    DiscoveryCatalogTarget,
    DiscoveryDescription,
    SourceBootstrap,
)
from foxlight_model_preservation.discovery_publish import (
    DiscoverySigningKeys,
    publish_catalog,
)
from foxlight_model_preservation.models import Catalog, CatalogEntry, TrustProfile
from foxlight_model_preservation.signing import (
    CATALOG_TYPE,
    digest,
    encode_record,
    public_record,
    sign_record,
)
from foxlight_model_preservation.storage import manifest_path
from pydantic import JsonValue
from securesystemslib.signer import (
    CryptoSigner,  # pyright: ignore[reportPrivateImportUsage]
)
from tuf.api.metadata import Metadata, Root  # pyright: ignore[reportPrivateImportUsage]

from skulk.api.main import API
from skulk.api.preservation import create_preservation_router
from skulk.store.discovery_contracts import (
    DiscoveryJob,
    DiscoveryPage,
    DiscoveryQuery,
    DiscoveryRequest,
    DiscoveryResults,
    SourceReview,
)
from skulk.store.model_store import ModelStore
from skulk.store.preservation_contracts import (
    KitUploadStatus,
    PreservationPlan,
    PreservationStart,
)
from skulk.store.preservation_host import PreservationHost
from skulk.store.preservation_journal import PreservationHostError
from skulk.store.tests.test_preservation_sharing import public_kit, settled


@dataclass(frozen=True)
class StaticServer:
    url: str
    requests: list[str]
    server: ThreadingHTTPServer


@contextmanager
def serve(path: Path) -> Iterator[StaticServer]:
    """Expose synthetic public fixtures with actual bounded HTTP listeners."""
    requests: list[str] = []

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *args: object, **kwargs: object) -> None:
            """Keep fixture traffic out of test logs."""

        def do_GET(self) -> None:
            """Observe exact requested paths before the ordinary file response."""
            requests.append(self.path)
            super().do_GET()

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(Handler, directory=str(path))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield StaticServer(f"http://127.0.0.1:{server.server_port}", requests, server)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def source_tree(
    path: Path,
    manifest: str,
    trust: TrustProfile,
    mirror: str,
    *,
    keys: bool = True,
    extra_manifests: tuple[str, ...] = (),
    description: str = "A preserved model for offline use",
    name: str = "Community fixture",
) -> bytes:
    """Use only the standalone public source producer and stock TUF root tooling."""
    online, offline, catalog_key = (Ed25519PrivateKey.generate() for _ in range(3))
    root = Metadata(Root(expires=datetime.now(UTC) + timedelta(days=30)))
    root.signed.add_key(CryptoSigner(offline).public_key, "root")
    for role in ("timestamp", "snapshot", "targets"):
        root.signed.add_key(CryptoSigner(online).public_key, role)
    root.sign(CryptoSigner(offline))
    catalog = Catalog(
        schema_version="foxlight.model-preservation.catalog.v1",
        description="Independent fixture source",
        entries=tuple(
            CatalogEntry(
                entry_id="community-model"
                if index == 0
                else f"community-model-{index}",
                manifest=identity,
                mirrors=(mirror,),
            )
            for index, identity in enumerate((manifest, *extra_manifests))
        ),
    )
    catalog_trust = TrustProfile(
        schema_version="foxlight.model-preservation.trust.v1",
        threshold=1,
        as_of=datetime.now(UTC).isoformat(),
        keys=(public_record(catalog_key),),
    )
    profile = DiscoveryCatalogTarget(
        catalog_id=digest(encode_record(catalog)),
        catalog_trust=catalog_trust,
        descriptions=tuple(
            DiscoveryDescription(
                entry_id=entry.entry_id,
                manifest=entry.manifest,
                name=name,
                summary=description,
                candidate_bundle_trust=trust if keys else None,
            )
            for entry in catalog.entries
        ),
    )
    publish_catalog(
        path,
        root.to_bytes(),
        encode_record(sign_record(catalog, (catalog_key,), CATALOG_TYPE)),
        profile,
        DiscoverySigningKeys(online, online, online),
        version=1,
        expires=datetime.now(UTC) + timedelta(days=1),
    )
    return root.to_bytes()


def application(host: PreservationHost) -> FastAPI:
    """Mount the production operator gate and canonical preservation router."""

    async def provider() -> PreservationHost:
        return host

    app = FastAPI()
    app.include_router(
        create_preservation_router(provider, API.require_preservation_operator)
    )
    return app


async def enroll(client: httpx.AsyncClient, encoded: bytes) -> SourceReview:
    """Use ordinary binary ingress and an explicit source-file review."""
    created = await client.post(
        "/preservation/kit-uploads",
        json={"size_bytes": len(encoded), "purpose": "source"},
    )
    assert created.status_code == 200, created.text
    upload = KitUploadStatus.model_validate_json(created.text)
    written = await client.put(
        f"/preservation/kit-uploads/{upload.upload_id}?offset=0",
        content=encoded,
        headers={"Content-Type": "application/octet-stream"},
    )
    assert written.status_code == 200, written.text
    reviewed = await client.post(
        "/preservation/discovery/reviews", json={"upload_id": upload.upload_id}
    )
    assert reviewed.status_code == 200, reviewed.text
    view = SourceReview.model_validate_json(reviewed.text)
    followed = await client.post(
        "/preservation/discovery/sources",
        json={
            "review_id": view.review_id,
            "confirmed": True,
            "allow_http": True,
            "allow_private": True,
        },
    )
    assert followed.status_code == 200, followed.text
    return view


async def metadata_settled(client: httpx.AsyncClient, job_id: str) -> DiscoveryJob:
    """Observe retained native jobs rather than running work in the browser."""
    async with asyncio.timeout(30):
        while True:
            response = await client.get("/preservation/discovery/jobs/" + job_id)
            assert response.status_code == 200, response.text
            job = DiscoveryJob.model_validate_json(response.text)
            if job.state not in {"queued", "running", "cancelling"}:
                return job
            await asyncio.sleep(0.025)


async def submit(
    client: httpx.AsyncClient, selection: DiscoveryRequest
) -> DiscoveryJob:
    response = await client.post(
        "/preservation/discovery/jobs", json=selection.model_dump(mode="json")
    )
    assert response.status_code == 202, response.text
    return DiscoveryJob.model_validate_json(response.text)


@pytest.mark.asyncio
async def test_native_source_to_verified_copy_then_offline_search_after_restart(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "model"
    fixture.mkdir()
    _, manifest, trust = public_kit(fixture)
    host_root = tmp_path / "core"
    host = PreservationHost(ModelStore(tmp_path / "models"), host_root)
    try:
        with serve(fixture / "archive") as models:
            root = source_tree(tmp_path / "source", manifest, trust, models.url)
            with serve(tmp_path / "source") as metadata:
                bootstrap = SourceBootstrap(
                    name="Independent source",
                    root=base64.b64encode(root).decode(),
                    endpoints=(metadata.url,),
                )
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=application(host)),
                    base_url="http://localhost",
                ) as client:
                    page = await client.get(
                        "/preservation/discovery",
                        params={"owner_id": "library-discovery"},
                    )
                    assert DiscoveryPage.model_validate_json(page.text).sources == ()
                    review = await enroll(client, encode_record(bootstrap))
                    assert not metadata.requests and not models.requests
                    selection = DiscoveryRequest(
                        owner_id="library-discovery",
                        operation_id="initial-refresh",
                        kind="refresh",
                        source_id=review.source_id,
                    )
                    queued = await submit(client, selection)
                    assert (
                        await metadata_settled(client, queued.job_id)
                    ).state == "succeeded"
                    attempts = tuple(metadata.requests)
                    # Reopening/replay is observational, including after success.
                    assert (await submit(client, selection)).job_id == queued.job_id
                    found = await client.post(
                        "/preservation/discovery/search", json={"query": "offline"}
                    )
                    results = DiscoveryResults.model_validate_json(found.text)
                    assert results.results[0].manifest_id == manifest
                    assert tuple(metadata.requests) == attempts and not models.requests
                    prep = DiscoveryRequest(
                        owner_id="library-discovery",
                        operation_id="review-model",
                        kind="model_metadata",
                        source_id=review.source_id,
                        entry_id="community-model",
                        manifest_id=manifest,
                        accept_source_keys=True,
                        allow_http=True,
                        allow_private=True,
                    )
                    reviewed = await metadata_settled(
                        client, (await submit(client, prep)).job_id
                    )
                    assert reviewed.state == "succeeded" and reviewed.upload_id, (
                        reviewed
                    )
                    assert models.requests == ["/" + manifest_path(manifest)]
                    # Source disappearance cannot prevent model-byte receiving.
                    metadata.server.shutdown()
                    metadata.server.server_close()
                    planned = await client.post(
                        "/preservation/plans",
                        json={
                            "kind": "receive",
                            "location_id": "invitation",
                            "upload_id": reviewed.upload_id,
                            "accepted_key_ids": [key.key_id for key in trust.keys],
                            "allow_http": True,
                            "allow_private": True,
                        },
                    )
                    assert planned.status_code == 200, planned.text
                    plan = PreservationPlan.model_validate_json(planned.text)
                    received = host.start(
                        PreservationStart(
                            plan_id=plan.plan_id,
                            operation_id="receive-discovered",
                            owner_id="library-receive",
                            accept_owner_capture=True,
                        )
                    )
                    completed = await settled(host, received.job_id)
                    assert completed.state == "succeeded" and completed.copy_id, (
                        completed
                    )
                    assert host.journal.copy(completed.copy_id).manifest_id == manifest
                    failed = await submit(
                        client,
                        selection.model_copy(
                            update={"operation_id": "offline-refresh"}
                        ),
                    )
                    assert (
                        await metadata_settled(client, failed.job_id)
                    ).state == "failed"
                    assert DiscoveryResults.model_validate_json(
                        (
                            await client.post(
                                "/preservation/discovery/search",
                                json={"query": "offline"},
                            )
                        ).text
                    ).results
                    bad_replay = await client.post(
                        "/preservation/discovery/jobs",
                        json=selection.model_copy(
                            update={"source_id": "a" * 64}
                        ).model_dump(mode="json"),
                    )
                    assert bad_replay.status_code == 409
        await host.close()
        host = PreservationHost(ModelStore(tmp_path / "models"), host_root)
        # All fixtures are gone; reopening does not replay any update or download.
        results = await host.discovery.search(DiscoveryQuery(query="offline"))
        assert results.results[0].manifest_id == manifest
        assert (await host.discovery.page("library-discovery", 0)).sources[
            0
        ].freshness == "retained"
        assert not (await host.page()).locations
    finally:
        await host.close()


@pytest.mark.asyncio
async def test_metadata_jobs_retain_replay_and_interrupt_without_restart_replay(
    tmp_path: Path,
) -> None:
    host = PreservationHost(ModelStore(tmp_path / "models"), tmp_path / "core")
    request = DiscoveryRequest(
        owner_id="owner", operation_id="queued", source_id="a" * 64, kind="refresh"
    )
    try:
        job, created = host.journal.reserve_discovery(request)
        assert created
        await host.close()
        host = PreservationHost(ModelStore(tmp_path / "models"), tmp_path / "core")
        assert host.journal.discovery_job(job.job_id)[0].state == "interrupted"
        assert host.discovery.start(request).state == "interrupted"
        with pytest.raises(PreservationHostError, match="request_changed"):
            host.discovery.start(request.model_copy(update={"source_id": "b" * 64}))
        for index in range(4):
            host.journal.reserve_discovery(
                request.model_copy(update={"operation_id": str(index)})
            )
        with pytest.raises(PreservationHostError, match="capacity"):
            host.journal.reserve_discovery(
                request.model_copy(update={"operation_id": "full"})
            )
    finally:
        await host.close()


@pytest.mark.asyncio
async def test_source_and_model_metadata_permissions_remain_separate(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "model"
    fixture.mkdir()
    _, manifest, trust = public_kit(fixture)
    host = PreservationHost(ModelStore(tmp_path / "models"), tmp_path / "core")
    try:
        with serve(fixture / "archive") as models:
            root = source_tree(tmp_path / "source", manifest, trust, models.url)
            with serve(tmp_path / "source") as metadata:
                bootstrap = SourceBootstrap(
                    name="Explicit source",
                    root=base64.b64encode(root).decode(),
                    endpoints=(metadata.url,),
                )
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=application(host)),
                    base_url="http://localhost",
                ) as client:
                    review = await enroll(client, encode_record(bootstrap))
                    refresh = DiscoveryRequest(
                        owner_id="library",
                        operation_id="refresh",
                        kind="refresh",
                        source_id=review.source_id,
                    )
                    assert (
                        await metadata_settled(
                            client, (await submit(client, refresh)).job_id
                        )
                    ).state == "succeeded"
                    selected = {
                        "owner_id": "library",
                        "operation_id": "selected",
                        "kind": "model_metadata",
                        "source_id": review.source_id,
                        "entry_id": "community-model",
                        "manifest_id": manifest,
                    }
                    assert (
                        await client.post("/preservation/discovery/jobs", json=selected)
                    ).status_code == 422
                    denied = await submit(
                        client,
                        DiscoveryRequest.model_validate_json(
                            json.dumps({**selected, "accept_source_keys": True})
                        ),
                    )
                    assert (
                        await metadata_settled(client, denied.job_id)
                    ).state == "failed"
                    assert not models.requests
                    attempts = tuple(metadata.requests)
                    removed = await client.delete(
                        "/preservation/discovery/sources/" + review.source_id
                    )
                    assert removed.status_code == 204
                    assert not (await host.discovery.search(DiscoveryQuery())).results
                    assert (await host.discovery.page("library", 0)).sources[
                        0
                    ].root_version == 1
                    await client.post(
                        "/preservation/discovery/sources",
                        json={
                            "review_id": review.review_id,
                            "confirmed": True,
                            "allow_http": True,
                            "allow_private": True,
                        },
                    )
                    assert (await host.discovery.search(DiscoveryQuery())).results
                    assert tuple(metadata.requests) == attempts
                    assert not (await host.page()).locations
    finally:
        await host.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["refresh", "model_metadata"])
async def test_terminal_discovery_receipt_follows_cleanup_and_source_unlock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: Literal["refresh", "model_metadata"],
) -> None:
    """A terminal receipt permits immediate source withdrawal after real cleanup."""
    fixture = tmp_path / "model"
    fixture.mkdir()
    _, manifest, trust = public_kit(fixture)
    host = PreservationHost(ModelStore(tmp_path / "models"), tmp_path / "core")
    cleanup_started = threading.Event()
    release_cleanup = threading.Event()
    original_unlink = Path.unlink

    def held_cleanup(path: Path, missing_ok: bool = False) -> None:
        if path.parent == host.root / "discovery-requests":
            cleanup_started.set()
            assert release_cleanup.wait(30), "test must release owned request cleanup"
        original_unlink(path, missing_ok=missing_ok)

    try:
        with serve(fixture / "archive") as models:
            root = source_tree(tmp_path / "source", manifest, trust, models.url)
            with serve(tmp_path / "source") as metadata:
                bootstrap = SourceBootstrap(
                    name="Withdrawal after cleanup",
                    root=base64.b64encode(root).decode(),
                    endpoints=(metadata.url,),
                )
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=application(host)),
                    base_url="http://localhost",
                ) as client:
                    review = await enroll(client, encode_record(bootstrap))
                    monkeypatch.setattr(Path, "unlink", held_cleanup)
                    selection = DiscoveryRequest(
                        owner_id="library",
                        operation_id="cleanup-order",
                        kind=kind,
                        source_id=review.source_id,
                        entry_id="community-model"
                        if kind == "model_metadata"
                        else None,
                        manifest_id=manifest if kind == "model_metadata" else None,
                        accept_source_keys=kind == "model_metadata",
                    )
                    submitted = await submit(client, selection)
                    assert await asyncio.to_thread(cleanup_started.wait, 30)
                    observed = await client.get(
                        "/preservation/discovery/jobs/" + submitted.job_id
                    )
                    assert observed.json()["state"] == "running", observed.text
                    busy = await client.delete(
                        "/preservation/discovery/sources/" + review.source_id
                    )
                    assert busy.status_code == 409
                    assert busy.json()["detail"] == "discovery_busy_try_again"
                    release_cleanup.set()
                    completed = await metadata_settled(client, submitted.job_id)
                    assert completed.state == (
                        "succeeded" if kind == "refresh" else "failed"
                    )
                    removed = await client.delete(
                        "/preservation/discovery/sources/" + review.source_id
                    )
                    assert removed.status_code == 204, removed.text
                    assert not models.requests
    finally:
        release_cleanup.set()
        await host.close()


@pytest.mark.asyncio
async def test_source_review_upload_caps_and_all_operator_gates(tmp_path: Path) -> None:
    host = PreservationHost(ModelStore(tmp_path / "models"), tmp_path / "core")
    app = application(host)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://localhost"
        ) as client:
            oversized = await client.post(
                "/preservation/kit-uploads",
                json={"purpose": "source", "size_bytes": 256 * 1024 + 1},
            )
            assert oversized.status_code == 409
            assert tuple(host.uploads.root.iterdir()) == ()
        requests = (
            ("POST", "/preservation/discovery/reviews", {"upload_id": "a" * 32}),
            (
                "POST",
                "/preservation/discovery/sources",
                {"review_id": "a" * 64, "confirmed": True},
            ),
            ("DELETE", "/preservation/discovery/sources/" + "a" * 64, None),
            ("GET", "/preservation/discovery?owner_id=library", None),
            ("POST", "/preservation/discovery/search", {"query": "private query"}),
            (
                "POST",
                "/preservation/discovery/jobs",
                {
                    "owner_id": "library",
                    "operation_id": "blocked",
                    "kind": "refresh",
                    "source_id": "a" * 64,
                },
            ),
            ("GET", "/preservation/discovery/jobs/" + "a" * 64, None),
            ("POST", "/preservation/discovery/jobs/" + "a" * 64 + "/cancel", None),
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("192.0.2.10", 1234)),
            base_url="http://localhost",
        ) as remote:
            for method, path, body in requests:
                response = await remote.request(method, path, json=body)
                assert response.status_code == 403, response.text
        assert host.journal.discovery_jobs("library") == ()
        schema = cast(dict[str, JsonValue], app.openapi())
        paths = schema["paths"]
        assert isinstance(paths, dict)
        for path, methods in paths.items():
            if "/discovery" in path:
                assert isinstance(methods, dict)
                for operation in methods.values():
                    assert isinstance(operation, dict)
                    assert (
                        operation["summary"]
                        and operation["description"]
                        and operation["tags"]
                    )
    finally:
        await host.close()


@pytest.mark.asyncio
async def test_queued_cancel_and_forced_stop_hold_metadata_ownership(
    tmp_path: Path,
) -> None:
    from contextlib import suppress

    fixture = tmp_path / "model"
    fixture.mkdir()
    _, manifest, trust = public_kit(fixture)
    root = source_tree(tmp_path / "source", manifest, trust, "https://unused.invalid")
    entered, release = threading.Event(), threading.Event()

    class SlowHandler(SimpleHTTPRequestHandler):
        def log_message(self, *args: object, **kwargs: object) -> None:
            """Do not retain fixture addresses."""

        def do_GET(self) -> None:
            """Stall actual HTTP before headers to exercise the parent stop ladder."""
            entered.set()
            release.wait(30)
            with suppress(BrokenPipeError, ConnectionResetError):
                super().do_GET()

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(SlowHandler, directory=str(tmp_path / "source"))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host = PreservationHost(ModelStore(tmp_path / "models"), tmp_path / "core")
    try:
        bootstrap = SourceBootstrap(
            name="Stalled source",
            root=base64.b64encode(root).decode(),
            endpoints=(f"http://127.0.0.1:{server.server_port}",),
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application(host)),
            base_url="http://localhost",
        ) as client:
            review = await enroll(client, encode_record(bootstrap))
            selection = DiscoveryRequest(
                owner_id="library",
                operation_id="running",
                kind="refresh",
                source_id=review.source_id,
            )
            first = await submit(client, selection)
            async with asyncio.timeout(10):
                while not entered.is_set():
                    await asyncio.sleep(0.025)
            second = await submit(
                client, selection.model_copy(update={"operation_id": "queued"})
            )
            await client.post(
                "/preservation/discovery/jobs/" + second.job_id + "/cancel"
            )
            stopped = await client.post(
                "/preservation/discovery/jobs/" + first.job_id + "/cancel"
            )
            assert stopped.status_code == 200
            assert (await metadata_settled(client, first.job_id)).state == "interrupted"
            assert (await metadata_settled(client, second.job_id)).state == "cancelled"
            # Only a source-cache reservation exists; no model/signing enrollment.
            assert not (await host.page()).locations
            for child in psutil.Process().children():
                with suppress(psutil.NoSuchProcess):
                    assert first.job_id not in " ".join(child.cmdline())
    finally:
        release.set()
        await host.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.asyncio
async def test_metadata_worker_exits_when_parent_pipe_is_lost(tmp_path: Path) -> None:
    from foxlight_model_preservation.discovery import (
        bootstrap_identity,
        follow_source,
        list_sources,
        refresh_source,
    )

    from skulk.store.discovery_worker import DiscoveryWorkerRequest
    from skulk.store.preservation_host import preservation_worker_command

    fixture = tmp_path / "model"
    fixture.mkdir()
    _, manifest, trust = public_kit(fixture)
    root = source_tree(tmp_path / "source", manifest, trust, "https://unused.invalid")
    with serve(tmp_path / "source") as metadata:
        bootstrap = SourceBootstrap(
            name="Parent lifetime",
            root=base64.b64encode(root).decode(),
            endpoints=(metadata.url,),
        )
        cache = tmp_path / "cache"
        source = follow_source(
            cache,
            encode_record(bootstrap),
            bootstrap_identity(bootstrap),
            allow_http=True,
            allow_private=True,
        )
        accepted = refresh_source(cache, source.source_id)
        selection = DiscoveryRequest(
            owner_id="lifetime",
            operation_id="parent-eof",
            kind="refresh",
            source_id=source.source_id,
        )
        request_path = tmp_path / "request.json"
        request_path.write_text(
            DiscoveryWorkerRequest(cache=cache, selection=selection).model_dump_json()
        )
        process = await asyncio.create_subprocess_exec(
            *preservation_worker_command(request_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        assert process.stdin is not None
        process.stdin.close()
        try:
            await asyncio.wait_for(process.wait(), 5)
            assert process.returncode == 75
            assert list_sources(cache)[0].catalog_id == accepted.catalog_id
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()


@pytest.mark.asyncio
async def test_discovered_skulk_model_restores_companions_with_separate_code_consent(
    tmp_path: Path,
) -> None:
    import shutil

    from foxlight_model_preservation.publication import (
        PUBLISHER_NOTICE_PATH,
        PublicationRequest,
        PublicMemberTerms,
        PublisherNotice,
    )
    from foxlight_model_preservation.storage import verify_store

    from skulk.store.installed_cards import read_installed_card
    from skulk.store.preservation_contracts import (
        PreservationJob,
        PreservationPlanRequest,
    )
    from skulk.store.preservation_provenance import retained_recovery_card
    from skulk.store.tests.test_preservation_delivery import metadata_upload
    from skulk.store.tests.test_preservation_export import (
        make_preservation_installation,
    )

    installed = make_preservation_installation(
        tmp_path / "original", "assistant", code=True
    )
    producer = PreservationHost(
        ModelStore(installed.primary.parent), tmp_path / "producer"
    )
    try:
        await producer.setup_device()
        await producer.setup_publisher()
        model = (await producer.page()).models[0]
        preserve = await producer.plan(
            PreservationPlanRequest(
                kind="preserve",
                location_id="device",
                model_id=model.model_id,
                generation_id=model.generation_id,
            )
        )
        kept = await settled(
            producer,
            producer.start(
                PreservationStart(plan_id=preserve.plan_id, operation_id="keep")
            ).job_id,
        )
        assert kept.state == "succeeded" and kept.copy_id
        source = await producer.publication_source(kept.copy_id)
        terms = PublicationRequest(
            name="Complete community assistant",
            workloads=source.workloads,
            runtime_requirements=source.runtime_requirements,
            members=tuple(
                PublicMemberTerms(
                    member_id=member.member_id,
                    license_id="test-only",
                    terms_reference="LICENSE",
                    decision="Synthetic fixture redistribution",
                    notice_paths=member.notice_paths or (PUBLISHER_NOTICE_PATH,),
                )
                for member in source.members
            ),
            notices=tuple(
                PublisherNotice(
                    member_id=member.member_id,
                    text="Synthetic fixture license and attribution.\n",
                )
                for member in source.members
                if not member.notice_paths
            ),
        )
        upload = metadata_upload(producer, encode_record(terms), "publication")
        publish = await producer.plan(
            PreservationPlanRequest(
                kind="publish",
                location_id="device",
                copy_id=kept.copy_id,
                upload_id=upload,
            )
        )
        published = await settled(
            producer,
            producer.start(
                PreservationStart(
                    plan_id=publish.plan_id,
                    operation_id="publish",
                    public_participation=True,
                )
            ).job_id,
        )
        assert published.state == "succeeded" and published.copy_id
        copy = producer.journal.copy(published.copy_id)
        trust = producer.public_trust(copy.copy_id)
        source_path = next((producer.root / "public-copies").iterdir())
        original = verify_store(source_path, copy.manifest_id, trust)
        assert len(original.members) == 2
        shutil.copytree(source_path, tmp_path / "independent-bytes")
    finally:
        await producer.close()
    shutil.rmtree(tmp_path / "producer")
    shutil.rmtree(tmp_path / "original")
    # The normal ModelStoreServer creates its empty authoritative store first.
    (tmp_path / "models").mkdir()
    receiver = PreservationHost(
        ModelStore(tmp_path / "models"),
        tmp_path / "receiver",
        custom_cards=tmp_path / "cards",
    )
    try:
        with serve(tmp_path / "independent-bytes") as models:
            root = source_tree(tmp_path / "source", copy.manifest_id, trust, models.url)
            with serve(tmp_path / "source") as metadata:
                bootstrap = SourceBootstrap(
                    name="Independent operator",
                    root=base64.b64encode(root).decode(),
                    endpoints=(metadata.url,),
                )
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=application(receiver)),
                    base_url="http://localhost",
                ) as client:
                    review = await enroll(client, encode_record(bootstrap))
                    refresh = DiscoveryRequest(
                        owner_id="new-library",
                        operation_id="refresh",
                        kind="refresh",
                        source_id=review.source_id,
                    )
                    assert (
                        await metadata_settled(
                            client, (await submit(client, refresh)).job_id
                        )
                    ).state == "succeeded"
                    selected = DiscoveryRequest(
                        owner_id="new-library",
                        operation_id="selected",
                        kind="model_metadata",
                        source_id=review.source_id,
                        entry_id="community-model",
                        manifest_id=copy.manifest_id,
                        accept_source_keys=True,
                        allow_http=True,
                        allow_private=True,
                    )
                    prepared = await metadata_settled(
                        client, (await submit(client, selected)).job_id
                    )
                    assert prepared.state == "succeeded" and prepared.upload_id
                    # Selection fetches only signed metadata, never model objects.
                    assert models.requests == ["/" + manifest_path(copy.manifest_id)]
                    receive_response = await client.post(
                        "/preservation/plans",
                        json={
                            "kind": "receive",
                            "location_id": "invitation",
                            "upload_id": prepared.upload_id,
                            "accepted_key_ids": [key.key_id for key in trust.keys],
                            "allow_http": True,
                            "allow_private": True,
                        },
                    )
                    assert receive_response.status_code == 200, receive_response.text
                    receive = PreservationPlan.model_validate_json(
                        receive_response.text
                    )
                    started = await client.post(
                        "/preservation/jobs",
                        json={
                            "plan_id": receive.plan_id,
                            "operation_id": "receive",
                            "owner_id": "new-library-receiving",
                            "accept_owner_capture": True,
                        },
                    )
                    received = await settled(
                        receiver,
                        PreservationJob.model_validate_json(started.text).job_id,
                    )
                    assert received.state == "succeeded" and received.copy_id
                    assert not receiver.store.list_models()
        # Both suppliers are gone before runtime admission and recovery.
        assert not (tmp_path / "producer").exists()
        receiving_copy = receiver.journal.copy(received.copy_id)
        # Reopen the HTTP client; no page owns the retained copy or job.
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application(receiver)),
            base_url="http://localhost",
        ) as reopened:
            restore_response = await reopened.post(
                "/preservation/plans",
                json={
                    "kind": "recover",
                    "location_id": receiving_copy.location_id,
                    "copy_id": receiving_copy.copy_id,
                    "historical": True,
                },
            )
            assert restore_response.status_code == 200, restore_response.text
            restore = PreservationPlan.model_validate_json(restore_response.text)
            assert restore.member_count == 2 and restore.requires_code_permission
            refused = await reopened.post(
                "/preservation/jobs",
                json={
                    "plan_id": restore.plan_id,
                    "operation_id": "restore",
                    "accept_owner_capture": True,
                },
            )
            assert (
                refused.status_code == 409
                and "repository_code_permission_required" in refused.text
            )
            assert not receiver.store.list_models()
            allowed = await reopened.post(
                "/preservation/jobs",
                json={
                    "plan_id": restore.plan_id,
                    "operation_id": "restore",
                    "accept_owner_capture": True,
                    "allow_repository_code": True,
                },
            )
            restored = await settled(
                receiver, PreservationJob.model_validate_json(allowed.text).job_id
            )
            assert restored.state == "succeeded", restored
            assert any(member.retained_card is not None for member in original.members)
            for member in original.members:
                alias = member.origin_repository
                assert alias in {"example/base", "example/companion"}
                path = receiver.store.get_store_path(alias)
                assert path is not None
                record = read_installed_card(path)
                assert record is not None and record.preservation is not None
                assert record.preservation.manifest_id == copy.manifest_id
                from skulk.store.installed_cards import verify_installed_card

                assert verify_installed_card(path, record, verify_hashes=True)
                if member.retained_card is not None:
                    retained = retained_recovery_card(path, record)
                    assert (
                        retained is not None
                        and retained.model_dump(mode="json") == member.retained_card
                    )
    finally:
        await receiver.close()


@pytest.mark.asyncio
async def test_parent_deadline_stops_child_before_terminal_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    from contextlib import suppress

    from skulk.store import discovery_host

    host = PreservationHost(ModelStore(tmp_path / "models"), tmp_path / "core")
    # A controlled nonresponsive process tests the supervisor independently of
    # HTTP/DNS timing; the user-facing journey uses the actual known worker.
    monkeypatch.setattr(discovery_host, "METADATA_WORKER_SECONDS", 0.2)

    def blocked_command(path: Path) -> tuple[str, ...]:
        return sys.executable, "-c", "import time; time.sleep(30)", str(path)

    monkeypatch.setattr(host.discovery, "command", blocked_command)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application(host)),
            base_url="http://localhost",
        ) as client:
            job = await submit(
                client,
                DiscoveryRequest(
                    owner_id="deadline",
                    operation_id="finite",
                    kind="refresh",
                    source_id="a" * 64,
                ),
            )
            stopped = await metadata_settled(client, job.job_id)
            assert (
                stopped.state == "interrupted"
                and stopped.error_code == "discovery_deadline_exceeded"
            )
            for child in psutil.Process().children():
                with suppress(psutil.NoSuchProcess):
                    assert job.job_id not in " ".join(child.cmdline())
    finally:
        await host.close()


@pytest.mark.asyncio
async def test_search_projection_bounds_preserve_full_signed_statements(
    tmp_path: Path,
) -> None:
    from foxlight_model_preservation.discovery import (
        bootstrap_identity,
        follow_source,
        refresh_source,
        search_sources,
    )

    fixture = tmp_path / "model"
    fixture.mkdir()
    _, manifest, trust = public_kit(fixture)
    host = PreservationHost(ModelStore(tmp_path / "models"), tmp_path / "core")
    text = "🦊" * 800
    try:
        identities = tuple(digest(f"model-{index}".encode()) for index in range(3))
        for index in range(16):
            path = tmp_path / f"source-{index}"
            root = source_tree(
                path,
                manifest,
                trust,
                "https://unused.invalid",
                extra_manifests=identities,
                description=text,
                name="🦊" * 200,
            )
            bootstrap = SourceBootstrap(
                name="🦊" * 200,
                root=base64.b64encode(root).decode(),
                endpoints=(str(path),),
            )
            source = follow_source(
                host.discovery.cache,
                encode_record(bootstrap),
                bootstrap_identity(bootstrap),
                allow_local=True,
            )
            refresh_source(host.discovery.cache, source.source_id)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application(host)),
            base_url="http://localhost",
        ) as client:
            response = await client.post("/preservation/discovery/search", json={})
            assert response.status_code == 200 and len(response.content) < 48000, (
                response.text
            )
            page = DiscoveryResults.model_validate_json(response.text)
            assert len(page.results) == 4 and page.next_offset is None
            assert all(
                len(result.suppliers) == 4 and result.supplier_count == 16
                for result in page.results
            )
            assert all(
                supplier.summary.endswith("…")
                for result in page.results
                for supplier in result.suppliers
            )
            following = await client.post(
                "/preservation/discovery/search",
                json={"query": page.results[0].manifest_id, "supplier_offset": 2},
            )
            suppliers = DiscoveryResults.model_validate_json(following.text).results[0]
            assert len(suppliers.suppliers) == 4 and suppliers.supplier_count == 16
            source_page = await client.get(
                "/preservation/discovery", params={"owner_id": "projection"}
            )
            assert source_page.status_code == 200 and len(source_page.content) < 48000
            assert (
                len(DiscoveryPage.model_validate_json(source_page.text).sources) == 16
            )
            seen = set[str]()
            for offset in range(0, 16, 4):
                response = await client.post(
                    "/preservation/discovery/search",
                    json={
                        "query": page.results[0].manifest_id,
                        "supplier_offset": offset,
                    },
                )
                choice = DiscoveryResults.model_validate_json(response.text).results[0]
                seen.update(supplier.source_id for supplier in choice.suppliers)
            assert len(seen) == 16
        originals = search_sources(host.discovery.cache, "", limit=1000)
        assert all(
            location.description and location.description.summary == text
            for result in originals
            for location in result.locations
        )
    finally:
        await host.close()
