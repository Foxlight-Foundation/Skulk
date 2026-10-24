"""Real protected Core jobs use only the chosen proxy and retain original evidence."""

from __future__ import annotations

import asyncio
import base64
import json
import re
import shutil
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from foxlight_model_preservation.discovery_models import SourceBootstrap
from foxlight_model_preservation.invitations import create_invitation
from foxlight_model_preservation.models import Recipe, TrustProfile
from foxlight_model_preservation.signing import encode_record, public_record
from foxlight_model_preservation.storage import (
    manifest_path,
    pack,
    restore,
    verify_store,
)
from pydantic import JsonValue

from skulk.store.discovery_contracts import (
    DiscoveryPage,
    DiscoveryRequest,
    DiscoveryResults,
    SourceReview,
)
from skulk.store.model_store import ModelStore
from skulk.store.preservation_contracts import (
    I2PProxySettings,
    KitUploadStatus,
    PreservationPlan,
    PreservationPlanRequest,
    PreservationStart,
    ReceivingPolicy,
)
from skulk.store.preservation_host import PreservationHost
from skulk.store.preservation_journal import PreservationHostError
from skulk.store.tests.test_discovery import (
    application,
    metadata_settled,
    serve,
    source_tree,
    submit,
)
from skulk.store.tests.test_preservation_delivery import metadata_upload
from skulk.store.tests.test_preservation_sharing import settled


@contextmanager
def proxy(
    root: Path,
    *,
    blocked: threading.Event | None = None,
    release: threading.Event | None = None,
    ranges: list[int] | None = None,
) -> Iterator[tuple[int, list[str]]]:
    """A real loopback HTTP proxy fixture accepting only the selected I2P name."""
    requests: list[str] = []
    expected = base64.b32encode(b"i" * 32).decode().rstrip("=").lower() + ".b32.i2p"

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *args: object, **kwargs: object) -> None:
            """Keep destination/path history inside this explicit fixture only."""

        def do_GET(self) -> None:
            """Record the actual absolute-form request and refuse every other route."""
            requests.append(self.path)
            parsed = urlsplit(self.path)
            if parsed.hostname != expected or parsed.scheme != "http":
                self.send_error(403)
                return
            self.path = parsed.path
            if release is not None and self.path.startswith("/model/objects/"):
                body = (root / self.path.lstrip("/")).read_bytes()
                requested = self.headers.get("Range")
                matched = re.fullmatch(r"bytes=([0-9]+)-", requested or "")
                offset = int(matched[1]) if matched else 0
                if ranges is not None and matched:
                    ranges.append(offset)
                self.send_response(206 if matched else 200)
                self.send_header("Content-Length", str(len(body) - offset))
                if matched:
                    self.send_header(
                        "Content-Range", f"bytes {offset}-{len(body) - 1}/{len(body)}"
                    )
                self.end_headers()
                try:
                    if (
                        len(body) > 65536
                        and blocked is not None
                        and not blocked.is_set()
                    ):
                        self.wfile.write(body[offset : offset + 65536])
                        self.wfile.flush()
                        blocked.set()
                        release.wait(10)
                        self.wfile.write(body[offset + 65536 :])
                    else:
                        self.wfile.write(body[offset:])
                except OSError:
                    pass
                return
            super().do_GET()

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(Handler, directory=str(root))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port, requests
    finally:
        if release is not None:
            release.set()
        server.shutdown()
        server.server_close()
        thread.join(5)


def archive(root: Path) -> tuple[str, TrustProfile]:
    """Capture synthetic weights plus a required companion with opaque card data."""
    key = Ed25519PrivateKey.generate()
    sources: dict[str, Path] = {}
    members: list[dict[str, JsonValue]] = []
    for name, role, deps in (
        ("weights", "primary", ["tokenizer"]),
        ("tokenizer", "tokenizer", []),
    ):
        path = root / name
        path.mkdir()
        (path / "data.bin").write_bytes(
            name.encode() * (50000 if name == "weights" else 1000)
        )
        (path / "LICENSE").write_text("Synthetic fixture only.\n")
        sources[name] = path
        dependencies: list[JsonValue] = [*deps]
        members.append(
            {
                "member_id": name,
                "role": role,
                "format": "test-fixture",
                "origin_repository": "test/" + name,
                "origin_revision": "a" * 40,
                "artifact_identity": "fixture:" + name,
                "dependencies": dependencies,
                "retained_card": {"future_property": {"keep": [1, 2, 3]}},
                "intake": {
                    "audience": "public",
                    "license_id": "test-only",
                    "terms_reference": "LICENSE",
                    "decision": "synthetic",
                    "notice_paths": ["LICENSE"],
                },
                "files": [{"path": "data.bin"}, {"path": "LICENSE"}],
            }
        )
    recipe = Recipe.model_validate_json(
        json.dumps(
            {
                "schema_version": "foxlight.model-preservation.v1",
                "name": "I2P fixture",
                "root_member": "weights",
                "workloads": ["fixture"],
                "runtime_requirements": ["future-reader"],
                "members": members,
            }
        )
    )
    trust = TrustProfile(
        schema_version="foxlight.model-preservation.trust.v1",
        keys=(public_record(key),),
        threshold=1,
        as_of="2026-10-24T00:00:00Z",
    )
    identity = pack(recipe, sources, root / "model", (key,))
    for path in sources.values():
        shutil.rmtree(path)
    return identity, trust


async def follow_i2p(
    client: httpx.AsyncClient, encoded: bytes, port: int
) -> SourceReview:
    """Use public upload/review/follow routes with one explicit proxy-only grant."""
    created = await client.post(
        "/preservation/kit-uploads",
        json={"size_bytes": len(encoded), "purpose": "source"},
    )
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
    review = SourceReview.model_validate_json(reviewed.text)
    followed = await client.post(
        "/preservation/discovery/sources",
        json={
            "review_id": review.review_id,
            "confirmed": True,
            "allow_i2p": True,
            "i2p_proxy": {"host": "127.0.0.1", "port": port},
        },
    )
    assert followed.status_code == 200, followed.text
    return review


@pytest.mark.asyncio
async def test_real_i2p_core_source_receive_restart_and_offline_recovery(
    tmp_path: Path,
) -> None:
    root = tmp_path / "http-root"
    root.mkdir()
    identity, trust = archive(root)
    original = (root / "model" / manifest_path(identity)).read_bytes()
    hostname = base64.b32encode(b"i" * 32).decode().rstrip("=").lower() + ".b32.i2p"
    base = "http://" + hostname
    signed_root = source_tree(root / "source", identity, trust, base + "/model")
    host_root = tmp_path / "core"
    models = tmp_path / "models"
    host = PreservationHost(ModelStore(models), host_root)
    copy_id = ""
    policy_id = ""
    try:
        with proxy(root) as (port, requests), serve(tmp_path) as trap:
            bootstrap = SourceBootstrap(
                name="I2P source",
                root=base64.b64encode(signed_root).decode(),
                endpoints=(trap.url, base + "/source"),
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=application(host)),
                base_url="http://localhost",
            ) as client:
                review = await follow_i2p(client, encode_record(bootstrap), port)
                assert not requests and not trap.requests
                page = DiscoveryPage.model_validate_json(
                    (
                        await client.get(
                            "/preservation/discovery", params={"owner_id": "library"}
                        )
                    ).text
                )
                source = page.sources[0]
                assert (
                    source.transport == "i2p"
                    and source.i2p_proxy == I2PProxySettings(port=port)
                )
                assert source.i2p_policy_id
                policy_id = source.i2p_policy_id
                selection = DiscoveryRequest(
                    owner_id="library",
                    operation_id="refresh",
                    kind="refresh",
                    source_id=review.source_id,
                    transport="i2p",
                    allow_i2p=True,
                    i2p_policy_id=policy_id,
                )
                refreshed = await metadata_settled(
                    client, (await submit(client, selection)).job_id
                )
                assert refreshed.state == "succeeded", refreshed
                # An old HTTP client and a stale proxy policy both refuse before
                # a worker can contact any supplier or the direct-network trap.
                before = len(requests)
                for rejected in (
                    DiscoveryRequest(
                        owner_id="library",
                        operation_id="old-http",
                        kind="refresh",
                        source_id=review.source_id,
                    ),
                    selection.model_copy(
                        update={
                            "operation_id": "stale-policy",
                            "i2p_policy_id": "f" * 64,
                        }
                    ),
                ):
                    refused = await metadata_settled(
                        client, (await submit(client, rejected)).job_id
                    )
                    assert refused.state == "failed"
                    assert refused.error_code in {
                        "source_transport_changed",
                        "source_i2p_policy_changed",
                    }
                assert len(requests) == before and not trap.requests
                results = DiscoveryResults.model_validate_json(
                    (
                        await client.post(
                            "/preservation/discovery/search", json={"query": "offline"}
                        )
                    ).text
                )
                assert results.results[0].manifest_id == identity
                assert results.results[0].suppliers[0].transport == "i2p"
                assert len(requests) == before and not trap.requests
                metadata = await metadata_settled(
                    client,
                    (
                        await submit(
                            client,
                            DiscoveryRequest(
                                owner_id="library",
                                operation_id="metadata",
                                kind="model_metadata",
                                source_id=review.source_id,
                                entry_id="community-model",
                                manifest_id=identity,
                                accept_source_keys=True,
                                transport="i2p",
                                allow_i2p=True,
                                i2p_policy_id=policy_id,
                            ),
                        )
                    ).job_id,
                )
                assert metadata.state == "succeeded" and metadata.upload_id, metadata
                shutil.rmtree(root / "source")
                planned = await client.post(
                    "/preservation/plans",
                    json={
                        "kind": "receive",
                        "location_id": "invitation",
                        "upload_id": metadata.upload_id,
                        "accepted_key_ids": [key.key_id for key in trust.keys],
                        "allow_i2p": True,
                        "i2p_proxy": {"host": "127.0.0.1", "port": port},
                    },
                )
                assert planned.status_code == 200, planned.text
                plan = PreservationPlan.model_validate_json(planned.text)
                assert plan.receiving and plan.receiving.transport == "i2p"
                completed = await settled(
                    host,
                    host.start(
                        PreservationStart(
                            plan_id=plan.plan_id,
                            operation_id="receive",
                            owner_id="library",
                            accept_owner_capture=True,
                        )
                    ).job_id,
                )
                assert completed.state == "succeeded" and completed.copy_id, completed
                before_preview = len(requests)
                retained = await client.get(
                    "/preservation/jobs/" + completed.job_id + "/invitation"
                )
                assert retained.status_code == 200, retained.text
                assert retained.json()["receiving"] == plan.receiving.model_dump(
                    mode="json"
                )
                fresh = await client.get(
                    "/preservation/kit-uploads/" + metadata.upload_id + "/invitation"
                )
                assert fresh.status_code == 200 and "receiving" not in fresh.json()
                assert len(requests) == before_preview
                copy_id = completed.copy_id
                assert not trap.requests
                assert all(urlsplit(value).hostname == hostname for value in requests)
    finally:
        await host.close()
    shutil.rmtree(root)
    host = PreservationHost(ModelStore(models), host_root)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application(host)),
            base_url="http://localhost",
        ) as client:
            page = DiscoveryPage.model_validate_json(
                (
                    await client.get(
                        "/preservation/discovery", params={"owner_id": "library"}
                    )
                ).text
            )
            assert (
                page.sources[0].transport == "i2p"
                and page.sources[0].i2p_policy_id == policy_id
            )
            matches = DiscoveryResults.model_validate_json(
                (
                    await client.post(
                        "/preservation/discovery/search", json={"query": "offline"}
                    )
                ).text
            )
            assert matches.results[0].manifest_id == identity
        copy = host.journal.copy(copy_id)
        # The public copy remains portable after the Core/source/proxy are lost.
        assert copy.manifest_id == identity
        _, payload = host.journal.plan(completed.plan_id)
        destination = Path(str(payload["destination"]))
        verified = verify_store(destination, identity, trust)
        assert len(verified.members) == 2
        for member in verified.members:
            assert member.retained_card is not None
            assert member.retained_card["future_property"] == {"keep": [1, 2, 3]}
        assert (destination / manifest_path(identity)).read_bytes() == original
        restore(destination, identity, trust, tmp_path / "restored")
        assert (
            tmp_path / "restored/members/tokenizer/data.bin"
        ).read_bytes() == b"tokenizer" * 1000
    finally:
        await host.close()


@pytest.mark.parametrize("host", ["localhost", "0.0.0.0", "192.0.2.1", "proxy.example"])
def test_i2p_proxy_is_always_numeric_loopback(host: str) -> None:
    with pytest.raises(ValueError):
        I2PProxySettings.model_validate_json(json.dumps({"host": host, "port": 4444}))


def test_existing_http_request_replay_projections_exclude_new_i2p_defaults() -> None:
    request = DiscoveryRequest(
        owner_id="library", operation_id="legacy", kind="refresh", source_id="a" * 64
    )
    assert not {"allow_i2p", "i2p_policy_id", "transport"}.intersection(
        request.model_dump(mode="json")
    )
    policy = ReceivingPolicy(
        mirrors=("https://model.example",), max_bytes=1000, deadline_seconds=10
    )
    assert not {"allow_i2p", "i2p_proxy", "transport"}.intersection(
        policy.model_dump(mode="json")
    )
    with pytest.raises(ValueError):
        PreservationPlanRequest(
            kind="receive", location_id="invitation", allow_i2p=True
        )
    with pytest.raises(ValueError):
        DiscoveryRequest(
            owner_id="library",
            operation_id="mixed",
            kind="refresh",
            source_id="a" * 64,
            transport="i2p",
            allow_i2p=True,
            i2p_policy_id="b" * 64,
            allow_peer_network=True,
        )


@pytest.mark.asyncio
async def test_i2p_receive_stop_restart_and_exact_policy_resume(tmp_path: Path) -> None:
    """Stop a blocked real child; restart grants no replay and resume keeps routing/budget."""
    root = tmp_path / "publisher"
    root.mkdir()
    identity, trust = archive(root)
    blocked, release = threading.Event(), threading.Event()
    ranges: list[int] = []
    models = tmp_path / "models"
    host_root = tmp_path / "core"
    host = PreservationHost(ModelStore(models), host_root)
    hostname = base64.b32encode(b"i" * 32).decode().rstrip("=").lower() + ".b32.i2p"
    try:
        with proxy(root, blocked=blocked, release=release, ranges=ranges) as (
            port,
            requests,
        ):
            invitation = create_invitation(
                root / "model",
                identity,
                trust,
                ("https://direct-never-use.invalid", "http://" + hostname + "/model"),
            )
            upload = metadata_upload(host, encode_record(invitation), "invitation")
            connection = I2PProxySettings(port=port)
            plan = await host.plan(
                PreservationPlanRequest(
                    kind="receive",
                    location_id="invitation",
                    upload_id=upload,
                    accepted_key_ids=tuple(key.key_id for key in trust.keys),
                    allow_i2p=True,
                    i2p_proxy=connection,
                )
            )
            assert plan.receiving is not None
            assert plan.receiving.mirrors == ("http://" + hostname + "/model",)
            job = host.start(
                PreservationStart(
                    plan_id=plan.plan_id,
                    operation_id="receive-i2p",
                    owner_id="library",
                    accept_owner_capture=True,
                )
            )
            assert await asyncio.to_thread(blocked.wait, 10)
            partials = host_root / "work" / job.job_id / "public-copy/.partial"
            async with asyncio.timeout(5):
                while not any(p.stat().st_size > 0 for p in partials.glob("*")):
                    await asyncio.sleep(0.025)
            async with asyncio.timeout(10):
                await host.cancel(job.job_id)
                stopped = await settled(host, job.job_id)
            assert stopped.state == "cancelled"
            assert not (await host.page()).public_copies
            assert any(p.stat().st_size > 0 for p in partials.glob("*"))
            attempts = len(requests)
            release.set()
            await host.close()
            host = PreservationHost(ModelStore(models), host_root)
            await asyncio.sleep(0.1)
            assert (
                len(requests) == attempts
                and host.journal.job(job.job_id).state == "cancelled"
            )
            retained = host.receive_invitation(job.job_id)
            assert retained.receiving == plan.receiving
            assert retained.receiving is not None
            assert retained.receiving.i2p_proxy == connection
            assert len(requests) == attempts
            # A projection cannot renew consent: an unconfirmed resume still
            # refuses under the exact original policy below.
            with pytest.raises(
                PreservationHostError, match="receive_resume_i2p_policy_required"
            ):
                await host.plan(
                    PreservationPlanRequest(
                        kind="receive",
                        location_id="invitation",
                        resume_job_id=job.job_id,
                    )
                )
            with pytest.raises(
                PreservationHostError, match="receive_resume_i2p_policy_required"
            ):
                await host.plan(
                    PreservationPlanRequest(
                        kind="receive",
                        location_id="invitation",
                        resume_job_id=job.job_id,
                        allow_i2p=True,
                        i2p_proxy=I2PProxySettings(
                            port=port + 1 if port < 65535 else port - 1
                        ),
                    )
                )
            assert len(requests) == attempts
            review = await host.plan(
                PreservationPlanRequest(
                    kind="receive",
                    location_id="invitation",
                    resume_job_id=job.job_id,
                    allow_i2p=True,
                    i2p_proxy=connection,
                )
            )
            assert review.receiving == plan.receiving
            resumed = await host.resume_receive(
                job.job_id,
                PreservationStart(
                    plan_id=review.plan_id,
                    operation_id="resume-i2p",
                    owner_id="library",
                    accept_owner_capture=True,
                ),
            )
            assert resumed.job_id == job.job_id
            finished = await settled(host, job.job_id)
            assert finished.state == "succeeded" and finished.copy_id, finished
            assert any(offset > 0 for offset in ranges)
    finally:
        release.set()
        await host.close()
