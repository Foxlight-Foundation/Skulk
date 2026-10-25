"""Real retained-copy metadata, durable preparation, ranges and explicit removal."""

from __future__ import annotations

import asyncio
import hashlib
import shutil
import socket
import sqlite3
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import BinaryIO

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi import FastAPI
from foxlight_model_preservation import storage as archive_storage
from foxlight_model_preservation.kits import export_kit, import_kit
from foxlight_model_preservation.mirror_intake import retain_local
from foxlight_model_preservation.mirror_models import MirrorState, SourceGrant, Terms
from foxlight_model_preservation.mirror_state import MirrorLedger
from foxlight_model_preservation.models import TrustProfile
from foxlight_model_preservation.signing import (
    digest,
    encode_record,
    public_record,
    sign_manifest,
)
from foxlight_model_preservation.storage import manifest_path, restore, verify_store

from skulk.api.main import API
from skulk.api.preservation import create_preservation_router
from skulk.api.tests.test_mirrors_api import ready, settled
from skulk.store import mirror_kits
from skulk.store.mirror_contracts import MirrorJob, MirrorRequest, MirrorReview
from skulk.store.mirror_kits import KitSeal
from skulk.store.mirror_process import MirrorProcess
from skulk.store.preservation_host import PreservationHost
from skulk.store.preservation_journal import PreservationHostError, PreservationJournal
from skulk.store.tests.test_mirror_worker import archive, configuration
from skulk.store.tests.test_preservation_sharing import empty_host, import_copy


async def retained(tmp_path: Path) -> tuple[PreservationHost, str, TrustProfile, bytes]:
    """Retain a genuine signed closure, then destroy input and original copy records."""
    owner = empty_host(tmp_path)
    configured = await owner.mirrors.start(
        MirrorRequest(
            owner_id="library",
            operation_id="configure",
            kind="configure",
            configuration=configuration(),
        )
    )
    assert (await settled(owner, configured.job_id)).state == "succeeded"
    source, identity, trust = archive(tmp_path)
    wrapper = (source / manifest_path(identity)).read_bytes()
    kit = tmp_path / "input.foxkit"
    export_kit(source, identity, trust, kit)
    copy = await import_copy(owner, kit.read_bytes())
    selected = await owner.mirrors.page()
    job = await owner.mirrors.start(
        MirrorRequest(
            owner_id="library",
            operation_id="retain",
            kind="retain",
            policy_id=selected.policy_id,
            copy_id=copy.copy_id,
            manifest_id=identity,
            confirm_redistribution=True,
        )
    )
    assert (await settled(owner, job.job_id)).state == "succeeded"
    location = owner.journal.setting("location:" + copy.location_id)
    assert isinstance(location, dict) and isinstance(location["path"], str)
    shutil.rmtree(Path(location["path"]))
    with closing(
        sqlite3.connect(owner.root / "journal.sqlite3", isolation_level=None)
    ) as connection:
        _ = connection.execute("DELETE FROM copies WHERE id=?", (copy.copy_id,))
    for path in (source, tmp_path / "model", tmp_path / "tokenizer"):
        shutil.rmtree(path)
    kit.unlink()
    return owner, identity, trust, wrapper


def app(owner: PreservationHost) -> FastAPI:
    async def provider() -> PreservationHost:
        return owner

    application = FastAPI()
    application.include_router(
        create_preservation_router(provider, API.require_preservation_operator)
    )
    return application


@pytest.mark.asyncio
async def test_prepared_kit_survives_input_records_client_and_core_then_explicit_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner, identity, trust, wrapper = await retained(tmp_path)
    current = owner
    try:
        transport = httpx.ASGITransport(app=app(owner), client=("127.0.0.1", 1000))
        async with httpx.AsyncClient(
            transport=transport, base_url="http://localhost"
        ) as client:
            selected = await client.get("/preservation/mirror/models/" + identity)
            assert selected.status_code == 200, selected.text
            assert selected.json()["model"]["state"] == "retained"
            reviewed = await client.get(
                "/preservation/mirror/models/" + identity + "/review"
            )
            assert reviewed.status_code == 200, reviewed.text
            review = MirrorReview.model_validate_json(reviewed.text)
            assert review.name == "Synthetic complete closure"
            assert len(review.members) == 2
            assert {member.role for member in review.members} == {
                "primary",
                "tokenizer",
            }
            assert review.capture_key_ids == tuple(key.key_id for key in trust.keys)
            page = await owner.mirrors.page()
            assert page.models[0].name == "Synthetic complete closure"
            run = await owner.mirrors.start(
                MirrorRequest(
                    owner_id="library",
                    operation_id="run",
                    kind="run",
                    policy_id=page.policy_id,
                    public_participation=True,
                    seconds=60,
                )
            )
            running = await ready(owner, run.job_id)
            decision = MirrorRequest(
                owner_id="library",
                operation_id="prepare",
                kind="prepare_kit",
                policy_id=page.policy_id,
                manifest_id=identity,
            )
            response = await client.post(
                "/preservation/mirror/jobs", json=decision.model_dump(mode="json")
            )
            assert response.status_code == 202, response.text
            submitted = MirrorJob.model_validate_json(response.text)
        # Client loss does not cancel a queued/running worker or replay a decision.
        prepared = await settled(owner, submitted.job_id)
        assert prepared.state == "succeeded", prepared
        assert prepared.kit is not None and prepared.kit.state == "prepared"
        assert (await owner.mirrors.start(decision)).job_id == submitted.job_id
        assert owner.mirrors.job(running.job_id).activity is not None
        assert owner.mirrors.job(running.job_id).state != "succeeded"
        await owner.close()
        current = PreservationHost(owner.store, owner.root)
        assert (await current.mirrors.page()).controller_job_id is None
        restored_job = current.mirrors.job(submitted.job_id)
        assert restored_job.state == "succeeded" and restored_job.kit == prepared.kit
        transport = httpx.ASGITransport(app=app(current), client=("127.0.0.1", 1000))
        async with httpx.AsyncClient(
            transport=transport, base_url="http://localhost"
        ) as client:

            def forbidden(*args: object, **kwargs: object) -> None:
                raise AssertionError("download attempted a new full-byte kit hash")

            with monkeypatch.context() as metadata_only:
                metadata_only.setattr(
                    mirror_kits, "hashlib", SimpleNamespace(sha256=forbidden)
                )
                head = await client.head(
                    "/preservation/mirror/kits/" + submitted.job_id
                )
                assert head.status_code == 200 and head.content == b""
                assert head.headers["Content-Length"] == str(prepared.kit.size_bytes)
                part = await client.get(
                    "/preservation/mirror/kits/" + submitted.job_id,
                    headers={"Range": "bytes=0-127"},
                )
                assert part.status_code == 206 and len(part.content) == 128
                downloaded = await client.get(
                    "/preservation/mirror/kits/" + submitted.job_id,
                    headers={"Range": "bytes=128-", "If-Range": part.headers["ETag"]},
                )
                assert downloaded.status_code == 206
                suffix = await client.get(
                    "/preservation/mirror/kits/" + submitted.job_id,
                    headers={"Range": "bytes=-16"},
                )
                assert (
                    suffix.status_code == 206
                    and suffix.content == downloaded.content[-16:]
                )
                invalid = await client.get(
                    "/preservation/mirror/kits/" + submitted.job_id,
                    headers={"Range": "bytes=999999999999-"},
                )
                assert invalid.status_code == 416 and invalid.headers[
                    "Content-Range"
                ] == "bytes */" + str(prepared.kit.size_bytes)
                changed = await client.get(
                    "/preservation/mirror/kits/" + submitted.job_id,
                    headers={"Range": "bytes=128-", "If-Range": '"different"'},
                )
                assert (
                    changed.status_code == 200
                    and changed.content == part.content + downloaded.content
                )
            data = part.content + downloaded.content
            assert hashlib.sha256(data).hexdigest() == prepared.kit.sha256
            assert len(data) == prepared.kit.size_bytes
            kit = tmp_path / "independent.foxkit"
            kit.write_bytes(data)

            with monkeypatch.context() as offline:
                offline.setattr(socket.socket, "connect", forbidden)
                restored = tmp_path / "independent-archive"
                import_kit(kit, restored, expected_manifest_id=identity, trust=trust)
                materialized = tmp_path / "materialized"
                restore(restored, identity, trust, materialized)
                assert (
                    materialized / ".preservation/envelope.json"
                ).read_bytes() == wrapper
                manifest = verify_store(restored, identity, trust)
                assert len(manifest.members) == 2
                assert all(
                    m.retained_card == {"future": {"opaque": [1, True, "unchanged"]}}
                    for m in manifest.members
                )
            denied = await client.request(
                "DELETE",
                "/preservation/mirror/kits/" + submitted.job_id,
                json={"confirm_job_id": "0" * 64},
            )
            assert denied.status_code != 200
            removed = await client.request(
                "DELETE",
                "/preservation/mirror/kits/" + submitted.job_id,
                json={"confirm_job_id": submitted.job_id},
            )
            assert removed.status_code == 200, removed.text
            assert (
                removed.json()["kit_removed"]
                and removed.json()["kit"]["state"] == "removed"
            )
            assert (
                await client.get("/preservation/mirror/kits/" + submitted.job_id)
            ).status_code != 200
            assert (await current.mirrors.start(decision)).job_id == submitted.job_id
            assert current.mirrors.job(submitted.job_id).kit_removed
            assert (
                verify_store(
                    current.root / "managed-mirror/store", identity, trust
                ).name
                == "Synthetic complete closure"
            )
            assert kit.read_bytes() == data
    finally:
        await current.close()
        if current is not owner:
            await owner.close()


@pytest.mark.asyncio
async def test_exact_lookup_off_page_removed_unknown_and_storage_loss(
    tmp_path: Path,
) -> None:
    owner, identity, _trust, _wrapper = await retained(tmp_path)
    try:
        ledger = MirrorLedger(owner.root / "managed-mirror")
        held = ledger.read().models[identity]
        identities = [f"{i:064x}" for i in range(45)]

        def fill(state: MirrorState) -> MirrorState:
            models = {key: held for key in identities}
            models[identities[-1]] = held.model_copy(update={"state": "removed"})
            return state.model_copy(update={"models": models})

        ledger.update(fill)
        assert (await owner.mirrors.page()).next_offset == 20
        removed = await owner.mirrors.selection(identities[-1])
        assert (
            removed.available
            and removed.model is not None
            and removed.model.state == "removed"
        )
        kept = await owner.mirrors.selection(identities[25])
        assert (
            kept.available and kept.model is not None and kept.model.state == "retained"
        )
        unknown = await owner.mirrors.selection("f" * 64)
        assert unknown.available and unknown.model is None
        (ledger.path / "mirror.json").write_text("corrupt owner state")
        unavailable = await owner.mirrors.selection("f" * 64)
        assert unavailable.configured and not unavailable.available
    finally:
        await owner.close()


def stalled_seal_command(marker: Path) -> Callable[[Path], tuple[str, ...]]:
    """Run the real fixed worker, stalling only its seal after the kit copy commits."""
    script = "\n".join(
        [
            "import os, time, sys",
            "sys.stderr = open(" + repr(str(marker) + ".stderr") + ", 'w')",
            "from pathlib import Path",
            "import skulk.store.mirror_worker as worker",
            "def stalled(target, manifest_id, control):",
            "    Path(" + repr(str(marker)) + ").write_text(str(os.getpid()))",
            "    while True:",
            "        control.checkpoint()",
            "        time.sleep(0.02)",
            "worker.seal_kit = stalled",
            "from skulk.store.preservation_worker import main",
            "main()",
        ]
    )

    def command(request: Path) -> tuple[str, ...]:
        return sys.executable, "-c", script, str(request)

    return command


@pytest.mark.asyncio
async def test_stop_during_sealing_keeps_controller_and_model_and_allows_explicit_cleanup(
    tmp_path: Path,
) -> None:
    owner, identity, trust, _wrapper = await retained(tmp_path)
    marker = tmp_path / "sealing-started"
    try:
        page = await owner.mirrors.page()
        run = await owner.mirrors.start(
            MirrorRequest(
                owner_id="library",
                operation_id="serve",
                kind="run",
                policy_id=page.policy_id,
                public_participation=True,
                seconds=60,
            )
        )
        running = await ready(owner, run.job_id)
        owner.mirrors.command = stalled_seal_command(marker)
        prepared = await owner.mirrors.start(
            MirrorRequest(
                owner_id="library",
                operation_id="slow-kit",
                kind="prepare_kit",
                policy_id=page.policy_id,
                manifest_id=identity,
            )
        )
        async with asyncio.timeout(15):
            while not marker.exists():
                assert owner.mirrors.job(prepared.job_id).state in {
                    "queued",
                    "running",
                }, (
                    (tmp_path / "sealing-started.stderr").read_text()
                    if (tmp_path / "sealing-started.stderr").exists()
                    else "worker failed before diagnostics"
                )
                await asyncio.sleep(0.025)
        with pytest.raises(PreservationHostError, match="mirror_job_capacity_reached"):
            await owner.mirrors.start(
                MirrorRequest(
                    owner_id="library",
                    operation_id="third-active-operation",
                    kind="model",
                    policy_id=page.policy_id,
                    manifest_id=identity,
                    action="pin",
                )
            )
        stopped = await owner.mirrors.cancel(prepared.job_id)
        assert stopped.state == "cancelled" and stopped.kit is None
        assert owner.mirrors.job(running.job_id).activity is not None
        assert (
            owner.root / "mirror-kits" / prepared.job_id / "bundle.foxkit"
        ).is_file()
        removed = await owner.mirrors.remove_kit(prepared.job_id, prepared.job_id)
        assert removed.kit_removed and removed.kit is None
        assert not (owner.root / "mirror-kits" / prepared.job_id).exists()
        assert verify_store(owner.root / "managed-mirror/store", identity, trust)
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_close_stops_later_controller_before_earlier_preparation_drain_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner, identity, trust, _wrapper = await retained(tmp_path)
    marker = tmp_path / "earlier-sealing"
    entered, release = asyncio.Event(), asyncio.Event()
    closing: asyncio.Task[None] | None = None
    reopened: PreservationHost | None = None
    original_stop = MirrorProcess.stop
    try:
        page = await owner.mirrors.page()
        owner.mirrors.command = stalled_seal_command(marker)
        preparation = await owner.mirrors.start(
            MirrorRequest(
                owner_id="library",
                operation_id="prepare-first",
                kind="prepare_kit",
                policy_id=page.policy_id,
                manifest_id=identity,
            )
        )
        async with asyncio.timeout(15):
            while not marker.exists() or not marker.read_text().isdigit():
                assert owner.mirrors.job(preparation.job_id).state in {
                    "queued",
                    "running",
                }
                await asyncio.sleep(0.025)
        preparation_pid = int(marker.read_text())
        controller = await owner.mirrors.start(
            MirrorRequest(
                owner_id="library",
                operation_id="serve-second",
                kind="run",
                policy_id=page.policy_id,
                public_participation=True,
                seconds=120,
            )
        )
        active = await ready(owner, controller.job_id)
        assert active.activity is not None and active.activity.http_port is not None
        port = active.activity.http_port

        async def stalled_stop(process: MirrorProcess) -> None:
            if process.process.pid == preparation_pid:
                entered.set()
                await release.wait()
                raise subprocess.TimeoutExpired("synthetic preparation drain", 0)
            await original_stop(process)

        with monkeypatch.context() as failure:
            failure.setattr(MirrorProcess, "stop", stalled_stop)
            closing = asyncio.create_task(owner.close())
            await asyncio.wait_for(entered.wait(), 5)
            async with asyncio.timeout(5):
                while MirrorLedger(owner.root / "managed-mirror").running():
                    await asyncio.sleep(0.025)
                while True:
                    try:
                        _, connection = await asyncio.open_connection("127.0.0.1", port)
                    except ConnectionRefusedError:
                        break
                    connection.close()
                    await connection.wait_closed()
                    await asyncio.sleep(0.025)
            assert not closing.done()
            release.set()
            with pytest.raises(subprocess.TimeoutExpired):
                await closing
        with pytest.raises(
            PreservationHostError, match="preservation_host_already_running"
        ):
            unexpected = PreservationJournal(owner.root)
            unexpected.close()
        assert owner.mirrors.job(preparation.job_id).state == "cancelling"
        assert (
            owner.root / "mirror-kits" / preparation.job_id / "bundle.foxkit"
        ).is_file()
        await asyncio.wait_for(owner.close(), 15)
        assert verify_store(owner.root / "managed-mirror/store", identity, trust)
        reopened = PreservationHost(owner.store, owner.root)
        assert reopened.mirrors.job(preparation.job_id).state == "interrupted"
        assert (await reopened.mirrors.page()).controller_job_id is None
        assert (
            owner.root / "mirror-kits" / preparation.job_id / "bundle.foxkit"
        ).is_file()
    finally:
        release.set()
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)
        if reopened is not None:
            await reopened.close()
        await owner.close()


@pytest.mark.asyncio
async def test_core_close_cancels_download_and_releases_descriptor_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner, identity, _trust, _wrapper = await retained(tmp_path)
    entered = asyncio.Event()
    download: asyncio.Task[httpx.Response] | None = None
    try:
        page = await owner.mirrors.page()
        created = await owner.mirrors.start(
            MirrorRequest(
                owner_id="library",
                operation_id="kit",
                kind="prepare_kit",
                policy_id=page.policy_id,
                manifest_id=identity,
            )
        )
        job = await settled(owner, created.job_id)
        assert job.state == "succeeded", job

        async def stalled(
            descriptor: int, seal: KitSeal, offset: int, size: int
        ) -> bytes:
            entered.set()
            await asyncio.Event().wait()
            raise AssertionError("cancelled download resumed")

        monkeypatch.setattr(owner.mirrors, "kit_chunk", stalled)
        transport = httpx.ASGITransport(app=app(owner), client=("127.0.0.1", 1000))
        async with httpx.AsyncClient(
            transport=transport, base_url="http://localhost"
        ) as client:
            download = asyncio.create_task(
                client.get("/preservation/mirror/kits/" + job.job_id)
            )
            await asyncio.wait_for(entered.wait(), 5)
            await asyncio.wait_for(owner.close(), 10)
            # Close waits for the actual streaming lease and descriptor release.
            assert download.done()
            assert (owner.root / "mirror-kits" / job.job_id / "bundle.foxkit").is_file()
    finally:
        if download is not None:
            if not download.done():
                download.cancel()
            await asyncio.gather(download, return_exceptions=True)
        await owner.close()


@pytest.mark.asyncio
async def test_review_preserves_all_notice_members_and_future_roles_or_refuses_whole_oversized_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = empty_host(tmp_path)
    try:
        configured = await owner.mirrors.start(
            MirrorRequest(
                owner_id="library",
                operation_id="configure",
                kind="configure",
                configuration=configuration(),
            )
        )
        assert (await settled(owner, configured.job_id)).state == "succeeded"
        source, original_id, original_trust = archive(tmp_path)
        original = verify_store(source, original_id, original_trust)
        key = Ed25519PrivateKey.generate()
        trust = TrustProfile(
            schema_version="foxlight.model-preservation.trust.v1",
            keys=(public_record(key),),
            threshold=1,
            as_of="2026-10-25T00:00:00Z",
        )
        primary, companion = original.members
        notice = companion.model_copy(
            update={
                "member_id": "source-evidence",
                "role": "notices",
                "files": tuple(f for f in companion.files if f.path == "LICENSE"),
            }
        )
        primary = primary.model_copy(
            update={"dependencies": (*primary.dependencies, "source-evidence")}
        )
        companion = companion.model_copy(update={"role": "future-role-" + "x" * 80})
        ledger = MirrorLedger(owner.root / "managed-mirror")
        identities: list[str] = []
        for workloads in (("fixture",), ("x" * 60000,)):
            manifest = original.model_copy(
                update={"members": (primary, companion, notice), "workloads": workloads}
            )
            identity = digest(encode_record(manifest))
            (source / manifest_path(identity)).write_bytes(
                encode_record(sign_manifest(manifest, (key,)))
            )
            retain_local(
                ledger,
                source,
                identity,
                SourceGrant(
                    trust=trust,
                    terms=(
                        Terms(license_id="fixture-only", terms_reference="LICENSE"),
                    ),
                ),
            )
            identities.append(identity)
        original_open = archive_storage.open_file

        @contextmanager
        def no_weights(
            root: int, path: str, *, write: bool = False
        ) -> Iterator[BinaryIO]:
            assert not path.startswith("objects/"), "metadata review read weight bytes"
            with original_open(root, path, write=write) as stream:
                yield stream

        monkeypatch.setattr(archive_storage, "open_file", no_weights)
        transport = httpx.ASGITransport(app=app(owner), client=("127.0.0.1", 1000))
        async with httpx.AsyncClient(
            transport=transport, base_url="http://localhost"
        ) as client:
            complete = await client.get(
                "/preservation/mirror/models/" + identities[0] + "/review"
            )
            assert complete.status_code == 200, complete.text
            review = MirrorReview.model_validate_json(complete.text)
            assert len(review.members) == 3
            assert review.members[1].role == companion.role
            assert review.members[2].member_id == "source-evidence"
            refused = await client.get(
                "/preservation/mirror/models/" + identities[1] + "/review"
            )
            assert (
                refused.status_code != 200
                and "metadata_preview_too_large" in refused.text
            )
            assert "members" not in refused.json()
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["HEAD", "DELETE"])
async def test_kit_headers_and_removal_authorize_before_owner_creation(
    method: str,
) -> None:
    async def forbidden() -> PreservationHost:
        raise AssertionError("untrusted request constructed an owner")

    application = FastAPI()
    application.include_router(
        create_preservation_router(forbidden, API.require_preservation_operator)
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application, client=("203.0.113.9", 1000)),
        base_url="http://untrusted.example",
    ) as client:
        response = await client.request(
            method,
            "/preservation/mirror/kits/" + "1" * 64,
            json={"confirm_job_id": "1" * 64} if method == "DELETE" else None,
        )
        assert response.status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "/preservation/mirror/models/" + "1" * 64,
        "/preservation/mirror/models/" + "1" * 64 + "/review",
        "/preservation/mirror/kits/" + "1" * 64,
    ],
)
async def test_recovery_reads_authorize_before_host_creation(path: str) -> None:
    async def forbidden() -> PreservationHost:
        raise AssertionError("untrusted request constructed an owner")

    application = FastAPI()
    application.include_router(
        create_preservation_router(forbidden, API.require_preservation_operator)
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application, client=("203.0.113.9", 1000)),
        base_url="http://untrusted.example",
    ) as client:
        assert (await client.get(path)).status_code == 403
