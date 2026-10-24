"""Actual store-server startup and standing intent, with no reads-based autoplay."""

from __future__ import annotations

import asyncio
import subprocess
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Literal

import pytest
from foxlight_model_preservation.kits import export_kit
from foxlight_model_preservation.mirror_models import (
    HeldModel,
    MirrorState,
    SourceGrant,
    Terms,
)
from foxlight_model_preservation.mirror_state import MirrorLedger

import skulk.store.mirror_host as mirror_host
import skulk.store.model_store_server as store_server
from skulk.store.mirror_contracts import MirrorJob, MirrorRequest
from skulk.store.model_store import ModelStore
from skulk.store.model_store_server import ModelStoreServer
from skulk.store.preservation_host import PreservationHost
from skulk.store.preservation_io import PreservationIO
from skulk.store.preservation_journal import PreservationHostError
from skulk.store.tests.test_mirror_worker import archive, configuration
from skulk.store.tests.test_preservation_sharing import empty_host, import_copy


async def complete(host: PreservationHost, request: MirrorRequest) -> MirrorJob:
    """Submit to the real owner and wait through actual owned child drain."""
    job = await host.mirrors.start(request)
    async with asyncio.timeout(30):
        while host.mirrors.job(job.job_id).state in {"queued", "running", "cancelling"}:
            await asyncio.sleep(0.025)
    result = host.mirrors.job(job.job_id)
    assert result.state == "succeeded", result
    return result


async def listening(host: PreservationHost, identity: str) -> MirrorJob:
    """Observe only the current owned HTTP listener."""
    async with asyncio.timeout(30):
        while True:
            job = host.mirrors.job(identity)
            assert job.state in {"queued", "running"}, job
            if job.activity and job.activity.running and job.activity.http_port:
                return job
            await asyncio.sleep(0.025)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["finite", "resume"])
async def test_store_startup_requires_approved_grant_and_prior_running_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: Literal["finite", "resume"]
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(store_server, "SKULK_DATA_HOME", data)
    store = ModelStore(models)
    servers: list[ModelStoreServer] = []

    def server() -> ModelStoreServer:
        instance = ModelStoreServer(store, host="127.0.0.1", port=0)
        servers.append(instance)
        return instance

    try:
        first = server()
        await first.start()
        owner = await first.preservation_host()
        assert not (await owner.mirrors.page()).configured
        await complete(
            owner,
            MirrorRequest(
                owner_id="library",
                operation_id="configure",
                kind="configure",
                configuration=configuration(),
            ),
        )
        current = await owner.mirrors.page()
        expires = int(time.time()) + 300
        await complete(
            owner,
            MirrorRequest(
                owner_id="library",
                operation_id="grant",
                kind="grant_resume",
                policy_id=current.policy_id,
                resume_expires=expires,
                confirm_resume=True,
            ),
        )
        assert (await owner.mirrors.page()).controller_job_id is None
        assert not MirrorLedger(data / "preservation/managed-mirror").running()
        await first.stop()

        second = server()
        await second.start()  # A saved grant alone never starts sharing.
        owner = await second.preservation_host()
        current = await owner.mirrors.page()
        assert current.controller_job_id is None and current.restart_expires == expires
        original = await owner.mirrors.start(
            MirrorRequest(
                owner_id="library",
                operation_id="explicit-run",
                kind="run",
                policy_id=current.policy_id,
                mode=mode,
                public_participation=True,
                seconds=60,
            )
        )
        await listening(owner, original.job_id)
        before = MirrorLedger(data / "preservation/managed-mirror").read()
        await second.stop()  # Core shutdown drains but does not invent/revoke intent.

        third = server()
        owner = await third.preservation_host()
        assert owner.mirrors.job(original.job_id).state == "interrupted"
        assert owner.mirrors.job(original.job_id).activity is None
        # Construction and repeated read requests must not consume the standing grant.
        for _ in range(3):
            assert (await owner.mirrors.page()).controller_job_id is None
        assert not MirrorLedger(data / "preservation/managed-mirror").running()
        await third.start()
        if mode == "finite":
            assert (await owner.mirrors.page()).controller_job_id is None
            assert not MirrorLedger(data / "preservation/managed-mirror").running()
            return
        async with asyncio.timeout(30):
            while (current := await owner.mirrors.page()).controller_job_id is None:
                await asyncio.sleep(0.025)
        resumed_id = current.controller_job_id
        assert resumed_id != original.job_id
        await listening(owner, resumed_id)
        after = MirrorLedger(data / "preservation/managed-mirror").read()
        assert after.restart == before.restart and after.restart is not None
        assert after.restart.expires == expires
        assert after.egress_month_used == before.egress_month_used
        stopped = await owner.mirrors.cancel(resumed_id)
        assert stopped.state == "cancelled" and stopped.activity is None
        assert owner.journal.setting("mirror-running-intent") is None
        await third.stop()
        fourth = server()
        await fourth.start()
        owner = await fourth.preservation_host()
        assert (await owner.mirrors.page()).controller_job_id is None
        assert not MirrorLedger(data / "preservation/managed-mirror").running()
    finally:
        for instance in reversed(servers):
            await instance.stop()


@pytest.mark.asyncio
async def test_cancel_queued_before_worker_creation_is_terminal_and_not_replayed(
    tmp_path: Path,
) -> None:
    models = tmp_path / "models"
    models.mkdir()
    host = PreservationHost(ModelStore(models), tmp_path / "owner")
    request = MirrorRequest(
        owner_id="library",
        operation_id="queued",
        kind="configure",
        configuration=configuration(),
    )
    try:
        job = await host.mirrors.start(request)
        cancelled = await host.mirrors.cancel(job.job_id)
        assert cancelled.state == "cancelled"
        assert not (await host.mirrors.page()).configured
        assert (await host.mirrors.start(request)).state == "cancelled"
        assert not host.journal.jobs()
    finally:
        await host.close()


@pytest.mark.asyncio
async def test_stop_drains_network_before_stalled_pause_and_client_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = empty_host(tmp_path)
    entered, release = threading.Event(), threading.Event()
    stopping: asyncio.Task[MirrorJob] | None = None
    try:
        await complete(
            host,
            MirrorRequest(
                owner_id="library",
                operation_id="config",
                kind="configure",
                configuration=configuration(),
            ),
        )
        page = await host.mirrors.page()
        request = MirrorRequest(
            owner_id="library",
            operation_id="run",
            kind="run",
            policy_id=page.policy_id,
            public_participation=True,
        )
        running = await host.mirrors.start(request)
        await listening(host, running.job_id)
        original = MirrorLedger.update

        def delayed(
            ledger: MirrorLedger, action: Callable[[MirrorState], MirrorState]
        ) -> MirrorState:
            entered.set()
            if not release.wait(15):
                raise AssertionError("test pause did not release")
            return original(ledger, action)

        monkeypatch.setattr(MirrorLedger, "update", delayed)
        stopping = asyncio.create_task(host.mirrors.cancel(running.job_id))
        assert await asyncio.to_thread(entered.wait, 10)
        assert host.mirrors.job(running.job_id).state == "cancelled"
        assert not MirrorLedger(host.root / "managed-mirror").running()
        assert (await host.mirrors.page()).activity is None
        stopping.cancel()
        await asyncio.sleep(0.05)
        assert (
            not stopping.done()
        )  # The actual private filesystem effect is still owned.
        denied = await host.mirrors.start(
            request.model_copy(update={"operation_id": "too-early"})
        )
        assert (
            denied.state == "failed" and denied.error_code == "mirror_busy_stop_first"
        )
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await stopping
        assert host.journal.setting("mirror-running-intent") is None
        assert MirrorLedger(host.root / "managed-mirror").read().paused
    finally:
        release.set()
        if stopping is not None:
            await asyncio.gather(stopping, return_exceptions=True)
        await host.close()


@pytest.mark.asyncio
async def test_stop_drains_workers_while_both_filesystem_effect_slots_stall(
    tmp_path: Path,
) -> None:
    host = empty_host(tmp_path)
    entered = (threading.Event(), threading.Event())
    release = threading.Event()
    effects: list[asyncio.Task[None]] = []
    stopping: asyncio.Task[MirrorJob] | None = None

    def stalled(index: int) -> None:
        entered[index].set()
        if not release.wait(20):
            raise AssertionError("test filesystem effects did not release")

    try:
        await complete(
            host,
            MirrorRequest(
                owner_id="library",
                operation_id="config",
                kind="configure",
                configuration=configuration(),
            ),
        )
        page = await host.mirrors.page()
        running = await host.mirrors.start(
            MirrorRequest(
                owner_id="library",
                operation_id="run",
                kind="run",
                policy_id=page.policy_id,
                public_participation=True,
            )
        )
        await listening(host, running.job_id)
        ledger = MirrorLedger(host.root / "managed-mirror")
        active = await listening(host, running.job_id)
        assert active.activity is not None and active.activity.http_port is not None
        port = active.activity.http_port
        effects = [
            asyncio.create_task(host.mirrors.storage.run(lambda i=i: stalled(i)))
            for i in range(2)
        ]
        for event in entered:
            assert await asyncio.to_thread(event.wait, 5)
        stopping = asyncio.create_task(host.mirrors.cancel(running.job_id))
        # A mounted archive may consume both filesystem slots indefinitely.
        # Signal and group drain must still end actual public participation;
        # private command-file cleanup and pause may correctly remain pending.
        async with asyncio.timeout(5):
            while ledger.running():
                await asyncio.sleep(0.025)
            while True:
                try:
                    _, connection = await asyncio.open_connection("127.0.0.1", port)
                except ConnectionRefusedError:
                    break
                connection.close()
                await connection.wait_closed()
                await asyncio.sleep(0.025)
        assert host.journal.setting("mirror-running-intent") is None
        assert host.journal.setting("mirror-stopped") is True
        assert not ledger.running()
        assert not stopping.done()
        release.set()
        assert (await stopping).state == "cancelled"
    finally:
        release.set()
        await asyncio.gather(*effects, return_exceptions=True)
        if stopping is not None:
            await asyncio.gather(stopping, return_exceptions=True)
        await host.close()


@pytest.mark.asyncio
async def test_close_drains_controller_before_unrelated_cancelled_model_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = empty_host(tmp_path)
    entered, release = threading.Event(), threading.Event()
    selected: str | None = None
    cancelled: asyncio.Task[MirrorJob] | None = None
    closing: asyncio.Task[None] | None = None
    original_cleanup = mirror_host.remove_mirror_request

    def stalled() -> None:
        entered.set()
        if not release.wait(15):
            raise AssertionError("test model cleanup did not release")

    async def cleanup(path: Path, storage: PreservationIO) -> None:
        if path.stem == selected:
            await storage.run(stalled)
        await original_cleanup(path, storage)

    try:
        await complete(
            host,
            MirrorRequest(
                owner_id="library",
                operation_id="config",
                kind="configure",
                configuration=configuration(),
            ),
        )
        source, identity, trust = archive(tmp_path)
        kit = tmp_path / "complete.foxkit"
        export_kit(source, identity, trust, kit)
        copy = await import_copy(host, kit.read_bytes())
        page = await host.mirrors.page()
        await complete(
            host,
            MirrorRequest(
                owner_id="library",
                operation_id="retain",
                kind="retain",
                policy_id=page.policy_id,
                copy_id=copy.copy_id,
                manifest_id=identity,
                confirm_redistribution=True,
            ),
        )
        page = await host.mirrors.page()
        controller = await host.mirrors.start(
            MirrorRequest(
                owner_id="library",
                operation_id="run",
                kind="run",
                policy_id=page.policy_id,
                public_participation=True,
            ),
        )
        active = await listening(host, controller.job_id)
        assert active.activity is not None and active.activity.http_port is not None
        port = active.activity.http_port
        monkeypatch.setattr(mirror_host, "remove_mirror_request", cleanup)
        model = await host.mirrors.start(
            MirrorRequest(
                owner_id="library",
                operation_id="pin",
                kind="model",
                policy_id=page.policy_id,
                manifest_id=identity,
                action="pin",
            ),
        )
        selected = model.job_id
        assert await asyncio.to_thread(entered.wait, 5)
        cancelled = asyncio.create_task(host.mirrors.cancel(model.job_id))
        async with asyncio.timeout(5):
            while host.mirrors.job(model.job_id).state != "cancelling":
                await asyncio.sleep(0.025)
        closing = asyncio.create_task(host.close())
        async with asyncio.timeout(5):
            while MirrorLedger(host.root / "managed-mirror").running():
                await asyncio.sleep(0.025)
            while True:
                try:
                    _, connection = await asyncio.open_connection("127.0.0.1", port)
                except ConnectionRefusedError:
                    break
                connection.close()
                await connection.wait_closed()
                await asyncio.sleep(0.025)
        assert not closing.done() and not cancelled.done()
        release.set()
        await closing
        assert (await cancelled).state == "interrupted"
    finally:
        release.set()
        if cancelled is not None:
            await asyncio.gather(cancelled, return_exceptions=True)
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)
        await host.close()


@pytest.mark.asyncio
async def test_successful_stop_retry_reconciles_retained_failed_task_and_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = empty_host(tmp_path)
    try:
        await complete(
            host,
            MirrorRequest(
                owner_id="library",
                operation_id="configure",
                kind="configure",
                configuration=configuration(),
            ),
        )
        page = await host.mirrors.page()
        job = await host.mirrors.start(
            MirrorRequest(
                owner_id="library",
                operation_id="first",
                kind="run",
                policy_id=page.policy_id,
                public_participation=True,
                seconds=1,
            ),
        )
        await listening(host, job.job_id)

        def unconfirmed(_identity: int) -> bool:
            return True

        with monkeypatch.context() as observation:
            observation.setattr(
                "skulk.utils.process_groups._group_is_running", unconfirmed
            )
            async with asyncio.timeout(10):
                while (
                    host.mirrors.job(job.job_id).error_code != "mirror_stop_incomplete"
                ):
                    await asyncio.sleep(0.025)
        assert host.mirrors.job(job.job_id).state == "cancelling"
        assert not MirrorLedger(host.root / "managed-mirror").running()
        stopped = await host.mirrors.cancel(job.job_id)
        assert stopped.state == "cancelled" and stopped.activity is None
        assert stopped.error_code is None
        assert not (host.root / "mirror-requests" / (job.job_id + ".json")).exists()
        current = await host.mirrors.page()
        assert current.controller_job_id is None and current.paused
        following = await host.mirrors.start(
            MirrorRequest(
                owner_id="library",
                operation_id="following",
                kind="run",
                policy_id=current.policy_id,
                public_participation=True,
            ),
        )
        await listening(host, following.job_id)
        assert (await host.mirrors.cancel(following.job_id)).state == "cancelled"
    finally:
        # This controlled fault has only an expired HTTP worker; the retry has
        # drained its real group even when the pre-fix metadata owner is stuck.
        with suppress(PreservationHostError, subprocess.TimeoutExpired):
            await host.close()
        await host.mirrors.storage.close()
        host.journal.close()


@pytest.mark.asyncio
async def test_pending_metadata_probe_projects_each_requested_cursor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = empty_host(tmp_path)
    entered, release = threading.Event(), threading.Event()
    try:
        await complete(
            host,
            MirrorRequest(
                owner_id="library",
                operation_id="config",
                kind="configure",
                configuration=configuration(),
            ),
        )
        _, _, trust = archive(tmp_path)
        original = MirrorLedger.read
        path = host.root / "managed-mirror"
        state = original(MirrorLedger(path))
        # Metadata-only pending selections exercise paging, not completeness or
        # publication. No synthetic record is written to the actual mirror.
        held = HeldModel(
            origin="local",
            envelope_sha256="e" * 64,
            operation_id="a" * 64,
            grant=SourceGrant(
                trust=trust,
                terms=(Terms(license_id="fixture", terms_reference="LICENSE"),),
            ),
        )
        identifiers = tuple(f"{index:064x}" for index in range(25))
        snapshot = state.model_copy(update={"models": dict.fromkeys(identifiers, held)})

        def delayed(ledger: MirrorLedger) -> MirrorState:
            if ledger.path == path:
                entered.set()
                if not release.wait(10):
                    raise AssertionError("test metadata probe did not release")
                return snapshot
            return original(ledger)

        monkeypatch.setattr(MirrorLedger, "read", delayed)
        first = asyncio.create_task(host.mirrors.page(0))
        assert await asyncio.to_thread(entered.wait, 5)
        second = asyncio.create_task(host.mirrors.page(20))
        await asyncio.sleep(0.01)
        release.set()
        page0, page20 = await asyncio.gather(first, second)
        assert tuple(model.manifest_id for model in page0.models) == identifiers[:20]
        assert tuple(model.manifest_id for model in page20.models) == identifiers[20:]
        assert page0.next_offset == 20 and page20.next_offset is None
        assert page0.model_count == page20.model_count == 25
    finally:
        release.set()
        await host.close()
