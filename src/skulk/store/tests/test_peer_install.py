"""Core-owned setup: consent, replay, actual child lifetime and independently checked bytes."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import httpx
import pytest
from foxlight_model_preservation.peer_install import managed_executable
from pydantic import ValidationError

from skulk.store.model_store import ModelStore
from skulk.store.peer_install_contracts import (
    RuntimeSetupJob,
    RuntimeSetupRequest,
    RuntimeSetupView,
)
from skulk.store.peer_install_host import RuntimeSetupHost
from skulk.store.preservation_host import PreservationHost
from skulk.store.preservation_journal import PreservationHostError
from skulk.store.tests.test_discovery import application


async def settled(host: PreservationHost, job_id: str) -> RuntimeSetupJob:
    """Wait only for a host-observed terminal receipt."""
    async with asyncio.timeout(40):
        while (job := host.peers.installation.job(job_id, "library")).state in {
            "queued",
            "running",
            "cancelling",
        }:
            await asyncio.sleep(0.01)
    return job


def worker(tmp_path: Path, mode: str) -> tuple[str, ...]:
    """A controlled child for actual lifetime tests; production never accepts a command."""
    script = tmp_path / "controlled-worker.py"
    script.write_text(
        "import pathlib,sys,time,json\n"
        "pathlib.Path(sys.argv[1]).with_suffix('.started').touch()\n"
        + (
            "print(json.dumps({'event':'succeeded'}),flush=True)\n"
            if mode == "claim"
            else "sys.stdin.readline()\ntime.sleep(0.1)\n"
        )
    )
    return sys.executable, str(script)


def select_worker(setup: RuntimeSetupHost, tmp_path: Path, mode: str) -> None:
    """Install a typed test child factory without exposing a production API."""

    def command(request: Path) -> tuple[str, ...]:
        return (*worker(tmp_path, mode), str(request))

    setup.command = command


def observe_children(
    monkeypatch: pytest.MonkeyPatch,
) -> list[asyncio.subprocess.Process]:
    """Observe actual spawned children without inspecting host internals."""
    children: list[asyncio.subprocess.Process] = []
    actual = asyncio.create_subprocess_exec

    async def capture(
        *args: str, stdin: int, stdout: int, stderr: int
    ) -> asyncio.subprocess.Process:
        process = await actual(*args, stdin=stdin, stdout=stdout, stderr=stderr)
        children.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", capture)
    return children


async def selection(
    host: PreservationHost, operation: str = "setup"
) -> RuntimeSetupRequest:
    """Bind explicit acquisition consent to this Core's exact offline plan."""
    view = await host.peers.installation.view("library")
    assert view.plan_id is not None
    return RuntimeSetupRequest(
        owner_id="library", operation_id=operation, plan_id=view.plan_id, confirmed=True
    )


async def test_offline_review_and_api_consent_have_no_paths_or_implicit_effects(
    tmp_path: Path,
) -> None:
    host = PreservationHost(ModelStore(tmp_path / "models"), tmp_path / "core")
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application(host)),
            base_url="http://localhost",
        ) as client:
            response = await client.get(
                "/preservation/discovery/peer/runtime", params={"owner_id": "library"}
            )
            assert response.status_code == 200
            view = RuntimeSetupView.model_validate_json(response.text)
            assert view.supported and view.job is None and view.plan is not None
            assert str(tmp_path) not in response.text
            assert not host.peers.installation.runtime_directory.exists()
            for extra in (
                {"runtime_directory": str(tmp_path)},
                {"url": "https://example.invalid/a"},
                {"confirmed": False},
            ):
                body = {
                    "owner_id": "library",
                    "operation_id": "setup",
                    "plan_id": view.plan_id,
                    "confirmed": True,
                    **extra,
                }
                refused = await client.post(
                    "/preservation/discovery/peer/runtime/jobs", json=body
                )
                assert refused.status_code == 422
            assert host.journal.setting("peer-setup") is None
            assert not host.peers.installation.runtime_directory.exists()
            hostile = await client.get(
                "/preservation/discovery/peer/runtime",
                params={"owner_id": "library"},
                headers={"origin": "https://untrusted.invalid"},
            )
            assert hostile.status_code == 403
    finally:
        await host.close()


async def test_worker_success_claim_is_not_installation_and_replay_never_restarts(
    tmp_path: Path,
) -> None:
    host = PreservationHost(ModelStore(tmp_path / "models"), tmp_path / "core")
    setup = host.peers.installation
    select_worker(setup, tmp_path, "claim")
    try:
        request = await selection(host)
        job = await setup.start(request)
        result = await settled(host, job.job_id)
        assert result.state == "failed"
        assert await setup.start(request) == result
        with pytest.raises(PreservationHostError, match="request_changed"):
            await setup.start(request.model_copy(update={"plan_id": "0" * 64}))
        with pytest.raises(PreservationHostError, match="not_found"):
            setup.job(job.job_id, "another-client")
        assert not setup.gate.locked()
        assert host.journal.setting("peer-setup") is None
    finally:
        await host.close()


async def test_close_during_spawn_drains_actual_child_and_restart_does_not_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    core = tmp_path / "core"
    host = PreservationHost(ModelStore(tmp_path / "models"), core)
    setup = host.peers.installation
    select_worker(setup, tmp_path, "wait")
    spawned, release = asyncio.Event(), asyncio.Event()
    actual = asyncio.create_subprocess_exec
    children: list[asyncio.subprocess.Process] = []

    async def delayed(
        *args: str, stdin: int, stdout: int, stderr: int
    ) -> asyncio.subprocess.Process:
        process = await actual(*args, stdin=stdin, stdout=stdout, stderr=stderr)
        children.append(process)
        spawned.set()
        await release.wait()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed)
    request = await selection(host)
    job = await setup.start(request)
    await asyncio.wait_for(spawned.wait(), 10)
    closing = asyncio.create_task(host.close())
    await asyncio.sleep(0.05)
    assert not closing.done()
    release.set()
    await asyncio.wait_for(closing, 10)
    assert all(child.returncode is not None for child in children)
    reopened = PreservationHost(ModelStore(tmp_path / "models"), core)
    try:
        record = reopened.peers.installation.job(job.job_id, "library")
        assert record.state == "interrupted"
        assert await reopened.peers.installation.start(request) == record
        assert len(children) == 1
    finally:
        await reopened.close()


async def test_immediate_close_retains_interruption_even_before_task_first_turn(
    tmp_path: Path,
) -> None:
    core = tmp_path / "core"
    host = PreservationHost(ModelStore(tmp_path / "models"), core)
    request = await selection(host)
    job = await host.peers.installation.start(request)
    await host.close()
    reopened = PreservationHost(ModelStore(tmp_path / "models"), core)
    try:
        assert (
            reopened.peers.installation.job(job.job_id, "library").state
            == "interrupted"
        )
        assert await reopened.peers.installation.start(
            request
        ) == reopened.peers.installation.job(job.job_id, "library")
    finally:
        await reopened.close()


async def test_cancel_drains_worker_and_holds_peer_gate_until_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = PreservationHost(ModelStore(tmp_path / "models"), tmp_path / "core")
    setup = host.peers.installation
    children = observe_children(monkeypatch)
    select_worker(setup, tmp_path, "wait")
    try:
        job = await setup.start(await selection(host))
        async with asyncio.timeout(10):
            while not children:
                await asyncio.sleep(0.01)
        assert setup.gate.locked()
        process = children[0]
        await setup.cancel(job.job_id, "library")
        result = await settled(host, job.job_id)
        assert result.state == "cancelled" and process.returncode is not None
        assert not setup.gate.locked()
    finally:
        await host.close()


async def test_actual_native_setup_requires_no_peer_grant(tmp_path: Path) -> None:
    if os.environ.get("KUBO_SETUP_NETWORK") != "1":
        pytest.skip("explicit real distribution acquisition qualification")
    host = PreservationHost(ModelStore(tmp_path / "models"), tmp_path / "core")
    try:
        request = await selection(host)
        job = await host.peers.installation.start(request)
        result = await settled(host, job.job_id)
        assert result.state == "succeeded"
        assert managed_executable(host.peers.installation.runtime_directory) is not None
        async with asyncio.timeout(15):
            while (await host.peers.status()).runtime != "available":
                await asyncio.sleep(0.1)
        assert host.journal.setting("peer-setup") is None
        assert not (host.root / "peer-cache").exists()
        assert await host.peers.installation.start(request) == result
    finally:
        await host.close()


def test_setup_contract_refuses_unreviewed_execution() -> None:
    with pytest.raises(ValidationError):
        RuntimeSetupRequest(
            owner_id="library", operation_id="a", plan_id="0" * 64, confirmed=False
        )


async def test_stale_plan_and_active_peer_session_refuse_before_worker(
    tmp_path: Path,
) -> None:
    host = PreservationHost(ModelStore(tmp_path / "models"), tmp_path / "core")
    setup = host.peers.installation
    try:
        request = await selection(host)
        with pytest.raises(PreservationHostError, match="plan_changed"):
            await setup.start(request.model_copy(update={"plan_id": "0" * 64}))
        async with setup.gate:
            with pytest.raises(PreservationHostError, match="peer_busy"):
                await setup.start(request)
        assert (await setup.view("library")).job is None
        assert not setup.runtime_directory.exists()
    finally:
        await host.close()


async def test_browser_closure_and_replay_keep_same_actual_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = PreservationHost(ModelStore(tmp_path / "models"), tmp_path / "core")
    setup = host.peers.installation
    children = observe_children(monkeypatch)
    select_worker(setup, tmp_path, "wait")
    transport = httpx.ASGITransport(app=application(host))
    try:
        request = await selection(host)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://localhost"
        ) as client:
            started = await client.post(
                "/preservation/discovery/peer/runtime/jobs",
                json=request.model_dump(mode="json"),
            )
            assert started.status_code == 200
            job = RuntimeSetupJob.model_validate_json(started.text)
        async with asyncio.timeout(10):
            while not children:
                await asyncio.sleep(0.01)
        process = children[0]
        assert process.returncode is None
        async with httpx.AsyncClient(
            transport=transport, base_url="http://localhost"
        ) as reopened:
            view = RuntimeSetupView.model_validate_json(
                (
                    await reopened.get(
                        "/preservation/discovery/peer/runtime",
                        params={"owner_id": "library"},
                    )
                ).text
            )
            assert view.job is not None and view.job.job_id == job.job_id
            replay = await reopened.post(
                "/preservation/discovery/peer/runtime/jobs",
                json=request.model_dump(mode="json"),
            )
            assert RuntimeSetupJob.model_validate_json(replay.text).job_id == job.job_id
            assert children == [process]
            assert (
                await reopened.delete(
                    f"/preservation/discovery/peer/runtime/jobs/{job.job_id}",
                    params={"owner_id": "library"},
                )
            ).status_code == 200
        assert (await settled(host, job.job_id)).state == "cancelled"
        assert process.returncode is not None
    finally:
        await host.close()


async def test_actual_setup_worker_eof_ends_blocked_acquisition(tmp_path: Path) -> None:
    script = tmp_path / "blocked-acquisition.py"
    script.write_text(
        "import time\n"
        "from pathlib import Path\n"
        "import skulk.store.peer_install_worker as worker\n"
        "from foxlight_model_preservation.peer_install import runtime_plan\n"
        "from foxlight_model_preservation.signing import digest,encode_record\n"
        "worker.install_runtime=lambda *a,**k:time.sleep(60)\n"
        "raise SystemExit(worker.run_runtime_setup_worker(worker.RuntimeSetupWorkerRequest(runtime_directory=Path('.'),plan_id=digest(encode_record(runtime_plan())))))\n"
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable, str(script), stdin=asyncio.subprocess.PIPE
    )
    assert process.stdin is not None
    process.stdin.close()
    assert await asyncio.wait_for(process.wait(), 10) == 75


async def test_shutdown_wins_inflight_admission_and_drains_after_caller_cancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    import skulk.store.peer_install_host as module

    host = PreservationHost(ModelStore(tmp_path / "models"), tmp_path / "core")
    request = await selection(host)
    entered, release = threading.Event(), threading.Event()
    original = module.runtime_plan

    def delayed_plan():
        entered.set()
        assert release.wait(10)
        return original()

    monkeypatch.setattr(module, "runtime_plan", delayed_plan)
    admission = asyncio.create_task(host.peers.installation.start(request))
    while not entered.is_set():
        await asyncio.sleep(0.01)
    closing = asyncio.create_task(host.close())
    await asyncio.sleep(0.05)
    closing.cancel()
    release.set()
    with pytest.raises(PreservationHostError, match="host_closed"):
        await admission
    await asyncio.wait_for(closing, 10)
    assert not host.peers.installation.runtime_directory.exists()
