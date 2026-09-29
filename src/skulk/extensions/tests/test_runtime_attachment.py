"""Restart attachment, foreign-profile fencing and interrupted local write recovery."""

import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError

from skulk.connectivity.tailscale import TailscaleStatus
from skulk.extensions.managed import ManagedConnection, load_managed_owners
from skulk.extensions.managed_attachment import ManagedAttachment
from skulk.extensions.runtime_artifacts import QualifiedHost
from skulk.extensions.runtime_attachment import (
    AttachmentJournal,
    AttachmentRequest,
    HostSettings,
    OwnerBinding,
    read_serve_host,
)
from skulk.extensions.runtime_controller import LifecycleRequest
from skulk.extensions.runtime_files import RuntimeLock, read_private, write_private
from skulk.extensions.runtime_manager import (
    InstallationRequest,
    RuntimeManager,
    SubmitRequest,
    manager_request,
)
from skulk.extensions.tests.test_runtime_install import artifacts
from skulk.extensions.tests.test_runtime_service import OWNER_SOURCE, running

PROFILE = "1" * 32
HOST = QualifiedHost("macos-arm64", "3.13.13", "1.5.2", "a" * 64)


def setup(root: Path, monkeypatch: pytest.MonkeyPatch) -> RuntimeManager:
    """Provision one synthetic profile and equal measured core builds."""
    write_private(
        root / "host.json",
        HostSettings(
            transport_node_id="before-reboot",
            profile_id=PROFILE,
        )
        .model_dump_json()
        .encode(),
    )
    monkeypatch.setattr("skulk.extensions.runtime_manager.measure_host", lambda: HOST)
    monkeypatch.setattr(
        "skulk.extensions.managed_attachment.measure_host", lambda: HOST
    )
    monkeypatch.setattr("skulk.extensions.runtime_install.measure_host", lambda: HOST)
    return RuntimeManager(root)


async def test_new_api_lifetime_refreshes_owner_without_replacing_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An actual owner restarts with the new transport, preserving all durable bytes."""
    manager = setup(tmp_path, monkeypatch)
    await manager.start()
    bridge = ManagedAttachment(tmp_path, PROFILE)
    bridge.retain("before-reboot")
    replacement = ManagedAttachment(tmp_path, PROFILE)
    replacement.retain("after-reboot")
    try:
        await bridge.ensure()
        await manager_request(
            tmp_path,
            InstallationRequest(action="register", plugin_id="managed.fixture"),
        )
        controller = manager.controllers["managed.fixture"]
        root = controller.root
        metadata, trust, _ = artifacts(tmp_path / "source", owner_source=OWNER_SOURCE)
        write_private(root / "publisher-trust.json", trust.model_dump_json().encode())
        staged = await controller.selector.installer.stage(
            metadata, tmp_path / "source"
        )
        await manager_request(
            tmp_path,
            SubmitRequest(
                plugin_id="managed.fixture",
                request=LifecycleRequest(
                    operation_id="2" * 32,
                    action="activate",
                    expected_revision=0,
                    runtime_digest=staged.runtime_digest,
                ),
            ),
        )
        assert controller.work is not None
        await controller.work
        assert controller.service is not None
        await running(controller.service)
        process = controller.service.process
        assert process is not None
        selection = controller.selector.current()
        preserved = {
            name: ("original-" + name).encode()
            for name in (
                "identity.json",
                "configuration.json",
                "receipts.json",
                "approval-reservations.json",
            )
        }
        for name, content in preserved.items():
            write_private(root / name, content)
        with pytest.raises(BlockingIOError):
            await replacement.ensure()
        assert process.returncode is None
        for request in (
            AttachmentRequest(
                profile_id="f" * 32,
                transport_node_id="foreign",
                skulk_build_sha256="a" * 64,
            ),
            AttachmentRequest(
                profile_id=PROFILE,
                transport_node_id="other-build",
                skulk_build_sha256="b" * 64,
            ),
        ):
            assert "error" in await manager_request(tmp_path, request)
            assert process.returncode is None
        await bridge.release()
        assert process.returncode is None  # API loss never owns independent cleanup.
        await replacement.ensure()
        assert process.returncode == 0
        successor = manager.controllers["managed.fixture"]
        assert successor.service is not None
        await running(successor.service)
        assert successor.selector.current() == selection
        assert (
            OwnerBinding.model_validate_json(
                read_private(root / "owner.json")
            ).transport_node_id
            == "after-reboot"
        )
        assert {name: read_private(root / name) for name in preserved} == preserved
        assert (
            AttachmentJournal.model_validate_json(
                read_private(tmp_path / "attachment.json")
            ).state
            == "complete"
        )
    finally:
        if bridge.users:
            await bridge.release()
        await replacement.release()
        await manager.close()
    RuntimeLock(tmp_path, "attachment.lock").close()


@pytest.mark.parametrize("failure", ["second-owner", "host", "completion"])
async def test_restart_recovers_partially_written_attachment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    """Disk faults retain exact local intent; a new manager finishes it before owners load."""
    manager = setup(tmp_path, monkeypatch)
    await manager.start()
    lease = RuntimeLock(tmp_path, "attachment.lock")
    try:
        for identifier in ("managed.first", "managed.second"):
            await manager_request(
                tmp_path, InstallationRequest(action="register", plugin_id=identifier)
            )
        request = AttachmentRequest(
            profile_id=PROFILE,
            transport_node_id="after-reboot",
            skulk_build_sha256="a" * 64,
        )

        def fail_write(path: Path, content: bytes) -> None:
            failed = (
                failure == "second-owner"
                and path == tmp_path / "installations/managed.second/owner.json"
                or failure == "host"
                and path == tmp_path / "host.json"
                or failure == "completion"
                and path == tmp_path / "attachment.json"
                and b'"state":"complete"' in content
            )
            if failed:
                raise OSError("synthetic disk fault")
            write_private(path, content)

        with monkeypatch.context() as fault:
            fault.setattr(
                "skulk.extensions.runtime_attachment.write_private", fail_write
            )
            assert "error" in await manager_request(tmp_path, request)
            assert not manager.controllers
            assert "error" in await manager_request(
                tmp_path,
                InstallationRequest(action="register", plugin_id="managed.third"),
            )
            assert not (tmp_path / "installations/managed.third").exists()
            journal = AttachmentJournal.model_validate_json(
                read_private(tmp_path / "attachment.json")
            )
            assert journal.state == "pending"
            assert set(manager.errors.values()) == {"attachment_recovery_required"}
        await manager.close()
        manager = RuntimeManager(tmp_path)
        await manager.start()
        assert manager.boot is not None
        await manager.boot
        assert set(manager.controllers) == {"managed.first", "managed.second"}
        assert manager.settings.transport_node_id == "after-reboot"
        assert "result" in await manager_request(tmp_path, request)
        for identifier in manager.controllers:
            binding = OwnerBinding.model_validate_json(
                read_private(tmp_path / "installations" / identifier / "owner.json")
            )
            assert binding.transport_node_id == "after-reboot"
    finally:
        lease.close()
        await manager.close()


async def test_foreign_binding_and_missing_bridge_refused_without_rewrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Possessing a manager address cannot silently adopt a foreign installation."""
    manager = setup(tmp_path, monkeypatch)
    await manager.start()
    request = AttachmentRequest(
        profile_id=PROFILE,
        transport_node_id="after-reboot",
        skulk_build_sha256="a" * 64,
    )
    try:
        assert "error" in await manager_request(tmp_path, request)
        await manager_request(
            tmp_path,
            InstallationRequest(action="register", plugin_id="managed.foreign"),
        )
        owner_path = tmp_path / "installations/managed.foreign/owner.json"
        foreign = (
            OwnerBinding(transport_node_id="foreign-host").model_dump_json().encode()
        )
        write_private(owner_path, foreign)
        lease = RuntimeLock(tmp_path, "attachment.lock")
        try:
            assert "error" in await manager_request(tmp_path, request)
        finally:
            lease.close()
        assert read_private(owner_path) == foreign
        assert manager.settings.transport_node_id == "before-reboot"
        assert not (tmp_path / "attachment.json").exists()
    finally:
        await manager.close()


async def test_registered_adapters_share_one_process_attachment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two locally registered installations renew once and retain one API fence."""
    manager = setup(tmp_path / "manager", monkeypatch)
    await manager.start()
    directory = tmp_path / "connections"
    for identifier in ("managed.first", "managed.second"):
        write_private(
            directory / (identifier + ".json"),
            ManagedConnection(
                plugin_id=identifier,
                state_root=str(manager.installations / identifier),
                manager_root=str(manager.root),
                profile_id=PROFILE,
            )
            .model_dump_json()
            .encode(),
        )
    owners = load_managed_owners(directory)
    assert len(owners) == 2
    attachment = owners[0].attachment
    assert attachment is not None and attachment is owners[1].attachment
    attachment.retain("new-api")
    attachment.retain("new-api")
    try:
        await attachment.ensure()
        await attachment.release()
        with pytest.raises(BlockingIOError):
            RuntimeLock(manager.root, "attachment.lock")
        assert manager.settings.transport_node_id == "new-api"
    finally:
        await attachment.release()
        await manager.close()
    RuntimeLock(manager.root, "attachment.lock").close()


def attach(serve_host: str | None) -> AttachmentRequest:
    """An attachment from the live bridge on the provisioned transport."""
    return AttachmentRequest(
        profile_id=PROFILE,
        transport_node_id="before-reboot",
        skulk_build_sha256="a" * 64,
        serve_host=serve_host,
    )


def test_an_attachment_without_an_address_keeps_its_previous_bytes() -> None:
    """A manager built before the address reads the request it always read."""
    assert "serve_host" not in attach(None).model_dump_json()
    assert '"serve_host":"100.70.1.2"' in attach("100.70.1.2").model_dump_json()
    for invalid in ("127.0.0.1", "::1", "0.0.0.0", "224.0.0.1", "node.example"):
        with pytest.raises(ValidationError, match="serve address"):
            attach(invalid)


async def test_a_new_serve_address_reaches_every_installation_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Owners restart once for a new address; an attachment without one keeps it."""
    manager = setup(tmp_path, monkeypatch)
    await manager.start()
    lease = RuntimeLock(tmp_path, "attachment.lock")
    try:
        assert manager.boot is not None
        await manager.boot
        for identifier in ("managed.first", "managed.second"):
            await manager_request(
                tmp_path, InstallationRequest(action="register", plugin_id=identifier)
            )
        first = dict(manager.controllers)
        assert "result" in await manager_request(tmp_path, attach("100.70.1.2"))
        restarted = dict(manager.controllers)
        assert set(restarted) == set(first)
        assert all(restarted[name] is not first[name] for name in first)
        for path in (
            tmp_path / "serve.json",
            *(tmp_path / "installations" / name / "serve.json" for name in first),
        ):
            assert read_serve_host(path) == "100.70.1.2"
        # No address seen keeps the current one, and the same one changes nothing.
        for serve_host in (None, "100.70.1.2"):
            assert "result" in await manager_request(tmp_path, attach(serve_host))
            assert all(manager.controllers[name] is restarted[name] for name in first)
        assert read_serve_host(tmp_path / "serve.json") == "100.70.1.2"
        await manager_request(
            tmp_path, InstallationRequest(action="register", plugin_id="managed.third")
        )
        assert (
            read_serve_host(tmp_path / "installations/managed.third/serve.json")
            == "100.70.1.2"
        )
        await manager.close()
        manager = RuntimeManager(tmp_path)
        assert manager.serve_host == "100.70.1.2"
    finally:
        lease.close()
        await manager.close()


async def test_an_interrupted_serve_address_write_is_finished_next_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A disk fault leaves the old address recorded, and the owners running."""
    manager = setup(tmp_path, monkeypatch)
    await manager.start()
    lease = RuntimeLock(tmp_path, "attachment.lock")
    try:
        assert manager.boot is not None
        await manager.boot
        for identifier in ("managed.first", "managed.second"):
            await manager_request(
                tmp_path, InstallationRequest(action="register", plugin_id=identifier)
            )

        def fail_write(path: Path, content: bytes) -> None:
            if path == tmp_path / "installations/managed.second/serve.json":
                raise OSError("synthetic disk fault")
            write_private(path, content)

        with monkeypatch.context() as fault:
            fault.setattr(
                "skulk.extensions.runtime_attachment.write_private", fail_write
            )
            assert "error" in await manager_request(tmp_path, attach("100.70.1.2"))
        assert set(manager.controllers) == {"managed.first", "managed.second"}
        assert manager.serve_host is None
        assert read_serve_host(tmp_path / "serve.json") is None
        assert "result" in await manager_request(tmp_path, attach("100.70.1.2"))
        for name in ("managed.first", "managed.second"):
            path = tmp_path / "installations" / name / "serve.json"
            assert read_serve_host(path) == "100.70.1.2"
        assert read_serve_host(tmp_path / "serve.json") == "100.70.1.2"
    finally:
        lease.close()
        await manager.close()


async def test_the_bridge_attaches_with_the_address_tailscale_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first answer reaches the manager; a failed query later keeps it."""
    manager = setup(tmp_path, monkeypatch)
    await manager.start()
    reports = [TailscaleStatus(running=True, self_ip="100.70.1.2")]
    answered = asyncio.Event()

    async def status() -> TailscaleStatus:
        await answered.wait()
        return reports[-1]

    monkeypatch.setattr(
        "skulk.extensions.managed_attachment.query_tailscale_status", status
    )
    bridge = ManagedAttachment(tmp_path, PROFILE)
    bridge.retain("before-reboot")
    try:
        assert manager.boot is not None
        await manager.boot
        await manager_request(
            tmp_path, InstallationRequest(action="register", plugin_id="managed.first")
        )
        # An attachment never waits for Tailscale.
        await bridge.ensure()
        assert manager.serve_host is None
        answered.set()
        assert bridge.serve_task is not None
        await bridge.serve_task
        await bridge.ensure()
        assert manager.serve_host == "100.70.1.2"
        path = tmp_path / "installations/managed.first/serve.json"
        assert read_serve_host(path) == "100.70.1.2"
        monkeypatch.setattr(
            "skulk.extensions.managed_attachment.SERVE_ADDRESS_REFRESH_SECONDS", 0.0
        )
        reports.append(TailscaleStatus(running=False))
        await bridge.ensure()
        assert bridge.serve_task is not None
        await bridge.serve_task
        assert bridge.serve_host == "100.70.1.2"
        assert read_serve_host(path) == "100.70.1.2"
    finally:
        await bridge.release()
        await manager.close()
    RuntimeLock(tmp_path, "attachment.lock").close()
