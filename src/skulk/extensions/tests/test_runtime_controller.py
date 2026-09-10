"""Durable owner operations with actual isolated processes and interrupted selection."""

import asyncio
from pathlib import Path

import pytest

from skulk.extensions.runtime_controller import LifecycleRequest, RuntimeController
from skulk.extensions.runtime_files import RuntimeLock, write_private
from skulk.extensions.tests.test_runtime_service import installed, running


async def test_revision_refusal_preserves_owner_and_disable_reconnect_never_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validate before stopping; duplicate accepted requests retain one transition."""
    selector = await installed(tmp_path, monkeypatch)
    controller = RuntimeController(selector.root)
    await controller.start()
    try:
        assert controller.service is not None
        await running(controller.service)
        process = controller.service.process
        assert process is not None
        stale = LifecycleRequest(
            operation_id="a" * 32, action="disable", expected_revision=0
        )
        with pytest.raises(ValueError, match="revision"):
            await controller.submit(stale)
        assert process.returncode is None
        selection = selector.current()
        assert selection is not None
        invalid = LifecycleRequest(
            operation_id="b" * 32,
            action="activate",
            expected_revision=1,
            runtime_digest="0" * 64,
        )
        with pytest.raises(FileNotFoundError):
            await controller.submit(invalid)
        assert process.returncode is None
        request = LifecycleRequest(
            operation_id="c" * 32, action="disable", expected_revision=1
        )
        accepted = await controller.submit(request)
        assert accepted.state == "accepted"
        assert controller.work is not None
        await controller.work
        complete = controller.operation(request.operation_id)
        assert complete.state == "complete"
        assert process.returncode == 0
        assert not complete.selection.enabled and complete.selection.revision == 2
        assert await controller.submit(request) == complete
        with pytest.raises(ValueError, match="identity"):
            await controller.submit(request.model_copy(update={"expected_revision": 2}))
        assert selector.current() == complete.selection
        assert (selector.root / "generations" / selection.runtime_digest).is_dir()
    finally:
        await controller.close()
    RuntimeLock(selector.root, "manager.lock").close()


async def test_owner_switch_is_completed_after_requester_disappears(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Accepted work survives losing the requesting task and preserves durable state."""
    selector = await installed(tmp_path, monkeypatch)
    controller = RuntimeController(selector.root)
    await controller.start()
    try:
        assert controller.service is not None
        await running(controller.service)
        previous = controller.service.process
        selection = selector.current()
        assert selection is not None
        write_private(selector.root / "receipts", b"synthetic outstanding cleanup")
        request = LifecycleRequest(
            operation_id="d" * 32,
            action="activate",
            expected_revision=1,
            runtime_digest=selection.runtime_digest,
        )
        submitted = asyncio.create_task(controller.submit(request))
        accepted = await submitted
        assert accepted.state == "accepted"
        assert controller.work is not None
        await controller.work
        assert controller.operation(request.operation_id).state == "complete"
        assert previous is not None and previous.returncode == 0
        assert controller.service is not None
        await running(controller.service)
        assert controller.service.process is not previous
        assert (
            selector.root / "receipts"
        ).read_bytes() == b"synthetic outstanding cleanup"
        assert selector.current() == accepted.selection
    finally:
        await controller.close()


@pytest.mark.parametrize("fault", ["before_selection", "after_selection"])
async def test_restart_completes_only_recorded_local_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    """Interrupted writes recover exact local intent once, with no provider operations."""
    selector = await installed(tmp_path, monkeypatch)
    controller = RuntimeController(selector.root)
    await controller.start()
    request = LifecycleRequest(
        operation_id="e" * 32, action="disable", expected_revision=1
    )
    original = selector.root / "runtime-selection.json"

    def fail_selection(path: Path, content: bytes) -> None:
        if path == original:
            if fault == "after_selection":
                write_private(path, content)
            raise OSError("synthetic publication interruption")
        write_private(path, content)

    try:
        with monkeypatch.context() as failure:
            failure.setattr(
                "skulk.extensions.runtime_selection.write_private", fail_selection
            )
            await controller.submit(request)
            assert controller.work is not None
            await controller.work
        assert controller.operation(request.operation_id).state == "recovery_required"
    finally:
        await controller.close()
    resumed = RuntimeController(selector.root)
    try:
        await resumed.start()
        complete = resumed.operation(request.operation_id)
        assert complete.state == "complete"
        assert selector.current() == complete.selection
        assert complete.selection.revision == 2
        assert not resumed.pending.exists()
        assert await resumed.submit(request) == complete
    finally:
        await resumed.close()


async def test_close_finishes_already_accepted_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Manager shutdown retains the accepted disable instead of reviving it at boot."""
    selector = await installed(tmp_path, monkeypatch)
    controller = RuntimeController(selector.root)
    await controller.start()
    request = LifecycleRequest(
        operation_id="f" * 32, action="disable", expected_revision=1
    )
    await controller.submit(request)
    await controller.close()
    assert controller.operation(request.operation_id).state == "complete"
    assert controller.service_task is None
    RuntimeLock(selector.root, "manager.lock").close()
    resumed = RuntimeController(selector.root)
    try:
        await resumed.start()
        assert resumed.service_task is not None
        await resumed.service_task
        assert resumed.service is not None and resumed.service.process is None
    finally:
        await resumed.close()


async def test_accepted_journal_failure_is_explicitly_recoverable_without_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed acknowledgement never appears as running work or needs a new ID."""
    selector = await installed(tmp_path, monkeypatch)
    controller = RuntimeController(selector.root)
    await controller.start()
    request = LifecycleRequest(
        operation_id="9" * 32, action="disable", expected_revision=1
    )
    record = controller.records / (request.operation_id + ".json")

    def failed_record(path: Path, value: bytes) -> None:
        write_private(path, value)
        if path == record:
            raise OSError("synthetic post-rename sync failure")

    try:
        with monkeypatch.context() as failure:
            failure.setattr(
                "skulk.extensions.runtime_controller.write_private", failed_record
            )
            with pytest.raises(OSError):
                await controller.submit(request)
        interrupted_work = controller.work
        assert interrupted_work is None
        assert controller.operation(request.operation_id).state == "recovery_required"
        assert (await controller.submit(request)).state == "recovery_required"
        recovered = await controller.recover(request.operation_id)
        assert recovered.state == "accepted"
        assert controller.work is not None
        await controller.work
        complete = controller.operation(request.operation_id)
        assert complete.state == "complete" and complete.selection.revision == 2
        assert await controller.recover(request.operation_id) == complete
    finally:
        await controller.close()


async def staged_controller(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[RuntimeController, str]:
    """Prepare an empty verified installation whose owner visibly marks any start."""
    from skulk.extensions.tests.test_runtime_install import artifacts
    from skulk.extensions.tests.test_runtime_service import OWNER_SOURCE

    source = OWNER_SOURCE.replace(
        "print('sensitive fixture output',flush=True)",
        "open(a.root+'/owner-started','w').write('started')\n"
        "print('sensitive fixture output',flush=True)",
    )
    metadata, trust, host = artifacts(root / "source", owner_source=source)
    monkeypatch.setattr("skulk.extensions.runtime_install.measure_host", lambda: host)
    controller = RuntimeController(root / "installed")
    write_private(controller.root / "publisher-trust.json", trust.model_dump_json().encode())
    staged = await controller.selector.installer.stage(metadata, root / "source")
    return controller, staged.runtime_digest


async def test_select_stopped_generation_requires_later_explicit_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A first selection and manager restart must not initialize private owner state."""
    controller, digest = await staged_controller(tmp_path, monkeypatch)
    request = LifecycleRequest(
        operation_id="1" * 32, action="select", expected_revision=0, runtime_digest=digest
    )
    await controller.start()
    try:
        await controller.submit(request)
        assert controller.work is not None
        await controller.work
        selected = controller.operation(request.operation_id)
        assert selected.state == "complete" and not selected.selection.enabled
        assert not (controller.root / "owner-started").exists()
        assert await controller.submit(request) == selected
        with pytest.raises(ValueError, match="identity"):
            await controller.submit(request.model_copy(update={"action": "activate"}))
    finally:
        await controller.close()
    resumed = RuntimeController(controller.root)
    try:
        await resumed.start()
        assert resumed.service_task is not None
        await resumed.service_task
        assert not (resumed.root / "owner-started").exists()
        await resumed.submit(LifecycleRequest(
            operation_id="2" * 32, action="activate", expected_revision=1, runtime_digest=digest
        ))
        assert resumed.work is not None
        await resumed.work
        assert resumed.service is not None
        await running(resumed.service)
        assert (resumed.root / "owner-started").read_text() == "started"
    finally:
        await resumed.close()


@pytest.mark.parametrize("fault", ["before_selection", "after_selection"])
@pytest.mark.parametrize("damage", ["none", "artifact", "revocation", "journal"])
async def test_stopped_selection_recovery_revalidates_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str, damage: str
) -> None:
    """Interrupted stopped selection cannot use disable's invalid-trust recovery."""
    from skulk.extensions.runtime_artifacts import RuntimeTrust
    from skulk.extensions.runtime_files import read_private

    controller, digest = await staged_controller(tmp_path, monkeypatch)
    request = LifecycleRequest(
        operation_id="3" * 32, action="select", expected_revision=0, runtime_digest=digest
    )
    await controller.start()
    selection_path = controller.root / "runtime-selection.json"

    def fail_selection(path: Path, content: bytes) -> None:
        if path == selection_path:
            if fault == "after_selection":
                write_private(path, content)
            raise OSError("synthetic selection interruption")
        write_private(path, content)

    try:
        with monkeypatch.context() as failure:
            failure.setattr("skulk.extensions.runtime_selection.write_private", fail_selection)
            await controller.submit(request)
            assert controller.work is not None
            await controller.work
        assert controller.operation(request.operation_id).state == "recovery_required"
    finally:
        await controller.close()
    if damage == "journal":
        from skulk.extensions.runtime_selection import SelectionOperation

        pending = controller.selector.pending
        payload = SelectionOperation.model_validate_json(read_private(pending))
        write_private(
            pending, payload.model_copy(update={"verify_runtime": False}).model_dump_json().encode()
        )
    elif damage == "artifact":
        artifact = controller.root / "generations" / digest / "artifacts/bundle.pyz"
        artifact.write_bytes(artifact.read_bytes() + b"tampered")
    elif damage == "revocation":
        path = controller.root / "publisher-trust.json"
        trust = RuntimeTrust.model_validate_json(read_private(path))
        write_private(path, trust.model_copy(update={"revision": 2, "revoked_artifacts": (digest,)}).model_dump_json().encode())
    resumed = RuntimeController(controller.root)
    try:
        await resumed.start()
        observed = resumed.operation(request.operation_id)
        assert (observed.state == "complete") == (damage == "none")
        assert not (resumed.root / "owner-started").exists()
        if resumed.service_task is not None:
            await resumed.service_task
        assert resumed.service is None or resumed.service.process is None
    finally:
        await resumed.close()
