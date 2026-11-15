"""Shared broker generation custody with actual protected caller stores."""

import asyncio
from pathlib import Path

import pytest
from fastapi import HTTPException

from skulk.api.native_inference import create_native_inference_router
from skulk.api.native_inference_service import NativeInferenceService
from skulk.api.tests.test_inference_access import application
from skulk.extensions.native_inference_calls import NativeInferenceCalls
from skulk.extensions.runtime_attachment import ServiceConnection
from skulk.extensions.runtime_files import write_private
from skulk.shared.types.common import NodeId


def service(root: Path) -> NativeInferenceService:
    """Establish real private custody, without an installed provider or socket."""
    application(root)
    return NativeInferenceService(
        NodeId("access-core"), connection_path=root / "connection.json"
    )


async def test_concurrent_adapters_share_one_broker_and_shutdown(
    tmp_path: Path,
) -> None:
    """Acquisition shares peers, while closing prevents any later resurrection."""
    shared = service(tmp_path)
    brokers = await asyncio.gather(*(shared.host() for _ in range(16)))
    assert all(value is brokers[0] for value in brokers)
    app, _ = application(tmp_path)
    app.include_router(
        create_native_inference_router(NodeId("access-core"), inference_service=shared)
    )
    async with app.router.lifespan_context(app):
        assert await shared.host() is brokers[0]
    assert brokers[0].closed and brokers[0].hosts.budget.used == 0
    await shared.close()
    with pytest.raises(HTTPException, match="stopped"):
        await shared.host()


@pytest.mark.parametrize("cancel_leader", [False, True])
async def test_replacement_drains_before_waiting_adapter_acquires(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel_leader: bool
) -> None:
    """A cancelled discovery cannot strand retirement or expose two live pools."""
    shared = service(tmp_path)
    previous = await shared.host()
    application(tmp_path / "replacement")
    write_private(
        tmp_path / "connection.json",
        ServiceConnection(
            manager_root=str(tmp_path / "replacement"), profile_id="a" * 32
        )
        .model_dump_json()
        .encode(),
    )
    entered, release = asyncio.Event(), asyncio.Event()
    original = NativeInferenceCalls.close

    async def close(broker: NativeInferenceCalls) -> None:
        """Stall only retirement, leaving real close and lease cleanup intact."""
        if broker is previous:
            entered.set()
            await release.wait()
        await original(broker)

    monkeypatch.setattr(NativeInferenceCalls, "close", close)
    leader = asyncio.create_task(shared.host())
    follower: asyncio.Task[NativeInferenceCalls] | None = None
    try:
        async with asyncio.timeout(2):
            await entered.wait()
            follower = asyncio.create_task(shared.host())
            await asyncio.sleep(0)
            assert not follower.done()
            if cancel_leader:
                leader.cancel()
                await asyncio.sleep(0)
                assert not leader.done() and not follower.done()
            release.set()
            outcomes = await asyncio.gather(leader, return_exceptions=True)
            replacement = await follower
            if cancel_leader:
                assert isinstance(outcomes[0], asyncio.CancelledError)
            else:
                assert outcomes[0] is replacement
            assert previous.closed and replacement is not previous
            assert await shared.host() is replacement
    finally:
        release.set()
        if not leader.done():
            leader.cancel()
        if follower is not None and not follower.done():
            follower.cancel()
        await asyncio.gather(
            leader, *(() if follower is None else (follower,)), return_exceptions=True
        )
        await shared.close()


@pytest.mark.parametrize("fault", ["missing", "invalid", "permissions"])
async def test_invalid_custody_withdraws_old_broker(tmp_path: Path, fault: str) -> None:
    """Failure to read current custody cannot leave the old pool reusable."""
    shared = service(tmp_path)
    previous = await shared.host()
    path = tmp_path / "connection.json"
    if fault == "missing":
        path.unlink()
    elif fault == "invalid":
        write_private(path, b"{}")
    else:
        path.chmod(0o644)
    try:
        with pytest.raises(HTTPException, match="unavailable"):
            await shared.host()
        assert previous.closed and previous.hosts.budget.used == 0
    finally:
        await shared.close()


def test_router_refuses_a_different_shared_issuer(tmp_path: Path) -> None:
    """Sharing transport custody never substitutes another server's principal."""
    with pytest.raises(ValueError, match="issuer differs"):
        create_native_inference_router(
            NodeId("different-core"), inference_service=service(tmp_path)
        )
