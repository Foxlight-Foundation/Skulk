"""Keep the plugin bridge's Tailscale observation off the machine running the tests."""

import asyncio
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from skulk.connectivity.tailscale import TailscaleStatus
from skulk.extensions.native_host_budget import NativePayloadBudget
from skulk.extensions.native_host_protocol import (
    AuthenticatedStream,
    admit_core,
    admit_owner,
)
from skulk.extensions.tests.test_native_host_protocol import Channels, binding


@pytest.fixture(autouse=True)
def no_tailnet(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report no Tailscale, so a developer's own tailnet never restarts test owners.

    A test that exercises the serve address replaces this with its own report.
    """

    async def not_running() -> TailscaleStatus:
        return TailscaleStatus(running=False)

    monkeypatch.setattr(
        "skulk.extensions.managed_attachment.query_tailscale_status", not_running
    )


@pytest.fixture(autouse=True)
def no_builtin_store(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every manager without the built-in capability store's network trust.

    Once a build ships the store's root, a manager would otherwise reach the
    real store on a catalog read. Tests of the store give it their own
    repository explicitly.
    """
    monkeypatch.setattr(
        "skulk.extensions.runtime_manager.EMBEDDED_CAPABILITY_STORE_ROOT",
        Path(__file__).with_name("no-capability-store-root.json"),
    )


@pytest.fixture
async def channels() -> AsyncIterator[Channels]:
    """Run real role proofs on a short Unix path, with separate shared budgets."""
    key = Ed25519PrivateKey.generate()
    expected = binding(key)
    core_budget, owner_budget = NativePayloadBudget(), NativePayloadBudget()
    accepted: asyncio.Future[AuthenticatedStream] = (
        asyncio.get_running_loop().create_future()
    )
    tasks: set[asyncio.Task[None]] = set()

    async def accept(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.current_task()
        assert task is not None
        tasks.add(task)
        try:
            keys = await admit_core(reader, writer, expected, lambda: None)
            accepted.set_result(AuthenticatedStream(reader, writer, keys, owner_budget))
        except BaseException as error:
            writer.close()
            if not accepted.done():
                accepted.set_exception(error)
        finally:
            tasks.discard(task)

    with tempfile.TemporaryDirectory(prefix="nhost-", dir="/tmp") as directory:
        socket = str(Path(directory) / "host")
        server = await asyncio.start_unix_server(accept, socket)
        core: AuthenticatedStream | None = None
        owner: AuthenticatedStream | None = None
        try:
            async with asyncio.timeout(3):
                reader, writer = await asyncio.open_unix_connection(socket)
                keys = await admit_owner(reader, writer, expected, key, lambda: None)
                core = AuthenticatedStream(reader, writer, keys, core_budget)
                owner = await accepted
            yield Channels(core, owner, core_budget, owner_budget)
        finally:
            if core is not None:
                await core.close()
            if owner is not None:
                await owner.close()
            server.close()
            await server.wait_closed()
            for task in list(tasks):
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            assert core_budget.used == owner_budget.used == 0
