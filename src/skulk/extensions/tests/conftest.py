"""Keep the plugin bridge's Tailscale observation off the machine running the tests."""

from pathlib import Path

import pytest

from skulk.connectivity.tailscale import TailscaleStatus


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
