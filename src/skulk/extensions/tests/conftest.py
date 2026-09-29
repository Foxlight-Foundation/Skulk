"""Keep the plugin bridge's Tailscale observation off the machine running the tests."""

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
