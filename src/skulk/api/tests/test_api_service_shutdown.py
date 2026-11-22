"""API shutdown drains independently owned brokers across failures and cancellation."""

from typing import final

import anyio
import pytest

from skulk.api.main import API


@final
class CloseProbe:
    """Observe completed asynchronous close calls without opening transports."""

    def __init__(self, name: str, closed: list[str], failure: str | None) -> None:
        """Retain the service identity, completion observations and chosen failure."""
        self.name = name
        self.closed = closed
        self.failure = failure

    async def close(self) -> None:
        """Yield to cancellation, record completion and optionally report a failure."""
        await anyio.sleep(0)
        self.closed.append(self.name)
        if self.name == self.failure:
            raise RuntimeError(self.name)


@pytest.mark.parametrize("failure", [None, "native", "music", "separation"])
async def test_api_closes_every_broker_after_a_service_failure(
    monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    """One failing broker cannot prevent either sibling from releasing its work."""
    api = API.__new__(API)
    closed: list[str] = []
    for field, name in (
        ("_native_inference_service", "native"),
        ("_music_edit_service", "music"),
        ("_separation_service", "separation"),
    ):
        monkeypatch.setattr(
            api, field, CloseProbe(name, closed, failure), raising=False
        )
    if failure is None:
        await api.close_inference_services()
    else:
        with pytest.raises(RuntimeError, match=failure):
            await api.close_inference_services()
    assert closed == ["native", "music", "separation"]


async def test_cancelled_api_still_drains_all_service_brokers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Listener cancellation cannot interrupt the asynchronous ownership drain."""
    api = API.__new__(API)
    closed: list[str] = []
    for field, name in (
        ("_native_inference_service", "native"),
        ("_music_edit_service", "music"),
        ("_separation_service", "separation"),
    ):
        monkeypatch.setattr(api, field, CloseProbe(name, closed, None), raising=False)
    with anyio.CancelScope() as scope:
        scope.cancel()
        await api.close_inference_services()
    assert closed == ["native", "music", "separation"]
