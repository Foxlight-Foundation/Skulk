# pyright: reportPrivateUsage=false, reportAny=false
"""Post-teardown placement settle window (planner lag tolerance).

After this API node tears an instance down, freed GPU memory lags in gossiped
telemetry, so a back-to-back placement preview can transiently find no fit. The
previews path re-runs within ``_POST_TEARDOWN_SETTLE_SECONDS`` of a teardown and
returns the first viable result; outside the window it returns immediately.
"""

import time
from typing import Any
from unittest.mock import AsyncMock

import anyio
import pytest

from skulk.api.main import API
from skulk.api.types.api import PlacementPreview, PlacementPreviewResponse
from skulk.shared.types.common import ModelId
from skulk.shared.types.worker.instances import InstanceMeta
from skulk.shared.types.worker.shards import Sharding


def _api() -> Any:
    api = object.__new__(API)
    api._last_teardown_monotonic = 0.0
    return api


def _errored() -> PlacementPreviewResponse:
    return PlacementPreviewResponse(
        previews=[
            PlacementPreview(
                model_id=ModelId("m"),
                sharding=Sharding.Pipeline,
                instance_meta=InstanceMeta.MlxRing,
                instance=None,
                error="No usable placement preview found",
            )
        ]
    )


def test_settle_remaining_window() -> None:
    api = _api()
    assert api._post_teardown_settle_remaining() == 0.0  # never torn down
    api._last_teardown_monotonic = time.monotonic()
    assert api._post_teardown_settle_remaining() > 0.0  # fresh teardown
    api._last_teardown_monotonic = time.monotonic() - 10_000
    assert api._post_teardown_settle_remaining() == 0.0  # long past


async def test_previews_retry_within_settle_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh teardown makes previews retry until a viable placement appears."""
    api = _api()
    api._last_teardown_monotonic = time.monotonic()  # in-window
    monkeypatch.setattr(anyio, "sleep", AsyncMock())  # no real delay

    # A viable preview just needs a non-None instance; model_construct skips
    # validation so we can use a lightweight sentinel rather than a full Instance.
    viable = PlacementPreviewResponse(
        previews=[
            PlacementPreview.model_construct(
                model_id=ModelId("m"),
                sharding=Sharding.Pipeline,
                instance_meta=InstanceMeta.MlxRing,
                instance=object(),
                error=None,
            )
        ]
    )

    calls = {"n": 0}

    async def fake_compute(*_a: Any, **_k: Any) -> tuple[PlacementPreviewResponse, bool]:
        calls["n"] += 1
        # (response, saw_memory_shortfall): errored-on-memory until the 3rd pass.
        return (_errored(), True) if calls["n"] < 3 else (viable, False)

    api._compute_placement_previews = fake_compute
    result = await api.get_placement_previews(ModelId("m"))
    assert any(p.instance is not None for p in result.previews)
    assert calls["n"] == 3  # retried twice through the memory lag, then viable


async def test_previews_no_retry_outside_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no recent teardown, previews return the first (all-errored) pass."""
    api = _api()
    api._last_teardown_monotonic = 0.0  # no recent teardown
    monkeypatch.setattr(anyio, "sleep", AsyncMock())

    calls = {"n": 0}

    async def fake_compute(*_a: Any, **_k: Any) -> tuple[PlacementPreviewResponse, bool]:
        calls["n"] += 1
        return _errored(), True  # memory shortfall, but out of window

    api._compute_placement_previews = fake_compute
    result = await api.get_placement_previews(ModelId("m"))
    assert all(p.instance is None for p in result.previews)
    assert calls["n"] == 1  # single pass, no retry


async def test_previews_no_retry_on_deterministic_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In-window but a non-memory error (backend/topology) does not retry,
    since waiting cannot change it, so it returns immediately."""
    api = _api()
    api._last_teardown_monotonic = time.monotonic()  # in-window
    monkeypatch.setattr(anyio, "sleep", AsyncMock())

    calls = {"n": 0}

    async def fake_compute(*_a: Any, **_k: Any) -> tuple[PlacementPreviewResponse, bool]:
        calls["n"] += 1
        return _errored(), False  # deterministic (non-memory) failure

    api._compute_placement_previews = fake_compute
    result = await api.get_placement_previews(ModelId("m"))
    assert all(p.instance is None for p in result.previews)
    assert calls["n"] == 1  # no retry despite being in-window
