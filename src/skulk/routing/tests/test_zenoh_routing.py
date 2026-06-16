"""Unit tests for the Router's data-plane transport selection (Zenoh, #279).

The Router routes the DATA topic over Zenoh only when a ZenohHandle is present
(the SKULK_ZENOH_DATA_PLANE flag); every other topic, and all topics when the
flag is off, stay on libp2p gossipsub. These tests pin that decision without
opening a network session.
"""

from typing import cast

from skulk_pyo3_bindings import NetworkingHandle, ZenohHandle

from skulk.routing.router import Router
from skulk.routing.topics import COMMANDS, DATA, GLOBAL_EVENTS


def _router(*, zenoh: bool) -> Router:
    # uses_zenoh only inspects topic identity and whether a handle is present;
    # the handles are never called here, so opaque stand-ins are sufficient.
    fake_net = cast(NetworkingHandle, object())
    fake_zenoh = cast(ZenohHandle, object()) if zenoh else None
    return Router(handle=fake_net, zenoh=fake_zenoh)


def test_data_routes_over_zenoh_when_enabled() -> None:
    router = _router(zenoh=True)
    assert router.uses_zenoh(DATA.topic) is True
    # Control/telemetry/election planes stay on gossipsub even with zenoh on.
    assert router.uses_zenoh(COMMANDS.topic) is False
    assert router.uses_zenoh(GLOBAL_EVENTS.topic) is False


def test_nothing_routes_over_zenoh_when_disabled() -> None:
    router = _router(zenoh=False)
    assert router.uses_zenoh(DATA.topic) is False
    assert router.uses_zenoh(COMMANDS.topic) is False
