"""Group-connect deadline tests (#265).

``mx.distributed.init(backend="ring", strict=True)`` blocks forever when a
neighbor socket fails the post-TCP rank handshake; the ConnectToGroup site
now runs under ``deadline_watchdog`` so a stalled group becomes a clean
runner death (wedge path, #260: first-failure give-up, fresh placement,
fresh ring port) instead of an eternal probe-timeout/cancel loop.
"""

import threading

import pytest

from skulk.worker.runner.bootstrap import (
    GROUP_CONNECT_DEADLINE_SECONDS_DEFAULT,
    GROUP_CONNECT_STALL_DIAGNOSIS,
    deadline_message,
    deadline_watchdog,
    resolve_group_connect_deadline_seconds,
)


def test_default_deadline(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SKULK_GROUP_CONNECT_DEADLINE_SECONDS", raising=False)
    monkeypatch.delenv("EXO_GROUP_CONNECT_DEADLINE_SECONDS", raising=False)
    assert (
        resolve_group_connect_deadline_seconds()
        == GROUP_CONNECT_DEADLINE_SECONDS_DEFAULT
    )


def test_operator_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SKULK_GROUP_CONNECT_DEADLINE_SECONDS", "45")
    assert resolve_group_connect_deadline_seconds() == 45.0


def test_invalid_override_falls_back(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SKULK_GROUP_CONNECT_DEADLINE_SECONDS", "-3")
    assert (
        resolve_group_connect_deadline_seconds()
        == GROUP_CONNECT_DEADLINE_SECONDS_DEFAULT
    )
    monkeypatch.setenv("SKULK_GROUP_CONNECT_DEADLINE_SECONDS", "soon")
    assert (
        resolve_group_connect_deadline_seconds()
        == GROUP_CONNECT_DEADLINE_SECONDS_DEFAULT
    )


def test_deadline_message_uses_network_diagnosis():
    # The group-connect site must not emit the GPU-wedge guidance — a ring
    # stall is a NETWORK condition; "test the GPU with a small matmul" would
    # send operators chasing the wrong subsystem.
    message = deadline_message(
        "Group connect for test-model", 120.0, GROUP_CONNECT_STALL_DIAGNOSIS
    )
    assert "distributed group never formed" in message
    assert "GPU" not in message
    # And the default keeps the Metal guidance for eval/warmup sites.
    default = deadline_message("Warmup of test-model", 300.0)
    assert "GPU may be wedged" in default


def test_watchdog_fires_on_stall_and_not_on_completion():
    fired = threading.Event()
    with deadline_watchdog(0.05, "stall", on_timeout=fired.set):
        assert fired.wait(2.0)

    fired_fast = threading.Event()
    with deadline_watchdog(5.0, "fast block", on_timeout=fired_fast.set):
        pass
    assert not fired_fast.wait(0.2)
