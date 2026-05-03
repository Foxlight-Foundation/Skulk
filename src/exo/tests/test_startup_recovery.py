"""Tests for the startup port preflight."""

from __future__ import annotations

import socket
from typing import cast

import pytest

from exo.startup_recovery import preflight_api_port


def _port_of(sock: socket.socket) -> int:
    sockname = cast(tuple[str, int], sock.getsockname())
    return sockname[1]


def _grab_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("0.0.0.0", 0))
        return _port_of(probe)


def test_preflight_returns_when_port_is_free():
    """Free port → returns cleanly, no exit."""
    port = _grab_free_port()
    preflight_api_port(port)


def test_preflight_exits_when_port_is_held():
    """Port already bound by another listener → SystemExit with EX_TEMPFAIL.

    SO_REUSEADDR makes simultaneous bind() calls on the same port succeed in
    some kernels until one of them listen()s. We hold the port with listen()
    so the preflight's probe bind reliably fails with EADDRINUSE, matching
    the production failure mode (a previous Skulk's API server still
    bound and listening).
    """
    held = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        held.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        held.bind(("0.0.0.0", 0))
        held.listen(1)
        with pytest.raises(SystemExit) as exit_info:
            preflight_api_port(_port_of(held))
        assert exit_info.value.code == 75
    finally:
        held.close()
