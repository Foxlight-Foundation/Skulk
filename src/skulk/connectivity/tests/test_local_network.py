# pyright: reportPrivateUsage=false
"""Tests for macOS Local Network Privacy detection."""

from __future__ import annotations

import errno
import socket
from unittest import mock

from skulk.connectivity import local_network


def test_non_darwin_is_unknown() -> None:
    with mock.patch("platform.system", return_value="Linux"):
        assert local_network.check_local_network_access() == "unknown"


def test_no_gateway_is_unknown() -> None:
    with (
        mock.patch("platform.system", return_value="Darwin"),
        mock.patch.object(local_network, "_default_gateway_ipv4", return_value=None),
    ):
        assert local_network.check_local_network_access() == "unknown"


def test_ehostunreach_is_blocked() -> None:
    """EHOSTUNREACH on a local-subnet connect is the Local Network denial signature."""
    sock = mock.Mock()
    sock.connect.side_effect = OSError(errno.EHOSTUNREACH, "No route to host")
    with (
        mock.patch("platform.system", return_value="Darwin"),
        mock.patch.object(local_network, "_default_gateway_ipv4", return_value="192.168.0.1"),
        mock.patch("socket.socket", return_value=sock),
    ):
        assert local_network.check_local_network_access() == "blocked"
    sock.close.assert_called_once()


def test_connection_refused_is_ok() -> None:
    """A reachable host with a closed port (ECONNREFUSED) means access is allowed."""
    sock = mock.Mock()
    sock.connect.side_effect = OSError(errno.ECONNREFUSED, "Connection refused")
    with (
        mock.patch("platform.system", return_value="Darwin"),
        mock.patch.object(local_network, "_default_gateway_ipv4", return_value="192.168.0.1"),
        mock.patch("socket.socket", return_value=sock),
    ):
        assert local_network.check_local_network_access() == "ok"


def test_successful_connect_is_ok() -> None:
    sock = mock.Mock()
    sock.connect.return_value = None
    with (
        mock.patch("platform.system", return_value="Darwin"),
        mock.patch.object(local_network, "_default_gateway_ipv4", return_value="192.168.0.1"),
        mock.patch("socket.socket", return_value=sock),
    ):
        assert local_network.check_local_network_access() == "ok"


def test_gateway_parser_extracts_ipv4() -> None:
    route_output = "   route to: default\n   gateway: 192.168.0.1\n   interface: en0\n"
    completed = mock.Mock(stdout=route_output)
    with mock.patch("subprocess.run", return_value=completed):
        assert local_network._default_gateway_ipv4() == "192.168.0.1"


def test_gateway_parser_rejects_ipv6() -> None:
    route_output = "   gateway: fe80::1%en0\n"
    completed = mock.Mock(stdout=route_output)
    with mock.patch("subprocess.run", return_value=completed):
        assert local_network._default_gateway_ipv4() is None


def test_blocked_message_is_actionable() -> None:
    msg = local_network.LOCAL_NETWORK_DENIED_MESSAGE
    assert "Local Network" in msg
    assert "System Settings" in msg
