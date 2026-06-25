"""Tests for the model-store advertise-host selection (routable IP, not hostname).

The store host broadcasts ``store_http_host`` for workers to build the download
URL. A bare hostname can mDNS-resolve to a Thunderbolt link-local address that
peers without a direct TB link cannot route to, so we advertise a routable IP.
"""

import socket
from typing import NamedTuple

import pytest

import skulk.main as main


class _FakeAddr(NamedTuple):
    """Mirror the snicaddr fields the code reads (family, address)."""

    family: int
    address: str


def _ifaddrs(*addresses: str) -> dict[str, list[_FakeAddr]]:
    """Build a fake ``psutil.net_if_addrs()`` return for the given IPv4 addrs."""
    return {
        f"if{i}": [_FakeAddr(family=socket.AF_INET, address=addr)]
        for i, addr in enumerate(addresses)
    }


def test_explicit_routable_ip_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main.psutil, "net_if_addrs", lambda: _ifaddrs("192.168.0.122"))
    # An operator-supplied routable IP literal is used verbatim.
    assert main._routable_store_advertise_host("10.0.0.5", "kite3") == "10.0.0.5"  # pyright: ignore[reportPrivateUsage]


def test_hostname_is_replaced_with_routable_lan_ip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The deployed bug: store_http_host=kite3.local resolves to a TB link-local.
    monkeypatch.setattr(
        main.psutil,
        "net_if_addrs",
        lambda: _ifaddrs("127.0.0.1", "169.254.201.94", "192.168.0.122"),
    )
    assert (
        main._routable_store_advertise_host("kite3.local", "kite3")  # pyright: ignore[reportPrivateUsage]
        == "192.168.0.122"
    )


def test_link_local_literal_is_replaced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main.psutil, "net_if_addrs", lambda: _ifaddrs("192.168.0.122"))
    assert (
        main._routable_store_advertise_host("169.254.1.1", "kite3")  # pyright: ignore[reportPrivateUsage]
        == "192.168.0.122"
    )


def test_lan_preferred_over_tailscale(monkeypatch: pytest.MonkeyPatch) -> None:
    # CGNAT (100.64/10, Tailscale) is also "private"; LAN must still win.
    monkeypatch.setattr(
        main.psutil,
        "net_if_addrs",
        lambda: _ifaddrs("100.88.129.94", "192.168.0.122"),
    )
    assert (
        main._routable_store_advertise_host(None, "kite3") == "192.168.0.122"  # pyright: ignore[reportPrivateUsage]
    )


def test_tailscale_used_when_no_lan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        main.psutil,
        "net_if_addrs",
        lambda: _ifaddrs("169.254.201.94", "100.88.129.94"),
    )
    assert (
        main._routable_store_advertise_host(None, "kite3") == "100.88.129.94"  # pyright: ignore[reportPrivateUsage]
    )


def test_falls_back_to_hostname_when_no_routable_ip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Only loopback + link-local present: nothing routable, keep the hostname.
    monkeypatch.setattr(
        main.psutil, "net_if_addrs", lambda: _ifaddrs("127.0.0.1", "169.254.201.94")
    )
    assert main._routable_store_advertise_host(None, "kite3") == "kite3"  # pyright: ignore[reportPrivateUsage]
