# pyright: reportAny=false
"""Tests for TailscaleStatus JSON parsing."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from skulk.connectivity import tailscale as tailscale_module
from skulk.connectivity.tailscale import parse_status_json

_FULL_STATUS: dict[str, Any] = {
    "Version": "1.66.1-t82d4e3b99-g7b76cfb8f",
    "BackendState": "Running",
    "Self": {
        "HostName": "my-node",
        "DNSName": "my-node.tailnet-abc.ts.net.",
        "TailscaleIPs": ["100.101.102.103", "fd7a:115c:a1e0::1"],
    },
}


def test_running_status_parses_correctly() -> None:
    status = parse_status_json(_FULL_STATUS)
    assert status.running is True
    assert status.self_ip == "100.101.102.103"
    assert status.hostname == "my-node"
    assert status.dns_name == "my-node.tailnet-abc.ts.net"
    assert status.tailnet == "tailnet-abc.ts.net"
    assert status.version == "1.66.1-t82d4e3b99-g7b76cfb8f"


def test_dns_name_trailing_dot_stripped() -> None:
    self_override: dict[str, Any] = {**_FULL_STATUS["Self"], "DNSName": "my-node.tailnet-abc.ts.net."}
    raw: dict[str, Any] = {**_FULL_STATUS, "Self": self_override}
    status = parse_status_json(raw)
    assert status.dns_name == "my-node.tailnet-abc.ts.net"
    assert status.dns_name is not None and not status.dns_name.endswith(".")


def test_tailnet_derived_from_dns_name() -> None:
    self_override: dict[str, Any] = {**_FULL_STATUS["Self"], "DNSName": "myhost.example-corp.ts.net"}
    raw: dict[str, Any] = {**_FULL_STATUS, "Self": self_override}
    status = parse_status_json(raw)
    assert status.tailnet == "example-corp.ts.net"


def test_not_running_returns_running_false() -> None:
    raw: dict[str, Any] = {"BackendState": "Stopped", "Self": {}}
    status = parse_status_json(raw)
    assert status.running is False


def test_missing_self_returns_nones() -> None:
    raw: dict[str, Any] = {"BackendState": "Running"}
    status = parse_status_json(raw)
    assert status.running is True
    assert status.self_ip is None
    assert status.hostname is None
    assert status.dns_name is None
    assert status.tailnet is None


def test_peer_ips_collected_from_peer_map() -> None:
    raw: dict[str, Any] = {
        "BackendState": "Running",
        "Self": {"TailscaleIPs": ["100.64.0.1"]},
        "Peer": {
            "keyA": {"TailscaleIPs": ["100.64.0.2", "fd7a::2"]},
            "keyB": {"TailscaleIPs": ["100.64.0.3"]},
        },
    }
    status = parse_status_json(raw)
    assert status.peer_ips == ["100.64.0.2", "100.64.0.3"]


def test_peer_ips_empty_when_no_peers() -> None:
    raw: dict[str, Any] = {"BackendState": "Running", "Self": {}}
    assert parse_status_json(raw).peer_ips == []


class _HungProcess:
    """Stand-in for a ``tailscale status`` child whose communicate() hangs.

    Raising TimeoutError from communicate() models the probe's wait_for
    expiring while the child is still alive (returncode None).
    """

    def __init__(self) -> None:
        self.killed = False
        self.returncode: int | None = None

    async def communicate(self) -> tuple[bytes, bytes]:
        raise TimeoutError

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        return -9


async def test_hung_probe_child_is_killed_and_reaped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A probe timeout must not orphan the child: the TTL-polled diagnostics
    path would otherwise leak one stuck subprocess per cache expiry."""
    process = _HungProcess()

    async def fake_exec(*_args: object, **_kwargs: object) -> _HungProcess:
        return process

    monkeypatch.setattr(tailscale_module, "_resolve_tailscale_binary", lambda: "/usr/bin/tailscale")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    status = await tailscale_module.query_tailscale_status()

    assert status.running is False
    assert process.killed is True
