# pyright: reportAny=false
"""Tests for TailscaleStatus JSON parsing."""

from __future__ import annotations

from typing import Any

from exo.connectivity.tailscale import parse_status_json

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
