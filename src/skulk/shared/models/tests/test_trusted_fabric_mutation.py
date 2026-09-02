"""Trusted-fabric admission for operator mutations.

The dashboard is served on the LAN listener and browsed from other machines,
so operator mutations accept a direct private-LAN or CGNAT socket peer in
addition to loopback: the cluster's standing trust posture, since such a peer
can already join the mesh as a full member. Forwarded requests and public
peers still fail closed.
"""

import pytest

from skulk.shared.models.remote_code_approval import (
    loopback_mutation_allowed,
    trusted_fabric_mutation_allowed,
)


@pytest.mark.parametrize(
    "host",
    ["127.0.0.1", "localhost", "192.168.0.55", "10.4.2.1", "172.16.9.9", "100.101.2.3"],
)
def test_direct_fabric_and_loopback_peers_are_admitted(host: str) -> None:
    assert trusted_fabric_mutation_allowed(host, None) is True


@pytest.mark.parametrize(
    "host",
    ["203.0.113.7", "8.8.8.8", "100.63.255.255", "172.32.0.1", "not-an-ip", None],
)
def test_public_and_unparseable_peers_are_rejected(host: str | None) -> None:
    assert trusted_fabric_mutation_allowed(host, None) is False


def test_forwarded_requests_fail_closed_even_from_the_fabric() -> None:
    """A forwarded request's true origin is unknowable; the relay never enters."""
    assert (
        trusted_fabric_mutation_allowed(
            "192.168.0.55", None, forwarding_headers_present=True
        )
        is False
    )


@pytest.mark.parametrize(
    ("origin", "allowed"),
    [
        ("http://192.168.0.122:52415", True),
        ("http://localhost:52415", True),
        ("https://10.0.0.9", True),
        ("http://evil.example.com", False),
        ("https://203.0.113.7", False),
        ("ftp://192.168.0.122", False),
    ],
)
def test_browser_origin_must_also_be_on_the_fabric(origin: str, allowed: bool) -> None:
    """A public page in a fabric browser must not ride the user's location."""
    assert trusted_fabric_mutation_allowed("192.168.0.55", origin) is allowed


def test_ipv6_non_loopback_is_rejected() -> None:
    """The fabric classification is IPv4, matching the Zenoh auto-bind policy."""
    assert trusted_fabric_mutation_allowed("fd00::1", None) is False
    assert trusted_fabric_mutation_allowed("::1", None) is True


def test_loopback_rule_remains_strict() -> None:
    """The narrower guard is untouched: fabric peers stay rejected there."""
    assert loopback_mutation_allowed("192.168.0.55", None) is False
    assert loopback_mutation_allowed("127.0.0.1", None) is True


@pytest.mark.parametrize(
    "origin_host",
    ["kite3.local", "my-node.tailnet-abc.ts.net"],
)
def test_own_hostname_dashboards_are_admitted(origin_host: str) -> None:
    """Dashboards are browsed by hostname; the node recognizes its own names."""
    assert (
        trusted_fabric_mutation_allowed(
            "192.168.0.55",
            f"http://{origin_host}:52415",
            self_host_names={"kite3.local", "kite3", "my-node.tailnet-abc.ts.net"},
        )
        is True
    )


def test_dns_rebound_hostname_is_rejected() -> None:
    """After a rebind, the attacker's name is in Origin AND Host; equality
    would prove nothing. The attacker cannot make their hostname one this
    node knows as its own, so the allowlist holds."""
    assert (
        trusted_fabric_mutation_allowed(
            "192.168.0.55",
            "http://evil.example.com:52415",
            self_host_names={"kite3.local", "kite3"},
        )
        is False
    )


def test_hostname_origin_without_self_names_is_rejected() -> None:
    assert (
        trusted_fabric_mutation_allowed(
            "192.168.0.55", "http://kite3.local:52415"
        )
        is False
    )


def test_self_hostname_match_is_case_insensitive() -> None:
    assert (
        trusted_fabric_mutation_allowed(
            "192.168.0.55",
            "http://KITE3.local:52415",
            self_host_names={"kite3.local"},
        )
        is True
    )
