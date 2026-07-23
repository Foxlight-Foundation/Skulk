"""Zenoh data-plane hardening helpers (#308 / #312 review).

The namespace derivation is the Zenoh isolation boundary, so distinct libp2p
namespaces must not collide on the same Zenoh namespace (or peers on different
libp2p namespaces could read each other's `data`). We hash unconditionally
(collision-resistant): a char-replacement sanitizer collapsed
"prod/main"/"prod_main" (P1), and a verbatim-when-safe split let a literal
"ns<sha256(victim)>" collide (P2). The namespace token mirrors exactly what
libp2p isolates on (swarm.rs), so one libp2p cluster cannot split across two
Zenoh namespaces (#312 review P2). The listener resolver keeps the zero-config
default on one peer-reachable interface rather than exposing every interface.
"""

import hashlib

import pytest

import skulk.main as main
from skulk.main import (
    _LIBP2P_NETWORK_VERSION,  # pyright: ignore[reportPrivateUsage]
    _derive_zenoh_namespace,  # pyright: ignore[reportPrivateUsage]
    _libp2p_namespace_token,  # pyright: ignore[reportPrivateUsage]
    _namespace_fingerprint,  # pyright: ignore[reportPrivateUsage]
    _resolve_zenoh_enabled,  # pyright: ignore[reportPrivateUsage]
    _resolve_zenoh_listen,  # pyright: ignore[reportPrivateUsage]
)


def _keyexpr_safe(s: str) -> bool:
    return bool(s) and all(c.isalnum() or c in "._-" for c in s)


def test_output_is_keyexpr_safe_and_deterministic() -> None:
    out = _derive_zenoh_namespace("foxlight-main")
    assert _keyexpr_safe(out)
    assert out == _derive_zenoh_namespace("foxlight-main")  # deterministic
    assert out == "ns" + hashlib.sha256(b"foxlight-main").hexdigest()


def test_distinct_namespaces_never_collapse() -> None:
    # The P1 collision (char-replacement) and general distinctness.
    assert _derive_zenoh_namespace("prod/main") != _derive_zenoh_namespace("prod_main")
    assert _derive_zenoh_namespace("foxlight-main") != _derive_zenoh_namespace("skulk")


def test_no_verbatim_hash_overlap() -> None:
    # The P2 collision: a fleet named literally like a hashed namespace must NOT
    # collide with whatever hashes to that value (everything is hashed now, so
    # the literal is itself hashed and cannot equal the raw hash of another).
    victim = "prod/main"
    derived_victim = _derive_zenoh_namespace(victim)
    attacker_literal = "ns" + hashlib.sha256(victim.encode()).hexdigest()
    assert _derive_zenoh_namespace(attacker_literal) != derived_victim


def test_libp2p_namespace_token_mirrors_swarm() -> None:
    # #312 review P2: the Zenoh namespace must derive from the SAME token libp2p
    # isolates on (swarm.rs), or one cluster splits across two Zenoh namespaces.
    # Since #659 the version ALWAYS contributes and a present override layers
    # on top (even when empty: Rust env::var is Ok("")), so a wire-version
    # bump re-keys both transports on every deployment shape.
    assert (
        _libp2p_namespace_token({"SKULK_LIBP2P_NAMESPACE": "prod"})
        == _LIBP2P_NETWORK_VERSION + "\0prod"
    )
    assert (
        _libp2p_namespace_token({"SKULK_LIBP2P_NAMESPACE": ""})
        == _LIBP2P_NETWORK_VERSION + "\0"
    )
    # Unset -> NETWORK_VERSION alone, NOT a Skulk-only "skulk" default.
    assert _libp2p_namespace_token({}) == _LIBP2P_NETWORK_VERSION
    # The legacy EXO_ env libp2p never reads must NOT influence the token.
    assert _libp2p_namespace_token({"EXO_LIBP2P_NAMESPACE": "legacy"}) == (
        _LIBP2P_NETWORK_VERSION
    )


def test_namespace_fingerprint_is_stable_and_non_routing() -> None:
    # #312 review: with no TLS the namespace is the isolation value, so logging
    # emits a fingerprint instead. It must be stable per namespace (operators
    # compare nodes) but neither equal to the namespace nor a prefix of it (a
    # peer must not be able to subscribe from what's logged).
    ns = _derive_zenoh_namespace("foxlight-main")
    fp = _namespace_fingerprint(ns)
    assert fp == _namespace_fingerprint(ns)  # stable
    assert fp != ns and fp not in ns  # not the namespace, not a prefix of it
    # Distinct namespaces yield distinct fingerprints.
    assert fp != _namespace_fingerprint(_derive_zenoh_namespace("other"))


def test_resolve_zenoh_enabled_defaults_on_for_fresh_install() -> None:
    # The regular E2E fleet qualifies Zenoh, so an unset flag must select the
    # same transport for a fresh installation. Listener presence is irrelevant.
    assert _resolve_zenoh_enabled("", "tcp/10.0.0.1:7447") is True
    assert _resolve_zenoh_enabled("", "") is True
    assert _resolve_zenoh_enabled("   ", "   ") is True


def test_resolve_zenoh_enabled_explicit_overrides() -> None:
    # Explicit on/off win regardless of listener presence.
    for truthy in ("1", "true", "TRUE", "yes", "on"):
        assert _resolve_zenoh_enabled(truthy, "") is True
    for falsy in ("0", "false", "No", "off", "OFF"):
        # #315 review: "off" must force gossipsub even with a listen configured.
        assert _resolve_zenoh_enabled(falsy, "tcp/10.0.0.1:7447") is False


def test_resolve_zenoh_enabled_rejects_garbage() -> None:
    # An unrecognized non-empty value must not silently select a transport.
    for bad in ("disable", "enabled", "maybe", "2"):
        with pytest.raises(ValueError, match="SKULK_ZENOH_DATA_PLANE"):
            _resolve_zenoh_enabled(bad, "tcp/10.0.0.1:7447")


def test_resolve_zenoh_listen_returns_explicit_value() -> None:
    assert _resolve_zenoh_listen("tcp/192.168.0.115:7447") == (
        "tcp/192.168.0.115:7447"
    )
    assert _resolve_zenoh_listen("tcp/203.0.113.8:7447") == (
        "tcp/203.0.113.8:7447"
    )
    assert _resolve_zenoh_listen("  tcp/127.0.0.1:7447  ") == (
        "tcp/127.0.0.1:7447"
    )


def test_resolve_zenoh_listen_uses_best_routable_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main, "_routable_local_ipv4", lambda: "192.168.0.115")
    assert _resolve_zenoh_listen("") == "tcp/192.168.0.115:7447"


def test_resolve_zenoh_listen_accepts_cgnat_fabric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main, "_routable_local_ipv4", lambda: "100.64.12.34")
    assert _resolve_zenoh_listen("") == "tcp/100.64.12.34:7447"


def test_resolve_zenoh_listen_rejects_automatic_public_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main, "_routable_local_ipv4", lambda: "203.0.113.8")
    assert _resolve_zenoh_listen("") == "tcp/127.0.0.1:7447"


def test_resolve_zenoh_listen_uses_loopback_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main, "_routable_local_ipv4", lambda: None)
    assert _resolve_zenoh_listen("   ") == "tcp/127.0.0.1:7447"
