"""Zenoh data-plane hardening helpers (#308 / #312 review).

The namespace derivation is the Zenoh isolation boundary, so distinct libp2p
namespaces must not collide on the same Zenoh namespace (or peers on different
libp2p namespaces could read each other's `data`). We hash unconditionally
(collision-resistant): a char-replacement sanitizer collapsed
"prod/main"/"prod_main" (P1), and a verbatim-when-safe split let a literal
"ns<sha256(victim)>" collide (P2). The namespace token mirrors exactly what
libp2p isolates on (swarm.rs), so one libp2p cluster cannot split across two
Zenoh namespaces (#312 review P2). The bind restriction fails fast when the
plane is enabled without an explicit listen endpoint (#308).
"""

import hashlib

import pytest

from skulk.main import (
    _LIBP2P_NETWORK_VERSION,  # pyright: ignore[reportPrivateUsage]
    _derive_zenoh_namespace,  # pyright: ignore[reportPrivateUsage]
    _libp2p_namespace_token,  # pyright: ignore[reportPrivateUsage]
    _namespace_fingerprint,  # pyright: ignore[reportPrivateUsage]
    _require_zenoh_listen,  # pyright: ignore[reportPrivateUsage]
    _resolve_zenoh_enabled,  # pyright: ignore[reportPrivateUsage]
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
    # Override present -> override value, even when empty (Rust env::var is Ok("")).
    assert _libp2p_namespace_token({"SKULK_LIBP2P_NAMESPACE": "prod"}) == "prod"
    assert _libp2p_namespace_token({"SKULK_LIBP2P_NAMESPACE": ""}) == ""
    # Unset -> NETWORK_VERSION default, NOT a Skulk-only "skulk" default.
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


def test_resolve_zenoh_enabled_soft_default_on() -> None:
    # Soft default-on (#315): the listen endpoint is the opt-in signal when the
    # flag is unset, so a bare node (no listen) stays on gossipsub and never hits
    # the #308 listen requirement.
    assert _resolve_zenoh_enabled("", "tcp/10.0.0.1:7447") is True
    assert _resolve_zenoh_enabled("", "") is False
    assert _resolve_zenoh_enabled("   ", "   ") is False


def test_resolve_zenoh_enabled_explicit_overrides() -> None:
    # Explicit on/off win regardless of listen presence (explicit-on with no
    # listen is a loud error later, in _require_zenoh_listen, not here).
    for truthy in ("1", "true", "TRUE", "yes", "on"):
        assert _resolve_zenoh_enabled(truthy, "") is True
    for falsy in ("0", "false", "No", "off", "OFF"):
        # #315 review: "off" must force gossipsub even with a listen configured.
        assert _resolve_zenoh_enabled(falsy, "tcp/10.0.0.1:7447") is False


def test_resolve_zenoh_enabled_rejects_garbage() -> None:
    # #315 review: an unrecognized non-empty value must NOT silently fall through
    # to the listen-based default and flip transports; it raises.
    for bad in ("disable", "enabled", "maybe", "2"):
        with pytest.raises(ValueError, match="SKULK_ZENOH_DATA_PLANE"):
            _resolve_zenoh_enabled(bad, "tcp/10.0.0.1:7447")


def test_require_zenoh_listen_returns_explicit_value() -> None:
    assert _require_zenoh_listen("tcp/192.168.0.115:7447") == "tcp/192.168.0.115:7447"
    assert _require_zenoh_listen("  tcp/127.0.0.1:7447  ") == "tcp/127.0.0.1:7447"


def test_require_zenoh_listen_rejects_empty() -> None:
    # #308 bind restriction: must fail fast rather than default to 0.0.0.0.
    for empty in ("", "   "):
        with pytest.raises(ValueError, match="SKULK_ZENOH_LISTEN"):
            _require_zenoh_listen(empty)
