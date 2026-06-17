"""The Zenoh namespace derivation must be injective (#308 / #312 review).

The derived value is the Zenoh data-plane isolation boundary, so distinct libp2p
namespaces must never collapse to the same Zenoh namespace (or peers on different
libp2p namespaces could read each other's `data`). We hash unconditionally:
a char-replacement sanitizer collapsed "prod/main"/"prod_main" (P1), and a
verbatim-when-safe split let a literal "ns<sha256(victim)>" collide (P2).
"""

import hashlib

from skulk.main import _derive_zenoh_namespace  # pyright: ignore[reportPrivateUsage]


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
