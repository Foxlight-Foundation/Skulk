"""The Zenoh namespace derivation must be injective (#308 / #312 review).

A char-replacement sanitizer would collapse distinct libp2p namespaces (e.g.
"prod/main" and "prod_main") to the same Zenoh namespace, letting peers on
different libp2p namespaces read each other's `data`. The derivation uses the
raw value verbatim when it is a valid key-expr segment, else a SHA-256 hash.
"""

from skulk.main import _derive_zenoh_namespace  # pyright: ignore[reportPrivateUsage]


def _keyexpr_safe(s: str) -> bool:
    return bool(s) and all(c.isalnum() or c in "._-" for c in s)


def test_safe_namespace_used_verbatim() -> None:
    assert _derive_zenoh_namespace("foxlight-main") == "foxlight-main"
    assert _derive_zenoh_namespace("skulk") == "skulk"


def test_unsafe_namespace_is_hashed_and_keyexpr_safe() -> None:
    out = _derive_zenoh_namespace("prod/main")
    assert out.startswith("nshash_")
    assert _keyexpr_safe(out)


def test_distinct_namespaces_do_not_collapse() -> None:
    # The exact collision the char-replacement sanitizer caused.
    assert _derive_zenoh_namespace("prod/main") != _derive_zenoh_namespace("prod_main")
    # "prod_main" is key-expr-safe, so it stays verbatim and can't equal a hash.
    assert _derive_zenoh_namespace("prod_main") == "prod_main"
