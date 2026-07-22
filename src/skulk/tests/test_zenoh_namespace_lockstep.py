# pyright: reportPrivateUsage=false
"""Lockstep guard: the Zenoh namespace token mirrors the Rust pnet key (#659).

Two nodes whose libp2p keys agree but whose Zenoh namespaces differ silently
drop all cross-node generation output, so the Python mirror of the Rust
derivation must never drift. This test parses the Rust source for the
authoritative values; a NETWORK_VERSION bump that forgets the Python
constant (or vice versa) fails here instead of on a live fleet.
"""

import re
from pathlib import Path

from skulk.main import (
    _LIBP2P_NAMESPACE_ENV_VAR,
    _LIBP2P_NETWORK_VERSION,
    _libp2p_namespace_token,
)

# tests/ -> skulk/ -> src/ -> repo root
_SWARM_RS = (
    Path(__file__).resolve().parents[3] / "rust" / "networking" / "src" / "swarm.rs"
)


def _rust_constant(pattern: str) -> str:
    source = _SWARM_RS.read_text()
    match = re.search(pattern, source)
    assert match is not None, f"pattern not found in swarm.rs: {pattern}"
    return match.group(1)


def test_network_version_matches_rust() -> None:
    rust_version = _rust_constant(
        r'pub const NETWORK_VERSION: &\[u8\] = b"([^"]+)";'
    )
    assert rust_version == _LIBP2P_NETWORK_VERSION


def test_namespace_env_var_matches_rust() -> None:
    rust_env_var = _rust_constant(
        r'pub const OVERRIDE_VERSION_ENV_VAR: &str = "([^"]+)";'
    )
    assert rust_env_var == _LIBP2P_NAMESPACE_ENV_VAR


def test_token_layers_namespace_over_version() -> None:
    """Version always contributes; a namespace layers on top (#659)."""
    assert _libp2p_namespace_token({}) == _LIBP2P_NETWORK_VERSION
    assert (
        _libp2p_namespace_token({_LIBP2P_NAMESPACE_ENV_VAR: "foxlight-main"})
        == _LIBP2P_NETWORK_VERSION + "foxlight-main"
    )
    # Presence, not truthiness: an empty var still selects the override arm,
    # matching Rust env::var semantics.
    assert (
        _libp2p_namespace_token({_LIBP2P_NAMESPACE_ENV_VAR: ""})
        == _LIBP2P_NETWORK_VERSION
    )
