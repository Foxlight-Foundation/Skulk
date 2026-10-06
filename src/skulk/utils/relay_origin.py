"""Validation of the relay origin a gateway registers its phone-pairing route with.

Kept free of operator and network imports so configuration loading can apply
the same rule as the registration client without pulling either in.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit


def validate_relay_origin(origin: str) -> str:
    """Return a canonical relay origin, refusing unsafe or ambiguous values.

    A relay origin is a scheme, host, and optional port with no path, query,
    fragment, or user information. HTTPS is required, except that a loopback
    development relay may use HTTP.

    Args:
        origin: Operator-supplied relay origin, for example
            ``https://relay.example``.

    Returns:
        The origin as ``scheme://host[:port]`` with a lowercase host and no
        trailing slash.

    Raises:
        ValueError: The origin is malformed, carries extra components, or would
            send registration in cleartext to a non-loopback host.
    """

    try:
        parts = urlsplit(origin.strip())
        port = parts.port
    except ValueError as exc:
        raise ValueError("relay origin must be a valid URL") from exc
    host = parts.hostname
    if (
        not host
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
        or parts.path not in ("", "/")
    ):
        raise ValueError(
            "relay origin must be scheme://host[:port] with no path, query, "
            "or credentials"
        )
    if parts.scheme == "http":
        if not _is_loopback_host(host):
            raise ValueError("relay origin must use HTTPS except on loopback")
    elif parts.scheme != "https":
        raise ValueError("relay origin must use HTTPS except on loopback")
    netloc = f"[{host}]" if ":" in host else host
    if port is not None:
        netloc = f"{netloc}:{port}"
    return f"{parts.scheme}://{netloc}"


def _is_loopback_host(host: str) -> bool:
    """Return whether a URL host names this machine."""

    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
