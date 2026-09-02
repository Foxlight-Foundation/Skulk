"""Authorization and immutable-identity checks for repository model code."""

import base64
import hashlib
import json
from collections.abc import Set as AbstractSet
from ipaddress import ip_address, ip_network
from urllib.parse import urlsplit

from skulk.shared.models.model_cards import ModelCard
from skulk.store.config import SkulkConfig, load_skulk_config

MODEL_TRUST_FAILURE_MARKER = "model_trust_denied"
"""Stable runner-failure marker for deterministic, non-retriable trust denial."""


def approved_remote_code_identities(
    config: SkulkConfig | None = None,
) -> frozenset[str]:
    """Return legacy exact-card approvals retained for compatibility.

    Args:
        config: Parsed cluster configuration. When omitted, read the converged
            local ``skulk.yaml`` copy used by every node and runner process.

    Returns:
        Historical immutable identities. Current authorization does not consult
        this set. Missing or unreadable configuration yields an empty set.
    """
    if config is None:
        try:
            config = load_skulk_config()
        except Exception:
            return frozenset()
    if config is None or config.model_trust is None:
        return frozenset()
    return frozenset(config.model_trust.approved_remote_code_identities)


def remote_code_execution_requires_approval(card: ModelCard) -> bool:
    """Return whether repository Python needs a second operator decision.

    Model entry is now the authorization boundary: publishing an immutable
    signed registry card or explicitly adding a model authorizes the executable
    repository content selected by that card. The historical cluster allow-list
    remains readable for wire/configuration compatibility, but no valid catalog
    card requires a second approval ceremony.

    Args:
        card: Effective card selected for download or execution.

    Returns:
        Always ``False``. Artifact and revision integrity are enforced
        separately by :func:`require_remote_code_approval` and installed-card
        verification.
    """
    del card
    return False


def remote_code_is_automatically_trusted(card: ModelCard) -> bool:
    """Return whether the card's entry path authorizes repository code.

    Signed registry publication authorizes every provenance class; provenance
    describes how truth was established, not whether the runtime may execute
    it. Operator-added cards are authorized by the add action, and bundled cards
    are authorized by the Skulk release that ships them.

    Args:
        card: Effective card selected for download or execution.

    Returns:
        ``True`` when a card that may execute repository code was authorized by
        registry publication, explicit addition, or bundled distribution.
    """
    may_execute_repository_code = card.trust_remote_code or card.vision is not None
    if not may_execute_repository_code:
        return False
    if card.source_revision is None:
        return False
    try:
        card.require_immutable_external_companions(
            context="executable model cards"
        )
    except ValueError:
        return False
    if card.is_custom or card.registry_card_id is None:
        return True
    return card.registry_snapshot_id is not None


def remote_code_trust_identity(card: ModelCard) -> str:
    """Return the legacy immutable identity exposed to older clients.

    Signed cards use their registry identity. Unsigned and custom cards use a
    digest of the complete effective card. Current authorization does not
    consult this identity, but it remains stable for wire compatibility.
    """
    if not card.is_custom and card.registry_card_id is not None:
        return card.registry_card_id
    payload = card.model_dump(
        mode="json",
        exclude={"registry_snapshot_id", "qualification_only"},
    )
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    digest = base64.b32encode(hashlib.sha256(canonical).digest()).decode().lower()
    return f"local_{digest.rstrip('=')}"


def remote_code_approval_required(
    card: ModelCard,
    approved_identities: AbstractSet[str] | None = None,
) -> bool:
    """Return whether repository code is blocked on legacy cluster approval.

    Args:
        card: Exact effective model card being downloaded or launched.
        approved_identities: Injectable cluster allow-list. When omitted, read
            the converged local cluster configuration.

    Returns:
        Always ``False``. Publication/addition is the authorization decision;
        the allow-list argument is accepted only for call-site compatibility.
    """
    del approved_identities
    return remote_code_execution_requires_approval(card)


def require_remote_code_approval(
    card: ModelCard,
    approved_identities: AbstractSet[str] | None = None,
) -> None:
    """Validate immutable execution identity before repository code can run.

    Args:
        card: Exact effective model card being downloaded or launched.
        approved_identities: Deprecated cluster allow-list retained for source
            compatibility. It no longer participates in authorization.

    Raises:
        PermissionError: If signed registry truth or an operator-added custom
            card references mutable executable repository content.
    """
    del approved_identities
    if (
        card.is_custom
        and (card.trust_remote_code or card.vision is not None)
        and card.source_revision is None
    ):
        raise PermissionError(
            f"{MODEL_TRUST_FAILURE_MARKER}: executable custom model card lacks "
            "an immutable source revision; re-add the model through the "
            f"operator flow: {card.model_id}"
        )
    if (
        not card.is_custom
        and card.registry_card_id is None
        and (card.trust_remote_code or card.vision is not None)
        and card.source_revision is None
    ):
        raise PermissionError(
            f"{MODEL_TRUST_FAILURE_MARKER}: executable bundled model card lacks "
            "an immutable source revision; use the signed registry card or "
            f"update the bundled definition: {card.model_id}"
        )
    try:
        card.require_immutable_external_companions(
            context="executable model cards"
        )
    except ValueError as exc:
        raise PermissionError(
            f"{MODEL_TRUST_FAILURE_MARKER}: model card references a mutable "
            f"external companion: {exc}"
        ) from exc
    if (
        card.registry_card_id is not None
        and (card.trust_remote_code or card.vision is not None)
        and (
            card.registry_snapshot_id is None
            or card.source_revision is None
        )
    ):
        raise PermissionError(
            f"{MODEL_TRUST_FAILURE_MARKER}: published model card lacks an "
            "immutable signed execution identity: "
            f"{card.registry_card_id}"
        )


_TRUSTED_FABRIC_IPV4_NETWORKS = (
    ip_network("10.0.0.0/8"),
    ip_network("172.16.0.0/12"),
    ip_network("192.168.0.0/16"),
    ip_network("100.64.0.0/10"),
)
"""Private-LAN plus CGNAT overlay ranges, matching the Zenoh auto-bind policy.

The same address classes the data plane treats as the trusted fabric: a
browser or peer on one of these networks is a cluster-adjacent operator
surface, while anything public still requires loopback or the authenticated
gateway.
"""


def _is_trusted_fabric_host(host: str | None) -> bool:
    """Whether a socket or Origin host is loopback or on the trusted fabric."""
    if host == "localhost":
        return True
    if host is None:
        return False
    try:
        parsed = ip_address(host)
    except ValueError:
        return False
    if parsed.is_loopback:
        return True
    if parsed.version != 4:
        return False
    return any(parsed in network for network in _TRUSTED_FABRIC_IPV4_NETWORKS)


def trusted_fabric_mutation_allowed(
    client_host: str | None,
    origin: str | None,
    *,
    forwarding_headers_present: bool = False,
    request_host_authority: str | None = None,
) -> bool:
    """Return whether a request came directly from a trusted-fabric peer.

    The operator dashboard is served on the LAN listener and browsed from
    other machines on that LAN, so operator mutations accept a direct private
    LAN or CGNAT overlay socket peer in addition to loopback. This is the
    cluster's standing trust posture: a peer on those networks can already
    join the mesh as a full member, so gating the dashboard's mutations to
    loopback provided ceremony rather than a boundary.

    Args:
        client_host: Socket peer host reported by the ASGI server.
        origin: Optional browser Origin header.
        forwarding_headers_present: Whether a proxy-origin header was supplied.
        request_host_authority: The request's own ``Host`` header, used to
            accept a same-origin browser request when the dashboard is opened
            by hostname (``kite3.local``, a MagicDNS name) rather than an IP
            literal. Same-origin proves the page was served by this very node,
            so a cross-site page cannot satisfy it: a browser's cross-site
            request carries the target's Host but the attacker page's Origin.

    Returns:
        ``True`` only for a direct loopback or trusted-fabric peer and, for
        browser requests, an Origin that is either on the same trust classes
        or exactly same-origin with the request itself. Proxy-shaped requests
        still fail closed: a forwarded request's true origin is unknowable,
        and the public relay must never reach these handlers.
    """

    if forwarding_headers_present or not _is_trusted_fabric_host(client_host):
        return False
    if origin is None:
        return True
    try:
        parsed_origin = urlsplit(origin)
    except ValueError:
        return False
    if parsed_origin.scheme not in {"http", "https"}:
        return False
    if _is_trusted_fabric_host(parsed_origin.hostname):
        return True
    # Hostname Origins: accept only an exact same-origin match against the
    # request's own authority. Skulk serves plain HTTP on a non-default port,
    # so both sides always carry an explicit port and a lowercase string
    # comparison of the authority is exact.
    if request_host_authority is None:
        return False
    return parsed_origin.netloc.lower() == request_host_authority.strip().lower()


def loopback_mutation_allowed(
    client_host: str | None,
    origin: str | None,
    *,
    forwarding_headers_present: bool = False,
) -> bool:
    """Return whether a request came directly from a loopback control surface.

    Args:
        client_host: Socket peer host reported by the ASGI server.
        origin: Optional browser Origin header.
        forwarding_headers_present: Whether a proxy-origin header was supplied.

    Returns:
        ``True`` only for a direct loopback peer and, for browser requests, a
        loopback origin. Proxy-shaped requests fail closed because a local
        reverse proxy otherwise appears indistinguishable from localhost.
    """

    def is_loopback(host: str | None) -> bool:
        if host == "localhost":
            return True
        if host is None:
            return False
        try:
            return ip_address(host).is_loopback
        except ValueError:
            return False

    if forwarding_headers_present or not is_loopback(client_host):
        return False
    if origin is None:
        return True
    try:
        parsed_origin = urlsplit(origin)
    except ValueError:
        return False
    return parsed_origin.scheme in {"http", "https"} and is_loopback(
        parsed_origin.hostname
    )
