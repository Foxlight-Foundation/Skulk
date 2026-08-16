"""Cluster-wide trust policy for model artifacts that execute repository code."""

import base64
import hashlib
import json
from collections.abc import Set as AbstractSet
from ipaddress import ip_address
from urllib.parse import urlsplit

from skulk.shared.models.model_cards import ModelCard
from skulk.store.config import SkulkConfig, load_skulk_config

MODEL_TRUST_FAILURE_MARKER = "model_trust_denied"
"""Stable runner-failure marker for deterministic, non-retriable trust denial."""


def approved_remote_code_identities(
    config: SkulkConfig | None = None,
) -> frozenset[str]:
    """Return exact model-card identities approved for the whole cluster.

    Args:
        config: Parsed cluster configuration. When omitted, read the converged
            local ``skulk.yaml`` copy used by every node and runner process.

    Returns:
        The operator-approved immutable identities. Missing or unreadable
        configuration fails closed as an empty set.
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
    """Return whether repository Python needs an explicit operator decision.

    MLX vision processor discovery currently includes loaders that enable
    ``trust_remote_code`` internally. Treat that platform behavior as an
    execution risk even when the artifact card itself does not request remote
    code. An immutable Foxlight-provenance card from the signed registry is the
    trust decision for its exact pinned artifact; agent, community, custom and
    unsigned cards retain an explicit model-by-model cluster approval boundary.
    """
    return (
        card.trust_remote_code or card.vision is not None
    ) and not remote_code_is_automatically_trusted(card)


def remote_code_is_automatically_trusted(card: ModelCard) -> bool:
    """Return whether signed Foxlight provenance authorizes the pinned card.

    All four fields are load-bearing. In particular, copied provenance text or
    a mutable repository revision cannot acquire automatic execution authority.
    A changed immutable card receives a different ``registry_card_id`` and is
    therefore evaluated independently.
    """
    return (
        not card.is_custom
        and card.registry_card_id is not None
        and card.registry_snapshot_id is not None
        and card.registry_provenance == "foxlight"
        and card.source_revision is not None
    )


def remote_code_trust_identity(card: ModelCard) -> str:
    """Return the immutable identity used by cluster model-trust settings.

    Signed cards use their registry identity. Unsigned and custom cards use a
    digest of the complete effective card, so a revision or policy change
    cannot inherit approval from an earlier definition.
    """
    if card.registry_card_id is not None:
        return card.registry_card_id
    payload = card.model_dump(
        mode="json",
        exclude={"registry_snapshot_id"},
    )
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    digest = base64.b32encode(hashlib.sha256(canonical).digest()).decode().lower()
    return f"local_{digest.rstrip('=')}"


def remote_code_approval_required(
    card: ModelCard,
    approved_identities: AbstractSet[str] | None = None,
) -> bool:
    """Return whether repository code is blocked on cluster approval.

    Args:
        card: Exact effective model card being downloaded or launched.
        approved_identities: Injectable cluster allow-list. When omitted, read
            the converged local cluster configuration.

    Returns:
        ``True`` when the card can execute repository code and its exact
        identity has not been approved by the operator.
    """
    approvals = (
        approved_remote_code_identities()
        if approved_identities is None
        else approved_identities
    )
    return remote_code_execution_requires_approval(card) and (
        remote_code_trust_identity(card) not in approvals
    )


def require_remote_code_approval(
    card: ModelCard,
    approved_identities: AbstractSet[str] | None = None,
) -> None:
    """Fail closed before downloading or executing unapproved repository code.

    Args:
        card: Exact effective model card being downloaded or launched.
        approved_identities: Injectable cluster allow-list. When omitted, read
            the converged local cluster configuration.

    Raises:
        PermissionError: If a signed card references mutable processor code or
            the operator has not approved the exact card identity.
    """
    if (
        card.registry_card_id is not None
        and card.vision is not None
        and card.vision.processor_repo is not None
        and card.vision.processor_revision is None
    ):
        raise PermissionError(
            f"{MODEL_TRUST_FAILURE_MARKER}: signed model card references an "
            "unpinned vision processor repository: "
            f"{card.registry_card_id}"
        )
    if remote_code_approval_required(card, approved_identities):
        trust_identity = remote_code_trust_identity(card)
        raise PermissionError(
            f"{MODEL_TRUST_FAILURE_MARKER}: model card requires cluster "
            "operator approval for repository code: "
            f"{trust_identity}"
        )


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
