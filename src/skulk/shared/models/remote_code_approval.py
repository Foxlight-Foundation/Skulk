"""Trust policy for model artifacts that execute repository code."""

import base64
import hashlib
import json
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlsplit

from filelock import FileLock
from pydantic import BaseModel, ConfigDict, Field

from skulk.shared.constants import SKULK_MODEL_REMOTE_CODE_APPROVALS_PATH
from skulk.shared.models.model_cards import ModelCard

MODEL_TRUST_FAILURE_MARKER = "model_trust_denied"
"""Stable runner-failure marker for deterministic, non-retriable trust denial."""


class RemoteCodeApprovals(BaseModel):
    """Durable allow-list keyed by immutable registry card identity."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    schema_version: int = Field(default=1, ge=1)
    card_ids: frozenset[str] = frozenset()


class RemoteCodeApprovalStore:
    """Read and atomically mutate one node's remote-code allow-list."""

    def __init__(self, path: Path = SKULK_MODEL_REMOTE_CODE_APPROVALS_PATH) -> None:
        """Bind the store to an explicit local configuration path."""
        self._path = path
        self._lock = FileLock(str(path.with_suffix(path.suffix + ".lock")))

    def approved_card_ids(self) -> frozenset[str]:
        """Return immutable card ids approved on this node."""
        with self._lock:
            return self._read().card_ids

    def is_approved(self, card_id: str) -> bool:
        """Return whether an immutable registry card is locally approved."""
        return card_id in self.approved_card_ids()

    def approve(self, card_id: str) -> None:
        """Atomically approve one immutable registry card on this node."""
        with self._lock:
            approvals = self._read()
            self._write(
                approvals.model_copy(update={"card_ids": approvals.card_ids | {card_id}})
            )

    def revoke(self, card_id: str) -> None:
        """Atomically revoke one immutable registry card on this node."""
        with self._lock:
            approvals = self._read()
            self._write(
                approvals.model_copy(update={"card_ids": approvals.card_ids - {card_id}})
            )

    def _read(self) -> RemoteCodeApprovals:
        """Read strict local state, treating a missing file as empty."""
        if not self._path.exists():
            return RemoteCodeApprovals()
        return RemoteCodeApprovals.model_validate_json(
            self._path.read_bytes(), strict=False
        )

    def _write(self, approvals: RemoteCodeApprovals) -> None:
        """Replace approvals without exposing partial or world-readable state."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_name(f".{self._path.name}.tmp")
        temporary.write_text(approvals.model_dump_json(indent=2) + "\n")
        temporary.chmod(0o600)
        temporary.replace(self._path)


REMOTE_CODE_APPROVALS = RemoteCodeApprovalStore()
"""Process-wide facade over the node-local approval file."""


def remote_code_execution_requires_approval(card: ModelCard) -> bool:
    """Return whether repository Python needs explicit node-local approval.

    MLX vision processor discovery currently includes loaders that enable
    ``trust_remote_code`` internally. Treat that platform behavior as an
    execution risk even when the artifact card itself does not request remote
    code. An immutable Foxlight-provenance card from the signed registry is the
    trust decision for its exact pinned artifact; agent, community, custom and
    unsigned cards retain the explicit local approval boundary.
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
        card.registry_card_id is not None
        and card.registry_snapshot_id is not None
        and card.registry_provenance == "foxlight"
        and card.source_revision is not None
    )


def remote_code_trust_identity(card: ModelCard) -> str:
    """Return the immutable identity used by the local approval store.

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


def remote_code_approval_required(card: ModelCard) -> bool:
    """Return whether a repository-code card is blocked on local approval."""
    return (
        remote_code_execution_requires_approval(card)
        and not REMOTE_CODE_APPROVALS.is_approved(remote_code_trust_identity(card))
    )


def require_remote_code_approval(card: ModelCard) -> None:
    """Fail closed before downloading or executing unapproved repository code."""
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
    if remote_code_approval_required(card):
        trust_identity = remote_code_trust_identity(card)
        raise PermissionError(
            f"{MODEL_TRUST_FAILURE_MARKER}: model card requires node-local "
            "remote-code approval: "
            f"{trust_identity}"
        )


def remote_code_approval_mutation_allowed(
    client_host: str | None,
    origin: str | None,
    *,
    forwarding_headers_present: bool = False,
) -> bool:
    """Return whether an HTTP peer may mutate this node's approval file.

    The socket peer must be loopback. Browser requests must additionally come
    from a loopback origin so permissive inference CORS cannot turn a remote
    webpage into a localhost approval authority. A local reverse proxy also
    appears as a loopback peer, so any forwarding header makes the mutation
    ineligible rather than attempting to trust or interpret proxy identity.
    """

    def _is_loopback(host: str | None) -> bool:
        if host == "localhost":
            return True
        if host is None:
            return False
        try:
            return ip_address(host).is_loopback
        except ValueError:
            return False

    if forwarding_headers_present or not _is_loopback(client_host):
        return False
    if origin is None:
        return True
    try:
        parsed_origin = urlsplit(origin)
    except ValueError:
        return False
    return parsed_origin.scheme in {"http", "https"} and _is_loopback(
        parsed_origin.hostname
    )
