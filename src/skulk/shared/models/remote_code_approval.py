"""Node-local approval store for registry artifacts that execute repository code."""

from pathlib import Path

from filelock import FileLock
from pydantic import BaseModel, ConfigDict, Field

from skulk.shared.constants import SKULK_MODEL_REMOTE_CODE_APPROVALS_PATH
from skulk.shared.models.model_cards import ModelCard


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


def remote_code_approval_required(card: ModelCard) -> bool:
    """Return whether a registry card is blocked on this node's approval."""
    return (
        card.registry_card_id is not None
        and card.trust_remote_code
        and not REMOTE_CODE_APPROVALS.is_approved(card.registry_card_id)
    )


def require_remote_code_approval(card: ModelCard) -> None:
    """Fail closed before downloading or executing unapproved repository code."""
    if remote_code_approval_required(card):
        raise PermissionError(
            "model card requires node-local remote-code approval: "
            f"{card.registry_card_id}"
        )
