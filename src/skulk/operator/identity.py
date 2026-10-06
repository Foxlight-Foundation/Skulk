# pyright: reportUnusedFunction=false
"""Persistent identities for the Skulk operator security boundary.

Runtime libp2p peer IDs are intentionally ephemeral today. The operator app
therefore uses a separate per-installation identity for durable node references
and a cluster identity whose private key is persisted only by the encrypted
authority store.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, final
from uuid import uuid4

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import UUID4, Field, field_validator, model_validator

from skulk.shared.constants import SKULK_CONFIG_HOME
from skulk.utils.pydantic_ext import FrozenModel

_IDENTITY_FORMAT_VERSION = 1
_DEFAULT_CLUSTER_NAME = "Cluster"
_MAX_CLUSTER_NAME_LENGTH = 80


def _utc_now() -> datetime:
    """Return the current timezone-aware UTC time."""

    return datetime.now(tz=timezone.utc)


def _base64url_encode(value: bytes) -> str:
    """Encode bytes as unpadded URL-safe base64."""

    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _base64url_decode(value: str) -> bytes:
    """Decode an unpadded URL-safe base64 value."""

    padding = "=" * (-len(value) % 4)
    return base64.b64decode(
        f"{value}{padding}",
        altchars=b"-_",
        validate=True,
    )


def cluster_identity_fingerprint(public_key: bytes) -> str:
    """Return the stable display fingerprint for a cluster public key.

    Args:
        public_key: Raw Ed25519 public-key bytes.

    Returns:
        A URL-safe SHA-256 fingerprint prefixed with its algorithm.
    """

    return f"sha256:{_base64url_encode(hashlib.sha256(public_key).digest())}"


@final
class NodeInstallationIdentity(FrozenModel):
    """Persistent identity for one Skulk installation on one host."""

    format_version: Literal[1] = Field(
        default=_IDENTITY_FORMAT_VERSION,
        description="On-disk node-installation identity format version.",
    )
    node_install_id: UUID4 = Field(
        description=(
            "Stable random installation ID used by operator history and authority "
            "membership; it is independent of the runtime libp2p peer ID."
        ),
    )
    created_at: datetime = Field(
        description="Timezone-aware UTC timestamp when this identity was created.",
    )

    @field_validator("created_at")
    @classmethod
    def _created_at_is_timezone_aware(cls, value: datetime) -> datetime:
        """Reject ambiguous timestamps that cannot support recovery auditing."""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value


@final
class ClusterPublicIdentity(FrozenModel):
    """Public, durable identity of one operator-managed Skulk cluster."""

    format_version: Literal[1] = Field(
        default=_IDENTITY_FORMAT_VERSION,
        description="Cluster identity format version.",
    )
    cluster_id: UUID4 = Field(
        description="Stable random cluster ID used in pairing and compatibility checks.",
    )
    name: str = Field(
        default=_DEFAULT_CLUSTER_NAME,
        min_length=1,
        max_length=_MAX_CLUSTER_NAME_LENGTH,
        description="Operator-visible cluster name; defaults to Cluster.",
    )
    public_key: str = Field(
        description="Unpadded URL-safe base64 Ed25519 cluster public key.",
    )
    fingerprint: str = Field(
        description="SHA-256 fingerprint derived from the cluster public key.",
    )
    created_at: datetime = Field(
        description="Timezone-aware UTC timestamp when the cluster identity was created.",
    )

    @field_validator("name")
    @classmethod
    def _normalize_name(cls, value: str) -> str:
        """Normalize whitespace while rejecting an empty display name."""

        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("cluster name must not be blank")
        return normalized

    @field_validator("created_at")
    @classmethod
    def _created_at_is_timezone_aware(cls, value: datetime) -> datetime:
        """Reject ambiguous timestamps that cannot support recovery auditing."""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value

    @model_validator(mode="after")
    def _fingerprint_matches_public_key(self) -> "ClusterPublicIdentity":
        """Reject corrupted or substituted public identity records."""

        try:
            public_key = _base64url_decode(self.public_key)
        except (ValueError, UnicodeError) as exc:
            raise ValueError("public_key is not valid URL-safe base64") from exc
        if len(public_key) != 32:
            raise ValueError("public_key must contain one raw Ed25519 public key")
        expected = cluster_identity_fingerprint(public_key)
        if self.fingerprint != expected:
            raise ValueError("cluster identity fingerprint does not match public key")
        return self


@dataclass(frozen=True)
class ClusterIdentityMaterial:
    """New cluster public identity plus its unpersisted private key.

    The private key exists only long enough for an encrypted authority store to
    commit it. Callers must never serialize this dataclass or log its contents.
    """

    public_identity: ClusterPublicIdentity
    private_key: bytes


def create_cluster_identity(
    name: str = _DEFAULT_CLUSTER_NAME,
) -> ClusterIdentityMaterial:
    """Create a new cluster identity without persisting plaintext key material.

    Args:
        name: Initial operator-visible cluster name.

    Returns:
        Public identity and raw private key material for immediate encrypted
        persistence by :class:`EncryptedAuthorityStore`.
    """

    private_key = Ed25519PrivateKey.generate()
    private_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    public_identity = ClusterPublicIdentity(
        cluster_id=uuid4(),
        name=name,
        public_key=_base64url_encode(public_bytes),
        fingerprint=cluster_identity_fingerprint(public_bytes),
        created_at=_utc_now(),
    )
    return ClusterIdentityMaterial(
        public_identity=public_identity,
        private_key=private_bytes,
    )


@final
class OperatorIdentityRepository:
    """Atomic filesystem repository for non-secret per-host operator identity.

    Cluster private keys do not belong here. They are accepted only by the
    encrypted authority store. The repository creates its directory with mode
    ``0700`` and identity file with mode ``0600`` on POSIX platforms.
    """

    def __init__(self, root: Path | None = None) -> None:
        """Create a repository rooted under Skulk's configuration directory.

        Args:
            root: Explicit root for tests or alternate installations. The
                default is ``SKULK_CONFIG_HOME / "operator"``.
        """

        self._root = root if root is not None else SKULK_CONFIG_HOME / "operator"
        self._node_identity_path = self._root / "node-installation.json"

    @property
    def root(self) -> Path:
        """Return the directory containing operator identity state."""

        return self._root

    def load_or_create_node_identity(self) -> NodeInstallationIdentity:
        """Load the host identity or create it exactly once under a file lock.

        Returns:
            The durable node-installation identity.

        Side effects:
            Creates the protected repository and identity file on first use.
        """

        from filelock import FileLock

        self._ensure_root()
        lock_path = self._node_identity_path.with_suffix(".lock")
        with FileLock(lock_path):
            if self._node_identity_path.exists():
                return self._read_node_identity()
            identity = NodeInstallationIdentity(
                node_install_id=uuid4(),
                created_at=_utc_now(),
            )
            self._atomic_write_json(
                self._node_identity_path,
                identity.model_dump(mode="json", by_alias=True),
            )
            return identity

    def _read_node_identity(self) -> NodeInstallationIdentity:
        """Read and validate the existing host identity record."""

        return NodeInstallationIdentity.model_validate_json(
            self._node_identity_path.read_text(encoding="utf-8")
        )

    def _ensure_root(self) -> None:
        """Create and harden the repository directory."""

        self._root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if os.name == "posix":
            self._root.chmod(0o700)

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
        """Durably replace one JSON record without exposing a partial write."""

        temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            descriptor = os.open(
                temporary_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
            if os.name == "posix":
                path.chmod(0o600)
                directory_descriptor = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
        finally:
            temporary_path.unlink(missing_ok=True)
