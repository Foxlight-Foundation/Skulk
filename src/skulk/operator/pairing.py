"""Single-use host-authorized pairing for one designated Skulk gateway."""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
from typing import Literal, cast, final
from uuid import UUID, uuid4

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import AnyHttpUrl, Field, field_validator

from skulk.operator.authority import (
    AuthorityCommitConflictError,
    AuthorityNotInitializedError,
    EncryptedAuthorityStore,
)
from skulk.operator.identity import ClusterPublicIdentity, create_cluster_identity
from skulk.operator.key_provider import LocalFileAuthorityKeyProvider
from skulk.utils.pydantic_ext import FrozenModel

_PAIRING_RECORD_TYPE = "operator_pairing_session"
_PAIRING_LIFETIME = timedelta(minutes=5)
_ACCESS_TOKEN_LIFETIME = timedelta(minutes=15)
_REFRESH_TOKEN_LIFETIME = timedelta(days=30)
_PAIRING_SIGNATURE_CONTEXT = b"skulk-device-pairing-v1\x00"

type OperatorScope = Literal[
    "cluster:read",
    "models:read",
    "chat:write",
    "operations:write",
]

_DEFAULT_SCOPES: tuple[OperatorScope, ...] = (
    "cluster:read",
    "models:read",
    "chat:write",
    "operations:write",
)


class PairingError(RuntimeError):
    """Base class for safe pairing failures."""


class PairingSessionNotFoundError(PairingError):
    """Raised when a pairing nonce does not name a pending session."""


class PairingSessionExpiredError(PairingError):
    """Raised when a pairing session has passed its five-minute lifetime."""


class PairingSessionStateError(PairingError):
    """Raised when a pairing transition is repeated or out of order."""


class PairingProofError(PairingError):
    """Raised when a device cannot prove possession of its proposed key."""


class PairingGatewayNotInitializedError(PairingError):
    """Raised when this API node has not been designated through local pairing."""


class PairingPackage(FrozenModel):
    """Short-lived public package encoded in a Skulk pairing QR code."""

    version: Literal[1] = Field(description="Pairing package format version.")
    cluster_id: UUID = Field(description="Stable identity of the target cluster.")
    cluster_name: str = Field(description="Operator-visible cluster name.")
    cluster_fingerprint: str = Field(
        description="Display fingerprint of the cluster identity key."
    )
    exchange_url: AnyHttpUrl = Field(
        description="Direct or relayed URL serving the pairing exchange."
    )
    expires_at: datetime = Field(description="UTC expiry of the pairing package.")
    nonce: str = Field(
        min_length=32,
        max_length=128,
        description="High-entropy single-use pairing capability.",
    )

    @field_validator("expires_at")
    @classmethod
    def _expiry_is_timezone_aware(cls, value: datetime) -> datetime:
        """Reject ambiguous package expiries."""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("expires_at must include a timezone")
        return value

    def as_url(self) -> str:
        """Return the package as a compact custom-scheme QR payload."""

        encoded = _base64url_encode(
            self.model_dump_json(by_alias=True).encode("utf-8")
        )
        return f"skulk://pair?payload={encoded}"


class PairingChallengeRequest(FrozenModel):
    """Candidate device identity submitted before credential exchange."""

    device_name: str = Field(
        min_length=1,
        max_length=80,
        description="Operator-visible name of the candidate device.",
    )
    device_public_key: str = Field(
        min_length=40,
        max_length=64,
        description="Unpadded URL-safe base64 Ed25519 device public key.",
    )

    @field_validator("device_name")
    @classmethod
    def _normalize_device_name(cls, value: str) -> str:
        """Normalize device names while rejecting whitespace-only input."""

        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("device_name must not be blank")
        return normalized


class PairingChallengeResponse(FrozenModel):
    """Cluster challenge that the candidate device must sign."""

    challenge: str = Field(
        description="Unpadded URL-safe base64 random challenge."
    )
    expires_at: datetime = Field(description="UTC expiry inherited from the session.")


class PairingExchangeRequest(FrozenModel):
    """Device proof completing a pending pairing session."""

    signature: str = Field(
        min_length=80,
        max_length=100,
        description="Ed25519 signature over the domain-separated pairing challenge.",
    )


class PairingExchangeResponse(FrozenModel):
    """One-time credential response returned after successful pairing."""

    device_id: UUID = Field(description="Stable identifier assigned to the device.")
    cluster: ClusterPublicIdentity = Field(
        description="Validated cluster identity bound to the credential."
    )
    access_token: str = Field(
        description="Opaque short-lived bearer credential returned only once."
    )
    access_token_expires_at: datetime = Field(
        description="UTC expiry of the access credential."
    )
    refresh_token: str = Field(
        description="Opaque rotating refresh credential returned only once."
    )
    refresh_token_expires_at: datetime = Field(
        description="UTC expiry of the refresh credential."
    )
    scopes: tuple[OperatorScope, ...] = Field(
        description="Canonical API scopes granted to the paired device."
    )


class _StoredPairingSession(FrozenModel):
    """Encrypted durable state for one pairing capability and resulting device."""

    state: Literal["pending", "challenged", "consumed", "revoked"]
    nonce_hash: str
    created_at: datetime
    expires_at: datetime
    exchange_url: str
    device_name: str | None = None
    device_public_key: str | None = None
    challenge: str | None = None
    device_id: UUID | None = None
    access_token_hash: str | None = None
    access_token_expires_at: datetime | None = None
    refresh_token_hash: str | None = None
    refresh_token_expires_at: datetime | None = None
    scopes: tuple[OperatorScope, ...] = ()


def _utc_now() -> datetime:
    """Return the current timezone-aware UTC time."""

    return datetime.now(tz=timezone.utc)


def _base64url_encode(value: bytes) -> str:
    """Encode bytes as unpadded URL-safe base64."""

    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _base64url_decode(value: str) -> bytes:
    """Decode strict unpadded URL-safe base64."""

    padding = "=" * (-len(value) % 4)
    return base64.b64decode(
        f"{value}{padding}",
        altchars=b"-_",
        validate=True,
    )


def _opaque_digest(value: str) -> str:
    """Return a non-reversible identifier for a bearer value."""

    return _base64url_encode(hashlib.sha256(value.encode("ascii")).digest())


def pairing_signature_message(
    *,
    cluster_id: UUID,
    nonce: str,
    challenge: str,
) -> bytes:
    """Build the exact message a candidate device signs during pairing.

    Args:
        cluster_id: Target cluster identity from the QR package.
        nonce: Single-use pairing capability from the QR package.
        challenge: Random challenge returned by the gateway.

    Returns:
        Domain-separated bytes suitable for Ed25519 signing.
    """

    return (
        _PAIRING_SIGNATURE_CONTEXT
        + str(cluster_id).encode("ascii")
        + b"\x00"
        + nonce.encode("ascii")
        + b"\x00"
        + challenge.encode("ascii")
    )


@final
class OperatorPairingService:
    """Persist and validate pairing sessions on one designated gateway."""

    def __init__(
        self,
        store: EncryptedAuthorityStore,
        key_provider: LocalFileAuthorityKeyProvider,
        *,
        now: Callable[[], datetime] = _utc_now,
    ) -> None:
        """Create a pairing service with injected persistence and time.

        Args:
            store: Encrypted local authority journal.
            key_provider: Local data-key provider used only during explicit
                gateway initialization.
            now: Time provider for deterministic expiry tests.
        """

        self._store = store
        self._key_provider = key_provider
        self._now = now

    @classmethod
    def from_default_paths(cls) -> "OperatorPairingService":
        """Create the production service for the designated gateway paths."""

        key_provider = LocalFileAuthorityKeyProvider()
        return cls(EncryptedAuthorityStore(key_provider), key_provider)

    def create_session(
        self,
        *,
        exchange_url: str,
        cluster_name: str = "Cluster",
    ) -> PairingPackage:
        """Create one host-authorized five-minute pairing package.

        Args:
            exchange_url: Direct or relayed base URL reachable by the phone.
            cluster_name: Name used only when initializing a new gateway.

        Returns:
            Public package safe to render as a QR code.
        """

        if self._store.path.exists():
            self._key_provider.load_data_key(self._key_provider.active_key_id)
        else:
            self._key_provider.ensure_data_key()
        try:
            identity = self._store.cluster_identity()
        except AuthorityNotInitializedError:
            material = create_cluster_identity(cluster_name)
            self._store.initialize_cluster(
                material.public_identity,
                material.private_key,
            )
            identity = material.public_identity

        now = self._now()
        nonce = secrets.token_urlsafe(32)
        nonce_hash = _opaque_digest(nonce)
        expires_at = now + _PAIRING_LIFETIME
        package = PairingPackage(
            version=1,
            cluster_id=UUID(str(identity.cluster_id)),
            cluster_name=identity.name,
            cluster_fingerprint=identity.fingerprint,
            exchange_url=AnyHttpUrl(exchange_url),
            expires_at=expires_at,
            nonce=nonce,
        )
        session = _StoredPairingSession(
            state="pending",
            nonce_hash=nonce_hash,
            created_at=now,
            expires_at=expires_at,
            exchange_url=str(package.exchange_url),
        )
        self._append_session(nonce_hash, session)
        return package

    def create_challenge(
        self,
        nonce: str,
        request: PairingChallengeRequest,
    ) -> PairingChallengeResponse:
        """Bind a candidate device key and issue one random challenge.

        Args:
            nonce: Single-use pairing capability from the QR package.
            request: Candidate device name and public key.

        Returns:
            Challenge that expires with the pairing session.

        Raises:
            PairingSessionNotFoundError: The nonce is unknown.
            PairingSessionExpiredError: The session expired.
            PairingSessionStateError: A challenge was already issued or used.
            PairingProofError: The proposed public key is malformed.
        """

        self._decode_device_public_key(request.device_public_key)
        nonce_hash = _opaque_digest(nonce)
        session, session_commit_index = self._load_session(nonce_hash)
        self._require_pending(session)
        challenge = _base64url_encode(secrets.token_bytes(32))
        updated = session.model_copy(
            update={
                "state": "challenged",
                "device_name": request.device_name,
                "device_public_key": request.device_public_key,
                "challenge": challenge,
            }
        )
        self._append_session(
            nonce_hash,
            updated,
            expected_record_commit_index=session_commit_index,
        )
        return PairingChallengeResponse(
            challenge=challenge,
            expires_at=session.expires_at,
        )

    def exchange(
        self,
        nonce: str,
        request: PairingExchangeRequest,
    ) -> PairingExchangeResponse:
        """Verify device-key possession and consume one pairing session.

        Args:
            nonce: Single-use pairing capability from the QR package.
            request: Ed25519 proof over the issued challenge.

        Returns:
            One-time access and refresh credentials.

        Raises:
            PairingProofError: Signature validation fails.
            PairingSessionStateError: The session is not awaiting exchange.
            PairingSessionExpiredError: The session expired.
        """

        nonce_hash = _opaque_digest(nonce)
        session, session_commit_index = self._load_session(nonce_hash)
        self._require_unexpired(session)
        if (
            session.state != "challenged"
            or session.device_public_key is None
            or session.challenge is None
            or session.device_name is None
        ):
            raise PairingSessionStateError(
                "pairing session is not awaiting device proof"
            )
        public_key = self._decode_device_public_key(session.device_public_key)
        try:
            signature = _base64url_decode(request.signature)
            public_key.verify(
                signature,
                pairing_signature_message(
                    cluster_id=UUID(str(self._store.cluster_identity().cluster_id)),
                    nonce=nonce,
                    challenge=session.challenge,
                ),
            )
        except (InvalidSignature, ValueError, UnicodeError) as exc:
            raise PairingProofError("device pairing proof is invalid") from exc

        now = self._now()
        device_id = uuid4()
        access_token = secrets.token_urlsafe(32)
        refresh_token = secrets.token_urlsafe(48)
        access_expires_at = now + _ACCESS_TOKEN_LIFETIME
        refresh_expires_at = now + _REFRESH_TOKEN_LIFETIME
        consumed = session.model_copy(
            update={
                "state": "consumed",
                "device_id": device_id,
                "access_token_hash": _opaque_digest(access_token),
                "access_token_expires_at": access_expires_at,
                "refresh_token_hash": _opaque_digest(refresh_token),
                "refresh_token_expires_at": refresh_expires_at,
                "scopes": _DEFAULT_SCOPES,
            }
        )
        self._append_session(
            nonce_hash,
            consumed,
            expected_record_commit_index=session_commit_index,
        )
        return PairingExchangeResponse(
            device_id=device_id,
            cluster=self._store.cluster_identity(),
            access_token=access_token,
            access_token_expires_at=access_expires_at,
            refresh_token=refresh_token,
            refresh_token_expires_at=refresh_expires_at,
            scopes=_DEFAULT_SCOPES,
        )

    def _load_session(self, nonce_hash: str) -> tuple[_StoredPairingSession, int]:
        """Load and validate one encrypted session by its nonce digest."""

        try:
            record, payload = self._store.read_latest_record_payload(
                _PAIRING_RECORD_TYPE,
                nonce_hash,
            )
        except AuthorityNotInitializedError as exc:
            if not self._store.path.exists():
                raise PairingGatewayNotInitializedError(
                    "operator gateway is not initialized"
                ) from exc
            raise PairingSessionNotFoundError("pairing session was not found") from exc
        try:
            session = _StoredPairingSession.model_validate_json(
                json.dumps(payload, separators=(",", ":"), allow_nan=False)
            )
        except ValueError as exc:
            raise PairingError("stored pairing session is invalid") from exc
        return session, record.commit_index

    def _append_session(
        self,
        nonce_hash: str,
        session: _StoredPairingSession,
        *,
        expected_record_commit_index: int = 0,
    ) -> None:
        """Append one encrypted session transition with local CAS."""

        records = self._store.records()
        expected_index = records[-1].commit_index if records else 1
        payload = cast(
            Mapping[str, object],
            session.model_dump(mode="json", by_alias=True),
        )
        try:
            self._store.append(
                expected_commit_index=expected_index,
                expected_record_commit_index=expected_record_commit_index,
                authority_term=1,
                record_type=_PAIRING_RECORD_TYPE,
                record_id=nonce_hash,
                payload=payload,
            )
        except AuthorityCommitConflictError as exc:
            raise PairingSessionStateError(
                "pairing state changed concurrently; restart pairing"
            ) from exc

    def _require_pending(self, session: _StoredPairingSession) -> None:
        """Require an unexpired session that has not issued a challenge."""

        self._require_unexpired(session)
        if session.state != "pending":
            raise PairingSessionStateError("pairing session was already used")

    def _require_unexpired(self, session: _StoredPairingSession) -> None:
        """Reject an expired session without modifying its durable record."""

        if self._now() >= session.expires_at:
            raise PairingSessionExpiredError("pairing session expired")

    @staticmethod
    def _decode_device_public_key(value: str) -> Ed25519PublicKey:
        """Decode and validate one raw Ed25519 device public key."""

        try:
            raw = _base64url_decode(value)
            if len(raw) != 32:
                raise ValueError("wrong key length")
            return Ed25519PublicKey.from_public_bytes(raw)
        except (ValueError, UnicodeError) as exc:
            raise PairingProofError("device public key is invalid") from exc
