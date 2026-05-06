"""Companion app pairing and read-only overview contracts."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import os
import secrets
import socket
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, final

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)
from pydantic import Field, ValidationError

from exo.connectivity.remote_access import RemoteAccessInfo
from exo.shared.constants import (
    SKULK_COMPANION_CLUSTER_KEY,
    SKULK_COMPANION_CREDENTIALS,
)
from exo.shared.types.common import NodeId
from exo.shared.types.events import Event
from exo.shared.types.state import State
from exo.shared.types.worker.instances import InstanceId
from exo.utils.pydantic_ext import CamelCaseModel, FrozenModel

PAIRING_VERSION: Literal[1] = 1
PAIRING_SESSION_TTL = timedelta(minutes=5)
MAX_PAIRING_SESSIONS = 64
COMPANION_READ_SCOPES: tuple[str, ...] = (
    "cluster:read",
    "nodes:read",
    "models:read",
    "events:read",
)


@final
class CompanionPairingSessionRequest(CamelCaseModel):
    cluster_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=80,
        description="Optional consumer-facing cluster name to display in the app.",
    )


@final
class CompanionPairingQrPayload(FrozenModel):
    version: Literal[1]
    cluster_id: str
    cluster_name: str
    pairing_nonce: str
    expires_at: datetime
    exchange_url: str
    cluster_public_key: str
    lan_url: str | None = None
    tailscale_url: str | None = None
    preferred_url: str | None = None


@final
class CompanionPairingSessionResponse(FrozenModel):
    qr_payload: CompanionPairingQrPayload


@final
class CompanionPairingExchangeRequest(CamelCaseModel):
    client_name: str | None = Field(default=None, min_length=1, max_length=80)


@final
class CompanionClusterMetadata(FrozenModel):
    cluster_id: str
    cluster_name: str
    cluster_public_key: str
    node_id: NodeId
    base_url: str | None


@final
class CompanionPairingExchangeResponse(FrozenModel):
    credential_id: str
    token: str
    scopes: Sequence[str]
    issued_at: datetime
    expires_at: datetime | None
    cluster: CompanionClusterMetadata


@final
class CompanionNodeOverview(FrozenModel):
    node_id: NodeId
    display_name: str
    online: bool
    model_id: str | None = None
    exo_version: str | None = None
    lan_ips: Sequence[str] = ()


@final
class CompanionRunningModel(FrozenModel):
    instance_id: InstanceId
    model_id: str
    node_ids: Sequence[NodeId]


@final
class CompanionConnectionOverview(FrozenModel):
    authenticated: bool
    scopes: Sequence[str]
    transport_url: str | None
    tailscale_running: bool


@final
class CompanionEventOverview(FrozenModel):
    event_type: str


@final
class CompanionCredentialRecord(FrozenModel):
    credential_id: str
    token_hash: str
    scopes: Sequence[str]
    issued_at: datetime
    cluster_name: str
    client_name: str | None = None
    revoked: bool = False


@final
class CompanionCredentialStore(FrozenModel):
    credentials: Sequence[CompanionCredentialRecord] = ()


@final
class CompanionOverviewResponse(FrozenModel):
    cluster: CompanionClusterMetadata
    connection: CompanionConnectionOverview
    nodes: Sequence[CompanionNodeOverview]
    running_models: Sequence[CompanionRunningModel]
    recent_events: Sequence[CompanionEventOverview]


@final
class _PairingSession:
    def __init__(
        self,
        *,
        nonce: str,
        expires_at: datetime,
        cluster_name: str,
        base_url: str | None,
    ) -> None:
        self.nonce = nonce
        self.expires_at = expires_at
        self.cluster_name = cluster_name
        self.base_url = base_url
        self.exchanged = False


@final
class CompanionCredential:
    def __init__(
        self,
        *,
        credential_id: str,
        token_hash: str,
        scopes: Sequence[str],
        issued_at: datetime,
        cluster_name: str,
        client_name: str | None = None,
        revoked: bool = False,
    ) -> None:
        self.credential_id = credential_id
        self.token_hash = token_hash
        self.scopes = tuple(scopes)
        self.issued_at = issued_at
        self.cluster_name = cluster_name
        self.client_name = client_name
        self.revoked = revoked


class CompanionAuthError(Exception):
    """Raised when a companion credential is missing or invalid."""


class CompanionPairingError(Exception):
    """Raised when a pairing session cannot be created or exchanged."""


@final
class CompanionPairingManager:
    """In-process companion pairing and credential registry.

    Credentials are intentionally stored as SHA-256 hashes. The cleartext bearer
    token is returned once during nonce exchange and must be stored by the
    mobile app in platform secure storage.
    """

    def __init__(
        self,
        *,
        node_id: NodeId,
        key_path: Path = SKULK_COMPANION_CLUSTER_KEY,
        credentials_path: Path = SKULK_COMPANION_CREDENTIALS,
    ) -> None:
        self._node_id = node_id
        self._key_path = key_path
        self._credentials_path = credentials_path
        self._sessions: dict[str, _PairingSession] = {}
        self._credentials: dict[str, CompanionCredential] = self._load_credentials()
        self._credential_hash_index: dict[str, CompanionCredential] = {
            credential.token_hash: credential
            for credential in self._credentials.values()
            if not credential.revoked
        }
        self._cluster_private_key = self._load_or_create_cluster_key()
        public_key = self._cluster_private_key.public_key().public_bytes(
            encoding=Encoding.Raw,
            format=PublicFormat.Raw,
        )
        self._cluster_public_key = f"ed25519:{base64.urlsafe_b64encode(public_key).decode().rstrip('=')}"
        self._cluster_id = hashlib.sha256(self._cluster_public_key.encode()).hexdigest()[:24]

    @property
    def cluster_id(self) -> str:
        return self._cluster_id

    @property
    def cluster_public_key(self) -> str:
        return self._cluster_public_key

    def create_session(
        self,
        *,
        request: CompanionPairingSessionRequest,
        remote_access: RemoteAccessInfo,
    ) -> CompanionPairingSessionResponse:
        self._prune_expired_sessions()
        nonce = secrets.token_urlsafe(32)
        now = _utc_now()
        expires_at = now + PAIRING_SESSION_TTL
        cluster_name = request.cluster_name or default_cluster_name()
        base_url = remote_access.preferred_url
        if base_url is None:
            raise CompanionPairingError("pairing_url_unavailable")
        session = _PairingSession(
            nonce=nonce,
            expires_at=expires_at,
            cluster_name=cluster_name,
            base_url=base_url,
        )
        self._sessions[nonce] = session
        self._trim_pairing_sessions()
        exchange_url = (
            f"{base_url}/v1/companion/pairing-sessions/{nonce}/exchange"
            if base_url
            else f"/v1/companion/pairing-sessions/{nonce}/exchange"
        )
        return CompanionPairingSessionResponse(
            qr_payload=CompanionPairingQrPayload(
                version=PAIRING_VERSION,
                cluster_id=self.cluster_id,
                cluster_name=cluster_name,
                pairing_nonce=nonce,
                expires_at=expires_at,
                exchange_url=exchange_url,
                cluster_public_key=self.cluster_public_key,
                lan_url=remote_access.local.url,
                tailscale_url=remote_access.tailscale.url,
                preferred_url=remote_access.preferred_url,
            )
        )

    def exchange_session(
        self,
        *,
        nonce: str,
        request: CompanionPairingExchangeRequest,
    ) -> CompanionPairingExchangeResponse:
        session = self._sessions.get(nonce)
        if session is None:
            raise CompanionAuthError("invalid_code")
        if session.exchanged:
            raise CompanionAuthError("already_used")
        if session.expires_at <= _utc_now():
            del self._sessions[nonce]
            raise CompanionAuthError("expired_code")

        credential_id = secrets.token_urlsafe(16)
        token = secrets.token_urlsafe(48)
        issued_at = _utc_now()
        credential = CompanionCredential(
            credential_id=credential_id,
            token_hash=_hash_token(token),
            scopes=COMPANION_READ_SCOPES,
            issued_at=issued_at,
            cluster_name=session.cluster_name,
            client_name=request.client_name,
        )
        self._credentials[credential_id] = credential
        try:
            self._save_credentials()
        except Exception:
            del self._credentials[credential_id]
            raise
        self._credential_hash_index[credential.token_hash] = credential
        session.exchanged = True
        return CompanionPairingExchangeResponse(
            credential_id=credential_id,
            token=token,
            scopes=credential.scopes,
            issued_at=issued_at,
            expires_at=None,
            cluster=CompanionClusterMetadata(
                cluster_id=self.cluster_id,
                cluster_name=session.cluster_name,
                cluster_public_key=self.cluster_public_key,
                node_id=self._node_id,
                base_url=session.base_url,
            ),
        )

    def authenticate_bearer(self, bearer_token: str | None) -> CompanionCredential:
        if not bearer_token:
            raise CompanionAuthError("missing_token")
        token_hash = _hash_token(bearer_token)
        credential = self._credential_hash_index.get(token_hash)
        if credential is not None and not credential.revoked:
            return credential
        raise CompanionAuthError("auth_failed")

    def revoke_credential(self, credential_id: str) -> bool:
        credential = self._credentials.get(credential_id)
        if credential is None:
            return False
        credential.revoked = True
        self._credential_hash_index.pop(credential.token_hash, None)
        self._save_credentials()
        return True

    def _prune_expired_sessions(self) -> None:
        now = _utc_now()
        expired = [
            nonce
            for nonce, session in self._sessions.items()
            if session.expires_at <= now
        ]
        for nonce in expired:
            del self._sessions[nonce]

    def _trim_pairing_sessions(self) -> None:
        if len(self._sessions) <= MAX_PAIRING_SESSIONS:
            return
        excess = len(self._sessions) - MAX_PAIRING_SESSIONS
        ordered = sorted(self._sessions.values(), key=lambda session: session.expires_at)
        for session in ordered[:excess]:
            self._sessions.pop(session.nonce, None)

    def _load_or_create_cluster_key(self) -> Ed25519PrivateKey:
        if self._key_path.exists():
            try:
                private_key = Ed25519PrivateKey.from_private_bytes(
                    self._key_path.read_bytes()
                )
            except (OSError, ValueError):
                with contextlib.suppress(OSError):
                    self._key_path.unlink()
            else:
                with contextlib.suppress(OSError):
                    self._key_path.chmod(0o600)
                return private_key
        private_key = Ed25519PrivateKey.generate()
        private_bytes = private_key.private_bytes(
            encoding=Encoding.Raw,
            format=PrivateFormat.Raw,
            encryption_algorithm=NoEncryption(),
        )
        _atomic_write_private_file(self._key_path, private_bytes)
        return private_key

    def _load_credentials(self) -> dict[str, CompanionCredential]:
        if not self._credentials_path.exists():
            return {}
        try:
            store = CompanionCredentialStore.model_validate_json(
                self._credentials_path.read_text()
            )
        except (OSError, UnicodeDecodeError, ValidationError):
            return {}
        return {
            record.credential_id: CompanionCredential(
                credential_id=record.credential_id,
                token_hash=record.token_hash,
                scopes=record.scopes,
                issued_at=record.issued_at,
                cluster_name=record.cluster_name,
                client_name=record.client_name,
                revoked=record.revoked,
            )
            for record in store.credentials
        }

    def _save_credentials(self) -> None:
        store = CompanionCredentialStore(
            credentials=tuple(
                CompanionCredentialRecord(
                    credential_id=credential.credential_id,
                    token_hash=credential.token_hash,
                    scopes=credential.scopes,
                    issued_at=credential.issued_at,
                    cluster_name=credential.cluster_name,
                    client_name=credential.client_name,
                    revoked=credential.revoked,
                )
                for credential in self._credentials.values()
            )
        )
        _atomic_write_private_file(
            self._credentials_path,
            store.model_dump_json(indent=2).encode(),
        )


def build_companion_overview(
    *,
    node_id: NodeId,
    state: State,
    remote_access: RemoteAccessInfo,
    credential: CompanionCredential,
    manager: CompanionPairingManager,
    recent_events: Sequence[Event],
) -> CompanionOverviewResponse:
    node_ids = set(state.topology.list_nodes())
    node_ids.update(state.node_identities.keys())
    node_ids.update(state.node_network.keys())
    node_ids.add(node_id)

    nodes = [
        _build_node_overview(
            node_id=known_node_id,
            local_node_id=node_id,
            state=state,
        )
        for known_node_id in sorted(node_ids)
    ]
    running_models = [
        CompanionRunningModel(
            instance_id=instance_id,
            model_id=str(instance.shard_assignments.model_id),
            node_ids=tuple(instance.shard_assignments.node_to_runner.keys()),
        )
        for instance_id, instance in state.instances.items()
    ]
    return CompanionOverviewResponse(
        cluster=CompanionClusterMetadata(
            cluster_id=manager.cluster_id,
            cluster_name=credential.cluster_name,
            cluster_public_key=manager.cluster_public_key,
            node_id=node_id,
            base_url=remote_access.preferred_url,
        ),
        connection=CompanionConnectionOverview(
            authenticated=True,
            scopes=credential.scopes,
            transport_url=remote_access.preferred_url,
            tailscale_running=remote_access.tailscale.running,
        ),
        nodes=nodes,
        running_models=running_models,
        recent_events=tuple(
            CompanionEventOverview(event_type=event.__class__.__name__)
            for event in recent_events
        ),
    )


def default_cluster_name() -> str:
    hostname = socket.gethostname().split(".")[0]
    return f"{hostname} Skulk" if hostname else "Skulk Cluster"


def bearer_token_from_authorization(authorization: str | None) -> str | None:
    if authorization is None:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token.strip()


def _build_node_overview(
    *,
    node_id: NodeId,
    local_node_id: NodeId,
    state: State,
) -> CompanionNodeOverview:
    identity = state.node_identities.get(node_id)
    network = state.node_network.get(node_id)
    lan_ips = tuple(iface.ip_address for iface in network.interfaces) if network else ()
    return CompanionNodeOverview(
        node_id=node_id,
        display_name=identity.friendly_name if identity else str(node_id),
        online=node_id in state.last_seen or node_id == local_node_id,
        model_id=identity.model_id if identity else None,
        exo_version=identity.exo_version if identity else None,
        lan_ips=lan_ips,
    )


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _atomic_write_private_file(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        with contextlib.suppress(OSError):
            path.chmod(0o600)
    except Exception:
        with contextlib.suppress(OSError):
            os.close(fd)
        with contextlib.suppress(OSError):
            tmp_path.unlink()
        raise
