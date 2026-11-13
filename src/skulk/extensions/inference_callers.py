"""Core authentication and private enrollment for inference API clients.

Credentials establish original caller identity only. Policy grants, signed
destination checks and a protected owner dispatch fence remain independent.
There is no provider-account credential, author-supplied identity or router here.
"""

import hmac
import re
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import final

from pydantic import BaseModel, ConfigDict, Field

from skulk.extensions.caller_identity import AuthenticatedOwner
from skulk.extensions.native_inference_authority import (
    InferenceAccessPolicy,
    InferenceCredential,
    InferenceCredentialState,
    InferenceGrant,
    InferencePrincipal,
    PrincipalAccess,
    inference_access,
    inference_revision,
)
from skulk.extensions.runtime_attachment import HostSettings
from skulk.extensions.runtime_files import RuntimeLock, read_private, write_private

_TOKEN = re.compile(r"ski_([a-f0-9]{32})_([a-f0-9]{64})")
_ZERO = "0" * 64


class _Contract(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


@final
class ClientSummary(_Contract):
    """Safe owner-visible credential metadata, without a bearer or its digest."""

    identifier: str = Field(description="Server-issued client identifier.")
    label: str = Field(description="Owner-assigned label.")
    expires_unix_ms: int | None = Field(
        description="Optional expiry in Unix milliseconds."
    )


@final
class ClientInventory(_Contract):
    """Current credential revision and active enrollment metadata."""

    revision: str = Field(description="Expected revision for enrollment or revocation.")
    clients: tuple[ClientSummary, ...] = Field(
        description="Enrolled clients; expired clients cannot authenticate."
    )


@final
@dataclass(frozen=True, slots=True, repr=False)
class AuthenticatedInferenceCaller:
    """Process-local result of actual Core authentication, never HTTP input."""

    principal: InferencePrincipal
    credential_revision: str | None


@final
@dataclass(frozen=True, slots=True, repr=False)
class InferenceAuthorization:
    """Current authority snapshot to attenuate with native composition admission."""

    caller: AuthenticatedInferenceCaller
    revision: str
    access: PrincipalAccess
    credential_deadline_unix_ms: int | None


@final
class InferenceCallers:
    """Authenticate clients and fence enrollment to one actual manager profile."""

    def __init__(
        self,
        root: Path,
        profile_id: str,
        issuer_node_id: str,
        *,
        unix_ms: Callable[[], int] = lambda: time.time_ns() // 1_000_000,
    ) -> None:
        """Bind host-configured custody and an injectable expiry clock.

        No request may select this directory, issuer or profile. The Core
        manager owns it; capability backup/restore never replaces these files.
        """
        self.root, self.profile_id, self.issuer_node_id = (
            root,
            profile_id,
            issuer_node_id,
        )
        self.unix_ms = unix_ms

    def check(self) -> None:
        """Refuse a stale service attachment before reading or changing authority."""
        actual = HostSettings.model_validate_json(read_private(self.root / "host.json"))
        if (
            actual.profile_id != self.profile_id
            or actual.transport_node_id != self.issuer_node_id
        ):
            raise ValueError("inference authority attachment differs")

    def state(self) -> InferenceCredentialState | None:
        """Read only bounded private state; absence grants no client identity."""
        self.check()
        try:
            state = InferenceCredentialState.model_validate_json(
                read_private(self.root / "inference-clients.json")
            )
        except FileNotFoundError:
            return None
        if (
            state.profile_id != self.profile_id
            or state.issuer_node_id != self.issuer_node_id
        ):
            raise ValueError("inference credential custody differs")
        return state

    def inventory(self) -> ClientInventory:
        """Return metadata only; token hashes and digest seeds never leave custody."""
        state = self.state()
        return ClientInventory(
            revision=_ZERO if state is None else state.revision,
            clients=()
            if state is None
            else tuple(
                ClientSummary(
                    identifier=c.identifier,
                    label=c.label,
                    expires_unix_ms=c.expires_unix_ms,
                )
                for c in state.clients
            ),
        )

    def create(
        self, expected_revision: str, label: str, expires_unix_ms: int | None
    ) -> tuple[ClientSummary, str]:
        """Enroll one unprivileged client and return its bearer exactly once.

        The owner must separately grant exact destinations. All mutations share
        the owner dispatch fence; a stale management page cannot undo revocation.
        """
        lock = RuntimeLock(self.root, "native-policy.lock")
        try:
            state = self.state()
            if expected_revision != (_ZERO if state is None else state.revision):
                raise ValueError("inference credential revision changed")
            if expires_unix_ms is not None and expires_unix_ms <= self.unix_ms():
                raise ValueError("inference credential already expired")
            if state is None:
                state = InferenceCredentialState(
                    profile_id=self.profile_id,
                    issuer_node_id=self.issuer_node_id,
                    revision=_ZERO,
                    digest_key=secrets.token_hex(32),
                    clients=(),
                )
            identifier = secrets.token_hex(16)
            token = f"ski_{identifier}_{secrets.token_hex(32)}"
            client = InferenceCredential(
                identifier=identifier,
                revision=secrets.token_hex(32),
                label=label,
                token_digest=hmac.digest(
                    bytes.fromhex(state.digest_key), token.encode("ascii"), "sha256"
                ).hex(),
                expires_unix_ms=expires_unix_ms,
            )
            updated = InferenceCredentialState(
                profile_id=self.profile_id,
                issuer_node_id=self.issuer_node_id,
                revision=secrets.token_hex(32),
                digest_key=state.digest_key,
                clients=state.clients + (client,),
            )
            write_private(
                self.root / "inference-clients.json", updated.model_dump_json().encode()
            )
            return ClientSummary(
                identifier=identifier, label=label, expires_unix_ms=expires_unix_ms
            ), token
        finally:
            lock.close()

    def revoke(self, expected_revision: str, identifier: str) -> ClientInventory:
        """Remove one credential lifetime under the same physical dispatch fence."""
        lock = RuntimeLock(self.root, "native-policy.lock")
        try:
            state = self.state()
            if (
                state is None
                or expected_revision != state.revision
                or identifier not in {c.identifier for c in state.clients}
            ):
                raise ValueError("inference credential revision or identity changed")
            updated = InferenceCredentialState(
                profile_id=state.profile_id,
                issuer_node_id=state.issuer_node_id,
                revision=secrets.token_hex(32),
                digest_key=state.digest_key,
                clients=tuple(c for c in state.clients if c.identifier != identifier),
            )
            write_private(
                self.root / "inference-clients.json", updated.model_dump_json().encode()
            )
            return self.inventory()
        finally:
            lock.close()

    def authenticate(self, token: str) -> AuthenticatedInferenceCaller:
        """Validate a real bearer; no caller ID, gateway signature or JSON substitutes."""
        match = _TOKEN.fullmatch(token)
        if match is None:
            raise ValueError("inference credential refused")
        state = self.state()
        if state is None:
            raise ValueError("inference credential refused")
        digest = hmac.digest(
            bytes.fromhex(state.digest_key), token.encode("ascii"), "sha256"
        ).hex()
        matches = [
            c
            for c in state.clients
            if c.identifier == match[1] and hmac.compare_digest(c.token_digest, digest)
        ]
        if len(matches) != 1:
            raise ValueError("inference credential refused")
        client = matches[0]
        if (
            client.expires_unix_ms is not None
            and client.expires_unix_ms <= self.unix_ms()
        ):
            raise ValueError("inference credential refused")
        return AuthenticatedInferenceCaller(
            InferencePrincipal(
                identifier=f"client:{client.identifier}",
                kind="client",
                issuer_node_id=self.issuer_node_id,
            ),
            client.revision,
        )

    def owner(self, owner: AuthenticatedOwner) -> AuthenticatedInferenceCaller:
        """Project an already authenticated direct owner without adding permission."""
        self.check()
        if str(owner.issuer_node_id) != self.issuer_node_id:
            raise ValueError("inference owner issuer differs")
        return AuthenticatedInferenceCaller(
            InferencePrincipal(
                identifier=owner.identifier,
                kind=owner.kind,
                issuer_node_id=self.issuer_node_id,
            ),
            None,
        )

    def current(self, caller: AuthenticatedInferenceCaller) -> int | None:
        """Recheck credential lifetime after queueing; return its optional deadline."""
        self.check()
        principal = caller.principal
        if principal.issuer_node_id != self.issuer_node_id:
            raise ValueError("inference caller issuer differs")
        if principal.kind == "operator":
            if caller.credential_revision is not None:
                raise ValueError("inference owner credential differs")
            return None
        state = self.state()
        matches = (
            []
            if state is None
            else [
                c
                for c in state.clients
                if f"client:{c.identifier}" == principal.identifier
                and c.revision == caller.credential_revision
            ]
        )
        if len(matches) != 1:
            raise ValueError("inference caller was revoked")
        expiry = matches[0].expires_unix_ms
        if expiry is not None and expiry <= self.unix_ms():
            raise ValueError("inference caller expired")
        return expiry

    def policy(self) -> InferenceAccessPolicy:
        """Read a permission snapshot; an absent policy is explicitly disabled."""
        self.check()
        try:
            return InferenceAccessPolicy.model_validate_json(
                read_private(self.root / "native-inference-policy.json")
            )
        except FileNotFoundError:
            return InferenceAccessPolicy(revision=_ZERO, enabled=False, callers=())

    def authorize(
        self, caller: AuthenticatedInferenceCaller, grant: InferenceGrant
    ) -> InferenceAuthorization:
        """Require a current caller and exact destination without readiness coupling."""
        deadline = self.current(caller)
        policy = self.policy()
        access = inference_access(policy, caller.principal, grant)
        return InferenceAuthorization(
            caller,
            inference_revision(policy, caller.principal, caller.credential_revision),
            access,
            deadline,
        )
