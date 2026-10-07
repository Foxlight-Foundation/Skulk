"""Durable downloads from one owner-configured private signed release source."""

import asyncio
import hashlib
import os
import time
from itertools import islice
from pathlib import Path
from typing import Annotated, Final, Literal, Self, final
from urllib.parse import quote, urlsplit
from uuid import uuid4

import httpx
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    JsonValue,
    SecretStr,
    TypeAdapter,
    model_validator,
)

from skulk.extensions.runtime_artifacts import (
    Digest,
    RuntimePlatform,
    RuntimeTrust,
    VerifiedRuntime,
    canonical_json,
)
from skulk.extensions.runtime_attachment import ProfileIdentifier
from skulk.extensions.runtime_files import (
    RuntimeLock,
    private_directory,
    read_private,
    remove_private,
    write_private,
)
from skulk.extensions.runtime_install import RuntimeInstaller, finish_runtime_work

_OBJECT = TypeAdapter(dict[str, JsonValue])

LEGACY_STORE_BINDING_FILE: Final = "store-binding.json"
"""Where an earlier build marked an installation following the built-in store.

Follower status now lives in the source record itself; ``source()`` carries a
marker it finds into the record and removes it."""


class _LegacyStoreBinding(BaseModel):
    """The marker an earlier build wrote for a store-following installation."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    source: Literal["builtin_store"]


_TRANSITION_FILE: Final = "trust-transition.json"
"""Journal of a source record and trust being installed together."""


class StoreTrustRecord(BaseModel):
    """The built-in store trust an installation last took, as the store published it."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    revision: int = Field(ge=1, description="The store's own trust revision.")
    sha256: Digest = Field(
        description="Digest of the store trust document's canonical form."
    )


def store_trust_record(trust: RuntimeTrust) -> StoreTrustRecord:
    """The record naming ``trust`` exactly as the store published it."""
    return StoreTrustRecord(
        revision=trust.revision,
        sha256=hashlib.sha256(
            canonical_json(trust.model_dump(mode="json"))
        ).hexdigest(),
    )


def _source_document(source: "ReleaseSource") -> bytes:
    """The source record as written; follower fields only when they are set.

    Leaving the defaults out keeps a private installation's record readable by
    builds that predate them.
    """
    unset = {
        name
        for name, value in (
            ("follows_store", source.follows_store),
            ("store_trust", source.store_trust),
        )
        if not value
    }
    return source.model_dump_json(exclude=unset or None).encode()


def _source_trust(value: object) -> RuntimeTrust:
    # FastAPI validates parsed JSON as Python data. Re-enter JSON validation for
    # immutable tuple fields without weakening the underlying strict trust model.
    if isinstance(value, RuntimeTrust):
        return value
    return RuntimeTrust.model_validate_json(
        _OBJECT.dump_json(_OBJECT.validate_python(value, strict=True))
    )


class ReleaseSource(BaseModel):
    """Owner-local source settings; never accept a URL override on an install request."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    revision: int = Field(ge=1, description="Owner-maintained source revision.")
    base_url: str = Field(
        max_length=2048, description="Explicit HTTPS artifact directory."
    )
    metadata_filename: str = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,200}\.json$",
        description="Signed runtime metadata basename within the same directory.",
    )
    credential_reference: ProfileIdentifier | None = Field(
        default=None,
        description="Protected local feed credential reference, never its value.",
    )
    follows_store: bool = Field(
        default=False,
        description="Whether the installation's publisher trust follows the "
        "built-in capability store. Recorded with the source, so a change of "
        "trust authority is one durable write.",
    )
    store_trust: StoreTrustRecord | None = Field(
        default=None,
        description="The store trust revision the installation last took, while "
        "it follows the store; an older or different one at that revision is "
        "never applied after it.",
    )

    @model_validator(mode="after")
    def protected_origin(self) -> Self:
        """Reject ambiguous endpoints, URL credentials and query-based secret storage."""
        parsed = urlsplit(self.base_url)
        try:
            httpx.URL(self.base_url)
        except httpx.InvalidURL:
            raise ValueError("invalid release source address") from None
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or not parsed.path.endswith("/")
            or any(character.isspace() for character in self.base_url)
            or parsed.port == 0
        ):
            raise ValueError("release source requires an explicit HTTPS directory")
        return self


class _TrustTransition(BaseModel):
    """A source record and publisher trust to install as one step, journaled first."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    source: ReleaseSource
    trust: RuntimeTrust


class SourceUpdate(BaseModel):
    """Explicit owner trust and source configuration; feed credentials are write-only."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    expected_revision: int = Field(
        ge=0, description="Current source revision, zero before setup."
    )
    base_url: str | None = Field(
        default=None,
        max_length=2048,
        description="Trusted HTTPS release directory; omission retains the configured directory.",
    )
    metadata_filename: str | None = Field(
        default=None,
        max_length=206,
        description="Signed metadata basename; omission retains the configured name.",
    )
    trust: Annotated[RuntimeTrust, BeforeValidator(_source_trust)] | None = Field(
        default=None,
        description="Explicit owner publisher keys and revocation view; omission retains trust. Existing revocations cannot be removed through source configuration.",
    )
    token: SecretStr | None = Field(
        default=None,
        description="Write-only feed bearer; omission retains the prior reference.",
    )
    clear_token: bool = Field(
        default=False,
        description="Explicitly use an anonymous feed; old protected credential evidence is retained.",
    )

    @model_validator(mode="after")
    def valid_source(self) -> Self:
        """Validate source and credential shape before changing any local files."""
        ReleaseSource(
            revision=self.expected_revision + 1,
            base_url=self.base_url
            if self.base_url is not None
            else "https://unused.invalid/",
            metadata_filename=self.metadata_filename
            if self.metadata_filename is not None
            else "runtime.json",
        )
        if self.token is not None:
            token = self.token.get_secret_value()
            if (
                self.clear_token
                or not token
                or len(token) > 8192
                or not token.isascii()
                or any(character.isspace() for character in token)
            ):
                raise ValueError("invalid private feed credential")
        return self


class SourceStatus(BaseModel):
    """Public source readiness without stored token values or protected file paths."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    revision: int = Field(
        ge=0, description="Current source revision, zero if unconfigured."
    )
    configured: bool = Field(
        description="Whether a private release source is configured."
    )
    credential_reference: ProfileIdentifier | None = Field(
        default=None, description="Opaque feed credential reference."
    )
    credential_ready: bool = Field(
        description="Whether the configured credential is readable, or the source is anonymous."
    )
    trust_revision: int | None = Field(
        default=None, description="Configured publisher trust revision."
    )
    follows_store: bool = Field(
        default=False,
        description="Whether the publisher trust follows the built-in capability "
        "store: bound from a store listing and renewed by the host.",
    )


class ReleaseReview(BaseModel):
    """Verified immutable release identity and permissions available before installation."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    runtime_digest: Digest = Field(description="Digest of all signed runtime claims.")
    source_revision: int = Field(ge=1, description="Reviewed owner source revision.")
    publisher: str = Field(description="Trusted publisher identity.")
    bundle_id: str = Field(description="Signed stable plugin identity.")
    version: str = Field(description="Signed plugin release version.")
    sequence: int = Field(ge=1, description="Signed publisher release sequence.")
    platform: RuntimePlatform = Field(description="Verified compatible host platform.")
    python_requires: str = Field(description="Signed supported Python version range.")
    skulk_build_sha256: Digest = Field(
        description="Skulk build the publisher qualified the release against."
    )
    permissions: tuple[str, ...] = Field(
        description="Signed declared plugin permissions."
    )
    artifact_bytes: int = Field(
        ge=1, description="Total signed bundle and wheel byte count."
    )
    expires_at: int = Field(description="Signed release expiry as Unix seconds.")


class InstallRequest(BaseModel):
    """Exact reviewed release intent; no paths, executable, credential or activation."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    operation_id: ProfileIdentifier = Field(
        description="Retained installation operation ID."
    )
    runtime_digest: Digest = Field(description="Exact reviewed signed runtime digest.")
    expected_source_revision: int = Field(ge=1, description="Reviewed source revision.")


class InstallOperation(BaseModel):
    """Durable staging progress; accepted work belongs to the independent manager."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    attempt: int = Field(
        default=0,
        ge=0,
        le=8,
        description="Original attempt is zero; explicitly requested recovery attempts retain separate evidence.",
    )
    attempt_source_revision: int | None = Field(
        default=None,
        ge=1,
        description="Owner-reviewed current source revision for a recovery attempt; original intent remains unchanged.",
    )
    request: InstallRequest = Field(
        description="Original immutable installation request."
    )
    review: ReleaseReview = Field(description="Exact reviewed release being installed.")
    state: Literal[
        "accepted", "downloading", "staging", "staged", "recovery_required"
    ] = Field(description="Installation progress, separate from runtime activation.")
    downloaded_bytes: int = Field(
        default=0, ge=0, description="Verified downloaded artifact bytes."
    )
    error_code: (
        Literal["download_failed", "installation_failed", "installation_interrupted"]
        | None
    ) = Field(
        default=None,
        description="Sanitized local failure class; no raw network or installer output.",
    )

    @model_validator(mode="after")
    def exact_intent(self) -> Self:
        """Reject a journal that changes the reviewed release or source identity."""
        if (
            self.request.runtime_digest != self.review.runtime_digest
            or self.request.expected_source_revision != self.review.source_revision
            or self.downloaded_bytes > self.review.artifact_bytes
        ):
            raise ValueError("installation journal identity differs")
        return self


@final
class RuntimeDownloads:
    """Download and stage one installation without interrupting its active runtime.

    The manager owns this object under its existing process fence. Source and
    publisher trust are local owner settings. HTTP requests select only a reviewed
    digest; they cannot redirect a stored credential to another source.
    """

    def __init__(
        self, root: Path, *, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        """Open durable state without network access or automatic replay."""
        self.root = root
        self.directory = root / "downloads"
        private_directory(self.directory)
        self.installer = RuntimeInstaller(root)
        self.transport = transport
        self.work: asyncio.Task[None] | None = None
        self.active_id: str | None = None
        self.guard = asyncio.Lock()
        self.closed = False

    def source(self) -> ReleaseSource:
        """Read the fixed owner-configured source; missing credentials never fall back.

        A store-binding marker left by an earlier build is carried into the
        record (``follows_store``) and then removed, durably and in that
        order, before the record is returned: the record is the one place
        follower status lives, so no later write can disagree with a marker.
        """
        # A source and trust change interrupted after its journal landed is
        # completed before anything reads either: release operations never
        # see a source whose trust has not caught up, or the reverse.
        self._complete_transition()
        marker = self.root / LEGACY_STORE_BINDING_FILE
        try:
            source = ReleaseSource.model_validate_json(
                read_private(self.root / "release-source.json", 8192)
            )
        except FileNotFoundError:
            # A marker without a source names nothing to follow.
            if marker.exists() or marker.is_symlink():
                remove_private(marker)
            raise
        if not (marker.exists() or marker.is_symlink()):
            return source
        try:
            _LegacyStoreBinding.model_validate_json(read_private(marker, 4096))
            marked = True
        except (OSError, ValueError):
            # An earlier build read an unreadable marker as not following.
            marked = False
        if marked and not source.follows_store:
            source = source.model_copy(update={"follows_store": True})
            write_private(self.root / "release-source.json", _source_document(source))
        remove_private(marker)
        return source

    def _commit(self, source: ReleaseSource, trust: RuntimeTrust) -> None:
        """Install ``source`` and ``trust`` as one durable step.

        The pair is journaled first; the journal landing is the commit. Both
        files are then written and the journal removed. A crash anywhere after
        the journal is completed on the next read of the source, so a change
        of trust authority (joining or leaving the store, a renewal) is never
        half applied. The caller holds the installation fence.
        """
        journal = self.root / _TRANSITION_FILE
        write_private(
            journal,
            _TrustTransition(source=source, trust=trust).model_dump_json().encode(),
        )
        self._apply(source, trust)
        remove_private(journal)

    def _apply(self, source: ReleaseSource, trust: RuntimeTrust) -> None:
        write_private(
            self.root / "publisher-trust.json", trust.model_dump_json().encode()
        )
        write_private(self.root / "release-source.json", _source_document(source))

    def _complete_transition(self) -> None:
        """Finish a journaled source and trust change a crash interrupted.

        Raises:
            ValueError: The journal cannot be read, so release operations are
                refused until it is repaired; nothing is guessed.
        """
        journal = self.root / _TRANSITION_FILE
        if not (journal.exists() or journal.is_symlink()):
            return
        try:
            pending = _TrustTransition.model_validate_json(
                read_private(journal, 262144)
            )
        except (OSError, ValueError):
            raise ValueError(
                "an interrupted source and trust change cannot be read; "
                "local maintenance required"
            ) from None
        self._apply(pending.source, pending.trust)
        remove_private(journal)

    def followed_store_trust(self) -> StoreTrustRecord | None:
        """The store trust this installation last took, or ``None``."""
        try:
            source = self.source()
        except (OSError, ValueError):
            return None
        return source.store_trust if source.follows_store else None

    def follows_store(self) -> bool:
        """Whether this installation's trust follows the built-in capability store."""
        try:
            return self.source().follows_store
        except (OSError, ValueError):
            return False

    def source_status(self) -> SourceStatus:
        """Read source readiness without network I/O or disclosing its credential."""
        try:
            trust = RuntimeTrust.model_validate_json(
                read_private(self.root / "publisher-trust.json")
            )
            trust_revision = trust.revision
        except (OSError, ValueError):
            trust_revision = None
        try:
            source = self.source()
        except FileNotFoundError:
            return SourceStatus(
                revision=0,
                configured=False,
                credential_ready=False,
                trust_revision=trust_revision,
            )
        ready = True
        if source.credential_reference is not None:
            try:
                token = read_private(
                    self.root / "feed-credentials" / source.credential_reference, 8192
                ).decode("ascii")
                ready = bool(token) and not any(
                    character.isspace() for character in token
                )
            except (OSError, ValueError):
                ready = False
        return SourceStatus(
            revision=source.revision,
            configured=True,
            credential_reference=source.credential_reference,
            credential_ready=ready,
            trust_revision=trust_revision,
            follows_store=source.follows_store,
        )

    async def configure(
        self,
        update: SourceUpdate,
        *,
        carry_revocations: bool = True,
        follows_store: bool = False,
        store_trust: StoreTrustRecord | None = None,
        restore_trust: RuntimeTrust | None = None,
    ) -> SourceStatus:
        """Apply direct owner source/trust changes without borrowing any native vault.

        Credentials are provisioned before their reference is published. Trust
        may tighten before source replacement on a disk fault; runtime admission
        then fails closed and the unchanged source revision allows correction.

        Follower status is part of the source record, and the record and
        the trust are installed as one journaled step, so a change of trust
        authority is never half applied: no crash can leave a store-following
        record with an owner's trust, or an owner's record with the store's.

        Args:
            update: The reviewed source and trust change.
            carry_revocations: Keep every revocation the current trust holds
                when the trust revision rises. Only a binding to the built-in
                capability store passes ``False``: its trust applies as the
                store published it.
            follows_store: Whether the installation's trust follows the
                built-in store from now on. Only a store binding, or the
                restore of a store-following source, passes ``True``.
            store_trust: The store trust revision the trust was taken from,
                recorded with a following source.
            restore_trust: Earlier trust content to put back after a refused
                store binding, written at the next revision of the
                installation's history (its trust floor only rises) and
                restored as it was even if it has expired: a rollback returns
                the authority the installation had, never a wider one.
        """
        if self.guard.locked():
            raise ValueError("release source is busy")
        async with self.guard:
            if self.closed or (self.work is not None and not self.work.done()):
                raise ValueError("installation is busy or closed")
            lock = RuntimeLock(self.installer.installer)
            try:
                try:
                    previous = self.source()
                except FileNotFoundError:
                    previous = None
                if (previous.revision if previous else 0) != update.expected_revision:
                    raise ValueError("release source revision conflict")
                try:
                    trust = RuntimeTrust.model_validate_json(
                        read_private(self.root / "publisher-trust.json")
                    )
                except FileNotFoundError:
                    trust = None
                next_trust = update.trust if update.trust is not None else trust
                base_url = (
                    update.base_url
                    if update.base_url is not None
                    else previous.base_url
                    if previous
                    else None
                )
                metadata_filename = (
                    update.metadata_filename
                    if update.metadata_filename is not None
                    else previous.metadata_filename
                    if previous
                    else None
                )
                if next_trust is None or base_url is None or metadata_filename is None:
                    raise ValueError(
                        "initial source setup requires directory, metadata name and publisher trust"
                    )
                if next_trust.expires_at <= time.time() or (
                    trust is not None
                    and (
                        next_trust.revision < trust.revision
                        or (
                            next_trust.revision == trust.revision
                            and next_trust != trust
                        )
                    )
                ):
                    raise ValueError("publisher trust revision conflict or expiry")
                if (
                    carry_revocations
                    and trust is not None
                    and next_trust.revision > trust.revision
                ):
                    # Source credential rotation must never silently restore a
                    # publisher or artifact previously rejected by the owner.
                    next_trust = RuntimeTrust.model_validate(
                        {
                            **next_trust.model_dump(),
                            "revoked_publishers": tuple(
                                sorted(
                                    set(trust.revoked_publishers)
                                    | set(next_trust.revoked_publishers)
                                )
                            ),
                            "revoked_artifacts": tuple(
                                sorted(
                                    set(trust.revoked_artifacts)
                                    | set(next_trust.revoked_artifacts)
                                )
                            ),
                        }
                    )
                reference = previous.credential_reference if previous else None
                if update.clear_token:
                    reference = None
                elif update.token is not None:
                    reference = uuid4().hex
                    write_private(
                        self.root / "feed-credentials" / reference,
                        update.token.get_secret_value().encode("ascii"),
                    )
                elif (
                    previous is not None
                    and previous.base_url != base_url
                    and reference is not None
                ):
                    # A stored bearer is bound to the owner-approved source. A
                    # source move must explicitly supply its credential again.
                    raise ValueError(
                        "source change requires explicit credential replacement"
                    )
                if restore_trust is not None:
                    next_trust = restore_trust.model_copy(
                        update={
                            "revision": trust.revision + 1 if trust is not None else 1
                        }
                    )
                source = ReleaseSource(
                    revision=update.expected_revision + 1,
                    base_url=base_url,
                    metadata_filename=metadata_filename,
                    credential_reference=reference,
                    follows_store=follows_store,
                    store_trust=store_trust if follows_store else None,
                )
                self._commit(source, next_trust)
                return self.source_status()
            finally:
                lock.close()

    async def follow_trust(
        self, trust: RuntimeTrust, *, wait_seconds: float = 2.0
    ) -> bool:
        """Bring this installation's publisher trust up to ``trust``'s content.

        For an installation bound to the built-in capability store: its trust
        becomes the store's current trust (publishers, expiry, revocations) at
        the next revision of the installation's own history, which the
        installer's trust floor requires to rise. The source and its revision
        are untouched, so a reviewed install in flight keeps its fence. The
        running owner's periodic verification reads the new trust: a renewal
        keeps it running past the old expiry, a revocation stops it.

        Follower status is read again under the installation fence, so a
        source change that ended following can never be overwritten here. The
        store revision taken is recorded with the source in the same
        journaled step, and a trust older than it, or a different one at its
        revision, is refused: a catalog read racing a renewal can never push
        older store trust back onto the installation.

        Args:
            trust: The store's verified trust, still in force.
            wait_seconds: How long to wait for the installation fence, which
                the running owner's periodic verification holds for moments
                at a time; the caller shares one deadline across
                installations.

        Returns:
            Whether the installation's trust changed. ``False`` as well when
            the installation no longer follows the store, or already took a
            newer store trust.

        Raises:
            ValueError: The installation is busy, closed, or ``trust`` has
                expired; nothing was written.
            BlockingIOError: Another operation still held the installation
                fence when the wait ran out.
        """
        if trust.expires_at <= time.time():
            raise ValueError("store trust has expired")
        if self.guard.locked():
            raise ValueError("release source is busy")
        async with self.guard:
            if self.closed or (self.work is not None and not self.work.done()):
                raise ValueError("installation is busy or closed")
            # The running owner's periodic verification holds this fence for
            # moments at a time; wait that out rather than skip a renewal.
            deadline = time.monotonic() + wait_seconds
            while True:
                try:
                    lock = RuntimeLock(self.installer.installer)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise
                    await asyncio.sleep(0.05)
            try:
                try:
                    source = self.source()
                except FileNotFoundError:
                    return False
                if not source.follows_store:
                    return False
                offered = store_trust_record(trust)
                taken = source.store_trust
                if taken is not None and (
                    offered.revision < taken.revision
                    or (offered.revision == taken.revision and offered != taken)
                ):
                    # Older than the store trust this installation already
                    # took: a read that verified it before a renewal finished
                    # must not push it back, whatever the local revision.
                    return False
                try:
                    current: RuntimeTrust | None = RuntimeTrust.model_validate_json(
                        read_private(self.root / "publisher-trust.json")
                    )
                except FileNotFoundError:
                    current = None
                if (
                    taken == offered
                    and current is not None
                    and trust.model_copy(update={"revision": current.revision})
                    == current
                ):
                    return False
                self._commit(
                    source.model_copy(update={"store_trust": offered}),
                    trust.model_copy(
                        update={
                            "revision": current.revision + 1
                            if current is not None
                            else 1
                        }
                    ),
                )
                return True
            finally:
                lock.close()

    def _client(self, source: ReleaseSource) -> httpx.AsyncClient:
        headers = {"Accept": "application/octet-stream", "Accept-Encoding": "identity"}
        if source.credential_reference is not None:
            token = (
                read_private(
                    self.root / "feed-credentials" / source.credential_reference, 8192
                )
                .decode("ascii")
                .strip()
            )
            if not token or any(character.isspace() for character in token):
                raise ValueError("release credential unavailable")
            headers["Authorization"] = "Bearer " + token
        return httpx.AsyncClient(
            transport=self.transport,
            timeout=10,
            follow_redirects=False,
            trust_env=False,
            headers=headers,
        )

    @staticmethod
    def _review(runtime: VerifiedRuntime, source: ReleaseSource) -> ReleaseReview:
        release = runtime.claims.release
        return ReleaseReview(
            runtime_digest=runtime.digest,
            source_revision=source.revision,
            publisher=release.publisher,
            bundle_id=release.manifest.bundle_id,
            version=release.manifest.bundle_version,
            sequence=release.sequence,
            platform=runtime.claims.platform,
            python_requires=release.python_requires,
            skulk_build_sha256=release.skulk_build_sha256,
            permissions=release.permissions,
            artifact_bytes=release.artifact_size
            + sum(wheel.size for wheel in runtime.claims.wheels),
            expires_at=release.expires_at,
        )

    async def inspect(self) -> ReleaseReview:
        """Authenticate bounded metadata before artifact download or code execution."""
        if self.guard.locked():
            raise ValueError("release source is busy")
        async with self.guard:
            if self.closed:
                raise ValueError("release downloads closed")
            source = self.source()
            try:
                async with (
                    asyncio.timeout(20),
                    self._client(source) as client,
                    client.stream(
                        "GET", source.base_url + quote(source.metadata_filename)
                    ) as response,
                ):
                    self._response(response)
                    raw = bytearray()
                    async for chunk in response.aiter_bytes(chunk_size=65536):
                        if len(raw) + len(chunk) > 131072:
                            raise ValueError("release metadata exceeds bound")
                        raw.extend(chunk)
                runtime = await self.installer.inspect_metadata(bytes(raw))
                if source != self.source():
                    raise ValueError("release source changed")
                reviews = self.directory / "reviews"
                private_directory(reviews)
                destination = reviews / (runtime.digest + ".json")
                if (
                    not destination.exists()
                    and len(tuple(islice(reviews.iterdir(), 128))) >= 128
                ):
                    raise ValueError(
                        "release review history requires local maintenance"
                    )
                write_private(destination, runtime.metadata)
                return self._review(runtime, source)
            except (httpx.HTTPError, TimeoutError):
                raise ValueError("release metadata unavailable") from None

    @staticmethod
    def _response(response: httpx.Response) -> None:
        if (
            response.status_code != 200
            or response.headers.get("content-encoding", "identity") != "identity"
        ):
            raise ValueError("release download refused")

    def _save(self, operation: InstallOperation) -> None:
        raw = operation.model_dump_json().encode()
        write_private(
            self.directory / "operations" / (operation.request.operation_id + ".json"),
            raw,
        )
        write_private(self.directory / "current.json", raw)

    def operation(self, operation_id: str) -> InstallOperation:
        """Read retained work after reconnect; interrupted work requires explicit recovery."""
        # Public methods also validate paths when called by terminal code directly.
        if len(operation_id) != 32 or any(
            character not in "0123456789abcdef" for character in operation_id
        ):
            raise ValueError("invalid installation operation identity")
        operation = InstallOperation.model_validate_json(
            read_private(self.directory / "operations" / (operation_id + ".json"))
        )
        if operation.request.operation_id != operation_id:
            raise ValueError("installation operation identity differs")
        if operation.state in {"accepted", "downloading", "staging"} and not (
            self.active_id == operation_id
            and self.work is not None
            and not self.work.done()
        ):
            operation = operation.model_copy(
                update={
                    "state": "recovery_required",
                    "error_code": "installation_interrupted",
                }
            )
        return operation

    def current(self) -> InstallOperation | None:
        """Read the last installation reference without repeating a request."""
        try:
            retained = InstallOperation.model_validate_json(
                read_private(self.directory / "current.json")
            )
        except FileNotFoundError:
            return None
        return self.operation(retained.request.operation_id)

    async def submit(self, request: InstallRequest) -> InstallOperation:
        """Journal exact intent before acknowledging and supervise staging independently."""
        if self.guard.locked():
            raise ValueError("release source is busy")
        async with self.guard:
            if self.closed:
                raise ValueError("release downloads closed")
            try:
                prior = self.operation(request.operation_id)
            except FileNotFoundError:
                prior = None
            if prior is not None:
                if prior.request != request:
                    raise ValueError("installation operation identity differs")
                return prior
            if self.work is not None and not self.work.done():
                raise ValueError("installation is busy")
            source = self.source()
            if source.revision != request.expected_source_revision:
                raise ValueError("release source revision conflict")
            metadata = read_private(
                self.directory / "reviews" / (request.runtime_digest + ".json")
            )
            runtime = await self.installer.inspect_metadata(metadata)
            if runtime.digest != request.runtime_digest:
                raise ValueError("reviewed release differs")
            operations = self.directory / "operations"
            private_directory(operations)
            if len(tuple(islice(operations.iterdir(), 128))) >= 128:
                raise ValueError("installation history requires local maintenance")
            operation = InstallOperation(
                request=request, review=self._review(runtime, source), state="accepted"
            )
            self._save(operation)
            self.active_id = request.operation_id
            self.work = asyncio.create_task(self._install(operation, runtime, source))
            return operation

    async def _artifact(
        self,
        client: httpx.AsyncClient,
        source: ReleaseSource,
        directory: Path,
        name: str,
        size: int,
        digest: str,
    ) -> None:
        destination = directory / name
        temporary = directory / (name + ".partial")
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
        )
        total, checksum = 0, hashlib.sha256()
        with os.fdopen(descriptor, "wb") as target:
            async with client.stream("GET", source.base_url + quote(name)) as response:
                self._response(response)
                async for chunk in response.aiter_bytes(chunk_size=65536):
                    total += len(chunk)
                    if total > size:
                        raise ValueError("release artifact exceeds signed size")
                    checksum.update(chunk)
                    target.write(chunk)
            if total != size or checksum.hexdigest() != digest:
                raise ValueError("release artifact differs from signed bytes")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, destination)

    async def _install(
        self,
        operation: InstallOperation,
        runtime: VerifiedRuntime,
        source: ReleaseSource,
    ) -> None:
        try:
            operation = operation.model_copy(update={"state": "downloading"})
            self._save(operation)
            name = operation.request.operation_id
            if operation.attempt:
                name += f".{operation.attempt}"
            directory = self.directory / "artifacts" / name
            private_directory(directory)
            release = runtime.claims.release
            artifacts = [
                (
                    "bundle.pyz",
                    release.artifact_size,
                    release.manifest.executable_sha256,
                )
            ]
            artifacts.extend(
                (wheel.filename, wheel.size, wheel.sha256)
                for wheel in runtime.claims.wheels
            )
            async with asyncio.timeout(180), self._client(source) as client:
                for name, size, digest in artifacts:
                    if source != self.source():
                        raise ValueError("release source changed")
                    await self._artifact(client, source, directory, name, size, digest)
                    operation = operation.model_copy(
                        update={"downloaded_bytes": operation.downloaded_bytes + size}
                    )
                    self._save(operation)
            if source != self.source():
                raise ValueError("release source changed")
            operation = operation.model_copy(update={"state": "staging"})
            self._save(operation)
            staged = await self.installer.stage(
                runtime.metadata,
                directory,
                operation_id=operation.request.operation_id,
                recover=operation.attempt > 0,
                wait_for_ownership=True,
            )
            if staged.state != "staged":
                raise ValueError("offline installation requires recovery")
            operation = operation.model_copy(update={"state": "staged"})
            self._save(operation)
        except (
            OSError,
            ValueError,
            LookupError,
            httpx.HTTPError,
            TimeoutError,
            asyncio.CancelledError,
        ):
            self._save(
                operation.model_copy(
                    update={
                        "state": "recovery_required",
                        "error_code": "installation_failed"
                        if operation.state == "staging"
                        else "download_failed",
                    }
                )
            )

    async def recover(
        self, operation_id: str, expected_source_revision: int
    ) -> InstallOperation:
        """Explicitly resume the original signed release under the reviewed current source.

        Credential rotation may change the source revision, but cannot change the
        original runtime digest, permissions, identity or request. Prior download
        attempts and incomplete offline generations remain protected evidence.
        """
        if self.guard.locked():
            raise ValueError("release source is busy")
        async with self.guard:
            if self.closed:
                raise ValueError("release downloads closed")
            prior = self.operation(operation_id)
            if prior.state != "recovery_required":
                return prior
            if self.work is not None and not self.work.done():
                raise ValueError("installation is busy")
            if prior.attempt >= 8:
                raise ValueError(
                    "installation recovery history requires local maintenance"
                )
            source = self.source()
            if source.revision != expected_source_revision:
                raise ValueError("release source revision conflict")
            runtime = await self.installer.inspect_metadata(
                read_private(
                    self.directory
                    / "reviews"
                    / (prior.request.runtime_digest + ".json")
                )
            )
            if runtime.digest != prior.request.runtime_digest:
                raise ValueError("reviewed release differs")
            write_private(
                self.directory / "attempts" / operation_id / f"{prior.attempt}.json",
                prior.model_dump_json().encode(),
            )
            operation = prior.model_copy(
                update={
                    "attempt": prior.attempt + 1,
                    "attempt_source_revision": source.revision,
                    "state": "accepted",
                    "downloaded_bytes": 0,
                    "error_code": None,
                }
            )
            self._save(operation)
            self.active_id = operation_id
            self.work = asyncio.create_task(self._install(operation, runtime, source))
            return operation

    async def close(self) -> None:
        """Cancel bounded downloads and finish owned offline staging before releasing state."""
        self.closed = True
        if self.work is not None:
            self.work.cancel()

            async def finish() -> None:
                assert self.work is not None
                results = await asyncio.gather(self.work, return_exceptions=True)
                for result in results:
                    if isinstance(result, BaseException) and not isinstance(
                        result, asyncio.CancelledError
                    ):
                        raise ValueError("installation shutdown failed") from None

            await finish_runtime_work(asyncio.create_task(finish()))
