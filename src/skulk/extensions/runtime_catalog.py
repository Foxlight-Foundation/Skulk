"""The signed capability catalog a host reads: discovery and consent, never authority.

A catalog is a flat index signed by a publisher this host trusts for
discovery. Each entry names a release, where its signed record is served and
the facts an operator needs before any transfer: what the capability can do
and what it can spend. Reading the catalog selects nothing and installs
nothing; installing still goes through the verified release path, where the
release record itself is authenticated against installation trust.
"""

import asyncio
import hashlib
import json
import re
import time
from itertools import islice
from pathlib import Path
from typing import Annotated, Literal, Self, final
from urllib.parse import quote, urlsplit
from uuid import uuid4

import httpx
from pydantic import (
    AfterValidator,
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
    Identifier,
    ProtocolUnsupportedError,
    RuntimeTrust,
    canonical_json,
    canonical_platform,
    platform_matches,
)
from skulk.extensions.runtime_attachment import ProfileIdentifier
from skulk.extensions.runtime_files import (
    RuntimeLock,
    private_directory,
    read_private,
    write_private,
)

ACCEPTED_CATALOG_PROTOCOLS: tuple[int, ...] = (1,)
"""Catalog record protocols this host reads: the current and, once there is
one, the previous."""

_OBJECT: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue])


def _accepted_catalog_protocol(value: int) -> int:
    if value not in ACCEPTED_CATALOG_PROTOCOLS:
        window = " or ".join(str(item) for item in ACCEPTED_CATALOG_PROTOCOLS)
        raise ValueError(
            f"catalog protocol {value} is not accepted; this host accepts {window}"
        )
    return value


CatalogProtocol = Annotated[int, AfterValidator(_accepted_catalog_protocol)]


class _Contract(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


_Text = Annotated[str, Field(min_length=1, max_length=512)]


class _CatalogEntryClaims(_Contract):
    bundle_id: Identifier
    bundle_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    title: str | None = Field(default=None, max_length=120)
    publisher: Identifier
    sequence: int = Field(ge=1)
    feed_url: str = Field(max_length=512)
    release_sha256: Digest
    release_digest: Digest
    runtime_platform: Annotated[str, Field(max_length=64)] | None = None
    artifact_sha256: Digest
    artifact_size: int = Field(ge=1, le=67108864)
    transfer_bytes: int = Field(ge=1, le=67108864 + 536870912)
    platforms: tuple[Literal["darwin", "linux"], ...] = Field(
        min_length=1, max_length=2
    )
    skulk_build_sha256: Digest
    permissions: tuple[_Text, ...] = Field(min_length=1, max_length=16)
    descriptors: tuple[Annotated[str, Field(max_length=200)], ...] = Field(
        min_length=1, max_length=8
    )
    surfaces: tuple[Annotated[str, Field(max_length=64)], ...] = Field(
        default=(), max_length=4
    )
    operations: bool = False
    steward_risks: tuple[Literal["observation", "billable", "effect"], ...] = Field(
        default=(), max_length=3
    )
    expires_at: int = Field(gt=0)

    @model_validator(mode="after")
    def served_from_a_directory(self) -> Self:
        """A feed is an explicit HTTPS directory, like a configured release source."""
        if self.transfer_bytes < self.artifact_size:
            raise ValueError("catalog entry transfer size cannot be below its artifact")
        parsed = urlsplit(self.feed_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or not parsed.path.endswith("/")
            or any(character.isspace() for character in self.feed_url)
        ):
            raise ValueError("catalog entry feed requires an explicit HTTPS directory")
        return self


class _CatalogClaims(_Contract):
    protocol: CatalogProtocol
    publisher: Identifier
    revision: int = Field(ge=1)
    created_at: int = Field(gt=0)
    expires_at: int = Field(gt=0)
    entries: tuple[_CatalogEntryClaims, ...] = Field(max_length=64)

    @model_validator(mode="after")
    def coherent(self) -> Self:
        """The listing is the signer's own and names a bundle sequence once."""
        if not self.created_at < self.expires_at:
            raise ValueError("catalog expiry must follow creation")
        seen: set[tuple[str, int, str | None]] = set()
        for entry in self.entries:
            if entry.publisher != self.publisher:
                raise ValueError("catalog entries must belong to the catalog publisher")
            # One sequence may ship one artifact family per runtime platform;
            # aliases of a family (a distribution name and its libc family)
            # are the same listing, as runtime compatibility treats them.
            family = (
                canonical_platform(entry.runtime_platform)
                if entry.runtime_platform is not None
                else None
            )
            key = (entry.bundle_id, entry.sequence, family)
            if key in seen:
                raise ValueError("catalog lists one bundle sequence twice")
            seen.add(key)
        return self


class _SignedCatalog(_Contract):
    catalog: dict[str, JsonValue]
    signature: str = Field(pattern=r"^[a-f0-9]{128}$")


@final
class CatalogEntryReview(BaseModel):
    """One listed release as an operator sees it: consent facts, no addresses."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    bundle_id: str = Field(description="Bundle identity the release belongs to.")
    bundle_version: str = Field(description="Bundle version the release carries.")
    title: str | None = Field(default=None, description="Display title, if any.")
    publisher: str = Field(description="Publisher that signed the release.")
    sequence: int = Field(description="Publisher release sequence.")
    release_digest: str = Field(
        description="Digest of the signed claims, as inspection reviews them."
    )
    runtime_platform: str | None = Field(
        default=None,
        description="Exact artifact family of a runtime-bearing release; None for a plain one.",
    )
    artifact_sha256: str = Field(description="Digest of the executable artifact.")
    artifact_bytes: int = Field(description="Signed artifact size in bytes.")
    transfer_bytes: int = Field(
        description="Everything an install transfers: the artifact plus, for a "
        "runtime-bearing release, every wheel it lists."
    )
    platforms: tuple[str, ...] = Field(description="Operating systems supported.")
    skulk_build_sha256: str = Field(description="Skulk build the release binds to.")
    permissions: tuple[str, ...] = Field(description="Signed permission summaries.")
    descriptors: tuple[str, ...] = Field(description="Qualified capability ids.")
    surfaces: tuple[str, ...] = Field(description="Titles of declared surfaces.")
    operations: bool = Field(description="Whether durable operations are declared.")
    steward_risks: tuple[str, ...] = Field(
        description="Steward exposure risk classes: observation, billable, effect."
    )
    expires_at: int = Field(description="Release expiry as a Unix timestamp.")
    matches_host: bool = Field(
        description="Whether this host's Skulk build and platform match the release."
    )


@final
class CatalogReview(BaseModel):
    """A verified catalog as an operator sees it, without addresses or credentials."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    publisher: str = Field(description="Publisher that signed the catalog.")
    revision: int = Field(description="Catalog revision.")
    created_at: int = Field(description="Catalog creation as a Unix timestamp.")
    expires_at: int = Field(description="Catalog expiry as a Unix timestamp.")
    catalog_sha256: str = Field(
        description="Digest of the catalog document as fetched and retained."
    )
    entries: tuple[CatalogEntryReview, ...] = Field(description="Listed releases.")


@final
class VerifiedCatalog:
    """A catalog whose signature and window this host has checked."""

    def __init__(
        self,
        document: bytes,
        claims: _CatalogClaims,
        revoked_artifacts: frozenset[str] = frozenset(),
    ) -> None:
        self.document = document
        self.claims = claims
        self.sha256 = hashlib.sha256(document).hexdigest()
        # A listing whose release or artifact this host's discovery trust
        # revokes is not offered: revocation is the operator's word over the
        # publisher's, at discovery as at install.
        self.entries = tuple(
            entry
            for entry in claims.entries
            if entry.release_digest not in revoked_artifacts
            and entry.release_sha256 not in revoked_artifacts
            and entry.artifact_sha256 not in revoked_artifacts
        )

    def entry(
        self, bundle_id: str, sequence: int, runtime_platform: str | None
    ) -> _CatalogEntryClaims | None:
        """The listed, unrevoked release for one bundle sequence and artifact family.

        One sequence may be listed once per runtime platform; the family is
        part of the identity, so a plain release is found only with ``None``.
        """
        wanted = (
            canonical_platform(runtime_platform)
            if runtime_platform is not None
            else None
        )
        for entry in self.entries:
            listed = (
                canonical_platform(entry.runtime_platform)
                if entry.runtime_platform is not None
                else None
            )
            if (
                entry.bundle_id == bundle_id
                and entry.sequence == sequence
                and listed == wanted
            ):
                return entry
        return None

    def review(self, *, skulk_build_sha256: str, platform: str) -> CatalogReview:
        """Project the catalog for operators, naming what fits this host."""
        expected_os = (
            "darwin" if platform.startswith("macos") else platform.split("-")[0]
        )
        return CatalogReview(
            publisher=self.claims.publisher,
            revision=self.claims.revision,
            created_at=self.claims.created_at,
            expires_at=self.claims.expires_at,
            catalog_sha256=self.sha256,
            entries=tuple(
                CatalogEntryReview(
                    bundle_id=entry.bundle_id,
                    bundle_version=entry.bundle_version,
                    title=entry.title,
                    publisher=entry.publisher,
                    sequence=entry.sequence,
                    release_digest=entry.release_digest,
                    runtime_platform=entry.runtime_platform,
                    artifact_sha256=entry.artifact_sha256,
                    artifact_bytes=entry.artifact_size,
                    transfer_bytes=entry.transfer_bytes,
                    platforms=entry.platforms,
                    skulk_build_sha256=entry.skulk_build_sha256,
                    permissions=entry.permissions,
                    descriptors=entry.descriptors,
                    surfaces=entry.surfaces,
                    operations=entry.operations,
                    steward_risks=entry.steward_risks,
                    expires_at=entry.expires_at,
                    matches_host=entry.skulk_build_sha256 == skulk_build_sha256
                    and expected_os in entry.platforms
                    and (
                        entry.runtime_platform is None
                        or platform_matches(entry.runtime_platform, platform)
                    ),
                )
                for entry in self.entries
            ),
        )


def _claimed_publisher(catalog: dict[str, JsonValue]) -> str | None:
    publisher = catalog.get("publisher")
    return publisher if isinstance(publisher, str) else None


def verify_catalog(
    document: bytes, trust: RuntimeTrust, *, now: int
) -> VerifiedCatalog:
    """Authenticate a catalog document against discovery trust, then read its claims.

    The order matters: the signature is checked over the canonical payload
    before anything but the publisher name is interpreted, so a protocol
    refusal is only ever named for an authenticated catalog.
    """
    if len(document) > 262144:
        raise ValueError("catalog document exceeds bound")
    signed = _SignedCatalog.model_validate_json(document)
    payload = canonical_json(signed.catalog)
    publisher = _claimed_publisher(signed.catalog)
    public = trust.publishers.get(publisher) if publisher is not None else None
    if (
        publisher is None
        or public is None
        or publisher in trust.revoked_publishers
        or now >= trust.expires_at
    ):
        raise ValueError("catalog publisher trust refused")
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(public)).verify(
            bytes.fromhex(signed.signature), payload
        )
    except (InvalidSignature, ValueError):
        raise ValueError("catalog signature refused") from None
    offered = signed.catalog.get("protocol")
    if (
        isinstance(offered, int)
        and not isinstance(offered, bool)
        and offered not in ACCEPTED_CATALOG_PROTOCOLS
    ):
        raise ProtocolUnsupportedError("catalog", offered, ACCEPTED_CATALOG_PROTOCOLS)
    claims = _CatalogClaims.model_validate_json(payload)
    if not claims.created_at <= now < claims.expires_at:
        raise ValueError("catalog window refused")
    return VerifiedCatalog(document, claims, frozenset(trust.revoked_artifacts))


def _catalog_trust(value: object) -> RuntimeTrust:
    if isinstance(value, RuntimeTrust):
        return value
    return RuntimeTrust.model_validate_json(
        _OBJECT.dump_json(_OBJECT.validate_python(value, strict=True))
    )


class CatalogSource(BaseModel):
    """Host-local catalog address; never accept a URL override on a read request."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    revision: int = Field(ge=1, description="Owner-maintained catalog source revision.")
    base_url: str = Field(max_length=2048, description="Explicit HTTPS directory.")
    document_filename: str = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,200}\.json$",
        description="Signed catalog basename within the directory.",
    )
    credential_reference: ProfileIdentifier | None = Field(
        default=None,
        description="Protected local catalog credential reference, never its value.",
    )

    @model_validator(mode="after")
    def protected_origin(self) -> Self:
        """Reject ambiguous endpoints, URL credentials and query-based secret storage."""
        parsed = urlsplit(self.base_url)
        try:
            httpx.URL(self.base_url)
        except httpx.InvalidURL:
            raise ValueError("invalid catalog source address") from None
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
            raise ValueError("catalog source requires an explicit HTTPS directory")
        return self


class CatalogSourceUpdate(BaseModel):
    """Explicit owner catalog address and discovery trust; the credential is write-only."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    expected_revision: int = Field(
        ge=0, description="Current catalog source revision, zero before setup."
    )
    base_url: str | None = Field(
        default=None,
        max_length=2048,
        description="HTTPS catalog directory; omission retains the configured one.",
    )
    document_filename: str | None = Field(
        default=None,
        max_length=206,
        description="Signed catalog basename; omission retains the configured name.",
    )
    trust: Annotated[RuntimeTrust, BeforeValidator(_catalog_trust)] | None = Field(
        default=None,
        description="Publishers trusted for discovery; omission retains trust. Existing revocations cannot be removed here.",
    )
    token: SecretStr | None = Field(
        default=None, description="Write-only catalog bearer; omission retains it."
    )
    clear_token: bool = Field(
        default=False, description="Explicitly read the catalog anonymously."
    )

    @model_validator(mode="after")
    def valid_source(self) -> Self:
        """Validate the address and credential shape before touching local files."""
        CatalogSource(
            revision=self.expected_revision + 1,
            base_url=self.base_url
            if self.base_url is not None
            else "https://unused.invalid/",
            document_filename=self.document_filename
            if self.document_filename is not None
            else "catalog.json",
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
                raise ValueError("invalid catalog credential")
        return self


class CatalogSourceStatus(BaseModel):
    """Catalog source readiness without stored token values or protected paths."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    revision: int = Field(
        ge=0, description="Current catalog source revision, zero if unconfigured."
    )
    configured: bool = Field(description="Whether a catalog source is configured.")
    credential_reference: ProfileIdentifier | None = Field(
        default=None, description="Opaque catalog credential reference."
    )
    credential_ready: bool = Field(
        description="Whether the credential is readable, or the source is anonymous."
    )
    trust_revision: int | None = Field(
        default=None, description="Configured discovery trust revision."
    )


@final
class HostCatalog:
    """The one host-scoped catalog source, read on request and never automatically."""

    def __init__(
        self, root: Path, *, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        """Open host-scoped catalog state without network access."""
        self.root = root
        self.directory = root / "catalog"
        private_directory(self.directory)
        self.transport = transport
        self.guard = asyncio.Lock()
        self.closed = False

    def source(self) -> CatalogSource:
        """Read the configured catalog address; a missing credential never falls back."""
        return CatalogSource.model_validate_json(
            read_private(self.root / "catalog-source.json", 8192)
        )

    def trust(self) -> RuntimeTrust:
        """Read the publishers this host trusts for discovery, against its floor.

        The floor records the newest trust revision this host accepted; a
        restored older trust file (one that could drop a revocation) is
        refused rather than read.
        """
        raw = read_private(self.root / "catalog-trust.json")
        trust = RuntimeTrust.model_validate_json(raw)
        try:
            floor = _OBJECT.validate_json(
                read_private(self.root / "catalog-trust-floor.json", 4096)
            )
        except FileNotFoundError:
            return trust
        except ValueError:
            raise ValueError(
                "discovery trust floor unreadable; local maintenance required"
            ) from None
        revision, sha256 = floor.get("revision"), floor.get("sha256")
        if (
            not isinstance(revision, int)
            or isinstance(revision, bool)
            or revision < 1
            or not isinstance(sha256, str)
            or not re.fullmatch(r"[a-f0-9]{64}", sha256)
        ):
            raise ValueError(
                "discovery trust floor unreadable; local maintenance required"
            )
        if trust.revision < revision or (
            trust.revision == revision and hashlib.sha256(raw).hexdigest() != sha256
        ):
            raise ValueError("discovery trust rollback refused")
        return trust

    def source_status(self) -> CatalogSourceStatus:
        """Read readiness without network I/O or disclosing the credential."""
        try:
            trust_revision: int | None = self.trust().revision
        except (FileNotFoundError, ValueError):
            trust_revision = None
        try:
            source = self.source()
        except FileNotFoundError:
            return CatalogSourceStatus(
                revision=0,
                configured=False,
                credential_ready=False,
                trust_revision=trust_revision,
            )
        ready = source.credential_reference is None
        if source.credential_reference is not None:
            try:
                self._token(source.credential_reference)
                ready = True
            except (OSError, ValueError):
                ready = False
        return CatalogSourceStatus(
            revision=source.revision,
            configured=trust_revision is not None,
            credential_reference=source.credential_reference,
            credential_ready=ready,
            trust_revision=trust_revision,
        )

    async def configure(self, update: CatalogSourceUpdate) -> CatalogSourceStatus:
        """Apply owner catalog address and discovery trust changes under the host fence."""
        if self.guard.locked():
            raise ValueError("catalog source is busy")
        async with self.guard:
            if self.closed:
                raise ValueError("catalog closed")
            lock = RuntimeLock(self.root, "catalog.lock")
            try:
                try:
                    previous = self.source()
                except FileNotFoundError:
                    previous = None
                if (previous.revision if previous else 0) != update.expected_revision:
                    raise ValueError("catalog source revision conflict")
                try:
                    trust = self.trust()
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
                filename = (
                    update.document_filename
                    if update.document_filename is not None
                    else previous.document_filename
                    if previous
                    else "catalog.json"
                )
                if next_trust is None or base_url is None:
                    raise ValueError(
                        "initial catalog setup requires directory and discovery trust"
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
                    raise ValueError("discovery trust revision conflict or expiry")
                if trust is not None and next_trust.revision > trust.revision:
                    # A trust update never silently restores a publisher or
                    # artifact the owner previously revoked.
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
                        self.root / "catalog-credentials" / reference,
                        update.token.get_secret_value().encode("ascii"),
                    )
                elif (
                    previous is not None
                    and previous.base_url != base_url
                    and reference is not None
                ):
                    # A stored bearer is bound to the approved address; moving
                    # the catalog must supply its credential again.
                    raise ValueError(
                        "catalog source change requires explicit credential replacement"
                    )
                source = CatalogSource(
                    revision=update.expected_revision + 1,
                    base_url=base_url,
                    document_filename=filename,
                    credential_reference=reference,
                )
                if previous is not None and (
                    previous.base_url != source.base_url
                    or previous.document_filename != source.document_filename
                ):
                    # An owner-authorized move to another catalog starts a
                    # new revision history. The old record goes first: a
                    # crash between the writes leaves the old address without
                    # a floor (re-established on its next read), never the
                    # new address refused by the old one.
                    (self.root / "catalog-revision.json").unlink(missing_ok=True)
                trust_bytes = next_trust.model_dump_json().encode()
                write_private(self.root / "catalog-trust.json", trust_bytes)
                write_private(
                    self.root / "catalog-trust-floor.json",
                    json.dumps(
                        {
                            "revision": next_trust.revision,
                            "sha256": hashlib.sha256(trust_bytes).hexdigest(),
                        }
                    ).encode(),
                )
                write_private(
                    self.root / "catalog-source.json", source.model_dump_json().encode()
                )
                return self.source_status()
            finally:
                lock.close()

    def _token(self, reference: str) -> str:
        """The stored bearer, validated the way the fetch uses it."""
        token = (
            read_private(self.root / "catalog-credentials" / reference, 8192)
            .decode("ascii")
            .strip()
        )
        if not token or any(character.isspace() for character in token):
            raise ValueError("catalog credential unavailable")
        return token

    def _client(self, source: CatalogSource) -> httpx.AsyncClient:
        # The same policy as the release feed client: no redirects, no
        # environment proxies, identity encoding, a bearer only from the
        # protected credential file.
        headers = {"Accept": "application/json", "Accept-Encoding": "identity"}
        if source.credential_reference is not None:
            headers["Authorization"] = "Bearer " + self._token(
                source.credential_reference
            )
        return httpx.AsyncClient(
            transport=self.transport,
            timeout=10,
            follow_redirects=False,
            trust_env=False,
            headers=headers,
        )

    async def fetch(self, *, now: int | None = None) -> VerifiedCatalog:
        """Fetch and verify the catalog; retain the verified document by digest."""
        if self.guard.locked():
            raise ValueError("catalog source is busy")
        async with self.guard:
            if self.closed:
                raise ValueError("catalog closed")
            source = self.source()
            trust = self.trust()
            try:
                async with (
                    asyncio.timeout(20),
                    self._client(source) as client,
                    client.stream(
                        "GET", source.base_url + quote(source.document_filename)
                    ) as response,
                ):
                    if (
                        response.status_code != 200
                        or response.headers.get("content-encoding", "identity")
                        != "identity"
                    ):
                        raise ValueError("catalog download refused")
                    raw = bytearray()
                    async for chunk in response.aiter_bytes(chunk_size=65536):
                        if len(raw) + len(chunk) > 262144:
                            raise ValueError("catalog document exceeds bound")
                        raw.extend(chunk)
            except (httpx.HTTPError, TimeoutError):
                raise ValueError("catalog unavailable") from None
            verified = verify_catalog(
                bytes(raw), trust, now=now if now is not None else int(time.time())
            )
            if source != self.source():
                raise ValueError("catalog source changed")
            # A replayed older revision, or a different document at the
            # accepted revision, could hide newer releases or re-present
            # withdrawn ones; the accepted revision only moves forward.
            floor = self._floor(source, verified.claims.publisher)
            if floor is not None and (
                verified.claims.revision < floor[0]
                or (
                    verified.claims.revision == floor[0] and verified.sha256 != floor[1]
                )
            ):
                raise ValueError("catalog revision rollback refused")
            destination = self.directory / (verified.sha256 + ".json")
            floors = self._floors()
            if not destination.exists():
                self._prune(floors, keep=8)
            # The floor moves before the document lands: a crash in between
            # leaves a floor naming a document to fetch again, never a
            # document the floor would let an older catalog replace.
            floors[verified.claims.publisher] = {
                "revision": verified.claims.revision,
                "sha256": verified.sha256,
            }
            write_private(
                self.root / "catalog-revision.json",
                json.dumps(
                    {
                        "base_url": source.base_url,
                        "document_filename": source.document_filename,
                        "floors": floors,
                    }
                ).encode(),
            )
            write_private(destination, verified.document)
            return verified

    def _prune(self, floors: dict[str, JsonValue], *, keep: int) -> None:
        """Drop retained documents beyond the newest ``keep``, never an accepted floor.

        A catalog that updates regularly would otherwise fill its retention
        and stop; superseded documents are evidence with a short life.
        """
        accepted: set[str] = set()
        for entry in floors.values():
            sha256 = entry.get("sha256") if isinstance(entry, dict) else None
            if isinstance(sha256, str):
                accepted.add(sha256)
        retained = sorted(
            (path for path in islice(self.directory.iterdir(), 256)),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in retained[keep:]:
            if path.stem not in accepted:
                path.unlink(missing_ok=True)

    def _floors(self) -> dict[str, JsonValue]:
        """Accepted revisions per publisher at the configured catalog address.

        A missing record is first use. A damaged one fails closed: it is
        retained evidence, and reading past it could accept an older catalog.
        """
        try:
            raw = read_private(self.root / "catalog-revision.json", 16384)
        except FileNotFoundError:
            return {}
        try:
            document = _OBJECT.validate_json(raw)
            floors = document.get("floors")
            if not isinstance(floors, dict):
                raise ValueError("catalog revision record malformed")
        except ValueError:
            raise ValueError(
                "catalog revision record unreadable; local maintenance required"
            ) from None
        return floors

    def _floor(self, source: CatalogSource, publisher: str) -> tuple[int, str] | None:
        # The accepted revision belongs to one catalog address and one
        # publisher; a second trusted publisher at the same address keeps
        # its own floor, so serving one cannot erase the other's history.
        # Another address starts its own history (configure clears it).
        try:
            raw = read_private(self.root / "catalog-revision.json", 16384)
        except FileNotFoundError:
            return None
        try:
            document = _OBJECT.validate_json(raw)
        except ValueError:
            raise ValueError(
                "catalog revision record unreadable; local maintenance required"
            ) from None
        if (
            document.get("base_url") != source.base_url
            or document.get("document_filename") != source.document_filename
        ):
            # A move of the address clears this record; a record naming
            # another address is damaged evidence, not first use.
            raise ValueError(
                "catalog revision record unreadable; local maintenance required"
            )
        floors = document.get("floors")
        if not isinstance(floors, dict):
            raise ValueError(
                "catalog revision record unreadable; local maintenance required"
            )
        if publisher not in floors:
            # This publisher has no accepted revision here yet: first use.
            return None
        entry = floors[publisher]
        revision = entry.get("revision") if isinstance(entry, dict) else None
        sha256 = entry.get("sha256") if isinstance(entry, dict) else None
        if (
            isinstance(revision, int)
            and not isinstance(revision, bool)
            and revision >= 1
            and isinstance(sha256, str)
            and len(sha256) == 64
        ):
            return revision, sha256
        # A present but malformed floor is damaged evidence, not first use.
        raise ValueError(
            "catalog revision record unreadable; local maintenance required"
        )

    def retained(self, catalog_sha256: str) -> bytes:
        """The verified catalog document retained under ``catalog_sha256``."""
        if not re.fullmatch(r"[a-f0-9]{64}", catalog_sha256):
            raise ValueError("catalog digest required")
        return read_private(self.directory / (catalog_sha256 + ".json"), 262144)

    async def close(self) -> None:
        """Refuse further reads; nothing is owned in flight."""
        self.closed = True
