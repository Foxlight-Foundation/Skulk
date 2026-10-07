"""The signed capability catalog a host reads: discovery and consent, never authority.

A catalog is a flat index signed by a publisher this host trusts for
discovery. Each entry names a release, where its signed record is served and
the facts an operator needs before any transfer: what the capability can do
and what it can spend. Reading the catalog selects nothing and installs
nothing; installing still goes through the verified release path, where the
release record itself is authenticated against installation trust.

The host reads one source: a private catalog its owner configured, or the
built-in capability store when this build ships the store's trust root. The
store's discovery trust is not pasted by anyone; it is verified and renewed
through TUF (``capability_store``) and recorded like an owner's.
"""

import asyncio
import hashlib
import json
import re
import time
from collections.abc import Mapping
from itertools import islice
from pathlib import Path
from typing import Annotated, Final, Literal, Self, final
from urllib.parse import quote, urlsplit
from uuid import uuid4

import httpx
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    JsonValue,
    SecretStr,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from skulk.extensions.capability_store import (
    StoreTrustClient,
    StoreTrustUnavailableError,
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

ACCEPTED_CATALOG_PROTOCOLS: tuple[int, ...] = (1, 2)
"""Catalog record protocols this host reads: the current and the previous.

Protocol 2 listings carry the Skulk version range of the release they name,
so a host judges fit by its version. Protocol 1 listings predate that and
were published for one exact Skulk build, so they keep that rule."""

_OBJECT: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue])

CatalogRefusal = Literal[
    "catalog_unconfigured",
    "catalog_setup_incomplete",
    "catalog_source_conflict",
    "catalog_trust_update_refused",
    "catalog_credential_required",
    "catalog_credential_unavailable",
    "catalog_unreachable",
    "catalog_download_refused",
    "catalog_invalid",
    "catalog_trust_refused",
    "catalog_signature_refused",
    "catalog_window_refused",
    "catalog_rollback_refused",
    "catalog_superseded",
    "catalog_busy",
    "catalog_store_unavailable",
    "catalog_store_trust_unavailable",
    "catalog_source_reserved",
]
"""Why a catalog read, source change or install was refused, as the operator acts on it."""

_REFUSAL: TypeAdapter[CatalogRefusal] = TypeAdapter(CatalogRefusal)

CATALOG_REFUSED: Final = "catalog_refused"
"""Manager error naming a catalog refusal by its code."""


class CatalogRefusedError(ValueError):
    """A catalog refusal the operator can act on, named without disclosure.

    Carries only a code from a fixed vocabulary and, when the catalog server
    answered with something other than the catalog, that HTTP status. The
    message is for local logs and tests; no surface shows it. Every surface
    names the refusal with ``catalog_refusal_sentence`` instead, so the
    catalog address, its credential and the document never leave the manager.
    """

    def __init__(
        self, code: CatalogRefusal, message: str, *, status: int | None = None
    ) -> None:
        super().__init__(message)
        self.code: CatalogRefusal = code
        self.status = status


_REFUSAL_SENTENCES: Final[dict[CatalogRefusal, str]] = {
    "catalog_unconfigured": (
        "No plugin catalog is configured on this host. Configure the catalog "
        "source and discovery trust, then read the catalog again."
    ),
    "catalog_setup_incomplete": (
        "Setting up a catalog, or moving from the built-in store to a private "
        "catalog, needs both the catalog address and the discovery trust."
    ),
    "catalog_source_conflict": (
        "The catalog source changed since its revision was read. Read the "
        "catalog source status and apply the change at its current revision."
    ),
    "catalog_trust_update_refused": (
        "The discovery trust is expired, or older than the trust this host "
        "already holds. Supply the current discovery trust."
    ),
    "catalog_credential_required": (
        "Moving the catalog to another address needs its credential supplied again."
    ),
    "catalog_credential_unavailable": (
        "The catalog credential stored on this host cannot be read. Supply the "
        "credential again in the catalog source."
    ),
    "catalog_unreachable": (
        "This host could not reach the plugin catalog. Check that it can "
        "resolve and connect to the catalog address, then read the catalog "
        "again."
    ),
    "catalog_download_refused": (
        "The catalog server did not return the catalog as a plain download. "
        "Check the catalog address."
    ),
    "catalog_invalid": (
        "The catalog address did not return a valid signed plugin catalog. "
        "Check the catalog address; if it is right, the publisher must issue "
        "a corrected catalog."
    ),
    "catalog_trust_refused": (
        "This host's discovery trust does not accept the catalog's publisher, "
        "or the trust has expired. Update the discovery trust, then read the "
        "catalog again."
    ),
    "catalog_signature_refused": (
        "The catalog's signature does not match its publisher's trusted key, "
        "so nothing in it was used. Check the catalog address and the "
        "discovery trust."
    ),
    "catalog_window_refused": (
        "The catalog is outside its validity period: it has expired, or this "
        "host's clock is wrong. Check the host's clock; an expired catalog "
        "must be issued again by its publisher."
    ),
    "catalog_rollback_refused": (
        "The catalog server offered an older catalog than this host has "
        "already accepted, so it was refused. Read the catalog again later."
    ),
    "catalog_superseded": (
        "This host no longer holds that catalog listing, or has read a newer "
        "one since. Read the catalog again and review the release before "
        "installing."
    ),
    "catalog_busy": (
        "Another catalog read, install or source change is in progress on "
        "this host. Try again when it finishes."
    ),
    "catalog_store_unavailable": (
        "This build of Skulk does not include the built-in capability store. "
        "Configure a private catalog instead."
    ),
    "catalog_store_trust_unavailable": (
        "This host could not verify the built-in capability store's trust and "
        "holds no verified copy that is still current. Check that it can "
        "reach the store, then read the catalog again."
    ),
    "catalog_source_reserved": (
        "That address is the built-in capability store. Use the built-in "
        "store instead of configuring it as a private catalog."
    ),
}


def catalog_refusal_sentence(code: CatalogRefusal, status: int | None = None) -> str:
    """The one sentence every surface uses for a named catalog refusal.

    Args:
        code: The refusal code.
        status: The HTTP status the catalog server answered, when the refusal
            is ``catalog_download_refused`` because of it.

    Returns:
        A sentence naming what happened and what the operator can do next.
    """
    if code == "catalog_download_refused" and status is not None:
        return (
            f"The catalog server answered HTTP {status} instead of the catalog. "
            "Check the catalog address and its credential."
        )
    return _REFUSAL_SENTENCES[code]


def catalog_refusal_payload(refused: CatalogRefusedError) -> bytes:
    """Encode a catalog refusal as its fixed vocabulary for the manager socket.

    Args:
        refused: The refusal to encode.

    Returns:
        One JSON line with the error, the code and, when there is one, the
        HTTP status: nothing else about the catalog.
    """
    body: dict[str, JsonValue] = {"error": CATALOG_REFUSED, "code": refused.code}
    if refused.status is not None:
        body["status"] = refused.status
    return json.dumps(body, sort_keys=True).encode() + b"\n"


def catalog_refusal(response: Mapping[str, JsonValue]) -> CatalogRefusedError | None:
    """Rebuild a named catalog refusal from a manager reply, or nothing.

    Only the fixed vocabulary is read: a known code and an HTTP status
    integer. Anything else in the reply stays undisclosed.

    Args:
        response: A manager reply that carried no result.

    Returns:
        The refusal, or ``None`` when the reply names no catalog refusal.
    """
    if response.get("error") != CATALOG_REFUSED:
        return None
    try:
        code = _REFUSAL.validate_python(response.get("code"), strict=True)
    except ValidationError:
        return None
    status = response.get("status")
    if status is None:
        return CatalogRefusedError(code, code)
    if (
        isinstance(status, bool)
        or not isinstance(status, int)
        or not 100 <= status <= 599
    ):
        return None
    return CatalogRefusedError(code, code, status=status)


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
    skulk_requires: str | None = Field(default=None, max_length=128)
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
        if self.skulk_requires is not None:
            try:
                SpecifierSet(self.skulk_requires)
            except InvalidSpecifier:
                raise ValueError("catalog entry Skulk range is not a version range") from None
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
        # A protocol 2 listing always states its Skulk range; protocol 1 never
        # did, so a range in one would be a document no protocol describes.
        if any(
            (entry.skulk_requires is None) == (self.protocol >= 2)
            for entry in self.entries
        ):
            raise ValueError(
                "catalog protocol 2 entries state their Skulk range; protocol 1 entries do not"
            )
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
    skulk_build_sha256: str = Field(
        description="Skulk build the publisher qualified the release against; "
        "recorded for provenance, not required to match this host."
    )
    skulk_requires: str | None = Field(
        default=None,
        description="Skulk versions the release runs on (catalog protocol 2); "
        "null for a protocol 1 listing, which fits one exact Skulk build.",
    )
    permissions: tuple[str, ...] = Field(description="Signed permission summaries.")
    descriptors: tuple[str, ...] = Field(description="Qualified capability ids.")
    surfaces: tuple[str, ...] = Field(description="Titles of declared surfaces.")
    operations: bool = Field(description="Whether durable operations are declared.")
    steward_risks: tuple[str, ...] = Field(
        description="Steward exposure risk classes: observation, billable, effect."
    )
    expires_at: int = Field(description="Release expiry as a Unix timestamp.")
    matches_host: bool = Field(
        description="Whether the release fits this host: its platforms, and "
        "this host's Skulk version inside its range (for a protocol 1 listing, "
        "this host's exact Skulk build)."
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

    def review(
        self, *, skulk_version: str, skulk_build_sha256: str, platform: str
    ) -> CatalogReview:
        """Project the catalog for operators, naming what fits this host."""
        return CatalogReview(
            publisher=self.claims.publisher,
            revision=self.claims.revision,
            created_at=self.claims.created_at,
            expires_at=self.claims.expires_at,
            catalog_sha256=self.sha256,
            entries=tuple(
                self.entry_review(
                    entry,
                    skulk_version=skulk_version,
                    skulk_build_sha256=skulk_build_sha256,
                    platform=platform,
                )
                for entry in self.entries
            ),
        )

    @staticmethod
    def entry_review(
        entry: _CatalogEntryClaims,
        *,
        skulk_version: str,
        skulk_build_sha256: str,
        platform: str,
    ) -> CatalogEntryReview:
        """One listing as consent facts, with whether it fits this host.

        A listing states the Skulk versions its release runs on, so it fits
        any host whose version is in that range whatever build it runs, and a
        Skulk update never hides what the host can still install. A protocol 1
        listing has no range: it was published for one exact Skulk build and
        keeps that rule.
        """
        expected_os = (
            "darwin" if platform.startswith("macos") else platform.split("-")[0]
        )
        return CatalogEntryReview(
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
            skulk_requires=entry.skulk_requires,
            matches_host=expected_os in entry.platforms
            and (
                skulk_version in SpecifierSet(entry.skulk_requires)
                if entry.skulk_requires is not None
                else entry.skulk_build_sha256 == skulk_build_sha256
            )
            and (
                entry.runtime_platform is None
                or platform_matches(entry.runtime_platform, platform)
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
        raise CatalogRefusedError("catalog_invalid", "catalog document exceeds bound")
    try:
        signed = _SignedCatalog.model_validate_json(document)
    except ValidationError as malformed:
        raise CatalogRefusedError("catalog_invalid", str(malformed)) from None
    payload = canonical_json(signed.catalog)
    publisher = _claimed_publisher(signed.catalog)
    public = trust.publishers.get(publisher) if publisher is not None else None
    if (
        publisher is None
        or public is None
        or publisher in trust.revoked_publishers
        or now >= trust.expires_at
    ):
        raise CatalogRefusedError(
            "catalog_trust_refused", "catalog publisher trust refused"
        )
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(public)).verify(
            bytes.fromhex(signed.signature), payload
        )
    except (InvalidSignature, ValueError):
        raise CatalogRefusedError(
            "catalog_signature_refused", "catalog signature refused"
        ) from None
    offered = signed.catalog.get("protocol")
    if (
        isinstance(offered, int)
        and not isinstance(offered, bool)
        and offered not in ACCEPTED_CATALOG_PROTOCOLS
    ):
        raise ProtocolUnsupportedError("catalog", offered, ACCEPTED_CATALOG_PROTOCOLS)
    try:
        claims = _CatalogClaims.model_validate_json(payload)
    except ValidationError as invalid:
        # Signed by a trusted publisher, yet not a catalog this host reads:
        # the publisher's defect, named so the operator knows it is not theirs.
        raise CatalogRefusedError("catalog_invalid", str(invalid)) from None
    if not claims.created_at <= now < claims.expires_at:
        raise CatalogRefusedError("catalog_window_refused", "catalog window refused")
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
    configured: bool = Field(
        description="Whether a catalog source is configured. The built-in "
        "capability store counts as configured from the start, before its "
        "first read has verified its trust."
    )
    credential_reference: ProfileIdentifier | None = Field(
        default=None, description="Opaque catalog credential reference."
    )
    credential_ready: bool = Field(
        description="Whether the credential is readable, or the source is anonymous."
    )
    trust_revision: int | None = Field(
        default=None,
        description="Configured discovery trust revision; null for the built-in "
        "store until its first read.",
    )
    builtin_store: bool = Field(
        default=False,
        description="Whether the source is the built-in capability store, whose "
        "discovery trust Skulk verifies and renews itself.",
    )
    builtin_store_available: bool = Field(
        default=False,
        description="Whether this Skulk build includes the built-in capability "
        "store, so the host can switch to it.",
    )


class BuiltinCatalogSelection(BaseModel):
    """Owner request to read capabilities from the built-in store again."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    expected_revision: int = Field(
        ge=0, description="Current catalog source revision, zero before setup."
    )


class AcceptedFloor(_Contract):
    """The newest catalog revision this host accepted, and its document digest."""

    revision: int = Field(ge=1)
    sha256: Digest


class _CatalogState(_Contract):
    """Everything host-scoped about the catalog, replaced as one document.

    ``trust_floor`` is the current source's trust floor, written with its
    trust. ``floors`` holds every address's history: catalog revision floors
    keyed by address, document and publisher, and discovery trust floors
    keyed by ``_trust_floor_key`` per address.
    """

    source: CatalogSource | None = None
    trust: RuntimeTrust | None = None
    trust_floor: AcceptedFloor | None = None
    floors: dict[str, AcceptedFloor] = Field(default_factory=dict)


def _trust_digest(trust: RuntimeTrust) -> str:
    return hashlib.sha256(canonical_json(trust.model_dump(mode="json"))).hexdigest()


def _carry_revocations(trust: RuntimeTrust, held: RuntimeTrust) -> RuntimeTrust:
    """``trust`` with every revocation ``held`` carries added to its own.

    Only within a private catalog's own history: an owner's trust update
    never silently restores a publisher or artifact the owner revoked before
    it. The built-in store's trust is never merged: it applies as the store
    published it, and a source switch carries no revocation in either
    direction, since a publisher name means nothing across sources.
    """
    return RuntimeTrust.model_validate(
        {
            **trust.model_dump(),
            "revoked_publishers": tuple(
                sorted(set(held.revoked_publishers) | set(trust.revoked_publishers))
            ),
            "revoked_artifacts": tuple(
                sorted(set(held.revoked_artifacts) | set(trust.revoked_artifacts))
            ),
        }
    )


def bounded_floors(
    floors: dict[str, "AcceptedFloor"], key: str, prefix: str, *, bound: int = 32
) -> dict[str, "AcceptedFloor"]:
    """Refuse a new history beyond ``bound``; no accepted floor is ever dropped.

    A floor is rollback evidence, so the map keeps every one it has; a new
    address-and-publisher history past the bound is refused by name, which is
    the explicit maintenance path, while existing histories keep recording.
    """
    del prefix  # every history is kept; the address no longer orders eviction
    if len(floors) > bound:
        raise ValueError("catalog history requires local maintenance")
    return dict(floors)


def _floor_key(source: CatalogSource, publisher: str) -> str:
    return "\n".join((source.base_url, source.document_filename, publisher))


_TRUST_FLOOR_PREFIX: Final = "trust\n"
"""Key prefix of discovery trust floors inside the ``floors`` map.

Catalog revision floors are keyed by address, document and publisher, and an
address is an HTTPS URL, so no catalog floor key can start with this. Keeping
trust floors in the same map leaves the state document's shape unchanged:
builds that predate per-source trust floors still read it, and simply never
look these entries up."""


def _trust_floor_key(base_url: str, document_filename: str) -> str:
    """The floor key of one catalog address's discovery trust history."""
    return _TRUST_FLOOR_PREFIX + "\n".join((base_url, document_filename))


def _source_trust_floor(
    state: "_CatalogState", base_url: str, document_filename: str
) -> AcceptedFloor | None:
    """The newest discovery trust this host accepted for one catalog address.

    Each address keeps its own trust history, as each keeps its own catalog
    revision history. A state written before per-source trust floors held
    only the current source's floor, in ``trust_floor``; it counts for the
    current address.
    """
    floor = state.floors.get(_trust_floor_key(base_url, document_filename))
    current = state.source
    if (
        current is not None
        and (current.base_url, current.document_filename)
        == (base_url, document_filename)
        and state.trust_floor is not None
        and (floor is None or state.trust_floor.revision > floor.revision)
    ):
        return state.trust_floor
    return floor


def _below_floor(trust: RuntimeTrust, floor: AcceptedFloor | None) -> bool:
    """Whether ``trust`` is older than ``floor``, or another document at its revision."""
    return floor is not None and (
        trust.revision < floor.revision
        or (trust.revision == floor.revision and _trust_digest(trust) != floor.sha256)
    )


def _with_trust(
    state: "_CatalogState", source: CatalogSource, trust: RuntimeTrust
) -> "_CatalogState":
    """``state`` reading ``source`` under ``trust``, with each address's trust floor kept.

    ``trust_floor`` moves with the current trust, as it always has. The
    per-address floors only rise: the address being left keeps its floor
    (recorded here when a state from before per-source floors held it only in
    ``trust_floor``), so returning there meets its own history again, and a
    switch never overwrites another address's floor.
    """
    accepted = AcceptedFloor(revision=trust.revision, sha256=_trust_digest(trust))
    floors = dict(state.floors)
    if state.source is not None and state.trust_floor is not None:
        left = _trust_floor_key(state.source.base_url, state.source.document_filename)
        kept = floors.get(left)
        if kept is None or state.trust_floor.revision > kept.revision:
            floors[left] = state.trust_floor
    key = _trust_floor_key(source.base_url, source.document_filename)
    held = floors.get(key)
    if held is None or accepted.revision >= held.revision:
        floors[key] = accepted
    return state.model_copy(
        update={
            "source": source,
            "trust": trust,
            "trust_floor": accepted,
            "floors": bounded_floors(floors, key, ""),
        }
    )


_STORE_TRUST_SECONDS: Final = 8
"""Budget for verifying the built-in store's trust before a catalog read.

It sits inside the same request deadline as the catalog download, so the two
together stay under the manager's and the API's thirty seconds."""


@final
class HostCatalog:
    """The one host-scoped catalog source, read on request and never automatically.

    The source is either a private catalog the owner configured, with the
    discovery trust the owner supplied, or the built-in capability store when
    this build ships its trust root. A host with neither a source nor trust
    uses the built-in store: its first read verifies the store's publisher
    trust through TUF and records it in the same state document as an owner's
    configuration, so every floor and rollback check applies unchanged. While
    the source is the built-in store, each read renews that trust first and
    applies a newer revision, never an older one, exactly as published.
    """

    def __init__(
        self,
        root: Path,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        store: StoreTrustClient | None = None,
    ) -> None:
        """Open host-scoped catalog state without network access.

        Args:
            root: The manager root that holds catalog state.
            transport: HTTP transport for catalog reads, injectable for tests.
            store: The built-in capability store's trust client, or ``None``
                for a host without a built-in store.
        """
        self.root = root
        self.directory = root / "catalog"
        private_directory(self.directory)
        self.transport = transport
        self.store = store
        self.guard = asyncio.Lock()
        self.closed = False

    def _store_ready(self) -> bool:
        """Whether this build ships the built-in store's root."""
        return self.store is not None and self.store.available

    def _is_store(self, source: CatalogSource | None) -> bool:
        """Whether ``source`` is the built-in store's catalog address.

        The store is identified by its address, which ``configure`` reserves
        while the store is available, so no private catalog can share it. A
        build without the store's root treats that address like any other.
        """
        return (
            self.store is not None
            and self.store.available
            and source is not None
            and source.base_url == self.store.catalog_url
            and source.document_filename == self.store.catalog_document
        )

    def _uses_store(self, state: "_CatalogState") -> bool:
        """Whether reads come from the built-in store, seeded or not yet."""
        if state.source is None:
            return state.trust is None and self._store_ready()
        return self._is_store(state.source)

    async def _store_trust(
        self, *, offline: bool, now: int | None = None
    ) -> RuntimeTrust | None:
        """The store's verified trust in force, or ``None`` when there is none.

        Verification runs in a thread (python-tuf is synchronous) under its
        own budget. A refresh that outlives the budget keeps running and
        retains what it verifies; this call falls back to the store's last
        verified copy, read without the network or the refresh fence.
        """
        store = self.store
        if store is None:
            return None
        try:
            async with asyncio.timeout(_STORE_TRUST_SECONDS):
                return await asyncio.to_thread(store.load, offline=offline, now=now)
        except StoreTrustUnavailableError:
            return None
        except TimeoutError:
            try:
                return store.last_known_good(now=now)
            except StoreTrustUnavailableError:
                return None

    def on_builtin_store(self) -> bool:
        """Whether the host's catalog source is the built-in store, seeded or not yet."""
        return self._uses_store(self._state())

    async def builtin_store_trust(self, *, offline: bool) -> RuntimeTrust | None:
        """The built-in store's verified trust in force, whatever the current source.

        Installations bound from the store follow this trust even after the
        host's catalog moves to a private one: their publishers are the
        store's. Network access and expiry follow ``StoreTrustClient.load``.

        Args:
            offline: Use only the store's last verified trust.

        Returns:
            The trust, or ``None`` when this build has no store or no
            verified trust is in force.
        """
        if not self._store_ready():
            return None
        return await self._store_trust(offline=offline)

    async def apply_store_trust(self, trust: RuntimeTrust) -> bool:
        """Record verified store trust as the built-in source's discovery trust.

        Keeps the catalog in step with each verified renewal made outside a
        catalog read, so an install from a retained listing is checked
        against the trust the store's followers hold. Applies only while the
        source is the built-in store and only a strictly newer revision, as a
        read would. A catalog operation in progress already renews the trust
        itself, so this then changes nothing.

        Args:
            trust: The store's verified trust, still in force.

        Returns:
            Whether the catalog's discovery trust changed.
        """
        if self.closed or self.guard.locked():
            return False
        async with self.guard:
            state = self._state()
            if (
                state.source is None
                or not self._is_store(state.source)
                or state.trust is None
                or trust.revision <= state.trust.revision
                or time.time() >= trust.expires_at
            ):
                return False
            # The same helper as a read: the store address's own trust floor
            # rises with the trust, so a later switch away and back meets it.
            self._save(_with_trust(state, state.source, trust))
            return True

    def _store_source(self, revision: int) -> CatalogSource:
        """The built-in store's catalog address at ``revision``, read anonymously."""
        assert self.store is not None
        return CatalogSource(
            revision=revision,
            base_url=self.store.catalog_url,
            document_filename=self.store.catalog_document,
        )

    async def _renewed_store_state(
        self, state: "_CatalogState", *, offline: bool, now: int
    ) -> "_CatalogState":
        """Seed or renew the built-in store's trust in the state document.

        First use records the store's address and verified trust together.
        Later reads apply a renewed trust only when its revision is strictly
        newer, whether or not the held trust has expired: an older or
        altered trust never replaces the one the host accepted, even when
        the store client's own retained copy was lost. The
        renewed trust applies exactly as the store published it, its
        revocations included: TUF and the store client's floor already keep
        that document from moving backward, so nothing is merged in. With no
        verified trust available, a current held trust keeps serving; an
        expired or absent one is refused by name.
        """
        renewed = await self._store_trust(offline=offline, now=now)
        held = state.trust
        if state.source is None:
            if renewed is None:
                raise CatalogRefusedError(
                    "catalog_store_trust_unavailable", "store trust unavailable"
                )
            source, trust = self._store_source(1), renewed
            if _below_floor(
                trust,
                _source_trust_floor(state, source.base_url, source.document_filename),
            ):
                raise CatalogRefusedError(
                    "catalog_store_trust_unavailable", "store trust below its floor"
                )
        elif renewed is not None and (held is None or renewed.revision > held.revision):
            source, trust = state.source, renewed
        elif held is None or now >= held.expires_at:
            raise CatalogRefusedError(
                "catalog_store_trust_unavailable", "store trust expired"
            )
        else:
            return state
        updated = _with_trust(state, source, trust)
        self._save(updated)
        return updated

    def _state(self) -> "_CatalogState":
        """Read the one host-scoped catalog state document.

        Missing means first use. Anything unreadable fails closed: the state
        carries the floors that keep a replayed catalog or trust out.
        """
        try:
            raw = read_private(self.root / "catalog-state.json", 262144)
        except FileNotFoundError:
            # First use has no retained document either; a state document
            # lost beside remaining evidence would reset every floor.
            if any(True for _ in islice(self.directory.iterdir(), 1)):
                raise ValueError(
                    "catalog state missing while retained documents remain; "
                    "local maintenance required"
                ) from None
            return _CatalogState()
        try:
            state = _CatalogState.model_validate_json(raw)
        except ValueError:
            raise ValueError(
                "catalog state unreadable; local maintenance required"
            ) from None
        # Trust and its floor are always written together; a trust below its
        # floor, another trust at the floor's revision, or trust without a
        # floor is a rolled-back or edited document, never read past.
        if state.trust is not None and (
            state.trust_floor is None
            or state.trust.revision < state.trust_floor.revision
            or (
                state.trust.revision == state.trust_floor.revision
                and _trust_digest(state.trust) != state.trust_floor.sha256
            )
        ):
            raise ValueError("discovery trust rollback refused")
        return state

    def _save(self, state: "_CatalogState") -> None:
        # One document, replaced atomically: no ordering between the source,
        # the trust, its floor and the revision floors can be observed.
        write_private(
            self.root / "catalog-state.json", state.model_dump_json().encode()
        )

    def source(self) -> CatalogSource:
        """Read the configured catalog address; a missing credential never falls back."""
        source = self._state().source
        if source is None:
            raise FileNotFoundError("catalog source unconfigured")
        return source

    def trust(self) -> RuntimeTrust:
        """Read the publishers this host trusts for discovery, against its floor."""
        state = self._state()
        if state.trust is None:
            raise FileNotFoundError("discovery trust unconfigured")
        return state.trust

    def source_status(self) -> CatalogSourceStatus:
        """Read readiness without network I/O or disclosing the credential.

        A host that uses the built-in store reports it as configured even
        before the first read verifies the store's trust: there is nothing
        for the owner to supply, and that read seeds it.
        """
        state = self._state()
        available = self._store_ready()
        trust_revision = state.trust.revision if state.trust is not None else None
        if state.source is None:
            unseeded = self._uses_store(state)
            return CatalogSourceStatus(
                revision=0,
                configured=unseeded,
                credential_ready=unseeded,
                trust_revision=trust_revision,
                builtin_store=unseeded,
                builtin_store_available=available,
            )
        ready = state.source.credential_reference is None
        if state.source.credential_reference is not None:
            try:
                self._token(state.source.credential_reference)
                ready = True
            except (OSError, ValueError):
                ready = False
        return CatalogSourceStatus(
            revision=state.source.revision,
            configured=trust_revision is not None,
            credential_reference=state.source.credential_reference,
            credential_ready=ready,
            trust_revision=trust_revision,
            builtin_store=self._is_store(state.source),
            builtin_store_available=available,
        )

    async def configure(self, update: CatalogSourceUpdate) -> CatalogSourceStatus:
        """Apply owner catalog address and discovery trust changes under the host fence."""
        if self.guard.locked():
            raise CatalogRefusedError("catalog_busy", "catalog source is busy")
        async with self.guard:
            if self.closed:
                raise ValueError("catalog closed")
            lock = RuntimeLock(self.root, "catalog.lock")
            try:
                state = self._state()
                previous = state.source
                if (previous.revision if previous else 0) != update.expected_revision:
                    raise CatalogRefusedError(
                        "catalog_source_conflict", "catalog source revision conflict"
                    )
                trust = state.trust
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
                    raise CatalogRefusedError(
                        "catalog_setup_incomplete",
                        "initial catalog setup requires directory and discovery trust",
                    )
                if self._is_store(previous) and (
                    update.base_url is None or update.trust is None
                ):
                    # The store's address and its renewed trust are the
                    # store's own: a private catalog is a first setup, with
                    # its own address and the publishers its owner trusts.
                    raise CatalogRefusedError(
                        "catalog_setup_incomplete",
                        "a private catalog requires its directory and discovery trust",
                    )
                if (
                    self.store is not None
                    and self._store_ready()
                    and base_url == self.store.catalog_url
                ):
                    raise CatalogRefusedError(
                        "catalog_source_reserved",
                        "the built-in store's address is not a private catalog",
                    )
                # Each catalog address keeps its own trust history. Leaving the
                # store, a private trust is compared only with that address's
                # own history, never with the store's revisions; returning to
                # an address meets the floor it left behind.
                leaving_store = self._is_store(previous)
                if next_trust.expires_at <= time.time() or (
                    trust is not None
                    and not leaving_store
                    and (
                        next_trust.revision < trust.revision
                        or (
                            next_trust.revision == trust.revision
                            and next_trust != trust
                        )
                    )
                ):
                    raise CatalogRefusedError(
                        "catalog_trust_update_refused",
                        "discovery trust revision conflict or expiry",
                    )
                if _below_floor(
                    next_trust, _source_trust_floor(state, base_url, filename)
                ) or (
                    not leaving_store and _below_floor(next_trust, state.trust_floor)
                ):
                    raise CatalogRefusedError(
                        "catalog_trust_update_refused",
                        "discovery trust rollback refused",
                    )
                if (
                    trust is not None
                    and not self._is_store(previous)
                    and next_trust.revision > trust.revision
                ):
                    # A trust update never silently restores a publisher or
                    # artifact the owner previously revoked. The built-in
                    # store's revocations are the store's own and do not
                    # follow the host to a private catalog.
                    next_trust = _carry_revocations(next_trust, trust)
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
                    raise CatalogRefusedError(
                        "catalog_credential_required",
                        "catalog source change requires explicit credential replacement",
                    )
                source = CatalogSource(
                    revision=update.expected_revision + 1,
                    base_url=base_url,
                    document_filename=filename,
                    credential_reference=reference,
                )
                # Revision and trust floors are keyed by address and kept
                # across moves: another address starts its own history, and
                # returning to an old one meets its old floors again.
                self._save(_with_trust(state, source, next_trust))
                return self.source_status()
            finally:
                lock.close()

    async def use_builtin_store(
        self, expected_revision: int, *, offline: bool = False
    ) -> CatalogSourceStatus:
        """Make the built-in capability store the source again, under the host fence.

        The private catalog's address is replaced by the store's, read
        anonymously, and the discovery trust by the store's verified trust as
        the store published it: the private trust's revocations stay with
        the private catalog. The private trust's revision history does not
        bind the store's either: the store's own history is its retained TUF
        trust, which never moves backward. Revision and trust floors stay
        keyed by address: the store meets its own earlier floors again, and
        the private catalog's trust floor is kept for a later return there.
        Nothing is fetched from the catalog.

        Args:
            expected_revision: The source revision the owner reviewed.
            offline: Use only the store's retained trust, without network.

        Returns:
            Source readiness after the change.

        Raises:
            CatalogRefusedError: The build has no store, the revision is
                stale, another catalog operation is running, or no verified
                store trust is in force.
        """
        if self.guard.locked():
            raise CatalogRefusedError("catalog_busy", "catalog source is busy")
        async with self.guard:
            if self.closed:
                raise ValueError("catalog closed")
            if not self._store_ready():
                raise CatalogRefusedError(
                    "catalog_store_unavailable", "this build has no built-in store"
                )
            lock = RuntimeLock(self.root, "catalog.lock")
            try:
                state = self._state()
                previous = state.source
                if (previous.revision if previous else 0) != expected_revision:
                    raise CatalogRefusedError(
                        "catalog_source_conflict", "catalog source revision conflict"
                    )
                if self._uses_store(state):
                    # Already the store, seeded or not: nothing to change.
                    return self.source_status()
                renewed = await self._store_trust(offline=offline)
                if renewed is None:
                    raise CatalogRefusedError(
                        "catalog_store_trust_unavailable", "store trust unavailable"
                    )
                source = self._store_source(expected_revision + 1)
                # The store's own floor, not the private catalog's, binds it;
                # the private catalog's floor stays for a later return there.
                if _below_floor(
                    renewed,
                    _source_trust_floor(
                        state, source.base_url, source.document_filename
                    ),
                ):
                    raise CatalogRefusedError(
                        "catalog_store_trust_unavailable",
                        "store trust older than this host accepted",
                    )
                self._save(_with_trust(state, source, renewed))
                return self.source_status()
            finally:
                lock.close()

    def _token(self, reference: str) -> str:
        """The stored bearer, validated the way the fetch uses it."""
        try:
            token = (
                read_private(self.root / "catalog-credentials" / reference, 8192)
                .decode("ascii")
                .strip()
            )
        except (OSError, ValueError):
            token = ""
        if not token or any(character.isspace() for character in token):
            raise CatalogRefusedError(
                "catalog_credential_unavailable", "catalog credential unavailable"
            )
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

    async def fetch(
        self, *, now: int | None = None, offline: bool = False
    ) -> VerifiedCatalog:
        """Fetch and verify the catalog; retain the verified document by digest.

        When the source is the built-in store, its trust is seeded or renewed
        first (see ``_renewed_store_state``); ``offline`` keeps that step off
        the network. The catalog read itself is the operator's request and
        always reaches the catalog address.
        """
        if self.guard.locked():
            raise CatalogRefusedError("catalog_busy", "catalog source is busy")
        async with self.guard:
            if self.closed:
                raise ValueError("catalog closed")
            state = self._state()
            current_time = now if now is not None else int(time.time())
            if self._uses_store(state):
                state = await self._renewed_store_state(
                    state, offline=offline, now=current_time
                )
            if state.source is None or state.trust is None:
                raise CatalogRefusedError(
                    "catalog_unconfigured", "catalog source unconfigured"
                )
            source, trust = state.source, state.trust
            try:
                async with (
                    asyncio.timeout(20),
                    self._client(source) as client,
                    client.stream(
                        "GET", source.base_url + quote(source.document_filename)
                    ) as response,
                ):
                    if response.status_code != 200:
                        raise CatalogRefusedError(
                            "catalog_download_refused",
                            "catalog download refused",
                            status=response.status_code,
                        )
                    if response.headers.get("content-encoding", "identity") != (
                        "identity"
                    ):
                        raise CatalogRefusedError(
                            "catalog_download_refused", "catalog download refused"
                        )
                    raw = bytearray()
                    async for chunk in response.aiter_bytes(chunk_size=65536):
                        if len(raw) + len(chunk) > 262144:
                            raise CatalogRefusedError(
                                "catalog_invalid", "catalog document exceeds bound"
                            )
                        raw.extend(chunk)
            except (httpx.HTTPError, TimeoutError):
                raise CatalogRefusedError(
                    "catalog_unreachable", "catalog unavailable"
                ) from None
            verified = verify_catalog(bytes(raw), trust, now=current_time)
            current = self._state()
            if current.source != source:
                # A source change landed during the read: the same remedy as a
                # concurrent operation, read again once it is done.
                raise CatalogRefusedError("catalog_busy", "catalog source changed")
            # A replayed older revision, or a different document at the
            # accepted revision, could hide newer releases or re-present
            # withdrawn ones; the accepted revision only moves forward, per
            # catalog address and publisher.
            key = _floor_key(source, verified.claims.publisher)
            floor = current.floors.get(key)
            if floor is not None and (
                verified.claims.revision < floor.revision
                or (
                    verified.claims.revision == floor.revision
                    and verified.sha256 != floor.sha256
                )
            ):
                raise CatalogRefusedError(
                    "catalog_rollback_refused", "catalog revision rollback refused"
                )
            destination = self.directory / (verified.sha256 + ".json")
            floors = dict(current.floors)
            floors.pop(key, None)
            floors[key] = AcceptedFloor(
                revision=verified.claims.revision, sha256=verified.sha256
            )
            floors = bounded_floors(floors, key, _floor_key(source, ""))
            if not destination.exists():
                self._prune(floors, keep=8)
            # The floor moves before the document lands: a crash in between
            # leaves a floor naming a document to fetch again, never a
            # document the floor would let an older catalog replace.
            self._save(current.model_copy(update={"floors": floors}))
            write_private(destination, verified.document)
            return verified

    def _prune(self, floors: dict[str, "AcceptedFloor"], *, keep: int) -> None:
        """Drop retained documents beyond the newest ``keep``, never an accepted floor.

        A catalog that updates regularly would otherwise fill its retention
        and stop; superseded documents are evidence with a short life.
        """
        # Trust floors share the map but name trust documents, not catalogs.
        accepted = {
            floor.sha256
            for key, floor in floors.items()
            if not key.startswith(_TRUST_FLOOR_PREFIX)
        }
        retained = sorted(
            (path for path in islice(self.directory.iterdir(), 256)),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in retained[keep:]:
            if path.stem not in accepted:
                path.unlink(missing_ok=True)

    def retained(self, catalog_sha256: str) -> bytes:
        """The verified catalog document retained under ``catalog_sha256``."""
        if not re.fullmatch(r"[a-f0-9]{64}", catalog_sha256):
            raise ValueError("catalog digest required")
        document = read_private(self.directory / (catalog_sha256 + ".json"), 262144)
        if hashlib.sha256(document).hexdigest() != catalog_sha256:
            raise ValueError("retained catalog document differs from its digest")
        return document

    def accepted(self, catalog_sha256: str, *, now: int) -> VerifiedCatalog:
        """The retained listing under ``catalog_sha256``, verified again and still current.

        An install names the digest the operator reviewed. The document is
        verified again against the present discovery trust and clock, so a
        revocation or expiry since the read applies at consent as it did at
        discovery, and it must be the newest listing this host accepted from
        its configured source for that publisher: a listing a later read
        superseded is refused by name rather than installed from a stale page.
        """
        state = self._state()
        if state.source is None or state.trust is None:
            raise CatalogRefusedError(
                "catalog_unconfigured", "catalog source unconfigured"
            )
        try:
            document = self.retained(catalog_sha256)
        except FileNotFoundError:
            # Pruned after newer reads, or never read on this host: either
            # way the remedy is to read the catalog again.
            raise CatalogRefusedError(
                "catalog_superseded", "catalog listing is not retained"
            ) from None
        verified = verify_catalog(document, state.trust, now=now)
        floor = state.floors.get(_floor_key(state.source, verified.claims.publisher))
        if floor is None or floor.sha256 != catalog_sha256:
            raise CatalogRefusedError(
                "catalog_superseded",
                "catalog listing superseded; read the catalog again",
            )
        return verified

    def credential_for(self, feed_url: str) -> str | None:
        """The catalog bearer for a feed at the catalog's own origin, else ``None``.

        A credential was given for the catalog's origin and is presented only
        there: a listed feed at another origin is read anonymously until its
        installation is given a credential of its own.
        """
        source = self.source()
        if source.credential_reference is None:
            return None
        catalog, feed = urlsplit(source.base_url), urlsplit(feed_url)
        if (catalog.scheme, catalog.hostname, catalog.port) != (
            feed.scheme,
            feed.hostname,
            feed.port,
        ):
            return None
        return self._token(source.credential_reference)

    async def close(self) -> None:
        """Refuse further reads; nothing is owned in flight."""
        self.closed = True
