"""The signed list of engine packages a node may install, and what fits this node.

A node can only offer to install an engine it knows exists. The store
publishes one small document, the **package index**, that lists every
published package build: which engine, which platform and accelerator, which
channel, and the identity of its signed manifest. An engine publisher signs
it with the same key that signs manifests, under its own domain prefix, so a
manifest signature can never be replayed as an index and the reverse.

The index decides only what a node *offers*. It never causes an installation
by itself, and it is not trusted for the installation: when a package is
installed its own signed manifest is fetched and verified, and its identity
must be the one the index named. A wrong index can therefore make a node
offer something it then refuses to install; it cannot make a node run
anything a manifest signature does not cover.

A node keeps the last index it verified. That copy is also the revision
floor: an older index, or a different document at the same revision, is
refused and the kept copy stays in force. Reading what a node offers uses
only the kept copy, so it needs no network and adds no start-up wait.
"""

import contextlib
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Self, final

import httpx
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from filelock import FileLock
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    model_validator,
)

from skulk.extensions.runtime_artifacts import (
    Digest,
    Identifier,
    RuntimePlatform,
    RuntimeTrust,
    canonical_json,
)
from skulk.extensions.runtime_files import (
    private_directory,
    read_private,
    write_private,
)
from skulk.provisioning.engine_packages import (
    ComputeTarget,
    EngineHost,
    EngineHostRequirements,
    EngineId,
    EngineLane,
    EngineProtocolRange,
    Signature,
    VersionRange,
    engine_host_refusal,
)
from skulk.shared.backends import package_engine_of
from skulk.shared.constants import SKULK_DATA_HOME

ENGINE_INDEX_SIGNATURE_DOMAIN: Final = b"skulk-engine-index-v1\n"
"""Prefix of the signed bytes. It separates the index from a package manifest
and from every other document signed with the same kind of key."""

ENGINE_STORE_URL: Final = "https://skulkapps.foxlight.ai/engines/"
"""Directory of the built-in store that serves engine packages. Fixed in host
code: no index, manifest, request or setting can point a node elsewhere."""

ENGINE_INDEX_DOCUMENT: Final = "index-v1.json"
"""Basename of the signed package index under ``ENGINE_STORE_URL``."""

ENGINE_INDEX_DIR: Final = SKULK_DATA_HOME / "engine-index"
"""Where this node keeps the index it last verified."""

MAX_ENGINE_INDEX_BYTES: Final = 512 * 1024
"""Largest index document accepted before any parsing."""

MAX_ENGINE_INDEX_ENTRIES: Final = 256
"""Most package builds one index may list."""

_FETCH_SECONDS: Final = 8.0
_LOCK_SECONDS: Final = 20.0
_RETAINED_NAME: Final = "last-known-good-index.json"

EngineChannel = Literal["stable", "candidate"]
"""How far a package build has been promoted. A node installs from ``stable``
unless it was told to follow ``candidate`` as well."""

EngineIndexRefusal = Literal[
    "index_invalid",
    "trust_refused",
    "signature_refused",
    "window_refused",
    "rollback_refused",
    "index_unavailable",
]
"""Why an index was not used. Each code names one check."""


@final
class EngineIndexRefusedError(Exception):
    """An index was refused, or none in force is available.

    Attributes:
        code: The check that failed.
    """

    def __init__(self, code: EngineIndexRefusal, message: str) -> None:
        """Record the refusal code beside an operator-readable sentence."""
        super().__init__(message)
        self.code: Final = code


def _refuse(code: EngineIndexRefusal, message: str) -> EngineIndexRefusedError:
    return EngineIndexRefusedError(code, message)


def _relative_directory(value: str) -> str:
    """Require a plain relative directory path, with nothing that climbs."""
    parts = value.split("/")
    if (
        not value
        or value.startswith("/")
        or any(part in ("", ".", "..") for part in parts)
        or "\\" in value
    ):
        raise ValueError("directory must be a plain relative path")
    return value


class _Contract(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


@final
class EngineIndexEntry(_Contract):
    """One published package build, as the index lists it.

    The requirements repeat what the build's manifest signs, so a node can
    tell whether a build fits it without fetching every manifest. The
    manifest stays the authority at installation.
    """

    engine_id: EngineId = Field(description="publisher.engine identity.")
    engine_version: str = Field(
        min_length=1, max_length=64, description="Version of the engine inside."
    )
    generation: int = Field(ge=1, description="Package generation of this build.")
    channel: EngineChannel = Field(description="How far the build is promoted.")
    platform: RuntimePlatform = Field(description="Artifact family of the build.")
    lane: EngineLane = Field(description="Accelerator class the build executes on.")
    compute_targets: tuple[ComputeTarget, ...] = Field(
        default=(),
        max_length=16,
        description="Exact device targets; empty means any device of the lane.",
    )
    skulk_versions: VersionRange = Field(description="Skulk versions it runs on.")
    control_protocol: EngineProtocolRange = Field(
        description="Control-protocol versions the adapter speaks."
    )
    adapter_python_requires: VersionRange | None = Field(
        default=None, description="Interpreter range of a Python adapter."
    )
    manifest_digest: Digest = Field(
        description=(
            "Build identity of the package: SHA-256 of its manifest's canonical "
            "claims. An installation refuses a manifest with any other identity."
        )
    )
    directory: str = Field(
        min_length=1,
        max_length=200,
        pattern=r"^[A-Za-z0-9._/-]+$",
        description=(
            "Directory under the engine store that holds the build's "
            "`manifest.json`, archive and adapter."
        ),
    )
    download_bytes: int = Field(
        ge=1,
        le=16 * 1024**3,
        description="Bytes an installation fetches, for the notice shown first.",
    )
    conservative_overhead_bytes: int = Field(
        ge=0,
        le=64 * 1024**3,
        description=(
            "Memory added to a model's weight bytes for admission before the "
            "build is installed and can answer for itself."
        ),
    )

    @model_validator(mode="after")
    def consistent(self) -> Self:
        """Require a safe directory and distinct compute targets."""
        _relative_directory(self.directory)
        if len(set(self.compute_targets)) != len(self.compute_targets):
            raise ValueError("compute_targets must not repeat")
        return self

    @property
    def engine_name(self) -> str:
        """The engine part of the identity, as used in backend tags."""
        return self.engine_id.split(".", 1)[1]

    @property
    def backend_tag(self) -> str:
        """The backend tag this build would advertise, e.g. ``tensorfold-rocm``."""
        return f"{self.engine_name}-{self.lane}"

    @property
    def requirements(self) -> EngineHostRequirements:
        """What the build needs from a host, in the shared form."""
        return EngineHostRequirements(
            platform=self.platform,
            lane=self.lane,
            compute_targets=self.compute_targets,
            skulk_versions=self.skulk_versions,
            control_protocol=self.control_protocol,
            python_requires=self.adapter_python_requires,
        )


@final
class EngineIndexClaims(_Contract):
    """Everything a publisher signs about the packages it has published."""

    schema_version: Literal[1] = Field(description="Index schema version.")
    publisher: Identifier = Field(description="Signing publisher's trust ID.")
    revision: int = Field(
        ge=1, description="Rises with every published index; never reused."
    )
    issued_at: int = Field(gt=0, description="UTC Unix time of signing.")
    expires_at: int = Field(
        gt=0,
        description=(
            "UTC Unix time after which the index no longer offers anything. "
            "Installed packages keep working."
        ),
    )
    entries: tuple[EngineIndexEntry, ...] = Field(
        default=(), max_length=MAX_ENGINE_INDEX_ENTRIES
    )

    @model_validator(mode="after")
    def consistent(self) -> Self:
        """Require a real validity window and no build listed twice."""
        if self.expires_at <= self.issued_at:
            raise ValueError("expires_at must follow issued_at")
        digests = [entry.manifest_digest for entry in self.entries]
        if len(set(digests)) != len(digests):
            raise ValueError("a package build is listed more than once")
        return self


class _SignedEngineIndex(_Contract):
    index: dict[str, JsonValue]
    signature: Signature


@final
@dataclass(frozen=True)
class VerifiedEngineIndex:
    """An index whose signature, trust and validity window have passed.

    Attributes:
        claims: The signed claims, without any build the trust has revoked.
        digest: SHA-256 of the canonical claims.
        document: The signed document exactly as received, for retention.
    """

    claims: EngineIndexClaims
    digest: str
    document: bytes


def verify_engine_index(
    document: bytes, trust: RuntimeTrust, *, now: int
) -> VerifiedEngineIndex:
    """Check a signed package index against the engine trust.

    The order is the manifest's: the publisher named in the unverified
    document selects a key, the signature is checked, and only then are the
    claims parsed and believed. A build the trust has revoked is left out of
    the result rather than failing the whole index.

    Args:
        document: The signed index as received.
        trust: Publisher keys and revocations for engine packages.
        now: Current UTC Unix time.

    Returns:
        The verified index.

    Raises:
        EngineIndexRefusedError: With the code of the first failed check.
    """
    if len(document) > MAX_ENGINE_INDEX_BYTES:
        raise _refuse("index_invalid", "the index document is too large")
    try:
        signed = _SignedEngineIndex.model_validate_json(document)
    except ValidationError:
        raise _refuse(
            "index_invalid", "the document is not a signed engine index"
        ) from None
    publisher = signed.index.get("publisher")
    if now >= trust.expires_at:
        raise _refuse("trust_refused", "the engine trust document has expired")
    if (
        not isinstance(publisher, str)
        or publisher in trust.revoked_publishers
        or publisher not in trust.publishers
    ):
        raise _refuse("trust_refused", "the publisher is not trusted for engines")
    payload = canonical_json(signed.index)
    try:
        Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(trust.publishers[publisher])
        ).verify(
            bytes.fromhex(signed.signature), ENGINE_INDEX_SIGNATURE_DOMAIN + payload
        )
    except (InvalidSignature, ValueError):
        raise _refuse(
            "signature_refused", "the index signature does not verify"
        ) from None
    try:
        claims = EngineIndexClaims.model_validate_json(payload)
    except ValidationError as error:
        raise _refuse(
            "index_invalid",
            f"the signed claims do not fit the index schema "
            f"({error.error_count()} problems)",
        ) from None
    if not claims.issued_at <= now < claims.expires_at:
        raise _refuse("window_refused", "the index is outside its validity window")
    revoked = set(trust.revoked_artifacts)
    if revoked.intersection(entry.manifest_digest for entry in claims.entries):
        claims = claims.model_copy(
            update={
                "entries": tuple(
                    entry
                    for entry in claims.entries
                    if entry.manifest_digest not in revoked
                )
            }
        )
    return VerifiedEngineIndex(
        claims=claims, digest=hashlib.sha256(payload).hexdigest(), document=document
    )


def installable_engine_packages(
    index: VerifiedEngineIndex,
    host: EngineHost,
    *,
    channels: frozenset[EngineChannel] = frozenset({"stable"}),
) -> dict[str, EngineIndexEntry]:
    """Choose, per backend tag, the one build this host would install.

    A build is considered when its channel is followed and the host meets
    every stated requirement. Among those that fit one tag, the highest
    generation wins; at the same generation a build compiled for the host's
    exact device beats a generic one, and ``stable`` beats ``candidate``.

    A tag that a built-in engine owns, or that two engine IDs both claim, is
    offered by neither, as with installed packages.

    Args:
        index: A verified index.
        host: The node's own facts.
        channels: The channels this node follows.

    Returns:
        The chosen build by backend tag. Empty when nothing fits.
    """
    by_tag: dict[str, list[EngineIndexEntry]] = {}
    for entry in index.claims.entries:
        if entry.channel not in channels:
            continue
        if package_engine_of(entry.backend_tag) != entry.engine_name:
            # A built-in engine, or a reserved name, already owns the tag.
            continue
        if engine_host_refusal(entry.requirements, host) is not None:
            continue
        by_tag.setdefault(entry.backend_tag, []).append(entry)
    chosen: dict[str, EngineIndexEntry] = {}
    for tag in sorted(by_tag):
        fitting = by_tag[tag]
        if len({entry.engine_id for entry in fitting}) != 1:
            continue
        chosen[tag] = max(
            fitting,
            key=lambda entry: (
                entry.generation,
                bool(entry.compute_targets),
                entry.channel == "stable",
                entry.manifest_digest,
            ),
        )
    return chosen


class _RetainedIndex(_Contract):
    """The last index this node verified, bound to the bytes it verified."""

    document: str = Field(max_length=MAX_ENGINE_INDEX_BYTES)
    sha256: Digest = Field(description="Digest of the verified document bytes.")
    revision: int = Field(ge=1, description="Revision the document carries.")
    verified_at: int = Field(gt=0, description="Unix time it was verified.")


@final
class EngineIndexClient:
    """Fetch, verify and keep the store's package index.

    Construction touches neither the network nor the disk.

    Args:
        cache_dir: Owner-only directory for the kept index.
        base_url: HTTPS directory that serves the index.
        document: Basename of the index under ``base_url``.
        transport: Network transport, injectable for tests.
        timeout_seconds: Connect and between-bytes timeout.

    Raises:
        ValueError: ``base_url`` is not an HTTPS address.
    """

    def __init__(
        self,
        cache_dir: Path = ENGINE_INDEX_DIR,
        *,
        base_url: str = ENGINE_STORE_URL,
        document: str = ENGINE_INDEX_DOCUMENT,
        transport: httpx.BaseTransport | None = None,
        timeout_seconds: float = _FETCH_SECONDS,
    ) -> None:
        if not base_url.startswith("https://"):
            raise ValueError("engine store must be an HTTPS address")
        self.cache_dir: Final = cache_dir
        self.address: Final = base_url.rstrip("/") + "/" + document
        self.retained_path: Final = cache_dir / _RETAINED_NAME
        self._transport: Final = transport
        self._timeout_seconds: Final = timeout_seconds

    def load(
        self, trust: RuntimeTrust, *, offline: bool, now: int | None = None
    ) -> VerifiedEngineIndex:
        """The newest verified index still in force.

        Online, the store is asked first and a newer verified index replaces
        the kept one. Whatever the fetch refused (no network, a bad
        signature, an expired or older index, a different document at the
        kept revision), nothing it fetched is used and the kept copy decides.
        Offline, only the kept copy is read.

        Args:
            trust: The engine trust in force.
            offline: Never reach the network.
            now: Unix time for expiry checks; the current time by default.

        Returns:
            A verified index whose expiry is still ahead.

        Raises:
            EngineIndexRefusedError: ``index_unavailable`` when no verified
                index in force is available.

        Side effects:
            Writes the kept copy under ``cache_dir``.
        """
        current = now if now is not None else int(time.time())
        try:
            private_directory(self.cache_dir)
            with FileLock(
                str(self.cache_dir / "refresh.lock"), timeout=_LOCK_SECONDS
            ):
                if not offline:
                    # Security fallback boundary: nothing a refused fetch
                    # returned is used. The kept copy decides, and its
                    # absence or expiry is the refusal.
                    with contextlib.suppress(
                        EngineIndexRefusedError, httpx.HTTPError, OSError, ValueError
                    ):
                        return self._refresh(trust, current)
                return self._kept(trust, current)
        except (OSError, ValueError, TimeoutError) as error:
            raise _refuse(
                "index_unavailable", "no verified engine index is in force"
            ) from error

    def kept(self, trust: RuntimeTrust, *, now: int | None = None) -> VerifiedEngineIndex:
        """The kept index, checked again, without any network access.

        Args:
            trust: The engine trust in force.
            now: Unix time for expiry checks; the current time by default.

        Returns:
            The kept index when it is intact, still signed by a trusted
            publisher and still in force.

        Raises:
            EngineIndexRefusedError: ``index_unavailable`` otherwise.
        """
        try:
            return self._kept(trust, now if now is not None else int(time.time()))
        except (OSError, ValueError) as error:
            raise _refuse(
                "index_unavailable", "no verified engine index is in force"
            ) from error

    def _fetch(self) -> bytes:
        """Read the index document, stopping at its size bound."""
        received = bytearray()
        with (
            httpx.Client(
                transport=self._transport,
                timeout=self._timeout_seconds,
                # The address is fixed in host code; a redirect would let the
                # host that serves it choose another one.
                follow_redirects=False,
            ) as client,
            client.stream("GET", self.address) as response,
        ):
            if response.status_code != 200:
                raise ValueError("the engine store did not serve an index")
            for block in response.iter_bytes(65536):
                received.extend(block)
                if len(received) > MAX_ENGINE_INDEX_BYTES:
                    raise ValueError("the index document is too large")
        return bytes(received)

    def _refresh(self, trust: RuntimeTrust, now: int) -> VerifiedEngineIndex:
        """Fetch and verify the index, then keep it when it is not older."""
        document = self._fetch()
        index = verify_engine_index(document, trust, now=now)
        digest = hashlib.sha256(document).hexdigest()
        retained = self._retained_or_none()
        if retained is not None and (
            index.claims.revision < retained.revision
            or (index.claims.revision == retained.revision and digest != retained.sha256)
        ):
            raise _refuse("rollback_refused", "the engine index is older than the kept one")
        if retained is None or retained.sha256 != digest:
            write_private(
                self.retained_path,
                _RetainedIndex(
                    document=document.decode("utf-8"),
                    sha256=digest,
                    revision=index.claims.revision,
                    verified_at=now,
                )
                .model_dump_json()
                .encode(),
            )
        return index

    def _retained(self) -> _RetainedIndex:
        """The kept record, checked against its digest; raises when unusable."""
        record = _RetainedIndex.model_validate_json(
            read_private(self.retained_path, MAX_ENGINE_INDEX_BYTES * 2 + 4096)
        )
        if hashlib.sha256(record.document.encode()).hexdigest() != record.sha256:
            raise ValueError("kept engine index differs from its digest")
        return record

    def _retained_or_none(self) -> _RetainedIndex | None:
        """The kept record, or ``None`` when absent or unusable.

        An unusable record is local damage, not a remote rollback: it must
        not stop the index from renewing, so a refresh treats it as first use.
        """
        try:
            return self._retained()
        except (OSError, ValueError):
            return None

    def _kept(self, trust: RuntimeTrust, now: int) -> VerifiedEngineIndex:
        """Verify the kept document again under the trust now in force."""
        record = self._retained()
        try:
            index = verify_engine_index(record.document.encode(), trust, now=now)
        except EngineIndexRefusedError as error:
            raise ValueError("kept engine index is no longer acceptable") from error
        if index.claims.revision != record.revision:
            raise ValueError("kept engine index revision differs")
        return index
