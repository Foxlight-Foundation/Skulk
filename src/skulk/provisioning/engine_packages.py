"""Signed engine package manifests and their verification.

An engine package is one inference engine build for one platform, plus a small
adapter program, described by a manifest that a publisher signs. This module
is the contract between that publisher and Skulk: it defines the manifest,
checks its signature against an explicit trust document, decides whether the
package fits a host, and checks downloaded archives against the signed sizes
and digests.

Everything here works on bytes and files the caller supplies. Nothing is
downloaded, installed or started, and no engine is named: an engine reaches
Skulk only through a manifest. The same checks therefore serve an online
install and a package carried to an air-gapped cluster as files.

The trust document, canonical JSON form and platform names are the ones the
managed capability runtime already uses. The signed bytes are not: an engine
signature covers a domain prefix as well, so a signature made for a capability
release can never be accepted as an engine manifest.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated, Final, Literal, Self, final

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version
from pydantic import (
    AfterValidator,
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
    platform_matches,
)

ENGINE_SIGNATURE_DOMAIN: Final = b"skulk-engine-package-v1\n"
"""Prefix of the signed bytes. It separates engine manifests from every other
document signed with the same kind of key."""

MAX_ENGINE_MANIFEST_BYTES: Final = 64 * 1024
"""Largest signed manifest document accepted before any parsing."""

MAX_ENGINE_ARCHIVE_BYTES: Final = 3 * 1024**3
"""Largest single archive a manifest may name. A safety bound on what a node
will fetch, matching the native image packages; it is not a size target."""

MAX_ENGINE_UNPACKED_BYTES: Final = 8 * 1024**3
"""Largest unpacked size a manifest may declare for one archive."""

MAX_ENGINE_ADAPTER_BYTES: Final = 1024 * 1024
"""Largest adapter program. An adapter is a small translation layer, never an
engine."""

_HASH_BLOCK_BYTES: Final = 1024 * 1024

EngineId = Annotated[
    str,
    Field(
        max_length=64,
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*\.[a-z0-9]+(?:_[a-z0-9]+)*$",
    ),
]
"""``publisher.engine``, for example ``foxlight.tensorfold``. The engine part
is the engine name used in backend tags, so it admits underscores and no
hyphens."""

EngineLane = Literal["metal", "cuda", "rocm", "vulkan", "cpu"]
"""The accelerator class a build executes on. This is Skulk's own vocabulary:
it decides which memory pool admission counts against."""

ComputeTarget = Annotated[str, Field(max_length=32, pattern=r"^[a-z0-9_]+$")]
"""One exact compiled device target, such as ``sm_121`` or ``gfx1151``."""

Signature = Annotated[str, Field(pattern=r"^[a-f0-9]{128}$")]

EnginePackageRefusal = Literal[
    "manifest_invalid",
    "trust_unavailable",
    "trust_refused",
    "signature_refused",
    "window_refused",
    "artifact_revoked",
    "platform_mismatch",
    "lane_unavailable",
    "compute_target_mismatch",
    "skulk_version_unsupported",
    "protocol_unsupported",
    "adapter_python_unsupported",
    "archive_corrupt",
    "source_unavailable",
    "insufficient_space",
    "install_cancelled",
    "generation_missing",
    "generation_active",
    "installation_damaged",
]
"""Why a package was refused. Each code names one check, so an operator and a
test can tell a bad signature from a package that simply does not fit this
host."""


@final
class EnginePackageRefusedError(Exception):
    """A manifest or archive failed one named check.

    Attributes:
        code: The check that failed.
        message: An operator-readable sentence with no secret material.
    """

    def __init__(self, code: EnginePackageRefusal, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code: Final[EnginePackageRefusal] = code
        self.message: Final = message


def _relative_posix_path(value: str) -> str:
    """Refuse any path that could leave the package's own directory."""
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or "\\" in value
        or any(part in ("", ".", "..") for part in path.parts)
        or str(path) != value
    ):
        raise ValueError("path must be a normalized relative POSIX path")
    return value


def _specifier(value: str) -> str:
    """Require a parseable, non-empty version specifier."""
    try:
        parsed = SpecifierSet(value)
    except InvalidSpecifier as error:
        raise ValueError("invalid version specifier") from error
    if not str(parsed):
        raise ValueError("version specifier must not be empty")
    return value


RelativePath = Annotated[
    str, Field(min_length=1, max_length=255), AfterValidator(_relative_posix_path)
]
FileName = Annotated[
    str, Field(max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
]

ENGINE_ARCHIVE_SUFFIXES: Final = (".tar.gz", ".tgz", ".tar.xz", ".tar", ".zip")
"""Archive formats the installer unpacks. The file name states the format."""


def _archive_name(value: str) -> str:
    """Require an archive file name whose suffix names a supported format."""
    if not value.endswith(ENGINE_ARCHIVE_SUFFIXES):
        raise ValueError("archive must be a tar, tar.gz, tar.xz or zip file")
    return value


ArchiveFileName = Annotated[FileName, AfterValidator(_archive_name)]
VersionRange = Annotated[
    str, Field(min_length=1, max_length=128), AfterValidator(_specifier)
]


class _Contract(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


@final
class EngineArchive(_Contract):
    """The engine build itself: one archive, pinned by size and digest."""

    filename: ArchiveFileName = Field(
        description="Archive file name on the package host; its suffix names the format."
    )
    size: int = Field(
        ge=1, le=MAX_ENGINE_ARCHIVE_BYTES, description="Exact archive size in bytes."
    )
    sha256: Digest = Field(description="SHA-256 of the archive bytes.")
    unpacked_size: int = Field(
        ge=1,
        le=MAX_ENGINE_UNPACKED_BYTES,
        description="Declared unpacked size in bytes, for the free-space check.",
    )
    executable: RelativePath = Field(
        description="Engine executable, relative to the unpacked archive root."
    )


@final
class EngineAdapter(_Contract):
    """The adapter program: the only part of a package Skulk talks to."""

    filename: FileName = Field(description="Adapter file name on the package host.")
    size: int = Field(
        ge=1, le=MAX_ENGINE_ADAPTER_BYTES, description="Exact adapter size in bytes."
    )
    sha256: Digest = Field(description="SHA-256 of the adapter bytes.")
    kind: Literal["python", "native"] = Field(
        description=(
            "'python' is a single standard-library program run by the node's "
            "Skulk interpreter in isolated mode; 'native' is an executable."
        )
    )
    python_requires: VersionRange | None = Field(
        default=None,
        description="Interpreter versions a 'python' adapter runs on.",
    )

    @model_validator(mode="after")
    def python_adapter_states_its_range(self) -> Self:
        """Require an interpreter range for a Python adapter and none for a native one."""
        if (self.kind == "python") != (self.python_requires is not None):
            raise ValueError(
                "python_requires is required for a python adapter and "
                "forbidden for a native one"
            )
        return self


@final
class EngineExternalRequirement(_Contract):
    """A file the package needs but does not carry, pinned by digest.

    It lets a package depend on a library distribution its publisher may not
    redistribute. The node obtains exactly these bytes from the named address,
    or from a file on an air-gapped cluster.
    """

    name: Identifier = Field(description="Distribution name, for messages.")
    version: str = Field(min_length=1, max_length=64, description="Exact version.")
    filename: FileName = Field(description="Exact file name of the distribution.")
    size: int = Field(ge=1, le=MAX_ENGINE_ARCHIVE_BYTES, description="Exact size.")
    sha256: Digest = Field(description="SHA-256 of the distribution bytes.")
    url: str = Field(
        max_length=512,
        pattern=r"^https://[A-Za-z0-9.-]+/[A-Za-z0-9._~/+%-]+$",
        description=(
            "Exact HTTPS address of the file at its own publisher. The signed "
            "digest, not the address, decides whether the bytes are accepted."
        ),
    )


@final
class EngineProtocolRange(_Contract):
    """The control-protocol versions an adapter speaks, both ends included."""

    minimum: int = Field(ge=1, le=65535)
    maximum: int = Field(ge=1, le=65535)

    @model_validator(mode="after")
    def ordered(self) -> Self:
        """Require the minimum not to exceed the maximum."""
        if self.minimum > self.maximum:
            raise ValueError("protocol minimum exceeds maximum")
        return self


@final
class EngineProvenance(_Contract):
    """Where the engine build came from. Recorded, never used to decide fit."""

    repository: str = Field(
        max_length=128,
        pattern=r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$",
        description="Upstream source repository, as owner/name.",
    )
    version: str = Field(min_length=1, max_length=64, description="Upstream version.")
    source_commit: Annotated[str, Field(pattern=r"^[a-f0-9]{40}$")] = Field(
        description="Upstream commit the build was made from."
    )
    built_by: Literal["upstream", "publisher"] = Field(
        description=(
            "'upstream' means the archive is upstream's own release, mirrored "
            "byte for byte; 'publisher' means the manifest's publisher built it."
        )
    )


@final
class EnginePackageClaims(_Contract):
    """Everything a publisher signs about one engine build for one platform."""

    schema_version: Literal[1] = Field(description="Manifest schema version.")
    engine_id: EngineId = Field(description="publisher.engine identity.")
    publisher: Identifier = Field(description="Signing publisher's trust ID.")
    engine_version: str = Field(
        min_length=1, max_length=64, description="Version of the engine inside."
    )
    generation: int = Field(
        ge=1,
        description=(
            "Package generation for this engine and platform. It rises with "
            "every published change, including adapter-only ones."
        ),
    )
    platform: RuntimePlatform = Field(
        description="Operating system, C library where it matters, and machine."
    )
    lane: EngineLane = Field(description="Accelerator class the build executes on.")
    compute_targets: tuple[ComputeTarget, ...] = Field(
        default=(),
        max_length=16,
        description=(
            "Exact device targets the build was compiled for. Empty means any "
            "device of the lane."
        ),
    )
    skulk_versions: VersionRange = Field(
        description="Skulk versions the package runs on."
    )
    control_protocol: EngineProtocolRange = Field(
        description="Control-protocol versions the adapter speaks."
    )
    archive: EngineArchive = Field(description="The engine build.")
    adapter: EngineAdapter = Field(description="The adapter program.")
    external_requirements: tuple[EngineExternalRequirement, ...] = Field(
        default=(), max_length=8, description="Pinned files the package needs."
    )
    conservative_overhead_bytes: int = Field(
        ge=0,
        le=64 * 1024**3,
        description=(
            "Memory added to a model's weight bytes for admission until a "
            "node has answered with the adapter's own estimate."
        ),
    )
    provenance: EngineProvenance = Field(description="Origin of the engine build.")
    issued_at: int = Field(gt=0, description="UTC Unix time of signing.")
    expires_at: int = Field(
        gt=0,
        description=(
            "UTC Unix time after which the package may no longer be "
            "installed. An installed package keeps working."
        ),
    )

    @model_validator(mode="after")
    def consistent(self) -> Self:
        """Require a real validity window and distinct compute targets."""
        if self.expires_at <= self.issued_at:
            raise ValueError("expires_at must follow issued_at")
        if len(set(self.compute_targets)) != len(self.compute_targets):
            raise ValueError("compute_targets must not repeat")
        return self

    @property
    def engine_name(self) -> str:
        """The engine part of the identity, as used in backend tags."""
        return self.engine_id.split(".", 1)[1]

    @property
    def backend_tag(self) -> str:
        """The concrete backend tag this build advertises, e.g. ``tensorfold-rocm``."""
        return f"{self.engine_name}-{self.lane}"


class _SignedEnginePackage(_Contract):
    engine: dict[str, JsonValue]
    signature: Signature


@final
@dataclass(frozen=True)
class EngineHost:
    """What a node knows about itself when it considers a package.

    Attributes:
        platform: The host's artifact family, as ``current_platform`` names it.
        lanes: Accelerator classes the node has observed it can execute on.
        compute_targets: Exact device targets the node observed. Empty means
            detection found none, which is different from having none.
        skulk_version: The running Skulk release.
        control_protocols: Lowest and highest control-protocol version the
            host speaks, both included.
        python_version: The interpreter that would run a Python adapter.
    """

    platform: str
    lanes: frozenset[EngineLane]
    compute_targets: frozenset[str]
    skulk_version: str
    control_protocols: tuple[int, int]
    python_version: str


@final
@dataclass(frozen=True)
class VerifiedEnginePackage:
    """A manifest whose signature, trust and validity window have passed.

    Attributes:
        claims: The signed claims.
        digest: SHA-256 of the canonical claims. This is the package's build
            identity: it covers the engine archive, the adapter and every
            pinned requirement.
        document: The signed document exactly as received, for retention.
    """

    claims: EnginePackageClaims
    digest: str
    document: bytes


def _refuse(code: EnginePackageRefusal, message: str) -> EnginePackageRefusedError:
    return EnginePackageRefusedError(code, message)


def verify_engine_manifest(
    document: bytes, trust: RuntimeTrust, *, now: int
) -> VerifiedEnginePackage:
    """Check a signed manifest against an explicit trust document.

    The order matters: the publisher named in the unverified document selects
    a key, the signature is checked, and only then are the claims parsed and
    believed. Fit to a particular host is a separate question, answered by
    ``check_engine_host``.

    Args:
        document: The signed manifest as received.
        trust: Publisher keys and revocations for engine packages.
        now: Current UTC Unix time.

    Returns:
        The verified package and its build identity.

    Raises:
        EnginePackageRefusedError: With the code of the first failed check.
    """
    if len(document) > MAX_ENGINE_MANIFEST_BYTES:
        raise _refuse("manifest_invalid", "the manifest document is too large")
    try:
        signed = _SignedEnginePackage.model_validate_json(document)
    except ValidationError:
        raise _refuse(
            "manifest_invalid", "the document is not a signed engine manifest"
        ) from None
    publisher = signed.engine.get("publisher")
    if now >= trust.expires_at:
        raise _refuse("trust_refused", "the engine trust document has expired")
    if (
        not isinstance(publisher, str)
        or publisher in trust.revoked_publishers
        or publisher not in trust.publishers
    ):
        raise _refuse("trust_refused", "the publisher is not trusted for engines")
    payload = canonical_json(signed.engine)
    try:
        Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(trust.publishers[publisher])
        ).verify(bytes.fromhex(signed.signature), ENGINE_SIGNATURE_DOMAIN + payload)
    except (InvalidSignature, ValueError):
        raise _refuse(
            "signature_refused", "the manifest signature does not verify"
        ) from None
    try:
        claims = EnginePackageClaims.model_validate_json(payload)
    except ValidationError as error:
        raise _refuse(
            "manifest_invalid",
            f"the signed claims do not fit the manifest schema "
            f"({error.error_count()} problems)",
        ) from None
    if not claims.issued_at <= now < claims.expires_at:
        raise _refuse(
            "window_refused", "the manifest is outside its validity window"
        )
    digest = hashlib.sha256(payload).hexdigest()
    revoked = set(trust.revoked_artifacts)
    pinned = {digest, claims.archive.sha256, claims.adapter.sha256} | {
        requirement.sha256 for requirement in claims.external_requirements
    }
    if pinned & revoked:
        raise _refuse(
            "artifact_revoked", "the package or one of its files has been revoked"
        )
    return VerifiedEnginePackage(claims=claims, digest=digest, document=document)


def select_control_protocol(claims: EnginePackageClaims, host: EngineHost) -> int:
    """Pick the highest control-protocol version both sides speak.

    Raises:
        EnginePackageRefusedError: ``protocol_unsupported`` when the ranges
            do not overlap.
    """
    lowest = max(claims.control_protocol.minimum, host.control_protocols[0])
    highest = min(claims.control_protocol.maximum, host.control_protocols[1])
    if lowest > highest:
        raise _refuse(
            "protocol_unsupported",
            f"the adapter speaks control protocol "
            f"{claims.control_protocol.minimum} to {claims.control_protocol.maximum}; "
            f"this Skulk speaks {host.control_protocols[0]} to "
            f"{host.control_protocols[1]}",
        )
    return highest


def _version_in(specifier: str, version: str) -> bool:
    try:
        return SpecifierSet(specifier).contains(Version(version), prereleases=True)
    except InvalidVersion:
        return False


def adapter_python_supported(claims: EnginePackageClaims, python_version: str) -> bool:
    """Whether this interpreter can run the package's adapter.

    Always true for a native adapter. Checked at install and again at every
    launch, because the node's interpreter can change under an installed
    package when Skulk itself is updated.
    """
    required = claims.adapter.python_requires
    return required is None or _version_in(required, python_version)


def check_engine_host(package: VerifiedEnginePackage, host: EngineHost) -> int:
    """Decide whether a verified package can be installed on this host.

    These are technical requirements of the build: its platform, the
    accelerator class it executes on, the device targets it was compiled for,
    and the Skulk and protocol versions it was written against. Nothing here
    is a list of tested hardware. A host whose exact device target could not
    be detected is admitted, because the engine's own start-up is the final
    check and failed detection is not evidence of a mismatch.

    Args:
        package: A manifest that already passed ``verify_engine_manifest``.
        host: The node's own facts.

    Returns:
        The control-protocol version to use with this package.

    Raises:
        EnginePackageRefusedError: With the code of the first requirement the
            host does not meet.
    """
    claims = package.claims
    if not platform_matches(claims.platform, host.platform):
        raise _refuse(
            "platform_mismatch",
            f"the package is built for {claims.platform}; this host is "
            f"{host.platform}",
        )
    if claims.lane not in host.lanes:
        raise _refuse(
            "lane_unavailable",
            f"the package executes on {claims.lane}, which this host does not offer",
        )
    if (
        claims.compute_targets
        and host.compute_targets
        and not host.compute_targets.intersection(claims.compute_targets)
    ):
        raise _refuse(
            "compute_target_mismatch",
            f"the package is compiled for {', '.join(claims.compute_targets)}; "
            f"this host has {', '.join(sorted(host.compute_targets))}",
        )
    if not _version_in(claims.skulk_versions, host.skulk_version):
        raise _refuse(
            "skulk_version_unsupported",
            f"the package runs on Skulk {claims.skulk_versions}; this is "
            f"{host.skulk_version}",
        )
    protocol = select_control_protocol(claims, host)
    adapter = claims.adapter
    if adapter.python_requires is not None and not _version_in(
        adapter.python_requires, host.python_version
    ):
        raise _refuse(
            "adapter_python_unsupported",
            f"the adapter needs Python {adapter.python_requires}; this node "
            f"runs {host.python_version}",
        )
    return protocol


def verify_engine_file(path: Path, *, size: int, sha256: str) -> None:
    """Check one downloaded or imported file against its signed size and digest.

    The size is compared first so a truncated or padded file is refused
    without hashing it.

    Args:
        path: The file to check.
        size: Exact size the manifest signed.
        sha256: Digest the manifest signed.

    Raises:
        EnginePackageRefusedError: ``archive_corrupt`` when the file is
            missing, is not a regular file, or differs in size or content.
    """
    try:
        if path.is_symlink() or not path.is_file():
            raise _refuse("archive_corrupt", f"{path.name} is not a regular file")
        actual = path.stat().st_size
        if actual != size:
            raise _refuse(
                "archive_corrupt",
                f"{path.name} is {actual} bytes; the manifest signed {size}",
            )
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(_HASH_BLOCK_BYTES):
                digest.update(block)
    except OSError as error:
        raise _refuse(
            "archive_corrupt", f"{path.name} could not be read ({error.strerror})"
        ) from None
    if digest.hexdigest() != sha256:
        raise _refuse(
            "archive_corrupt", f"{path.name} does not match its signed digest"
        )


def installed_engine_claims(document: bytes) -> tuple[EnginePackageClaims, str]:
    """Read the claims and build identity from a manifest an installation kept.

    This checks neither the signature nor the validity window. It is only for
    the document a completed installation retained beside its files, which
    was verified when it was installed: an installed engine keeps working
    after its trust document or its manifest expires. Never use it to decide
    whether to install.

    Args:
        document: The signed manifest exactly as it was retained.

    Returns:
        The claims and the SHA-256 of their canonical form.

    Raises:
        EnginePackageRefusedError: ``installation_damaged`` when the retained
            document no longer parses as a manifest.
    """
    if len(document) > MAX_ENGINE_MANIFEST_BYTES:
        raise _refuse("installation_damaged", "the retained manifest is too large")
    try:
        signed = _SignedEnginePackage.model_validate_json(document)
        payload = canonical_json(signed.engine)
        claims = EnginePackageClaims.model_validate_json(payload)
    except ValidationError:
        raise _refuse(
            "installation_damaged", "the retained manifest is not readable"
        ) from None
    return claims, hashlib.sha256(payload).hexdigest()
