"""Generic verification of signed offline plugin runtimes without plugin imports."""

import hashlib
import importlib.metadata
import importlib.util
import io
import json
import platform
import stat
import sys
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, final

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, JsonValue

from skulk.extensions.runtime_files import read_private

Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Identifier = Annotated[
    str, Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9._:@-]+$")
]
# Preserve the signed/persisted Linux artifact identifier; it is not an OS-version gate.
RuntimePlatform = Annotated[
    str, Field(pattern=r"^[a-z0-9]+(?:[.-][a-z0-9_]+)*$", max_length=64)
]
"""An artifact family: operating system, C library where it matters, and machine.

This was a closed pair, which meant the installer refused every other host
before looking at a single wheel: a Grace Blackwell node on Linux aarch64 could
not install a capability runtime at all. The family is now derived from the
host, and each wheel's own tags are still checked against the live interpreter,
so a wheel that cannot run here is refused whatever the family says."""

LEGACY_PLATFORMS: dict[str, str] = {"ubuntu-24.04-x86_64": "linux-glibc-x86_64"}
"""Names already written into signed releases. The Ubuntu label never denoted
that distribution; it meant glibc Linux on x86_64."""


def canonical_platform(name: str) -> str:
    """Resolve a declared family, including the names already signed."""
    return LEGACY_PLATFORMS.get(name, name)


def _family_parts(name: str) -> tuple[str, str | None, str]:
    parts = canonical_platform(name).split("-", 2)
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    return parts[0], None, parts[-1]


def platform_matches(declared: str, host: str) -> bool:
    """Whether an artifact family describes this host.

    A legacy name matched every Linux x86_64 host whatever its C library, and a
    library reported as ``unknown`` means detection failed rather than that it
    differs; both compare on operating system and machine only. Two fully
    derived names must agree on the library as well.
    """
    left, right = _family_parts(declared), _family_parts(host)
    if left[0] != right[0] or left[2] != right[2]:
        return False
    if declared in LEGACY_PLATFORMS or host in LEGACY_PLATFORMS:
        return True
    return left[1] == right[1] or "unknown" in (left[1], right[1])


def current_platform() -> str:
    """Name this host's artifact family; only the operating system is closed."""
    machine = platform.machine().lower().replace("-", "_")
    architecture = {"amd64": "x86_64", "arm64": "aarch64"}.get(machine, machine)
    if sys.platform == "darwin":
        return f"macos-{'arm64' if machine == 'arm64' else architecture}"
    if sys.platform == "linux":
        library, _ = platform.libc_ver()
        return f"linux-{library or 'unknown'}-{architecture}"
    # Architecture and C library are open; the operating system is not. Skulk
    # itself runs on macOS and Linux, the release schema admits only those, and
    # naming a third one here would claim a host the runtime cannot serve.
    raise ValueError("managed runtimes are qualified on macOS and Linux only")


class _Contract(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


class RuntimeTrust(_Contract):
    """Explicit local publisher authority and current revocation view."""

    revision: int = Field(ge=1, description="Monotonic owner trust revision.")
    expires_at: int = Field(gt=0, description="UTC Unix expiry of this trust view.")
    publishers: dict[Identifier, Digest] = Field(
        min_length=1,
        max_length=16,
        description="Publisher IDs mapped to Ed25519 public keys.",
    )
    revoked_publishers: tuple[Identifier, ...] = Field(
        default=(), max_length=16, description="Publishers no longer authorized."
    )
    revoked_artifacts: tuple[Digest, ...] = Field(
        default=(),
        max_length=128,
        description="Revoked bundle, wheel or runtime digests.",
    )


class RuntimeWheel(_Contract):
    """One exact publisher-approved wheel; no version resolution is permitted."""

    filename: str = Field(
        pattern=r"^[A-Za-z0-9_.+-]+\.whl$",
        max_length=240,
        description="Safe wheel basename.",
    )
    sha256: Digest = Field(description="Exact artifact SHA-256.")
    size: int = Field(ge=1, le=67108864, description="Exact compressed byte count.")

    @property
    def distribution(self) -> tuple[str, str]:
        """Return canonical distribution identity from the wheel filename."""
        name, version, _, _ = parse_wheel_filename(self.filename)
        return str(name), str(version)


ACCEPTED_RELEASE_PROTOCOLS: tuple[int, ...] = (1,)
"""Release record protocols this host installs: the current and, once there is
one, the previous. A capability published against the previous protocol keeps
installing for one release cycle; anything else is refused by name."""

ACCEPTED_RUNTIME_PROTOCOLS: tuple[int, ...] = (2,)
"""Isolated runtime envelope protocols this host installs, on the same rule."""


class ProtocolUnsupportedError(ValueError):
    """A release outside this host's protocol window, named without disclosure.

    Carries only the protocol kind, the number offered and the numbers this host
    accepts, so the manager, the guided installer and the plugin routes can say
    exactly what to do (update the host, or pick a release published for it)
    without surfacing anything else from the release.
    """

    def __init__(self, kind: str, offered: int, accepted: tuple[int, ...]) -> None:
        window = " or ".join(str(item) for item in accepted)
        super().__init__(
            f"{kind} protocol {offered} is not accepted; this host accepts {window}"
        )
        self.kind = kind
        self.offered = offered
        self.accepted = accepted


def _accepted(name: str, accepted: tuple[int, ...]) -> Callable[[int], int]:
    def check(value: int) -> int:
        if value not in accepted:
            window = " or ".join(str(item) for item in accepted)
            raise ValueError(
                f"{name} protocol {value} is not accepted; this host accepts {window}"
            )
        return value

    return check


ReleaseProtocol = Annotated[
    int, AfterValidator(_accepted("release", ACCEPTED_RELEASE_PROTOCOLS))
]
RuntimeProtocol = Annotated[
    int, AfterValidator(_accepted("runtime", ACCEPTED_RUNTIME_PROTOCOLS))
]


class _ManifestClaims(BaseModel):
    # Plugin-specific policy remains opaque. The signature covers the original
    # canonical payload, never this partial interpretation of its common claims.
    model_config = ConfigDict(frozen=True, strict=True, extra="ignore")
    bundle_id: Identifier
    bundle_version: str
    skulk_requires: str = Field(max_length=128)
    executable: Literal["bundle.pyz"]
    executable_sha256: Digest


class _ReleaseClaims(_Contract):
    protocol: ReleaseProtocol
    publisher: Identifier
    sequence: int = Field(ge=1)
    created_at: int = Field(gt=0)
    expires_at: int = Field(gt=0)
    artifact_name: Literal["bundle.pyz"]
    artifact_size: int = Field(ge=1, le=67108864)
    manifest: _ManifestClaims
    platforms: tuple[Literal["darwin", "linux"], ...] = Field(
        min_length=1, max_length=2
    )
    python_requires: str = Field(max_length=128)
    skulk_build_sha256: Digest
    dependency_lock_sha256: Digest
    state_schema: Identifier
    compatible_state_schemas: tuple[Identifier, ...] = Field(max_length=8)
    permissions: tuple[Annotated[str, Field(min_length=1, max_length=512)], ...] = (
        Field(min_length=1, max_length=16)
    )


class _RuntimeClaims(_Contract):
    protocol: RuntimeProtocol
    implementation: Literal["cpython"]
    platform: RuntimePlatform
    release: _ReleaseClaims
    wheels: tuple[RuntimeWheel, ...] = Field(min_length=1, max_length=64)


class _Signed(_Contract):
    runtime: dict[str, JsonValue]
    signature: str = Field(pattern=r"^[a-f0-9]{128}$")


@final
@dataclass(frozen=True)
class QualifiedHost:
    """Locally measured execution environment, never taken from an API caller."""

    platform: RuntimePlatform
    python_version: str
    skulk_version: str
    skulk_build_sha256: str


@final
@dataclass(frozen=True)
class VerifiedRuntime:
    """Authenticated immutable bytes and their generic installation claims."""

    metadata: bytes
    payload: bytes
    claims: _RuntimeClaims

    @property
    def digest(self) -> str:
        """Bind every signed claim including opaque plugin policy."""
        return hashlib.sha256(self.payload).hexdigest()

    @property
    def inventory(self) -> dict[str, str]:
        """Return a fresh exact dependency inventory without installer tools."""
        return dict(sorted(wheel.distribution for wheel in self.claims.wheels))


def canonical_json(value: JsonValue) -> bytes:
    """Serialize JSON with the signed runtime's canonical field ordering."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def measure_host() -> QualifiedHost:
    """Measure actual core sources, native bindings, Python and supported platform."""
    target = current_platform()
    if sys.implementation.name != "cpython":
        raise ValueError("unsupported managed Python implementation")
    digest = hashlib.sha256()
    count = total = 0
    for package in ("skulk", "skulk_pyo3_bindings"):
        specification = importlib.util.find_spec(package)
        if specification is None or specification.origin is None:
            raise ValueError("qualified core build is unavailable")
        root = Path(specification.origin).parent
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if any(
                part in {"__pycache__", "tests", ".pytest_cache"}
                for part in relative.parts
            ):
                continue
            if path.is_symlink():
                raise ValueError("core build contains an unsupported file")
            if path.is_dir() or path.suffix in {".pyc", ".pyo"}:
                continue
            if not path.is_file():
                raise ValueError("core build contains an unsupported file")
            count += 1
            total += path.stat().st_size
            if count > 20000 or total > 536870912:
                raise ValueError("core build exceeds verification bound")
            with path.open("rb") as source:
                content = hashlib.file_digest(source, "sha256").digest()
            digest.update((package + "/" + relative.as_posix()).encode() + b"\0")
            digest.update(content)
    return QualifiedHost(
        target,
        platform.python_version(),
        importlib.metadata.version("skulk"),
        digest.hexdigest(),
    )


def _claimed_publisher(runtime: dict[str, JsonValue]) -> str | None:
    """The publisher a payload names, read without interpreting anything else."""
    release = runtime.get("release")
    publisher = release.get("publisher") if isinstance(release, dict) else None
    return publisher if isinstance(publisher, str) else None


def protocol_refusal_sentence(
    kind: str, offered: int, accepted: tuple[int, ...]
) -> str:
    """The one sentence every surface uses for a release outside the window."""
    window = " or ".join(str(item) for item in accepted)
    return (
        f"This host accepts {kind} protocol {window}; the release offers {offered}. "
        "Update Skulk on this host, or choose a release published for it."
    )


def _check_protocol_window(runtime: dict[str, JsonValue]) -> None:
    """Name a protocol outside the window before the claims are validated.

    The claim models refuse the same values; this runs first so the refusal is
    the typed one every surface can name. Anything malformed is left to the
    claim models' own validation.
    """
    release = runtime.get("release")
    offered_runtime = runtime.get("protocol")
    offered_release = release.get("protocol") if isinstance(release, dict) else None
    if (
        isinstance(offered_runtime, int)
        and not isinstance(offered_runtime, bool)
        and offered_runtime not in ACCEPTED_RUNTIME_PROTOCOLS
    ):
        raise ProtocolUnsupportedError(
            "runtime", offered_runtime, ACCEPTED_RUNTIME_PROTOCOLS
        )
    if (
        isinstance(offered_release, int)
        and not isinstance(offered_release, bool)
        and offered_release not in ACCEPTED_RELEASE_PROTOCOLS
    ):
        raise ProtocolUnsupportedError(
            "release", offered_release, ACCEPTED_RELEASE_PROTOCOLS
        )


def verify_runtime(
    metadata: bytes, trust: RuntimeTrust, host: QualifiedHost, *, now: int
) -> VerifiedRuntime:
    """Authenticate complete v2 metadata, then enforce local trust and compatibility.

    Plugin-specific manifest fields are signed opaque data. No plugin package,
    provider policy or private dependency is imported to inspect a release.
    """
    if len(metadata) > 131072:
        raise ValueError("runtime metadata exceeds bound")
    signed = _Signed.model_validate_json(metadata)
    payload = canonical_json(signed.runtime)
    # The publisher is read minimally so the signature can be checked before
    # anything else is interpreted: a named protocol refusal must come from an
    # authenticated payload, never from whatever a feed happens to serve.
    publisher = _claimed_publisher(signed.runtime)
    public = trust.publishers.get(publisher) if publisher is not None else None
    if (
        publisher is None
        or public is None
        or publisher in trust.revoked_publishers
        or now >= trust.expires_at
    ):
        raise ValueError("runtime publisher trust refused")
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(public)).verify(
            bytes.fromhex(signed.signature), payload
        )
    except (InvalidSignature, ValueError):
        raise ValueError("runtime signature refused") from None
    _check_protocol_window(signed.runtime)
    claims = _RuntimeClaims.model_validate_json(payload)
    release = claims.release
    runtime = VerifiedRuntime(metadata, payload, claims)
    family = canonical_platform(host.platform)
    expected_os = "darwin" if family.startswith("macos") else family.split("-")[0]
    if (
        not release.created_at <= now < release.expires_at
        or not platform_matches(claims.platform, host.platform)
        or expected_os not in release.platforms
        or host.python_version not in SpecifierSet(release.python_requires)
        or host.skulk_version not in SpecifierSet(release.manifest.skulk_requires)
        or host.skulk_build_sha256 != release.skulk_build_sha256
        or runtime.digest in trust.revoked_artifacts
        or release.manifest.executable_sha256 in trust.revoked_artifacts
        or any(wheel.sha256 in trust.revoked_artifacts for wheel in claims.wheels)
    ):
        raise ValueError("runtime compatibility or revocation refused")
    inventory = runtime.inventory
    if (
        len(inventory) != len(claims.wheels)
        or sum(wheel.size for wheel in claims.wheels) > 536870912
    ):
        raise ValueError("ambiguous or oversized runtime inventory")
    if (
        hashlib.sha256(canonical_json(dict(inventory))).hexdigest()
        != release.dependency_lock_sha256
    ):
        raise ValueError("runtime dependency inventory differs")
    return runtime


def verified_artifacts(runtime: VerifiedRuntime, directory: Path) -> dict[str, bytes]:
    """Verify exact runtime bytes and normalize unsupported archive failures."""
    try:
        return _verified_artifacts(runtime, directory)
    except (zipfile.BadZipFile, NotImplementedError, RuntimeError):
        raise ValueError("invalid or unsupported runtime archive") from None


RUNTIME_LAUNCHER_GROUP = "skulk.capability_runtime"
"""Entry-point group a signed runtime wheel uses to declare the owner,
management and setup launchers. A bundle used to carry those as fixed shims
that embedded the whole SDK; a wheel that declares them lets the bundle carry
only the capability, while the launchers still come from signed, hash-pinned
bytes installed into the generation runtime."""


def runtime_launchers(archive: zipfile.ZipFile) -> list[str]:
    """Names declared under the launcher group in a wheel's entry-point metadata.

    Read as data from ``*.dist-info/entry_points.txt`` so verification stays
    static: nothing is installed or imported to learn what the wheel offers.
    Repeats are kept so a caller can refuse an ambiguous declaration.
    """
    found: list[str] = []
    declarations = [
        item
        for item in archive.infolist()
        if len(PurePosixPath(item.filename).parts) == 2
        and PurePosixPath(item.filename).parts[0].endswith(".dist-info")
        and PurePosixPath(item.filename).parts[1] == "entry_points.txt"
    ]
    # A wheel has one dist-info; more than one entry_points.txt is malformed,
    # and reading only the single legitimate member bounds this scan.
    if len(declarations) > 1:
        raise ValueError("wheel declares entry points more than once")
    for item in declarations:
        if item.file_size > 65536:
            raise ValueError("wheel entry point declaration exceeds bound")
        section = None
        try:
            # importlib.metadata reads the installed copy as strict UTF-8, so a
            # byte tolerated here would only surface as an owner that never
            # starts; refuse it where refusal is cheap and visible.
            text = archive.read(item).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("wheel entry point declaration is not UTF-8") from error
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                # importlib.metadata drops blank and "#" lines and nothing
                # else; a ";" line is a declaration to it, so it is one here.
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1].strip()
            elif section == RUNTIME_LAUNCHER_GROUP and "=" in line:
                name, _, target = line.partition("=")
                if not valid_launcher_target(target):
                    # importlib.metadata would build an entry point with this
                    # value and fail at load(), after staging succeeded.
                    raise ValueError(
                        f"wheel declares an invalid launcher {name.strip()}"
                    )
                found.append(name.strip())
    return found


def valid_launcher_target(target: str) -> bool:
    """Whether ``module:attribute`` names something ``EntryPoint.load`` can resolve.

    The shape importlib accepts: dotted module and dotted attribute paths with
    optional whitespace around the colon. Extras are refused; a launcher is
    loaded from the pinned wheelhouse, never resolved against extras. The
    match runs through importlib's own pattern because its character class is
    narrower than ``str.isidentifier``: a combining mark passes the latter and
    would only fail at ``EntryPoint.load``, after staging.
    """
    match = importlib.metadata.EntryPoint.pattern.match(target.strip())
    if match is None or match.group("attr") is None or match.group("extras"):
        return False
    return all(
        part.isidentifier()
        for side in (match.group("module"), match.group("attr"))
        for part in side.split(".")
    )


def _verified_artifacts(runtime: VerifiedRuntime, directory: Path) -> dict[str, bytes]:
    """Verify exact bundle and wheels, archive paths, tags and complete dependencies."""
    release = runtime.claims.release
    bundle = read_private(directory / "bundle.pyz", release.artifact_size)
    if (
        len(bundle) != release.artifact_size
        or hashlib.sha256(bundle).hexdigest() != release.manifest.executable_sha256
    ):
        raise ValueError("runtime bundle differs")
    result = {"bundle.pyz": bundle}
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        entries = archive.infolist()
        if len(entries) > 20000 or sum(item.file_size for item in entries) > 268435456:
            raise ValueError("expanded bundle exceeds bound")
        if len({item.filename for item in entries}) != len(entries):
            raise ValueError("ambiguous bundle paths")
        for item in entries:
            path = PurePosixPath(item.filename)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in item.filename
                or stat.S_ISLNK(item.external_attr >> 16)
            ):
                raise ValueError("unsafe bundle path")
        owner = next(
            (item for item in entries if item.filename == "__owner__.py"), None
        )
        if owner is not None and not 0 < owner.file_size <= 65536:
            raise ValueError("bundle owner entrypoint has an invalid size")
    tags = set(sys_tags())
    expanded = 0
    inventory = runtime.inventory
    launchers: list[str] = []
    for wheel in runtime.claims.wheels:
        content = read_private(directory / wheel.filename, wheel.size)
        if (
            len(content) != wheel.size
            or hashlib.sha256(content).hexdigest() != wheel.sha256
        ):
            raise ValueError("runtime wheel differs")
        name, version, _, supported = parse_wheel_filename(wheel.filename)
        if not tags.intersection(supported):
            raise ValueError("wheel does not support this interpreter")
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            entries = archive.infolist()
            expanded += sum(item.file_size for item in entries)
            if (
                len(entries) > 20000
                or expanded > 536870912
                or sum(item.file_size for item in entries) > 268435456
            ):
                raise ValueError("expanded runtime exceeds bound")
            if len({item.filename for item in entries}) != len(entries):
                raise ValueError("ambiguous wheel paths")
            for item in entries:
                path = PurePosixPath(item.filename)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or "\\" in item.filename
                    or stat.S_ISLNK(item.external_attr >> 16)
                ):
                    raise ValueError("unsafe wheel path")
            metadata = [
                item
                for item in entries
                if item.filename.endswith(".dist-info/METADATA")
            ]
            if len(metadata) != 1 or metadata[0].file_size > 1048576:
                raise ValueError("missing or oversized wheel metadata")
            parsed = BytesParser().parsebytes(archive.read(metadata[0]))
            if (
                canonicalize_name(str(parsed.get("Name", ""))) != name
                or Version(str(parsed.get("Version", "0"))) != version
            ):
                raise ValueError("wheel identity differs")
            # Only after the entry-count, expanded-size and single-metadata
            # bounds hold: a malformed wheel must not get decompression work
            # out of the launcher scan before it is rejected.
            launchers += runtime_launchers(archive)
            for value in parsed.get_all("Requires-Dist", []):
                requirement = Requirement(str(value))
                if requirement.marker is not None and not requirement.marker.evaluate(
                    {"extra": ""}
                ):
                    continue
                dependency = str(canonicalize_name(requirement.name))
                if (
                    requirement.url is not None
                    or requirement.extras
                    or dependency not in inventory
                    or Version(inventory[dependency]) not in requirement.specifier
                ):
                    raise ValueError("incomplete or incompatible wheel dependency")
        result[wheel.filename] = content
    if len(launchers) != len(set(launchers)):
        # Which distribution answers importlib.metadata first is not bound by
        # the signed contract, so two declarations of one launcher would let
        # metadata order decide which owner starts. Refuse the ambiguity.
        raise ValueError("signed wheels declare the same runtime launcher twice")
    if owner is None and "owner" not in launchers:
        # Either shape is acceptable, never neither: a bundle with no shim must
        # be launched from a signed wheel that declares the owner launcher.
        raise ValueError(
            "bundle carries no owner entrypoint and no signed wheel declares one"
        )
    return result
