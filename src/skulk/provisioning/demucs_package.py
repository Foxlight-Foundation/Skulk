"""Shared sealed Demucs package contracts and bounded artifact verification.

The resident provisioner and standalone build tools use this same source. It
imports no Skulk runtime components. Bootstrap executes verified Python only
through the explicitly supplied supervised command runner.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import stat
import tarfile
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Protocol
from urllib.parse import unquote, urlsplit

from packaging.utils import canonicalize_name
from packaging.version import Version
from pydantic import BaseModel, ConfigDict, Field, model_validator

PYTHON_VERSION = "3.12.13"

MAX_PYTHON_ARCHIVE_BYTES = 128 * 1024 * 1024

PROFILE_TARGETS = {
    "macos-arm64": "macosx_14_0_arm64",
    "linux-amd64": "manylinux_2_28_x86_64",
    "linux-arm64": "manylinux_2_28_aarch64",
}

PROFILE_TRIPLES = {
    "macos-arm64": "aarch64-apple-darwin",
    "linux-amd64": "x86_64-unknown-linux-gnu",
    "linux-arm64": "aarch64-unknown-linux-gnu",
}

TARGETS = tuple(PROFILE_TARGETS.values())
MAX_PACKAGE_BYTES = 1024 * 1024 * 1024
MAX_EXPANDED_PYTHON_BYTES = 512 * 1024 * 1024
MAX_INSTALLED_BYTES = 4 * 1024 * 1024 * 1024
Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Checkpoint = Callable[[], None]


def _checkpoint(checkpoint: Checkpoint | None) -> None:
    if checkpoint is not None:
        checkpoint()


class PythonArchivePin(BaseModel):
    """One exact interpreter object, independently checked before packaging."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    profile: Literal["macos-arm64", "linux-amd64", "linux-arm64"]
    platform_tag: str = Field(description="Engine package's OS and CPU ABI tag.")
    filename: str = Field(description="Immutable release object basename.")
    url: str = Field(description="Pinned upstream release object, intake only.")
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    size_bytes: int = Field(gt=0, le=MAX_PYTHON_ARCHIVE_BYTES)
    version: Literal["3.12.13"]
    executable: Literal["python/bin/python3.12"]

    @model_validator(mode="after")
    def consistent_identity(self) -> PythonArchivePin:
        """Reject foreign interpreters, platforms and mutable or indirect URLs."""
        filename = (
            f"cpython-{PYTHON_VERSION}+20260807-{PROFILE_TRIPLES[self.profile]}"
            "-install_only_stripped.tar.gz"
        )
        url = urlsplit(self.url)
        if (
            self.platform_tag != PROFILE_TARGETS[self.profile]
            or self.filename != filename
            or url.scheme != "https"
            or url.netloc != "github.com"
            or unquote(url.path)
            != "/astral-sh/python-build-standalone/releases/download/20260807/"
            + filename
            or url.query
            or url.fragment
        ):
            raise ValueError("standalone Python identity does not match its target")
        return self


class PythonArchiveCatalog(BaseModel):
    """Closed three-platform pins, attributed to immutable upstream metadata."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    source_repository: Literal["https://github.com/astral-sh/uv"]
    source_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    source_path: Literal["crates/uv-python/download-metadata.json"]
    archives: tuple[PythonArchivePin, ...] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def complete_targets(self) -> PythonArchiveCatalog:
        """Require one and only one reviewed interpreter per supported platform."""
        if {entry.profile for entry in self.archives} != PROFILE_TARGETS.keys():
            raise ValueError("standalone Python catalog must cover all three targets")
        return self

    def for_target(self, target: str) -> PythonArchivePin:
        """Return the exact target's interpreter; never select by the build host."""
        for entry in self.archives:
            if entry.platform_tag == target:
                return entry
        raise ValueError("no pinned standalone Python for this engine target")


def read_python_catalog(root: Path) -> PythonArchiveCatalog:
    """Bound and validate the source-controlled interpreter object inventory."""
    with (root / "python.json").open("rb") as stream:
        data = stream.read(16385)
    if len(data) > 16384:
        raise ValueError("standalone Python catalog exceeds its bound")
    return PythonArchiveCatalog.model_validate_json(data)


def file_digest(path: Path, *, limit: int, checkpoint: Checkpoint | None = None) -> str:
    """Hash a bounded local object without loading a complete archive into RAM."""
    _checkpoint(checkpoint)
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            _checkpoint(checkpoint)
            size += len(block)
            if size > limit:
                raise ValueError("native archive exceeds its object bound")
            digest.update(block)
    return digest.hexdigest()


def verify_python_archive(
    path: Path, pin: PythonArchivePin, *, checkpoint: Checkpoint | None = None
) -> None:
    """Check the interpreter basename, complete size and digest against its pin."""
    if (
        path.name != pin.filename
        or path.stat().st_size != pin.size_bytes
        or file_digest(path, limit=pin.size_bytes, checkpoint=checkpoint) != pin.sha256
    ):
        raise ValueError("standalone Python differs from the pinned object")


def locked_dependencies(lock: bytes) -> dict[str, tuple[Version, frozenset[str]]]:
    """Read a closed version/hash lock with no URLs, markers or index directives."""
    if len(lock) > 2 * 1024 * 1024:
        raise ValueError("native dependency lock exceeds its bound")
    entries = lock.decode().replace("\\\n", " ").splitlines()
    result: dict[str, tuple[Version, frozenset[str]]] = {}
    for entry in entries:
        stripped = entry.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = re.fullmatch(
            r"([a-zA-Z0-9][a-zA-Z0-9_.-]*)==([^\s;]+)((?:\s+--hash=sha256:[a-f0-9]{64})+)",
            stripped,
        )
        if match is None:
            raise ValueError(
                "native lock requires exact versions and SHA-256 hashes only"
            )
        name = canonicalize_name(match[1])
        if name in result:
            raise ValueError("duplicate native dependency")
        result[name] = (
            Version(match[2]),
            frozenset(re.findall(r"--hash=sha256:([a-f0-9]{64})", match[3])),
        )
    if not result or len(result) > 128:
        raise ValueError("native lock must contain a bounded complete dependency set")
    return result


class BuildManifest(BaseModel):
    """Strict payload identity, anchored by the operator's complete package hash."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    protocol_version: Literal[1]
    python: Literal["3.12.13"]
    python_archive: PythonArchivePin
    adapter_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    source_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    source_repository: Literal["https://github.com/adefossez/demucs.git"]
    platform_tag: str
    variant: Literal["cpu"]
    members: dict[str, Digest] = Field(min_length=1, max_length=256)
    features: tuple[str, ...] = Field(max_length=16)
    payload_sha256: Digest


def safe_member(name: str) -> PurePosixPath:
    """Refuse absolute paths, traversal and alternative separators before writing."""
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or "\\" in name
        or "\x00" in name
        or any(part in ("", ".", "..") for part in name.rstrip("/").split("/"))
    ):
        raise ValueError("native archive member is not a safe relative path")
    return path


def extract_python(
    archive_path: Path,
    destination: Path,
    pin: PythonArchivePin,
    *,
    checkpoint: Checkpoint | None = None,
) -> Path:
    """Extract only bounded regular files and in-root links from verified Python."""
    verify_python_archive(archive_path, pin, checkpoint=checkpoint)
    destination.mkdir(parents=True, exist_ok=False)
    seen: set[str] = set()
    expanded = 0
    with tarfile.open(archive_path, "r|gz") as archive:
        for entry in archive:
            _checkpoint(checkpoint)
            path = safe_member(entry.name)
            if path.parts[0] != "python" or path.as_posix() in seen:
                raise ValueError(
                    "standalone Python contains a foreign or duplicate member"
                )
            seen.add(path.as_posix())
            expanded += entry.size
            if len(seen) > 8192 or expanded > MAX_EXPANDED_PYTHON_BYTES:
                raise ValueError("standalone Python exceeds its expanded bound")
            if not (entry.isfile() or entry.isdir() or entry.issym()):
                raise ValueError(
                    "standalone Python contains an unsupported member type"
                )
            # The standard data filter resolves link targets and existing parent
            # links against the destination. Legitimate relative links survive;
            # escaping links and writes through them are refused.
            archive.extract(entry, destination, filter="data")
    executable = destination / pin.executable
    if (
        executable.is_symlink()
        or not executable.is_file()
        or not os.access(executable, os.X_OK)
    ):
        raise ValueError("standalone Python lacks its pinned executable")
    return executable


def _candidate_manifest(archive: zipfile.ZipFile) -> BuildManifest:
    entries = archive.infolist()
    names = [entry.filename for entry in entries]
    if len(names) > 256 or len(set(names)) != len(names):
        raise ValueError("candidate has excessive or duplicate members")
    if sum(entry.file_size for entry in entries) > MAX_PACKAGE_BYTES:
        raise ValueError("candidate payload exceeds its expanded bound")
    build = archive.getinfo("skulk_demucs/build.json")
    if build.file_size > 128 * 1024:
        raise ValueError("candidate manifest exceeds its bound")
    manifest = BuildManifest.model_validate_json(archive.read(build))
    content = manifest.model_dump(mode="json", exclude={"payload_sha256"})
    identity = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    metadata_root = "skulk_demucs_cpu-0.1.0.dist-info"
    expected = set(manifest.members) | {
        "skulk_demucs/build.json",
        f"{metadata_root}/METADATA",
        f"{metadata_root}/WHEEL",
        f"{metadata_root}/RECORD",
    }
    if (
        identity != manifest.payload_sha256
        or set(names) != expected
        or manifest.platform_tag not in TARGETS
        or manifest.python_archive.platform_tag != manifest.platform_tag
    ):
        raise ValueError("candidate manifest identity or member inventory disagrees")
    for entry in entries:
        safe_member(entry.filename)
        if not stat.S_ISREG(entry.external_attr >> 16):
            raise ValueError("candidate payload must contain regular files only")
    return manifest


def unpack_candidate(
    wheel: Path,
    destination: Path,
    expected_sha256: str,
    *,
    checkpoint: Checkpoint | None = None,
) -> BuildManifest:
    """Verify the complete object and every payload member before extracting it."""
    if (
        file_digest(wheel, limit=MAX_PACKAGE_BYTES, checkpoint=checkpoint)
        != expected_sha256
    ):
        raise ValueError("candidate package differs from its expected digest")
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(wheel) as archive:
        manifest = _candidate_manifest(archive)
        entries = archive.infolist()
        for entry in entries:
            relative = safe_member(entry.filename)
            if not stat.S_ISREG(entry.external_attr >> 16):
                raise ValueError("candidate payload must contain regular files only")
            output = destination.joinpath(*relative.parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with archive.open(entry) as source, output.open("xb") as target:
                while block := source.read(1024 * 1024):
                    _checkpoint(checkpoint)
                    digest.update(block)
                    target.write(block)
            if (
                entry.filename in manifest.members
                and digest.hexdigest() != manifest.members[entry.filename]
            ):
                raise ValueError("candidate member differs from its manifest digest")
        return manifest


def verify_site_packages(
    site: Path,
    wheels: tuple[Path, ...],
    *,
    checkpoint: Checkpoint | None = None,
) -> None:
    """Check expanded code/data against trusted wheels, never installed RECORDs.

    Pip may rewrite RECORD and add its small installer receipts. Everything else
    in site-packages, including .pth files and native libraries, must match an
    original wheel member. Unexpected files and symlinks are refused.
    """
    expected: dict[str, str] = {}
    generated: set[str] = set()
    expanded = 0
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            wheel_names: set[str] = set()
            for entry in archive.infolist():
                _checkpoint(checkpoint)
                path = safe_member(entry.filename)
                name = path.as_posix()
                if name in wheel_names:
                    raise ValueError("native dependency has duplicate members")
                wheel_names.add(name)
                if entry.is_dir():
                    continue
                if path.parts[0].endswith(".data"):
                    # The pinned set has only a SymPy manual outside purelib.
                    # No executable script, header or alternative import root
                    # may escape the explicitly verified native environment.
                    if name.endswith(".data/data/share/man/man1/isympy.1"):
                        continue
                    raise ValueError("unadmitted dependency installation scheme")
                if name.endswith(".dist-info/RECORD"):
                    generated.update(
                        (path.parent / suffix).as_posix()
                        for suffix in (
                            "RECORD",
                            "INSTALLER",
                            "REQUESTED",
                            "direct_url.json",
                        )
                    )
                    continue
                expanded += entry.file_size
                if expanded > MAX_INSTALLED_BYTES or len(expected) > 65536:
                    raise ValueError("installed dependency set exceeds its bound")
                with archive.open(entry) as stream:
                    hasher = hashlib.sha256()
                    while block := stream.read(1024 * 1024):
                        _checkpoint(checkpoint)
                        hasher.update(block)
                    digest = hasher.hexdigest()
                if name in expected and expected[name] != digest:
                    raise ValueError(
                        "native distributions overwrite different import bytes"
                    )
                expected[name] = digest
    actual: set[str] = set()
    for path in site.rglob("*"):
        _checkpoint(checkpoint)
        if path.is_symlink():
            raise ValueError("installed native tree contains an unexpected symlink")
        if path.is_dir():
            continue
        name = path.relative_to(site).as_posix()
        actual.add(name)
        if len(actual) > 66048:
            raise ValueError("installed native tree exceeds its member bound")
        if name not in generated and (
            name not in expected
            or file_digest(path, limit=MAX_INSTALLED_BYTES, checkpoint=checkpoint)
            != expected[name]
        ):
            raise ValueError(
                f"installed native tree differs from trusted package bytes: {name}"
            )
    if not expected.keys() <= actual:
        raise ValueError("installed native tree is incomplete")


def verify_python_tree(
    archive_path: Path,
    destination: Path,
    pin: PythonArchivePin,
    *,
    checkpoint: Checkpoint | None = None,
) -> Path:
    """Verify cached interpreter bytes and links before executing that interpreter.

    The complete pinned archive remains the authority. Neither a mutable local
    receipt nor the interpreter's own version/import report can authorize reuse.
    This checks the executable, standard library, bundled bootstrap wheel and
    every relative link, and refuses injected files throughout the base tree.
    """
    verify_python_archive(archive_path, pin, checkpoint=checkpoint)
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError("cached standalone Python root is missing or redirected")
    root = destination.resolve()
    expected: set[str] = set()
    seen: set[str] = set()
    expanded = 0
    with tarfile.open(archive_path, "r|gz") as archive:
        for entry in archive:
            _checkpoint(checkpoint)
            relative = safe_member(entry.name)
            name = relative.as_posix()
            if relative.parts[0] != "python" or name in seen:
                raise ValueError(
                    "standalone Python contains a foreign or duplicate member"
                )
            seen.add(name)
            expanded += entry.size
            if len(seen) > 8192 or expanded > MAX_EXPANDED_PYTHON_BYTES:
                raise ValueError("standalone Python exceeds its expanded bound")
            expected.add(name)
            expected.update(
                parent.as_posix()
                for parent in relative.parents
                if parent != PurePosixPath(".")
            )
            path = destination.joinpath(*relative.parts)
            if not path.resolve().is_relative_to(root):
                raise ValueError("cached standalone Python member escapes its root")
            if entry.issym():
                if not path.is_symlink() or os.readlink(path) != entry.linkname:
                    raise ValueError(
                        "cached standalone Python link differs from its archive"
                    )
            elif entry.isdir():
                if path.is_symlink() or not path.is_dir():
                    raise ValueError(
                        "cached standalone Python directory is missing or redirected"
                    )
            elif entry.isfile():
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or path.stat().st_size != entry.size
                ):
                    raise ValueError(
                        "cached standalone Python file is missing or redirected"
                    )
                stream = archive.extractfile(entry)
                if stream is None:
                    raise ValueError("standalone Python archive lacks a regular file")
                digest = hashlib.sha256()
                with stream:
                    while block := stream.read(1024 * 1024):
                        _checkpoint(checkpoint)
                        digest.update(block)
                if (
                    file_digest(path, limit=entry.size, checkpoint=checkpoint)
                    != digest.hexdigest()
                ):
                    raise ValueError(
                        "cached standalone Python bytes differ from its archive"
                    )
            else:
                raise ValueError(
                    "standalone Python contains an unsupported member type"
                )
    pending = [destination]
    while pending:
        directory = pending.pop()
        # Iteration stays bounded even if a damaged directory has many injected
        # files. Directory symlinks are checked above and never followed here.
        with os.scandir(directory) as entries:
            for member in entries:
                _checkpoint(checkpoint)
                path = Path(member.path)
                if path.relative_to(destination).as_posix() not in expected:
                    raise ValueError(
                        "cached standalone Python contains an injected member"
                    )
                if member.is_dir(follow_symlinks=False):
                    pending.append(path)
    executable = destination / pin.executable
    if (
        executable.is_symlink()
        or not executable.is_file()
        or not os.access(executable, os.X_OK)
    ):
        raise ValueError("standalone Python lacks its pinned executable")
    return executable


def verify_cached_payload(
    wheel: Path,
    destination: Path,
    expected_sha256: str,
    *,
    checkpoint: Checkpoint | None = None,
) -> BuildManifest:
    """Check cached source files against a complete independently pinned package.

    The extracted lock, dependency wheels and interpreter archive cannot become
    their own authorities. No file from this tree is executed by this function.
    """
    if (
        file_digest(wheel, limit=MAX_PACKAGE_BYTES, checkpoint=checkpoint)
        != expected_sha256
    ):
        raise ValueError("candidate package differs from its expected digest")
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError("cached package payload is missing or redirected")
    root = destination.resolve()
    expected: set[str] = set()
    with zipfile.ZipFile(wheel) as archive:
        manifest = _candidate_manifest(archive)
        for entry in archive.infolist():
            _checkpoint(checkpoint)
            relative = safe_member(entry.filename)
            name = relative.as_posix()
            expected.add(name)
            expected.update(
                parent.as_posix()
                for parent in relative.parents
                if parent != PurePosixPath(".")
            )
            path = destination.joinpath(*relative.parts)
            if (
                path.is_symlink()
                or not path.is_file()
                or not path.resolve().is_relative_to(root)
            ):
                raise ValueError("cached package payload file is missing or redirected")
            if path.stat().st_size != entry.file_size:
                raise ValueError(
                    "cached package payload differs from its trusted bytes"
                )
            digest = manifest.members.get(name)
            if digest is None:
                with archive.open(entry) as source:
                    hasher = hashlib.sha256()
                    while block := source.read(1024 * 1024):
                        _checkpoint(checkpoint)
                        hasher.update(block)
                    digest = hasher.hexdigest()
            if (
                file_digest(path, limit=entry.file_size, checkpoint=checkpoint)
                != digest
            ):
                raise ValueError(
                    "cached package payload differs from its trusted bytes"
                )
    pending = [destination]
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                _checkpoint(checkpoint)
                path = Path(entry.path)
                if path.relative_to(destination).as_posix() not in expected:
                    raise ValueError(
                        "cached package payload contains an injected member"
                    )
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                elif entry.is_symlink():
                    raise ValueError(
                        "cached package payload contains a redirected member"
                    )
    return manifest


def verify_cached_runtime(
    wheel: Path,
    root: Path,
    expected_sha256: str,
    expected_payload_sha256: str,
    *,
    checkpoint: Checkpoint | None = None,
) -> tuple[Path, BuildManifest]:
    """Verify a stable cached environment before any of its Python code runs.

    Complete package/interpreter identities and expanded import bytes are the
    authorities. Local completion receipts are not trusted. This proves library
    custody only; selected-device probing and model readiness remain separate.
    """
    if root.is_symlink() or not root.is_dir():
        raise ValueError("cached native runtime generation is missing or redirected")
    payload = root / "payload"
    manifest = verify_cached_payload(
        wheel, payload, expected_sha256, checkpoint=checkpoint
    )
    if manifest.payload_sha256 != expected_payload_sha256:
        raise ValueError("cached native runtime belongs to a different build")
    base = verify_python_tree(
        payload / "skulk_demucs/python" / manifest.python_archive.filename,
        root / "interpreter",
        manifest.python_archive,
        checkpoint=checkpoint,
    )
    runtime = root / "runtime"
    configuration_path = runtime / "pyvenv.cfg"
    if runtime.is_symlink() or not runtime.is_dir() or configuration_path.is_symlink():
        raise ValueError("cached native virtual environment is missing or redirected")
    with configuration_path.open("rb") as stream:
        raw = stream.read(16385)
    if len(raw) > 16384:
        raise ValueError(
            "cached native virtual environment configuration exceeds its bound"
        )
    configuration: dict[str, str] = {}
    for line in raw.decode().splitlines():
        if not line:
            continue
        key, separator, value = line.partition(" = ")
        if (
            not separator
            or key in configuration
            or key
            not in {
                "home",
                "include-system-site-packages",
                "version",
                "executable",
                "command",
            }
        ):
            raise ValueError(
                "cached native virtual environment has unrecognized configuration"
            )
        configuration[key] = value
    if (
        configuration.get("include-system-site-packages") != "false"
        or configuration.get("version") != PYTHON_VERSION
        # CPython searches relative to the configured spelling of home. A
        # foreign/bin alias may resolve to this exact executable while selecting
        # foreign/lib instead. Only the known base spelling or its canonical
        # spelling is admitted, without resolving caller-controlled aliases.
        or configuration.get("home") not in {str(base.parent), str(base.parent.resolve())}
        or configuration.get("executable") not in {str(base), str(base.resolve())}
    ):
        raise ValueError(
            "cached native virtual environment selected a foreign interpreter"
        )
    binary_root = runtime / "bin"
    if binary_root.is_symlink() or not binary_root.is_dir():
        raise ValueError("cached native executable directory is missing or redirected")
    for count, entry in enumerate(binary_root.iterdir(), start=1):
        _checkpoint(checkpoint)
        if count > 512:
            raise ValueError("cached native executable directory exceeds its bound")
        # CPython honors adjacent ._pth files even in isolated mode, and a
        # bin/pyvenv.cfg takes precedence over the root configuration checked
        # above. Case folding also covers case-insensitive supported hosts.
        name = entry.name.casefold()
        if name.endswith("._pth") or name == "pyvenv.cfg":
            raise ValueError(
                "cached native executable has an untrusted startup override"
            )
    native = binary_root / "python"
    site = runtime / "lib/python3.12/site-packages"
    if (
        not native.is_file()
        or native.resolve() != base.resolve()
        or not os.access(native, os.X_OK)
    ):
        raise ValueError(
            "cached native executable does not select its verified interpreter"
        )
    if (
        site.is_symlink()
        or not site.is_dir()
        or not site.resolve().is_relative_to(runtime.resolve())
    ):
        raise ValueError("cached native import root is missing or redirected")
    dependencies = tuple(sorted((payload / "skulk_demucs/wheels").glob("*.whl")))
    verify_site_packages(site, (wheel, *dependencies), checkpoint=checkpoint)
    return native, manifest


class CommandRunner(Protocol):
    """Execute owned bootstrap commands with cancellation and bounded diagnostics."""

    def __call__(
        self,
        arguments: Sequence[str],
        *,
        environment: dict[str, str],
        timeout_seconds: int,
        capture: bool,
    ) -> str:
        """Run one bounded command; return stdout only when capture is requested."""
        ...


class InterpreterFacts(BaseModel):
    """Minimal facts queried from the isolated interpreter, not the controller."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    version: str
    platform: str
    machine: str
    prefix: str
    purelib: str


def install_fresh_runtime(
    wheel: Path,
    output: Path,
    expected_sha256: str,
    *,
    run: CommandRunner,
    checkpoint: Checkpoint | None = None,
) -> Path:
    """Bootstrap one fresh stable generation from sealed offline package bytes.

    ``run`` owns command supervision, timeouts and diagnostic bounds; ``checkpoint``
    may raise the owner's cancellation signal during filesystem verification.
    Return the private interpreter only after full expanded-tree verification.
    Failed output remains owned by the caller for supervised cleanup. This does
    not publish a cache pointer, advertise readiness or download model weights.
    """
    _checkpoint(checkpoint)
    output.mkdir(parents=True, exist_ok=False)
    payload = output / "payload"
    manifest = unpack_candidate(wheel, payload, expected_sha256, checkpoint=checkpoint)
    host_target = {
        ("Darwin", "arm64"): "macosx_14_0_arm64",
        ("Linux", "x86_64"): "manylinux_2_28_x86_64",
        ("Linux", "aarch64"): "manylinux_2_28_aarch64",
    }.get((platform.system(), platform.machine()))
    if host_target != manifest.platform_tag:
        raise ValueError("candidate cannot execute on this OS/CPU architecture")
    interpreter = extract_python(
        payload / "skulk_demucs" / "python" / manifest.python_archive.filename,
        output / "interpreter",
        manifest.python_archive,
        checkpoint=checkpoint,
    )
    environment = {
        name: value
        for name, value in os.environ.items()
        if name in {"PATH", "HOME", "SYSTEMROOT", "TMPDIR", "TMP", "TEMP", "LANG"}
    } | {
        "PIP_CONFIG_FILE": os.devnull,
        "PIP_NO_INDEX": "1",
        "PIP_NO_CACHE_DIR": "1",
        "PIP_NO_COMPILE": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    runtime = output / "runtime"
    _ = run(
        [str(interpreter), "-I", "-B", "-m", "venv", "--without-pip", str(runtime)],
        environment=environment,
        timeout_seconds=120,
        capture=False,
    )
    native = runtime / "bin" / "python"
    facts_code = (
        "import json,platform,sys,sysconfig; "
        "print(json.dumps(dict(version=platform.python_version(),platform=sys.platform,"
        "machine=platform.machine(),prefix=sys.prefix,purelib=sysconfig.get_path('purelib'))))"
    )
    facts = InterpreterFacts.model_validate_json(
        run(
            [str(native), "-I", "-B", "-c", facts_code],
            environment=environment,
            timeout_seconds=30,
            capture=True,
        )
    )
    site = Path(facts.purelib)
    if (
        facts.version != PYTHON_VERSION
        or Path(facts.prefix) != runtime
        or not site.is_relative_to(runtime)
    ):
        raise ValueError("candidate selected the wrong private interpreter")
    offline = payload / "skulk_demucs"
    lock = (offline / "requirements.lock").read_bytes()
    locked_dependencies(lock)
    pip_wheels = tuple(
        (
            output
            / "interpreter"
            / "python"
            / "lib"
            / "python3.12"
            / "ensurepip"
            / "_bundled"
        ).glob("*.whl")
    )
    if len(pip_wheels) != 1:
        raise ValueError("bundled interpreter has an unexpected bootstrap inventory")
    # ensurepip's installation compiles bytecode independently of -B and strips
    # PIP_* settings. Run the already verified bundled pip wheel as a zip import
    # instead: the final environment contains only sealed runtime distributions.
    pip_code = (
        "import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); "
        "runpy.run_module('pip',run_name='__main__')"
    )
    commands = (
        [
            "--require-hashes",
            "--find-links",
            str(offline / "wheels"),
            "-r",
            str(offline / "requirements.lock"),
        ],
        [str(wheel.resolve())],
    )
    for arguments in commands:
        _checkpoint(checkpoint)
        _ = run(
            [
                str(native),
                "-I",
                "-B",
                "-c",
                pip_code,
                str(pip_wheels[0]),
                "install",
                "--no-index",
                "--no-compile",
                *arguments,
            ],
            environment=environment,
            timeout_seconds=600,
            capture=False,
        )
    verified, verified_manifest = verify_cached_runtime(
        wheel, output, expected_sha256, manifest.payload_sha256, checkpoint=checkpoint
    )
    if verified != native or verified_manifest != manifest:
        raise ValueError("native installation changed its source identity")
    (output / "installation.json").write_text(
        json.dumps(
            {
                "package_sha256": expected_sha256,
                "payload_sha256": manifest.payload_sha256,
                "python_archive_sha256": manifest.python_archive.sha256,
                "platform_tag": manifest.platform_tag,
                "offline": True,
                "verified_expanded_tree": True,
                "python": facts.version,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )
    return native
