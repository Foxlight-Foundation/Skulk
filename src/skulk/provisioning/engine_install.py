"""Install, activate, roll back and remove engine package generations.

A verified manifest and the files it names become a **generation**: a
directory named by the manifest's digest that holds the unpacked engine, the
adapter, any pinned requirement files and the signed manifest itself.
Generations of one engine sit side by side. A small pointer file names the
one new placements use and the one kept for rollback; changing it touches no
generation and needs no restart, and anything already running keeps the
generation it was started from.

The files come from an ``EnginePackageSource``. One reads a directory, which
is how a cluster with no route to the store installs; one fetches over HTTPS.
Either way every file is checked against its signed size and digest before it
is used, and a generation appears under its final name only when it is
complete.

Nothing here consults trust. The caller verifies the manifest first; an
installed generation is later checked against the inventory taken at install,
so it keeps working offline and after its manifest expires.
"""

import contextlib
import hashlib
import os
import secrets
import shutil
import stat
import tarfile
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, Protocol, final

import httpx
from filelock import BaseFileLock, FileLock
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from skulk.extensions.runtime_artifacts import Digest
from skulk.provisioning.engine_packages import (
    EngineLane,
    EnginePackageClaims,
    EnginePackageRefusal,
    EnginePackageRefusedError,
    VerifiedEnginePackage,
    installed_engine_claims,
    verify_engine_file,
)
from skulk.shared.backends import (
    engine_packages_disabled,
    engine_packages_off_phrase,
)
from skulk.shared.constants import SKULK_ENGINES_DIR

ENGINE_PACKAGES_DIR: Final = SKULK_ENGINES_DIR / "packages"
"""Root of installed engine packages: ``<engine id>/<lane>/<manifest digest>``."""

_SPACE_RESERVE_BYTES: Final = 1024**3
"""Free space left untouched after an installation's own needs."""

_BLOCK_BYTES: Final = 1024 * 1024
_MAX_TREE_ENTRIES: Final = 50_000
_INVENTORY_BYTES: Final = 16 * 1024 * 1024
_LOCK_SECONDS: Final = 600.0
_MANIFEST_NAME: Final = "manifest.json"
_INVENTORY_NAME: Final = "installed.json"
_POINTER_NAME: Final = "active.json"
_ENGINE_DIR: Final = "engine"
_ADAPTER_DIR: Final = "adapter"
_REQUIREMENTS_DIR: Final = "requirements"
_TREE_DIRS: Final = (_ENGINE_DIR, _ADAPTER_DIR, _REQUIREMENTS_DIR)
_STAGING_PREFIX: Final = ".staging-"
_DAMAGED_PREFIX: Final = ".damaged-"
_REMOVING_PREFIX: Final = ".removing-"
_LEFTOVER_PREFIXES: Final = (_STAGING_PREFIX, _DAMAGED_PREFIX, _REMOVING_PREFIX)


def _refuse(code: EnginePackageRefusal, message: str) -> EnginePackageRefusedError:
    return EnginePackageRefusedError(code, message)


@final
@dataclass(frozen=True)
class EngineFileRequest:
    """One file an installation needs, as the manifest signed it.

    Attributes:
        filename: The file's name.
        size: Exact size in bytes.
        sha256: Digest of the bytes.
        url: The file's own HTTPS address when it lives outside the package
            host; ``None`` for the package's own archive and adapter.
    """

    filename: str
    size: int
    sha256: str
    url: str | None = None


class EnginePackageSource(Protocol):
    """Where an installation obtains the files a manifest names."""

    def fetch(
        self, request: EngineFileRequest, destination: Path, stop: threading.Event
    ) -> None:
        """Write the requested file to ``destination``.

        The installer checks size and digest afterwards, so a source need not.

        Raises:
            EnginePackageRefusedError: ``source_unavailable`` when the file
                cannot be obtained, ``install_cancelled`` when ``stop`` is set.
        """
        ...


@final
class DirectoryPackageSource:
    """Read package files from a local directory, with no network access.

    This is the path for an air-gapped cluster: the signed manifest, the
    archive, the adapter and every pinned requirement are carried in as
    files, and the same checks apply as for a download.
    """

    def __init__(self, directory: Path) -> None:
        """Serve files found directly inside ``directory``."""
        self.directory: Final = directory

    def fetch(
        self, request: EngineFileRequest, destination: Path, stop: threading.Event
    ) -> None:
        """Copy the named file out of the directory."""
        origin = self.directory / request.filename
        if origin.is_symlink() or not origin.is_file():
            raise _refuse(
                "source_unavailable", f"{request.filename} is not in the import folder"
            )
        try:
            with origin.open("rb") as reader, destination.open("wb") as writer:
                while block := reader.read(_BLOCK_BYTES):
                    if stop.is_set():
                        raise _refuse("install_cancelled", "installation cancelled")
                    writer.write(block)
        except OSError as error:
            raise _refuse(
                "source_unavailable",
                f"{request.filename} could not be copied ({error.strerror})",
            ) from None


@final
class HttpPackageSource:
    """Fetch package files over HTTPS.

    The package's own archive and adapter are read from ``base_url``. A pinned
    requirement is read from the exact address its manifest entry signs.
    Redirects are followed: the signed digest, not the address that finally
    serves the bytes, decides whether they are accepted.
    """

    def __init__(
        self,
        base_url: str,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        """Configure the source without touching the network.

        Args:
            base_url: HTTPS directory holding the package's archive and adapter.
            transport: Network transport, injectable for tests.
            timeout_seconds: Connect and between-bytes timeout.

        Raises:
            ValueError: ``base_url`` is not an HTTPS address.
        """
        if not base_url.startswith("https://"):
            raise ValueError("package host must be an HTTPS address")
        self.base_url: Final = base_url.rstrip("/") + "/"
        self._transport: Final = transport
        self._timeout_seconds: Final = timeout_seconds

    def fetch(
        self, request: EngineFileRequest, destination: Path, stop: threading.Event
    ) -> None:
        """Stream the file to ``destination``, stopping at its signed size."""
        address = request.url or self.base_url + request.filename
        received = 0
        try:
            with (
                httpx.Client(
                    transport=self._transport,
                    timeout=self._timeout_seconds,
                    follow_redirects=True,
                ) as client,
                client.stream("GET", address) as response,
                destination.open("wb") as writer,
            ):
                if response.status_code != 200:
                    raise _refuse(
                        "source_unavailable",
                        f"{request.filename} was answered with HTTP "
                        f"{response.status_code}",
                    )
                for block in response.iter_bytes(_BLOCK_BYTES):
                    if stop.is_set():
                        raise _refuse("install_cancelled", "installation cancelled")
                    received += len(block)
                    if received > request.size:
                        raise _refuse(
                            "archive_corrupt",
                            f"{request.filename} is larger than the manifest signed",
                        )
                    writer.write(block)
        except (httpx.HTTPError, OSError) as error:
            raise _refuse(
                "source_unavailable",
                f"{request.filename} could not be fetched ({type(error).__name__})",
            ) from None


class _Contract(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


class _InventoryEntry(_Contract):
    path: str = Field(description="POSIX path relative to the generation directory.")
    size: int = Field(ge=0)
    sha256: Digest | None = Field(description="Content digest; null for a link.")
    executable: bool
    link: str | None = Field(description="Link target, for a symbolic link.")


class _Inventory(_Contract):
    """What a completed installation contained, bound to its manifest."""

    schema_version: int = Field(ge=1, le=1)
    manifest_digest: Digest
    installed_at: int = Field(gt=0)
    files: tuple[_InventoryEntry, ...] = Field(max_length=_MAX_TREE_ENTRIES)


class _Pointer(_Contract):
    """Which generation new placements use, and which is kept for rollback."""

    schema_version: int = Field(ge=1, le=1)
    active: Digest
    previous: Digest | None


@final
@dataclass(frozen=True)
class InstalledEngine:
    """One installed, intact generation.

    Attributes:
        root: The generation directory.
        claims: The manifest claims it was installed from.
        digest: The manifest digest, which is also the directory name and the
            build identity stamped on a placement.
        executable: The engine executable.
        adapter: The adapter program.
        requirements: Pinned requirement files, by file name.
    """

    root: Path
    claims: EnginePackageClaims
    digest: str
    executable: Path
    adapter: Path
    requirements: dict[str, Path]


def _lane_directory(root: Path, engine_id: str, lane: EngineLane) -> Path:
    return root / engine_id / lane


def _private_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)


def _lock(lane_directory: Path) -> BaseFileLock:
    _private_directory(lane_directory)
    return FileLock(str(lane_directory / "install.lock"), timeout=_LOCK_SECONDS)


def _sweep(lane_directory: Path) -> None:
    """Remove what an interrupted installation or removal left behind.

    Called only while holding the lane's lock, so nothing it removes is in
    use by another installation.
    """
    for entry in lane_directory.iterdir():
        if entry.name.startswith(_LEFTOVER_PREFIXES):
            shutil.rmtree(entry, ignore_errors=True)


def _write_atomic(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}")
    try:
        with temporary.open("wb") as writer:
            writer.write(content)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        while block := reader.read(_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _tree(generation: Path, *, hashes: bool) -> tuple[_InventoryEntry, ...]:
    """List every file and link under the generation's content directories."""
    entries: list[_InventoryEntry] = []
    for top in _TREE_DIRS:
        base = generation / top
        if not base.is_dir():
            continue
        for directory, names, files in os.walk(base, followlinks=False):
            names.sort()
            for name in sorted(files) + [
                name for name in names if (Path(directory) / name).is_symlink()
            ]:
                path = Path(directory) / name
                relative = path.relative_to(generation).as_posix()
                info = path.lstat()
                if stat.S_ISLNK(info.st_mode):
                    entries.append(
                        _InventoryEntry(
                            path=relative,
                            size=0,
                            sha256=None,
                            executable=False,
                            link=os.readlink(path),
                        )
                    )
                elif stat.S_ISREG(info.st_mode):
                    entries.append(
                        _InventoryEntry(
                            path=relative,
                            size=info.st_size,
                            sha256=_hash_file(path) if hashes else None,
                            executable=bool(info.st_mode & stat.S_IXUSR),
                            link=None,
                        )
                    )
                else:
                    raise _refuse(
                        "installation_damaged",
                        f"{relative} is not a regular file or a link",
                    )
                if len(entries) > _MAX_TREE_ENTRIES:
                    raise _refuse(
                        "archive_corrupt", "the package holds too many files"
                    )
    return tuple(sorted(entries, key=lambda entry: entry.path))


def _safe_member_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise _refuse("archive_corrupt", "the archive names a path outside itself")
    return path


def _unpack_tar(
    archive_path: Path, destination: Path, limit: int, stop: threading.Event
) -> None:
    total = 0
    count = 0
    try:
        with tarfile.open(archive_path, "r:*") as archive:
            for member in archive:
                if stop.is_set():
                    raise _refuse("install_cancelled", "installation cancelled")
                # Refused here by name: the data filter below would quietly
                # rewrite an absolute path to a relative one.
                _safe_member_path(member.name)
                count += 1
                total += member.size if member.isfile() else 0
                if total > limit or count > _MAX_TREE_ENTRIES:
                    raise _refuse(
                        "archive_corrupt",
                        "the archive unpacks to more than the manifest declared",
                    )
                # The data filter refuses absolute paths, paths and links that
                # leave the destination, device files and set-id bits.
                archive.extract(member, destination, filter="data")
    except (tarfile.TarError, OSError) as error:
        raise _refuse(
            "archive_corrupt", f"the archive could not be unpacked ({type(error).__name__})"
        ) from None


def _unpack_zip(
    archive_path: Path, destination: Path, limit: int, stop: threading.Event
) -> None:
    total = 0
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            if len(members) > _MAX_TREE_ENTRIES:
                raise _refuse("archive_corrupt", "the package holds too many files")
            for member in members:
                if stop.is_set():
                    raise _refuse("install_cancelled", "installation cancelled")
                relative = _safe_member_path(member.filename)
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise _refuse("archive_corrupt", "the zip archive holds a link")
                target = destination.joinpath(*relative.parts)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                total += member.file_size
                if total > limit:
                    raise _refuse(
                        "archive_corrupt",
                        "the archive unpacks to more than the manifest declared",
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                written = 0
                with archive.open(member) as reader, target.open("wb") as writer:
                    while block := reader.read(_BLOCK_BYTES):
                        written += len(block)
                        if written > member.file_size:
                            raise _refuse(
                                "archive_corrupt", "a zip member exceeds its stated size"
                            )
                        writer.write(block)
                target.chmod(0o700 if mode & stat.S_IXUSR else 0o600)
    except (zipfile.BadZipFile, OSError) as error:
        raise _refuse(
            "archive_corrupt", f"the archive could not be unpacked ({type(error).__name__})"
        ) from None


def _installed(generation: Path, claims: EnginePackageClaims, digest: str) -> InstalledEngine:
    return InstalledEngine(
        root=generation,
        claims=claims,
        digest=digest,
        executable=generation / _ENGINE_DIR / claims.archive.executable,
        adapter=generation / _ADAPTER_DIR / claims.adapter.filename,
        requirements={
            requirement.filename: generation / _REQUIREMENTS_DIR / requirement.filename
            for requirement in claims.external_requirements
        },
    )


def verify_installed_engine(generation: Path, *, deep: bool = True) -> InstalledEngine:
    """Check an installed generation against the inventory taken at install.

    The directory name must be the digest of the manifest it holds, and its
    files must be exactly the ones recorded when it was installed. No trust
    document and no clock is involved.

    Args:
        generation: The generation directory.
        deep: Hash every file. Without it only names, sizes, links and
            executable bits are compared, which is enough to notice a
            missing or truncated file quickly.

    Returns:
        The intact generation.

    Raises:
        EnginePackageRefusedError: ``generation_missing`` when the directory
            does not exist, ``installation_damaged`` when it differs from
            its inventory.
    """
    if not generation.is_dir() or generation.is_symlink():
        raise _refuse("generation_missing", "that engine generation is not installed")
    try:
        document = (generation / _MANIFEST_NAME).read_bytes()
        raw_inventory = (generation / _INVENTORY_NAME).read_bytes()
    except OSError:
        raise _refuse(
            "installation_damaged", "the generation has lost its manifest or inventory"
        ) from None
    claims, digest = installed_engine_claims(document)
    if digest != generation.name or len(raw_inventory) > _INVENTORY_BYTES:
        raise _refuse(
            "installation_damaged", "the generation does not match its manifest"
        )
    try:
        inventory = _Inventory.model_validate_json(raw_inventory)
    except ValidationError:
        raise _refuse(
            "installation_damaged", "the generation's inventory is not readable"
        ) from None
    recorded = inventory.files
    if not deep:
        recorded = tuple(
            entry.model_copy(update={"sha256": None}) for entry in inventory.files
        )
    if inventory.manifest_digest != digest or _tree(generation, hashes=deep) != recorded:
        raise _refuse(
            "installation_damaged",
            "the generation's files differ from what was installed",
        )
    return _installed(generation, claims, digest)


def _requests(claims: EnginePackageClaims) -> tuple[EngineFileRequest, ...]:
    return (
        EngineFileRequest(
            claims.archive.filename, claims.archive.size, claims.archive.sha256
        ),
        EngineFileRequest(
            claims.adapter.filename, claims.adapter.size, claims.adapter.sha256
        ),
        *(
            EngineFileRequest(
                requirement.filename,
                requirement.size,
                requirement.sha256,
                requirement.url,
            )
            for requirement in claims.external_requirements
        ),
    )


def _stage(
    package: VerifiedEnginePackage,
    source: EnginePackageSource,
    staging: Path,
    stop: threading.Event,
) -> None:
    """Fetch, check and lay out one generation inside ``staging``."""
    claims = package.claims
    downloads = staging / "download"
    for name in (*_TREE_DIRS, "download"):
        _private_directory(staging / name)
    fetched: dict[str, Path] = {}
    for request in _requests(claims):
        if stop.is_set():
            raise _refuse("install_cancelled", "installation cancelled")
        if request.filename in fetched:
            raise _refuse("manifest_invalid", "the manifest names one file twice")
        path = downloads / request.filename
        source.fetch(request, path, stop)
        verify_engine_file(path, size=request.size, sha256=request.sha256)
        fetched[request.filename] = path
    archive_path = fetched[claims.archive.filename]
    engine = staging / _ENGINE_DIR
    if claims.archive.filename.endswith(".zip"):
        _unpack_zip(archive_path, engine, claims.archive.unpacked_size, stop)
    else:
        _unpack_tar(archive_path, engine, claims.archive.unpacked_size, stop)
    executable = engine / claims.archive.executable
    try:
        inside = executable.resolve(strict=True).is_relative_to(engine.resolve())
    except OSError:
        inside = False
    if not inside or not executable.is_file() or not os.access(executable, os.X_OK):
        raise _refuse(
            "archive_corrupt",
            f"the archive holds no executable at {claims.archive.executable}",
        )
    adapter = staging / _ADAPTER_DIR / claims.adapter.filename
    os.replace(fetched[claims.adapter.filename], adapter)
    adapter.chmod(0o700 if claims.adapter.kind == "native" else 0o600)
    for requirement in claims.external_requirements:
        kept = staging / _REQUIREMENTS_DIR / requirement.filename
        os.replace(fetched[requirement.filename], kept)
        kept.chmod(0o600)
    shutil.rmtree(downloads)
    _write_atomic(staging / _MANIFEST_NAME, package.document)
    inventory = _Inventory(
        schema_version=1,
        manifest_digest=package.digest,
        installed_at=int(time.time()),
        files=_tree(staging, hashes=True),
    )
    _write_atomic(staging / _INVENTORY_NAME, inventory.model_dump_json().encode())


def install_engine_package(
    package: VerifiedEnginePackage,
    source: EnginePackageSource,
    *,
    root: Path = ENGINE_PACKAGES_DIR,
    stop: threading.Event | None = None,
) -> InstalledEngine:
    """Install one verified package as a generation, beside any others.

    Installing does not activate. An intact generation that is already
    installed is returned without fetching anything. A damaged one is set
    aside and installed again.

    Args:
        package: A manifest that passed ``verify_engine_manifest`` and
            ``check_engine_host``.
        source: Where the package's files come from.
        root: Root of installed engine packages.
        stop: Set to cancel; the partial installation is removed.

    Returns:
        The installed generation.

    Raises:
        EnginePackageRefusedError: ``insufficient_space``,
            ``source_unavailable``, ``archive_corrupt`` or
            ``install_cancelled``. Nothing is left under the generation's
            final name in any of these cases. ``packages_disabled`` when
            engine packages are turned off on this node; nothing is fetched
            or written.

    Side effects:
        Writes under ``root``. Holds a per-engine, per-lane file lock for the
        duration, so two installations of the same engine never overlap.
    """
    if engine_packages_disabled():
        raise _refuse(
            "packages_disabled",
            "engine packages are turned off on this node "
            f"({engine_packages_off_phrase()})",
        )
    cancel = stop if stop is not None else threading.Event()
    claims = package.claims
    lane_directory = _lane_directory(root, claims.engine_id, claims.lane)
    target = lane_directory / package.digest
    with _lock(lane_directory):
        _sweep(lane_directory)
        if target.exists():
            with contextlib.suppress(EnginePackageRefusedError):
                return verify_installed_engine(target)
        needed = (
            claims.archive.unpacked_size
            + sum(request.size for request in _requests(claims))
            + _SPACE_RESERVE_BYTES
        )
        free = shutil.disk_usage(lane_directory).free
        if free < needed:
            raise _refuse(
                "insufficient_space",
                f"the installation needs {needed} bytes free and the engines "
                f"folder has {free}",
            )
        tag = f"{package.digest[:16]}-{secrets.token_hex(4)}"
        staging = lane_directory / f"{_STAGING_PREFIX}{tag}"
        _private_directory(staging)
        try:
            _stage(package, source, staging, cancel)
            if target.exists():
                # Present but not intact: set it aside rather than write into
                # files a running engine may still have open.
                os.rename(target, lane_directory / f"{_DAMAGED_PREFIX}{tag}")
            os.rename(staging, target)
        finally:
            _sweep(lane_directory)
        return verify_installed_engine(target, deep=False)


def installed_engine_generations(
    engine_id: str, lane: EngineLane, *, root: Path = ENGINE_PACKAGES_DIR
) -> tuple[str, ...]:
    """The digests of every generation directory present for one engine lane."""
    lane_directory = _lane_directory(root, engine_id, lane)
    if not lane_directory.is_dir():
        return ()
    return tuple(
        sorted(
            entry.name
            for entry in lane_directory.iterdir()
            if len(entry.name) == 64 and entry.is_dir() and not entry.is_symlink()
        )
    )


def installed_engine_generation(
    engine_id: str,
    lane: EngineLane,
    digest: str,
    *,
    root: Path = ENGINE_PACKAGES_DIR,
) -> InstalledEngine:
    """One installed generation by its digest, checked quickly against its inventory.

    Raises:
        EnginePackageRefusedError: ``generation_missing`` when no such
            generation is installed, ``installation_damaged`` when it is not
            intact.
    """
    target = _lane_directory(root, engine_id, lane) / digest
    if (
        len(digest) != 64
        or not set(digest) <= set("0123456789abcdef")
        or not target.is_dir()
        or target.is_symlink()
    ):
        raise _refuse("generation_missing", "that engine generation is not installed")
    return verify_installed_engine(target, deep=False)


def _read_pointer(lane_directory: Path) -> _Pointer | None:
    try:
        return _Pointer.model_validate_json(
            (lane_directory / _POINTER_NAME).read_bytes()
        )
    except FileNotFoundError:
        return None
    except (OSError, ValidationError):
        raise _refuse(
            "installation_damaged", "the active-generation record is not readable"
        ) from None


def _write_pointer(lane_directory: Path, pointer: _Pointer) -> None:
    _write_atomic(lane_directory / _POINTER_NAME, pointer.model_dump_json().encode())


def rollback_engine_generation(
    engine_id: str, lane: EngineLane, *, root: Path = ENGINE_PACKAGES_DIR
) -> str | None:
    """The digest a rollback would return to, or ``None`` when none is kept.

    Read-only, for reports. An unreadable record reads as ``None`` here; the
    lane's active generation reports that damage itself.
    """
    try:
        pointer = _read_pointer(_lane_directory(root, engine_id, lane))
    except EnginePackageRefusedError:
        return None
    return None if pointer is None else pointer.previous


def active_engine_generation(
    engine_id: str, lane: EngineLane, *, root: Path = ENGINE_PACKAGES_DIR
) -> str | None:
    """The digest new placements use, or ``None`` when the lane has none.

    Read-only and unchecked: it says what the record names, not whether that
    generation is intact. An unreadable record reads as ``None``.
    """
    try:
        pointer = _read_pointer(_lane_directory(root, engine_id, lane))
    except EnginePackageRefusedError:
        return None
    return None if pointer is None else pointer.active


def active_engine(
    engine_id: str, lane: EngineLane, *, root: Path = ENGINE_PACKAGES_DIR
) -> InstalledEngine | None:
    """The generation new placements use, or ``None`` when none is active.

    The generation is checked quickly against its inventory, without hashing.

    Raises:
        EnginePackageRefusedError: ``installation_damaged`` or
            ``generation_missing`` when the active generation is unusable.
    """
    lane_directory = _lane_directory(root, engine_id, lane)
    pointer = _read_pointer(lane_directory)
    if pointer is None:
        return None
    return verify_installed_engine(lane_directory / pointer.active, deep=False)


def activate_engine_generation(
    engine_id: str,
    lane: EngineLane,
    digest: str,
    *,
    root: Path = ENGINE_PACKAGES_DIR,
) -> InstalledEngine:
    """Make an installed generation the one new placements use.

    The generation is fully hashed first. The one that was active becomes the
    rollback generation. Nothing running is touched and no restart is needed.

    Raises:
        EnginePackageRefusedError: ``generation_missing`` or
            ``installation_damaged``; the active generation is then unchanged.
    """
    lane_directory = _lane_directory(root, engine_id, lane)
    with _lock(lane_directory):
        installed = verify_installed_engine(lane_directory / digest)
        pointer = _read_pointer(lane_directory)
        if pointer is None or pointer.active != digest:
            _write_pointer(
                lane_directory,
                _Pointer(
                    schema_version=1,
                    active=digest,
                    previous=None if pointer is None else pointer.active,
                ),
            )
        return installed


def rollback_engine(
    engine_id: str, lane: EngineLane, *, root: Path = ENGINE_PACKAGES_DIR
) -> InstalledEngine:
    """Return to the generation that was active before the current one.

    The two generations swap places, so a second rollback undoes the first.

    Raises:
        EnginePackageRefusedError: ``generation_missing`` when there is no
            earlier generation, ``installation_damaged`` when it is not
            intact; the active generation is then unchanged.
    """
    lane_directory = _lane_directory(root, engine_id, lane)
    with _lock(lane_directory):
        pointer = _read_pointer(lane_directory)
        if pointer is None or pointer.previous is None:
            raise _refuse(
                "generation_missing", "there is no earlier generation to return to"
            )
        installed = verify_installed_engine(lane_directory / pointer.previous)
        _write_pointer(
            lane_directory,
            _Pointer(schema_version=1, active=pointer.previous, previous=pointer.active),
        )
        return installed


def remove_engine_generation(
    engine_id: str,
    lane: EngineLane,
    digest: str,
    *,
    root: Path = ENGINE_PACKAGES_DIR,
) -> None:
    """Delete one generation that is not the active one.

    Removing the rollback generation clears it from the record. Model data is
    never under this directory and is not affected.

    Raises:
        EnginePackageRefusedError: ``generation_active`` for the active
            generation, ``generation_missing`` when it is not installed.
    """
    lane_directory = _lane_directory(root, engine_id, lane)
    with _lock(lane_directory):
        pointer = _read_pointer(lane_directory)
        if pointer is not None and pointer.active == digest:
            raise _refuse(
                "generation_active",
                "the active generation cannot be removed; activate another first",
            )
        target = lane_directory / digest
        if len(digest) != 64 or not target.is_dir() or target.is_symlink():
            raise _refuse(
                "generation_missing", "that engine generation is not installed"
            )
        doomed = lane_directory / (
            f"{_REMOVING_PREFIX}{digest[:16]}-{secrets.token_hex(4)}"
        )
        os.rename(target, doomed)
        if pointer is not None and pointer.previous == digest:
            _write_pointer(
                lane_directory,
                _Pointer(schema_version=1, active=pointer.active, previous=None),
            )
        shutil.rmtree(doomed, ignore_errors=True)


def remove_engine_package(
    engine_id: str, lane: EngineLane, *, root: Path = ENGINE_PACKAGES_DIR
) -> None:
    """Delete every generation of one engine lane and its active-generation record.

    The record goes first, so nothing is ever pointed at a generation that is
    being deleted. A lane whose record is unreadable can still be removed,
    which is how a damaged package is cleared. Model data is never under this
    directory and is not affected.

    Raises:
        EnginePackageRefusedError: ``generation_missing`` when the lane holds
            no generation and no record.
    """
    lane_directory = _lane_directory(root, engine_id, lane)
    absent = _refuse("generation_missing", "that engine package is not installed")
    if not lane_directory.is_dir() or lane_directory.is_symlink():
        raise absent
    with _lock(lane_directory):
        _sweep(lane_directory)
        generations = installed_engine_generations(engine_id, lane, root=root)
        record = lane_directory / _POINTER_NAME
        if not generations and not record.exists():
            raise absent
        record.unlink(missing_ok=True)
        for digest in generations:
            doomed = lane_directory / (
                f"{_REMOVING_PREFIX}{digest[:16]}-{secrets.token_hex(4)}"
            )
            os.rename(lane_directory / digest, doomed)
            shutil.rmtree(doomed, ignore_errors=True)
