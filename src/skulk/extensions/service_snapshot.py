"""Copy the exact existing Skulk runtime into stable, verified service storage."""

import asyncio
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import stat
import sys
import sysconfig
import venv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import final
from uuid import uuid4

from packaging.utils import canonicalize_name
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from skulk.extensions import service_bootstrap
from skulk.extensions.runtime_artifacts import Digest, QualifiedHost, measure_host
from skulk.extensions.runtime_files import (
    RuntimeLock,
    is_desktop_metadata,
    no_other_account_can_change,
    private_directory,
    read_private,
    write_private,
)
from skulk.extensions.runtime_install import finish_runtime_work
from skulk.extensions.runtime_integrity import copied_path_excess
from skulk.utils.dashboard_path import find_resources
from skulk.utils.owned_process import start_owned
from skulk.utils.runtime_files import examine, has_root_authority, open_unfollowed

_MAXIMUM_FILES = 200000
_MAXIMUM_BYTES = 17179869184
# Room kept free beyond the copy itself, so staging never leaves the volume
# that also holds models and the event log full.
_FREE_SPACE_MARGIN_BYTES = 2147483648


def _require_free_space(destination: Path, needed: int) -> None:
    """Refuse a copy that would leave the destination volume nearly full."""
    free = shutil.disk_usage(destination).free
    if free < needed + _FREE_SPACE_MARGIN_BYTES:
        gibibyte = 1024**3
        raise ValueError(
            "Not enough free disk space to set up plugins: "
            f"about {(needed + _FREE_SPACE_MARGIN_BYTES) / gibibyte:.1f} GB needed, "
            f"{free / gibibyte:.1f} GB free"
        )


# What ``venv`` itself puts in an empty environment on Windows, at its deepest.
_ENVIRONMENT_PATHS = ("Lib/site-packages/", "Scripts/deactivate.bat")


def _require_path_room(runtime: Path, members: Iterable[str]) -> None:
    """Refuse a copy whose deepest file would not fit below ``runtime``.

    Windows limits how long a path may be unless the machine has long paths
    switched on. Without this the copy stops partway with an error that names
    one file and says nothing of why.

    Only Windows has the limit, so the refusal is worded for Windows. There
    the plugin service's folder is inside the account's own folder and cannot
    be put elsewhere, so the refusal does not send the owner looking for a
    shorter folder, as it first did: long paths are the way out, and the only
    other is an account folder with a shorter name.

    Args:
        runtime: Where the environment would be built.
        members: The ``/``-separated paths the copy creates below it.
    """
    excess = copied_path_excess(runtime, (*_ENVIRONMENT_PATHS, *members))
    if excess:
        characters = "1 character" if excess == 1 else f"{excess} characters"
        raise ValueError(
            f"Skulk's own files need a path {characters} longer than Windows allows "
            "on this machine. Switch on long paths in Windows, which needs an "
            "administrator. The plugin service is kept inside your user folder "
            "and cannot be moved, so the only other way is a user folder with a "
            "shorter name."
        )


class ServiceSnapshot(BaseModel):
    """An immutable staged core service copy, independent of plugin generations."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    generation: str = Field(
        pattern=r"^[a-f0-9]{32}$",
        description="Locally generated immutable service-runtime ID.",
    )
    manifest_sha256: Digest = Field(
        description="Digest of the complete protected copy manifest."
    )
    skulk_build_sha256: Digest = Field(
        description="Exact qualified Skulk code and native binding identity."
    )
    copied_files: int = Field(
        ge=1, le=200000, description="Number of copied source and dependency files."
    )
    copied_bytes: int = Field(
        ge=1, le=17179869184, description="Total copied source and dependency bytes."
    )


@final
@dataclass(frozen=True)
class _SourceFile:
    relative: str
    source: Path
    digest: str
    size: int
    executable: bool


def _distributions() -> dict[str, str]:
    values: dict[str, str] = {}
    for distribution in importlib.metadata.distributions(
        path=[sysconfig.get_path("purelib")]
    ):
        name = str(canonicalize_name(distribution.metadata["Name"]))
        if name in values:
            raise ValueError("duplicate installed distribution")
        values[name] = distribution.version
    if not values or len(values) > 1024:
        raise ValueError("installed dependency inventory exceeds bound")
    return dict(sorted(values.items()))


PACKAGED_RUNTIME_MARKER = "skulk-packaged-runtime.json"
"""File in an interpreter's prefix that declares it dedicated to Skulk.

The packaged apps ship a relocatable standalone Python with Skulk installed in
its own site-packages instead of a virtual environment (a virtual environment
records an absolute path, and an app bundle can be moved). The marker tells
setup that every package in that interpreter belongs to Skulk, which is what a
virtual environment otherwise proves.
"""


def dedicated_skulk_interpreter() -> bool:
    """Whether this interpreter is isolated for Skulk: a venv or a packaged runtime."""
    if sys.prefix != sys.base_prefix:
        return True
    return (Path(sys.prefix) / PACKAGED_RUNTIME_MARKER).is_file()


def _site_prefix() -> str:
    """Where ``venv`` puts site-packages in a new environment, relative to it."""
    if sys.platform == "win32":
        return "Lib/site-packages/"
    return f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/"


def _sources() -> tuple[_SourceFile, ...]:
    if not dedicated_skulk_interpreter() or sysconfig.get_path(
        "purelib"
    ) != sysconfig.get_path("platlib"):
        raise ValueError(
            "service setup requires an isolated supported Skulk environment"
        )
    site_packages = Path(sysconfig.get_path("purelib"))
    site_prefix = _site_prefix()
    roots = [(site_packages, site_prefix, True)]
    for package in ("skulk", "skulk_pyo3_bindings"):
        specification = importlib.util.find_spec(package)
        if specification is None or specification.origin is None:
            raise ValueError("installed core package is unavailable")
        roots.append(
            (Path(specification.origin).parent, site_prefix + package + "/", False)
        )
    # Core imports need bundled declarative resources even without a dashboard.
    # Put them beside the new runtime's lib directory, where the normal resource
    # finder locates them without an environment override or checkout dependency.
    roots.append((find_resources(), "resources/", False))
    result: list[_SourceFile] = []
    total = 0
    for root, prefix, primary in roots:
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            if primary and relative.parts[0] in {"skulk", "skulk_pyo3_bindings"}:
                continue
            # Finder's metadata in the source checkout is not part of the
            # runtime; packing it would bind the runtime to a file Finder
            # rewrites whenever the folder is browsed.
            if is_desktop_metadata(path):
                continue
            member = examine(path)
            if member.kind == "link":
                raise ValueError("installed dependency has an unresolved link")
            if member.kind == "directory":
                continue
            if member.kind != "file" or not no_other_account_can_change(path):
                raise ValueError("installed dependency is not a protected regular file")
            info = member.info
            if primary and relative == Path("skulk.pth"):
                # The effective imported core is copied explicitly above. Never
                # retain the editable installation's checkout redirection.
                continue
            if primary and relative.parent == Path("."):
                if (
                    path.name in {"sitecustomize.py", "usercustomize.py"}
                    or path.suffix == ".egg-link"
                ):
                    raise ValueError("unsupported installed startup customization")
                if path.suffix == ".pth":
                    if path.name not in {"_virtualenv.pth", "distutils-precedence.pth"}:
                        raise ValueError("unsupported installed path extension")
                    # This new standard-library venv does not need the old
                    # installer's startup shims or their hidden import hooks.
                    continue
            total += info.st_size
            if len(result) >= _MAXIMUM_FILES or total > _MAXIMUM_BYTES:
                raise ValueError("core service copy exceeds bound")
            result.append(
                _SourceFile(
                    prefix + relative.as_posix(),
                    path,
                    service_bootstrap.digest_file(path),
                    info.st_size,
                    bool(info.st_mode & 0o100),
                )
            )
    names = [item.relative for item in result]
    if len(set(names)) != len(names):
        raise ValueError("core service source identity is ambiguous")
    return tuple(sorted(result, key=lambda item: item.relative))


def _protect(tree: Path) -> None:
    """Remove group and other write from everything staged under ``tree``.

    The standard library's venv builds its skeleton with the process umask,
    which on Ubuntu (0002) leaves directories and scripts group-writable, and
    the service bootstrap rightly refuses a runtime another account could
    change. The staged runtime belongs to the service alone, so it is made
    private before it is qualified and sealed.

    Windows has no such step: everything created inside the runtime's private
    folder takes its access from that folder, the account and SYSTEM only,
    and the seal refuses a member that grants anyone else.
    """
    if sys.platform == "win32":
        return
    for path in (tree, *tree.rglob("*")):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            continue
        mode = stat.S_IMODE(info.st_mode)
        if mode & 0o022:
            path.chmod(mode & ~0o022)


def _copy(source: tuple[_SourceFile, ...], destination: Path) -> None:
    for item in source:
        target = destination / item.relative
        if sys.platform == "win32":
            # No mode: a new folder then takes its access from the private
            # folder above it, the account and SYSTEM only. Python gives mode
            # 0o700 a meaning of its own on Windows, an access list that also
            # admits Administrators, which the seal refuses.
            target.parent.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = open_unfollowed(item.source)
        with os.fdopen(descriptor, "rb") as original, target.open("xb") as copied:
            info = os.fstat(original.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size != item.size:
                raise ValueError("core dependency changed during copy")
            shutil.copyfileobj(original, copied, length=1048576)
            copied.flush()
            os.fsync(copied.fileno())
        target.chmod(0o700 if item.executable else 0o600)
        if service_bootstrap.digest_file(target) != item.digest:
            raise ValueError("core dependency bytes changed during copy")


# The state directory arrives as an argument and is put in place before
# anything of Skulk's is imported: an owned child starts with a fixed
# environment that the caller cannot add to.
_QUALIFY = """import os,sys
os.environ['SKULK_HOME']=sys.argv[1]
import importlib.metadata,json
from packaging.utils import canonicalize_name
from skulk.extensions.runtime_artifacts import measure_host
host=measure_host()
print(json.dumps({'platform':host.platform,'python_version':host.python_version,
'skulk_version':host.skulk_version,'skulk_build_sha256':host.skulk_build_sha256,
'inventory':{str(canonicalize_name(d.metadata['Name'])):d.version
for d in importlib.metadata.distributions()}},sort_keys=True))
"""


async def _run(
    arguments: tuple[str, ...], directory: Path, lock: RuntimeLock, failure: str
) -> bytes:
    """Run one owned program to its end and return what it printed.

    The program cannot outlive the installer lock, and its whole tree has
    ended before this returns. What it printed is kept beside the runtime
    when it fails.

    Args:
        arguments: The program and its arguments.
        directory: The directory it starts in; evidence goes beside it.
        lock: The held installer lock.
        failure: What to say when the program exits with an error.
    """
    child = await start_owned(arguments, cwd=directory, lock=lock)
    output = bytearray()
    try:
        async with asyncio.timeout(60):
            while block := await child.stdout.read(16384):
                output.extend(block)
                if len(output) > 262144:
                    raise ValueError("service qualification output exceeds bound")
            await child.wait()
        if child.returncode != 0:
            raise ValueError(failure)
        return bytes(output)
    except (OSError, ValueError, TimeoutError):
        write_private(
            directory.parent / "qualification-evidence.log", bytes(output[:262144])
        )
        raise
    finally:
        try:
            if child.returncode is None:
                child.kill()
            await finish_runtime_work(asyncio.create_task(child.wait()))
        finally:
            await finish_runtime_work(asyncio.create_task(child.close()))


async def _build_environment(runtime: Path, lock: RuntimeLock) -> None:
    """Create the empty environment the copy goes into.

    Windows builds it in a child that runs the base interpreter by its real
    path, and asks for copies outright. The environment there holds no link
    to the base: its launcher finds the base through ``home`` in
    ``pyvenv.cfg``, which ``venv`` takes from how its interpreter was started,
    and the seal requires that to be the base's own folder. Asked for links,
    ``venv`` would also fall back to copies for an ordinary account without
    saying so.
    """
    if sys.platform == "win32":
        await _run(
            (
                str(service_bootstrap.current_base()),
                "-I",
                "-m",
                "venv",
                "--copies",
                "--without-pip",
                str(runtime),
            ),
            runtime,
            lock,
            "service environment could not be created",
        )
        return
    await asyncio.to_thread(
        venv.EnvBuilder(symlinks=True, with_pip=False).create, runtime
    )


async def _qualify(runtime: Path, lock: RuntimeLock) -> bytes:
    return await _run(
        (
            str(service_bootstrap.manager_interpreter(runtime)),
            "-I",
            "-B",
            "-c",
            _QUALIFY,
            str(runtime.parent / "qualification-state"),
        ),
        runtime,
        lock,
        "copied service runtime did not qualify",
    )


def service_source_identity() -> str:
    """Fingerprint the current core, dependency inventory and exact base interpreter.

    Local setup uses this identity to distinguish an unchanged service copy from
    an explicitly requested setup using a new Python or dependency environment.
    Runtime activation still verifies the complete copied tree independently.
    """
    host = measure_host()
    identity = {
        "platform": host.platform,
        "python_version": host.python_version,
        "skulk_version": host.skulk_version,
        "skulk_build_sha256": host.skulk_build_sha256,
        "inventory": _distributions(),
        **service_bootstrap.base_identity(service_bootstrap.current_base()),
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


async def stage_service_runtime(root: Path) -> ServiceSnapshot:
    """Stage and qualify an exact copy without altering the live Skulk environment.

    Dependencies are copied locally, never resolved or installed over the source.
    The effective core replaces editable .pth indirection. Completed copies retain
    a full tree seal. A copy that fails keeps its small evidence files but not
    its copied runtime, which is gigabytes and is never activated; a copy is
    refused outright when the volume lacks room for it. Cancellation waits for
    owned copy and qualification work to finish.
    """
    if has_root_authority():
        raise ValueError("service runtime preparation must run without root")
    private_directory(root)
    lock = RuntimeLock(root)
    try:

        async def prepare() -> ServiceSnapshot:
            host = await asyncio.to_thread(measure_host)
            inventory = _distributions()
            source = await asyncio.to_thread(_sources)
            generation_id = uuid4().hex
            parent = root / "core-runtimes"
            # Before anything is created: a refused copy leaves nothing behind.
            _require_path_room(
                parent / generation_id / "runtime", (item.relative for item in source)
            )
            private_directory(parent)
            await asyncio.to_thread(
                _require_free_space, parent, sum(item.size for item in source)
            )
            generation = parent / generation_id
            private_directory(generation)
            runtime = generation / "runtime"
            private_directory(runtime)
            try:
                return await _stage_into(
                    generation, runtime, source, host, inventory, lock
                )
            except Exception:
                # The unselected copy is never activated; keep the evidence
                # beside it, but give the volume its gigabytes back so a
                # retried setup cannot fill the disk one attempt at a time.
                await asyncio.to_thread(shutil.rmtree, runtime, True)
                raise

        async def _stage_into(
            generation: Path,
            runtime: Path,
            source: tuple[_SourceFile, ...],
            host: QualifiedHost,
            inventory: dict[str, str],
            lock: RuntimeLock,
        ) -> ServiceSnapshot:
            generation_id = generation.name
            await _build_environment(runtime, lock)
            await asyncio.to_thread(_copy, source, runtime)
            await asyncio.to_thread(_protect, runtime)
            bootstrap = Path(service_bootstrap.__file__).read_bytes()
            write_private(generation / "bootstrap.py", bootstrap)
            raw = await _qualify(runtime, lock)
            await asyncio.to_thread(_protect, runtime)
            expected = {
                "platform": host.platform,
                "python_version": host.python_version,
                "skulk_version": host.skulk_version,
                "skulk_build_sha256": host.skulk_build_sha256,
                "inventory": inventory,
            }
            if service_bootstrap.document(raw) != expected:
                raise ValueError("copied service identity differs")
            if (
                await asyncio.to_thread(measure_host) != host
                or _distributions() != inventory
                or await asyncio.to_thread(_sources) != source
            ):
                raise ValueError(
                    "source environment changed during service preparation"
                )
            base = service_bootstrap.current_base()
            entries = await asyncio.to_thread(
                service_bootstrap.runtime_tree, runtime, base
            )
            metadata = {
                "protocol": 1,
                "generation": generation_id,
                **service_bootstrap.base_identity(base),
                "bootstrap_sha256": hashlib.sha256(bootstrap).hexdigest(),
                "qualified_host": expected,
                "entries": entries,
            }
            payload = json.dumps(
                metadata, sort_keys=True, separators=(",", ":")
            ).encode()
            if len(payload) > 67108864:
                raise ValueError("service manifest exceeds bound")
            write_private(generation / "snapshot.json", payload)
            result = ServiceSnapshot(
                generation=generation_id,
                manifest_sha256=hashlib.sha256(payload).hexdigest(),
                skulk_build_sha256=host.skulk_build_sha256,
                copied_files=len(source),
                copied_bytes=sum(item.size for item in source),
            )
            write_private(generation / "staged.json", result.model_dump_json().encode())
            return result

        return await finish_runtime_work(asyncio.create_task(prepare()))
    finally:
        lock.close()


def selected_generation(root: Path) -> str | None:
    """The generation ``core-runtime.json`` selects for the next manager start.

    ``None`` when nothing is selected or the pointer is unreadable; callers
    treat that as "not this generation" rather than as an error.
    """
    document = TypeAdapter(dict[str, JsonValue])
    try:
        pointer = document.validate_json(read_private(root / "core-runtime.json", 4096))
    except (OSError, ValueError):
        return None
    generation = pointer.get("generation")
    return generation if isinstance(generation, str) else None


def activate_staged_runtime(root: Path, snapshot: ServiceSnapshot) -> None:
    """Select a staged generation for the next manager start, holding no lock.

    The caller owns the installer and manager fences: local setup takes both
    with the manager stopped, and the running manager takes the installer
    fence beside the manager fence it already holds when it activates its own
    successor before exiting for the OS service to restart it.
    """
    generation = root / "core-runtimes" / snapshot.generation
    if (
        ServiceSnapshot.model_validate_json(read_private(generation / "staged.json"))
        != snapshot
    ):
        raise ValueError("staged service identity differs")
    raw = read_private(generation / "snapshot.json", 67108864)
    if hashlib.sha256(raw).hexdigest() != snapshot.manifest_sha256:
        raise ValueError("service manifest integrity differs")
    metadata = service_bootstrap.document(raw)
    base = service_bootstrap.current_base()
    if not service_bootstrap.names_base(metadata, base) or metadata.get(
        "entries"
    ) != service_bootstrap.runtime_tree(generation / "runtime", base):
        raise ValueError("staged service runtime differs")
    bootstrap = read_private(generation / "bootstrap.py", 65536)
    if hashlib.sha256(bootstrap).hexdigest() != metadata.get("bootstrap_sha256"):
        raise ValueError("service bootstrap differs")
    write_private(root / "service-bootstrap.py", bootstrap)
    write_private(
        root / "core-runtime.json",
        json.dumps(
            {
                "generation": snapshot.generation,
                "manifest_sha256": snapshot.manifest_sha256,
            },
            sort_keys=True,
        ).encode(),
    )


def activate_service_runtime(root: Path, snapshot: ServiceSnapshot) -> None:
    """Select a staged generation with the manager stopped and both fences held.

    This owner-local setup step changes only the generic manager runtime. Plugin
    selections, credentials and independently supervised cleanup stay untouched.
    """
    installer = RuntimeLock(root)
    try:
        owner = RuntimeLock(root, "manager.lock")
        try:
            activate_staged_runtime(root, snapshot)
        finally:
            owner.close()
    finally:
        installer.close()
