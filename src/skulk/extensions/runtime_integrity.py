"""Bounded installed-runtime integrity checks performed before Python startup.

An installed runtime is a Python environment built by ``venv``, and the two
platform families lay one out differently:

- **POSIX.** The interpreter is ``runtime/bin/python``, a symbolic link that
  ends at the base interpreter. The seal records where each such link ends
  and the base's digest.
- **Windows.** An ordinary account may not create symbolic links, so the
  environment is built from copies. The interpreter is
  ``runtime\\Scripts\\python.exe``, a small launcher that finds the base through
  ``home`` in ``pyvenv.cfg``. The tree holds no link of any kind and the seal
  refuses one; it records the base in a field of its own.

Who owns a member and who can change it is asked through
:mod:`skulk.utils.runtime_files`, which has a side for each platform.

Windows also limits how long a path may be unless the machine has long paths
switched on. :func:`runtime_path_excess` says whether a runtime's files would
fit below a generation folder, so the installer can refuse before it writes
any of them.
"""

import base64
import functools
import hashlib
import io
import json
import os
import platform
import re
import stat
import sys
import zipfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Final, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from skulk.extensions.runtime_artifacts import Digest, canonical_json
from skulk.extensions.runtime_files import (
    is_desktop_metadata,
    read_private,
    write_private,
)
from skulk.utils.runtime_files import (
    create_file,
    examine,
    open_unfollowed,
    sync_directory,
)

# venv writes exactly these keys; anything else means the file is not one
# this installer produced, so adoption refuses rather than guessing.
_CONFIGURATION_KEYS = frozenset(
    {"home", "include-system-site-packages", "version", "executable", "command"}
)
_CONFIGURATION_LIMIT = 8192
_REBASE_JOURNAL = "interpreter-rebase.json"
# Only names this module creates while adopting, so an interrupted adoption's
# leftovers can be removed without touching anything a publisher installed.
_REBASE_LEFTOVER = re.compile(r"^\.(pyvenv\.cfg|python[0-9.]*)\.rebase-[0-9a-f]{32}$")


class _Entry(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    kind: Literal["file", "directory", "link"]
    mode: int = Field(ge=0, le=0o777)
    digest: Digest | None = None
    target: str | None = Field(default=None, max_length=4096)


class _Base(BaseModel):
    """The base interpreter of an environment that reaches it without a link."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    path: str = Field(max_length=4096)
    executable: Digest
    # The library beside the program. It holds nearly all of the interpreter,
    # so a base without it cannot start and is never recorded.
    library: Digest


class _Seal(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    runtime_digest: Digest
    entries: dict[str, _Entry] = Field(min_length=1, max_length=20000)
    # Windows only. A POSIX seal binds the base through its interpreter links.
    base: _Base | None = None


class _Rebase(BaseModel):
    """The sealed configuration and its replacement, kept until resealing ends."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    original: str = Field(max_length=_CONFIGURATION_LIMIT * 2)
    rewritten: str = Field(max_length=_CONFIGURATION_LIMIT * 2)


def runtime_interpreter(runtime: Path) -> Path:
    """The interpreter that starts an installed runtime's code.

    Args:
        runtime: The environment's directory, ``<generation>/runtime``.

    Returns:
        The path ``venv`` gives the environment's interpreter on this
        platform.
    """
    if sys.platform == "win32":
        return runtime / "Scripts" / "python.exe"
    return runtime / "bin" / "python"


def environment_command(runtime: Path) -> tuple[str, ...]:
    """The command that builds an empty environment on the current base.

    Windows asks for copies outright. Asked for links there, ``venv`` tries,
    fails for an ordinary account and falls back to copies, so the layout
    would depend on who ran it.

    Args:
        runtime: Where to build the environment.

    Returns:
        The program and its arguments.
    """
    return (
        str(_current_base()),
        "-I",
        "-m",
        "venv",
        "--copies" if sys.platform == "win32" else "--symlinks",
        str(runtime),
    )


# Windows without long paths refuses to create a directory whose path is 248
# characters or longer, and a file whose path is 260 or longer. Measured with
# ``venv`` on such a machine: it builds with its deepest directory at 247 and
# fails at 248. A compiled file (``.pyc``) whose path is too long is skipped
# without an error, so compiled files are not counted.
_DIRECTORY_LIMIT: Final = 247
_FILE_LIMIT: Final = 259
_SITE: Final = "Lib/site-packages/"
# What a generation holds whatever is installed into it: the installer's own
# record, the environment's folders and interpreter, and the longest name
# adoption gives a file it is replacing.
_FIXED_PATHS: Final = (
    "installer-evidence.log",
    "runtime/Lib/site-packages/",
    "runtime/Scripts/pythonw.exe",
    "runtime/.pyvenv.cfg.rebase-" + "0" * 32,
)

PATH_TOO_LONG: Final = "runtime_path_too_long"
"""Manager error naming an installation whose folder path is too long."""

_EXCESS_BOUND: Final = 32767


class RuntimePathTooLongError(ValueError):
    """A runtime whose files would not fit below its installation's folder.

    Carries one number, how many characters too long the deepest path would
    be, so the manager, the guided installer and the plugin routes can say
    what to change without disclosing any path.
    """

    def __init__(self, excess: int) -> None:
        super().__init__(
            f"the runtime needs a path {excess} characters longer than this machine allows"
        )
        self.excess = excess


def path_refusal_sentence(excess: int) -> str:
    """The one sentence every surface uses for an installation path that is too long.

    Args:
        excess: How many characters too long the deepest path would be.

    Returns:
        What happened and the two ways to fix it.
    """
    characters = "1 character" if excess == 1 else f"{excess} characters"
    return (
        f"This plugin's files need a path {characters} longer than Windows allows on "
        "this machine. Switch on long paths in Windows, or keep Skulk's files in a "
        "folder with a shorter path, then install again."
    )


def path_refusal_payload(refused: RuntimePathTooLongError) -> bytes:
    """Encode the refusal as a fixed vocabulary: a code and one integer."""
    return (
        json.dumps({"error": PATH_TOO_LONG, "excess": refused.excess}, sort_keys=True)
        + "\n"
    ).encode()


def path_refusal(response: Mapping[str, JsonValue]) -> RuntimePathTooLongError | None:
    """Rebuild the refusal from the manager's fixed vocabulary, or nothing.

    Args:
        response: A manager response.

    Returns:
        The refusal when the response names one with a usable number;
        ``None`` for anything else, which stays undisclosed.
    """
    excess = response.get("excess")
    if (
        response.get("error") != PATH_TOO_LONG
        or not isinstance(excess, int)
        or isinstance(excess, bool)
        or not 0 < excess <= _EXCESS_BOUND
    ):
        return None
    return RuntimePathTooLongError(excess)


def _paths_are_limited() -> bool:
    """Whether this process, and the programs it starts, are held to the short limit.

    Windows applies the short limit unless the machine has long paths switched
    on and the program says it can use them, which Python's does.
    """
    if sys.platform != "win32":
        return False
    import ctypes

    try:
        enabled = ctypes.WinDLL("ntdll").RtlAreLongPathsEnabled
    except (AttributeError, OSError):
        return True
    enabled.restype = ctypes.c_ubyte
    enabled.argtypes = ()
    return not enabled()


def _placed(member: str) -> str:
    """Where ``pip`` puts one member of a wheel, below a Windows environment."""
    head, _, rest = member.partition("/")
    if head.endswith(".data"):
        scheme, _, inner = rest.partition("/")
        if scheme in ("purelib", "platlib"):
            return _SITE + inner
        if scheme == "scripts":
            return "Scripts/" + inner
        if scheme == "data":
            return inner
    # Headers go elsewhere under a shorter name; counting them here can only
    # refuse a little early, never let a path through that does not fit.
    return _SITE + member


def _depth(paths: Iterable[str]) -> tuple[int, int]:
    """The longest directory and the longest file among ``/``-separated paths.

    A path ending in ``/`` is a directory. Every other path is a file, and the
    directory that holds it is counted too.
    """
    directory = file = 0
    for path in paths:
        if path.endswith("/"):
            directory = max(directory, _units(path) - 1)
        else:
            file = max(file, _units(path))
            directory = max(directory, _units(path.rpartition("/")[0]))
    return directory, file


def _units(text: str) -> int:
    """The length of a path as Windows counts it: in UTF-16 code units.

    A character outside the Basic Multilingual Plane, an emoji for one, is
    one character to Python and two units to Windows.
    """
    if text.isascii():
        return len(text)
    return len(text.encode("utf-16-le", "surrogatepass")) // 2


def _launchers(archive: zipfile.ZipFile) -> list[str]:
    """The program files ``pip`` generates in ``Scripts`` for a wheel's entry points.

    They are not members of the wheel: ``pip`` writes ``<name>.exe`` for each
    name under ``console_scripts`` and ``gui_scripts``. Read as data, within
    a bound; a declaration this cannot read is ``pip``'s to report.
    """
    found: list[str] = []
    for item in archive.infolist():
        head, _, name = item.filename.partition("/")
        if (
            not head.endswith(".dist-info")
            or name != "entry_points.txt"
            or item.file_size > 65536
        ):
            continue
        try:
            text = archive.read(item).decode("utf-8")
        except (UnicodeDecodeError, OSError, RuntimeError, zipfile.BadZipFile):
            continue
        section = None
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1].strip()
            elif section in ("console_scripts", "gui_scripts") and "=" in line:
                found.append("Scripts/" + line.partition("=")[0].strip() + ".exe")
    return found


def _wheel_depth(archive: zipfile.ZipFile, *, direct: bool) -> tuple[int, int]:
    """How deep one wheel's files go below a generation folder.

    Args:
        archive: The wheel.
        direct: Whether ``pip`` installs it from a file named on its command
            line, in which case it adds ``direct_url.json`` to the wheel's
            metadata folder.
    """
    members = archive.namelist()
    placed = [_placed(member) for member in members]
    if direct:
        placed += sorted(
            {
                _SITE + head + "/direct_url.json"
                for head in (member.partition("/")[0] for member in members)
                if head.endswith(".dist-info")
            }
        )
    placed += _launchers(archive)
    return _depth("runtime/" + path for path in placed)


@functools.cache
def _pip_depth() -> tuple[int, int]:
    """How deep the ``pip`` that ``venv`` puts in every runtime goes.

    ``venv`` installs the wheel that ships inside the standard library's
    ``ensurepip``. A base without one counts as nothing; ``venv`` then
    reports its own failure.
    """
    import ensurepip

    try:
        bundled = sorted(
            Path(ensurepip.__file__).parent.joinpath("_bundled").glob("pip-*.whl")
        )
        if not bundled:
            return 0, 0
        with zipfile.ZipFile(bundled[-1]) as archive:
            return _wheel_depth(archive, direct=False)
    except (OSError, zipfile.BadZipFile):
        return 0, 0


def _generation_depth(
    artifacts: Iterable[str], wheels: Iterable[bytes]
) -> tuple[int, int]:
    """The longest directory and file a runtime puts below its generation folder."""
    depths = [
        _depth(_FIXED_PATHS),
        _depth("artifacts/" + name for name in artifacts),
        _pip_depth(),
    ]
    for content in wheels:
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                depths.append(_wheel_depth(archive, direct=True))
        except zipfile.BadZipFile:
            # Not this check's refusal: ``pip`` reports an unreadable wheel.
            continue
    return max(depth[0] for depth in depths), max(depth[1] for depth in depths)


def runtime_path_excess(
    generation: Path, artifacts: Iterable[str] = (), wheels: Iterable[bytes] = ()
) -> int:
    """How many characters too long a runtime's deepest path would be.

    Counts the environment ``venv`` builds, the ``pip`` it installs there, the
    retained artifacts, every file of the given wheels and the program files
    ``pip`` generates for their entry points. Lengths are in UTF-16 code
    units, as Windows counts them. Compiled files are not counted: Python
    skips one whose path is too long.

    Args:
        generation: The generation folder the runtime would be installed in.
        artifacts: The names of the artifact files kept beside the runtime.
        wheels: The contents of the wheels installed into it, when they are
            at hand. Without them the answer covers everything else.

    Returns:
        Zero when everything fits, and always zero where no such limit
        applies: on POSIX, and on Windows with long paths switched on.
    """
    if not _paths_are_limited():
        return 0
    directory, file = _generation_depth(artifacts, wheels)
    below = _units(str(generation)) + 1
    return max(0, below + directory - _DIRECTORY_LIMIT, below + file - _FILE_LIMIT)


def _interpreter_names() -> frozenset[str]:
    """The links ``venv`` makes to the base interpreter; none on Windows."""
    if sys.platform == "win32":
        return frozenset()
    return frozenset(
        {
            "bin/python",
            "bin/python3",
            f"bin/python{sys.version_info.major}.{sys.version_info.minor}",
        }
    )


def _file_digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _inventory(directory: Path, base: Path | None) -> dict[str, _Entry]:
    """Inventory the runtime; with no base, record interpreter links unresolved.

    The unresolved form exists only for adoption, which must inspect a runtime
    whose interpreter moved or was replaced and so cannot resolve it yet.
    """
    root = examine(directory)
    if root.kind != "directory" or not root.owned or not root.exclusive:
        raise ValueError("installed runtime directory is unsafe")
    entries: dict[str, _Entry] = {
        ".": _Entry(kind="directory", mode=stat.S_IMODE(root.info.st_mode))
    }
    interpreter_names = _interpreter_names()
    total = 0
    # Do not follow directory links. Only the conventional lib64 alias and
    # interpreter aliases made by venv are accepted, never links into a checkout.
    for path in sorted(directory.rglob("*")):
        if len(entries) >= 20000:
            raise ValueError("installed runtime file count exceeds bound")
        # Browsing a generation in Finder must not unseal it.
        if is_desktop_metadata(path):
            continue
        relative = path.relative_to(directory).as_posix()
        member = examine(path)
        if not member.owned:
            raise ValueError("installed runtime ownership differs")
        mode = stat.S_IMODE(member.info.st_mode)
        if member.kind == "link":
            if sys.platform == "win32":
                # venv makes none there, and a junction needs no privilege.
                raise ValueError("installed runtime contains an unsupported link")
            target = os.readlink(path)
            if (relative == "lib64" and target == "lib") or (
                relative in interpreter_names and base is None
            ):
                entry = _Entry(kind="link", mode=mode, target=target)
            elif relative in interpreter_names and base is not None:
                resolved = path.resolve(strict=True)
                if resolved != base:
                    raise ValueError("installed interpreter is not the qualified base")
                entry = _Entry(
                    kind="link",
                    mode=mode,
                    target=str(resolved),
                    digest=_file_digest(resolved),
                )
            else:
                raise ValueError("installed runtime contains an unsupported link")
        elif not member.exclusive:
            raise ValueError("installed runtime is writable by another user")
        elif member.kind == "directory":
            entry = _Entry(kind="directory", mode=mode)
        elif member.kind == "file":
            total += member.info.st_size
            if total > 536870912:
                raise ValueError("installed runtime size exceeds bound")
            with os.fdopen(open_unfollowed(path), "rb") as source:
                opened = os.fstat(source.fileno())
                if (opened.st_ino, opened.st_dev) != (
                    member.info.st_ino,
                    member.info.st_dev,
                ):
                    raise ValueError("installed runtime changed during verification")
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            entry = _Entry(kind="file", mode=mode, digest=digest)
        else:
            raise ValueError("installed runtime contains an unsupported member")
        entries[relative] = entry
    if (
        sys.platform == "win32"
        and base is not None
        and (configuration := entries.get("pyvenv.cfg")) is not None
        and configuration.kind == "file"
        and _configured_home(directory / "pyvenv.cfg") != base.parent
    ):
        # The launcher starts whatever this names; no link says so here.
        raise ValueError("installed interpreter is not the qualified base")
    return entries


def _current_base() -> Path:
    """The interpreter a new environment is built on, and a sealed one must use.

    A Windows process started from an environment has that environment's
    launcher as its executable, which is not a link to anything, so the base
    is taken from where Python itself records it.
    """
    if sys.platform == "win32":
        return Path(getattr(sys, "_base_executable", sys.executable)).resolve(
            strict=True
        )
    return Path(sys.executable).resolve(strict=True)


def _sealed_base(base: Path) -> _Base | None:
    """What a seal made now records about the base outside its entries.

    Raises:
        OSError: When the base's program or its library cannot be read. A
            base that is missing either is refused, at sealing, at
            verification and at adoption alike.
    """
    if sys.platform != "win32":
        return None
    library = base.with_name(
        f"python{sys.version_info.major}{sys.version_info.minor}.dll"
    )
    return _Base(
        path=str(base),
        executable=_file_digest(base),
        library=_file_digest(library),
    )


def _read_seal(generation: Path) -> _Seal:
    return _Seal.model_validate_json(
        read_private(generation / "installed-files.json", 8388608)
    )


def _write_seal(generation: Path, seal: _Seal) -> None:
    record = seal.model_dump(mode="json")
    if seal.base is None:
        # A POSIX seal stays the bytes it has always been.
        del record["base"]
    write_private(generation / "installed-files.json", canonical_json(record))


def seal_runtime(generation: Path, runtime_digest: str) -> None:
    """Persist the exact installed files after trusted offline installation.

    Call only while holding the installer lock, before publishing staged.json.
    This protected local seal binds installation outputs to verified artifacts;
    it is not a replacement for publisher signatures or a same-user sandbox.
    Runtime processes must disable bytecode writes to keep the seal immutable.
    """
    base = _current_base()
    _write_seal(
        generation,
        _Seal(
            runtime_digest=runtime_digest,
            entries=_inventory(generation / "runtime", base),
            base=_sealed_base(base),
        ),
    )


def verify_installed_runtime(generation: Path, runtime_digest: str) -> None:
    """Refuse changed, missing or additional files before executing private Python.

    Includes executable bytes, bytecode, startup files, permissions, directory
    membership and the resolved qualified base interpreter. No installed code
    is imported to perform this verification.
    """
    seal = _read_seal(generation)
    base = _current_base()
    if (
        seal.runtime_digest != runtime_digest
        or seal.entries != _inventory(generation / "runtime", base)
        or seal.base != _sealed_base(base)
    ):
        raise ValueError("installed runtime integrity differs")


def _read_configuration(path: Path) -> bytes:
    descriptor = open_unfollowed(path)
    try:
        custody = examine(descriptor)
        if custody.kind != "file" or not custody.owned:
            raise ValueError("installed runtime integrity differs")
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            content = source.read(_CONFIGURATION_LIMIT + 1)
    finally:
        os.close(descriptor)
    if len(content) > _CONFIGURATION_LIMIT:
        raise ValueError("installed runtime configuration exceeds bound")
    return content


def _configured_home(path: Path) -> Path:
    """The folder ``pyvenv.cfg`` says the base interpreter is in."""
    for line in _read_configuration(path).decode("utf-8").splitlines():
        key, separator, value = line.partition(" = ")
        if separator and key == "home":
            return Path(value)
    raise ValueError("installed runtime configuration is not recognized")


def _rebased_configuration(original: bytes, base: Path) -> bytes:
    """Rewrite only the interpreter-derived values venv recorded.

    Python itself reads only ``home``; the other interpreter values are
    rewritten too so the file stays what venv would write for this base,
    line endings included.
    """
    rewritten: list[str] = []
    seen: set[str] = set()
    text = original.decode("utf-8")
    newline = "\r\n" if "\r\n" in text else "\n"
    for line in text.splitlines():
        key, separator, value = line.partition(" = ")
        if not separator or key not in _CONFIGURATION_KEYS or key in seen:
            raise ValueError("installed runtime configuration is not recognized")
        seen.add(key)
        if key == "home":
            value = str(base.parent)
        elif key == "include-system-site-packages":
            if value != "false":
                raise ValueError("installed runtime configuration is not recognized")
        elif key == "version":
            built = value.split(".")[:2]
            if built != [str(sys.version_info.major), str(sys.version_info.minor)]:
                raise ValueError(
                    "installed runtime was built for another Python version"
                )
            value = platform.python_version()
        elif key == "executable":
            value = os.path.realpath(base)
        else:
            _, marker, arguments = value.partition(" -m venv ")
            if not marker:
                raise ValueError("installed runtime configuration is not recognized")
            value = f"{base} -m venv {arguments}"
        rewritten.append(f"{key} = {value}")
    if "home" not in seen or "version" not in seen:
        raise ValueError("installed runtime configuration is not recognized")
    return (newline.join(rewritten) + newline).encode()


def _replace_file(path: Path, content: bytes, mode: int) -> None:
    temporary = path.with_name(f".{path.name}.rebase-{uuid4().hex}")
    # Created with the sealed mode itself, so the umask does not decide it.
    descriptor = create_file(temporary, mode)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(content)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _replace_link(path: Path, target: Path) -> None:
    temporary = path.with_name(f".{path.name}.rebase-{uuid4().hex}")
    try:
        temporary.symlink_to(target)
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _remove_rebase_leftovers(runtime: Path) -> None:
    for directory in (runtime, runtime / "bin"):
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            if _REBASE_LEFTOVER.match(path.name) and not path.is_dir():
                path.unlink()


def adopt_current_interpreter(generation: Path, runtime_digest: str) -> bool:
    """Re-point a sealed runtime at the running interpreter after it moved or changed.

    An application update replaces the bundled interpreter and moving the
    application changes its path. The seal records both, so without adoption
    every installed capability would refuse to start until purged. Adoption is
    deliberately narrow: every sealed member other than the interpreter links
    and ``pyvenv.cfg`` must still match the seal exactly, ``pyvenv.cfg`` must be
    the sealed bytes, the environment must target this same Python minor
    version, and the new base is the interpreter running this code, which is
    the base a fresh installation would record. Only interpreter-derived values
    change before the generation is resealed. An interrupted adoption resumes
    from a private journal holding the sealed configuration bytes.

    A Windows environment has no interpreter links: there adoption rewrites
    ``pyvenv.cfg`` and the seal's record of the base, and the launcher copies
    keep their sealed bytes.

    Call only while holding the installer lock. Returns whether the generation
    was resealed; raises ValueError when the runtime cannot be adopted, which
    leaves it for ordinary verification to refuse.
    """
    seal = _read_seal(generation)
    if seal.runtime_digest != runtime_digest:
        raise ValueError("installed runtime integrity differs")
    journal_path = generation / _REBASE_JOURNAL
    sealed_links = {
        name: entry
        for name, entry in seal.entries.items()
        if entry.kind == "link" and name != "lib64"
    }
    base = _current_base()
    base_digest = _file_digest(base)
    current_base = _sealed_base(base)
    if current_base is not None:
        if seal.base is None:
            raise ValueError("installed runtime integrity differs")
        unchanged = seal.base == current_base
    else:
        unchanged = not sealed_links or all(
            entry.target == str(base) and entry.digest == base_digest
            for entry in sealed_links.values()
        )
    if unchanged:
        journal_path.unlink(missing_ok=True)
        return False
    if not sealed_links.keys() <= _interpreter_names():
        raise ValueError("installed runtime was built for another Python version")
    sealed_configuration = seal.entries.get("pyvenv.cfg")
    if sealed_configuration is None or sealed_configuration.kind != "file":
        raise ValueError("installed runtime integrity differs")
    runtime = generation / "runtime"
    journal = (
        _Rebase.model_validate_json(read_private(journal_path))
        if journal_path.exists()
        else None
    )
    if journal is not None:
        _remove_rebase_leftovers(runtime)
    observed = _inventory(runtime, None)
    if observed.keys() != seal.entries.keys() or any(
        observed[name] != entry
        for name, entry in seal.entries.items()
        if name not in sealed_links and name != "pyvenv.cfg"
    ):
        raise ValueError("installed runtime integrity differs")
    if observed["pyvenv.cfg"].mode != sealed_configuration.mode:
        raise ValueError("installed runtime integrity differs")
    current = _read_configuration(runtime / "pyvenv.cfg")
    if hashlib.sha256(current).hexdigest() == sealed_configuration.digest:
        original = current
    elif (
        journal is not None
        and hashlib.sha256(base64.b64decode(journal.original)).hexdigest()
        == sealed_configuration.digest
        and current == base64.b64decode(journal.rewritten)
    ):
        original = base64.b64decode(journal.original)
    else:
        raise ValueError("installed runtime integrity differs")
    rewritten = _rebased_configuration(original, base)
    link_names = {Path(name).name for name in _interpreter_names()}
    for name in sealed_links:
        target = os.readlink(runtime / name)
        if not os.path.isabs(target) and target not in link_names:
            raise ValueError("installed runtime contains an unsupported link")
    write_private(
        journal_path,
        _Rebase(
            original=base64.b64encode(original).decode(),
            rewritten=base64.b64encode(rewritten).decode(),
        )
        .model_dump_json()
        .encode(),
    )
    if current != rewritten:
        _replace_file(runtime / "pyvenv.cfg", rewritten, sealed_configuration.mode)
    entries = dict(seal.entries)
    for name in sealed_links:
        path = runtime / name
        # venv makes one absolute link to the base and relative aliases to it.
        if os.path.isabs(os.readlink(path)) and os.readlink(path) != str(base):
            _replace_link(path, base)
        entries[name] = _Entry(
            kind="link",
            mode=stat.S_IMODE(path.lstat().st_mode),
            target=str(base),
            digest=base_digest,
        )
    entries["pyvenv.cfg"] = _Entry(
        kind="file",
        mode=sealed_configuration.mode,
        digest=hashlib.sha256(rewritten).hexdigest(),
    )
    _write_seal(
        generation,
        _Seal(runtime_digest=runtime_digest, entries=entries, base=current_base),
    )
    journal_path.unlink()
    return True


def verify_or_adopt_installed_runtime(generation: Path, runtime_digest: str) -> None:
    """Verify a generation, adopting a moved or updated interpreter only on failure.

    The ordinary verification runs first, so an unchanged generation costs
    nothing extra. When it fails and adoption reseals the generation, the
    complete verification runs again; otherwise the original refusal stands.
    Call only while holding the installer lock.
    """
    try:
        verify_installed_runtime(generation, runtime_digest)
    except (OSError, ValueError):
        if not adopt_current_interpreter(generation, runtime_digest):
            raise
        verify_installed_runtime(generation, runtime_digest)
