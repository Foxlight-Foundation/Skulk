"""Bounded installed-runtime integrity checks performed before Python startup."""

import base64
import hashlib
import os
import platform
import re
import stat
import sys
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from skulk.extensions.runtime_artifacts import Digest, canonical_json
from skulk.extensions.runtime_files import (
    is_desktop_metadata,
    read_private,
    write_private,
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


class _Seal(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    runtime_digest: Digest
    entries: dict[str, _Entry] = Field(min_length=1, max_length=20000)


class _Rebase(BaseModel):
    """The sealed configuration and its replacement, kept until resealing ends."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    original: str = Field(max_length=_CONFIGURATION_LIMIT * 2)
    rewritten: str = Field(max_length=_CONFIGURATION_LIMIT * 2)


def _interpreter_names() -> frozenset[str]:
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
    root_info = directory.lstat()
    if (
        not stat.S_ISDIR(root_info.st_mode)
        or root_info.st_uid != os.getuid()
        or root_info.st_mode & 0o022
    ):
        raise ValueError("installed runtime directory is unsafe")
    entries: dict[str, _Entry] = {
        ".": _Entry(kind="directory", mode=stat.S_IMODE(root_info.st_mode))
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
        info = path.lstat()
        if info.st_uid != os.getuid():
            raise ValueError("installed runtime ownership differs")
        mode = stat.S_IMODE(info.st_mode)
        if stat.S_ISLNK(info.st_mode):
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
        elif mode & 0o022:
            raise ValueError("installed runtime is writable by another user")
        elif stat.S_ISDIR(info.st_mode):
            entry = _Entry(kind="directory", mode=mode)
        elif stat.S_ISREG(info.st_mode):
            total += info.st_size
            if total > 536870912:
                raise ValueError("installed runtime size exceeds bound")
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(descriptor, "rb") as source:
                opened = os.fstat(source.fileno())
                if (opened.st_ino, opened.st_dev) != (info.st_ino, info.st_dev):
                    raise ValueError("installed runtime changed during verification")
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            entry = _Entry(kind="file", mode=mode, digest=digest)
        else:
            raise ValueError("installed runtime contains an unsupported member")
        entries[relative] = entry
    return entries


def _current_base() -> Path:
    return Path(sys.executable).resolve(strict=True)


def _read_seal(generation: Path) -> _Seal:
    return _Seal.model_validate_json(
        read_private(generation / "installed-files.json", 8388608)
    )


def _write_seal(generation: Path, seal: _Seal) -> None:
    write_private(
        generation / "installed-files.json",
        canonical_json(seal.model_dump(mode="json")),
    )


def seal_runtime(generation: Path, runtime_digest: str) -> None:
    """Persist the exact installed files after trusted offline installation.

    Call only while holding the installer lock, before publishing staged.json.
    This protected local seal binds installation outputs to verified artifacts;
    it is not a replacement for publisher signatures or a same-user sandbox.
    Runtime processes must disable bytecode writes to keep the seal immutable.
    """
    _write_seal(
        generation,
        _Seal(
            runtime_digest=runtime_digest,
            entries=_inventory(generation / "runtime", _current_base()),
        ),
    )


def verify_installed_runtime(generation: Path, runtime_digest: str) -> None:
    """Refuse changed, missing or additional files before executing private Python.

    Includes executable bytes, bytecode, startup files, permissions, directory
    membership and the resolved qualified base interpreter. No installed code
    is imported to perform this verification.
    """
    seal = _read_seal(generation)
    if seal.runtime_digest != runtime_digest or seal.entries != _inventory(
        generation / "runtime", _current_base()
    ):
        raise ValueError("installed runtime integrity differs")


def _read_configuration(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise ValueError("installed runtime integrity differs")
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            content = source.read(_CONFIGURATION_LIMIT + 1)
    finally:
        os.close(descriptor)
    if len(content) > _CONFIGURATION_LIMIT:
        raise ValueError("installed runtime configuration exceeds bound")
    return content


def _rebased_configuration(original: bytes, base: Path) -> bytes:
    """Rewrite only the interpreter-derived values venv recorded.

    Python itself reads only ``home``; the other interpreter values are
    rewritten too so the file stays what venv would write for this base.
    """
    rewritten: list[str] = []
    seen: set[str] = set()
    for line in original.decode("utf-8").splitlines():
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
    return ("\n".join(rewritten) + "\n").encode()


def _sync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _replace_file(path: Path, content: bytes, mode: int) -> None:
    temporary = path.with_name(f".{path.name}.rebase-{uuid4().hex}")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(content)
            target.flush()
            # The umask must not decide the sealed mode.
            os.fchmod(target.fileno(), mode)
            os.fsync(target.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _replace_link(path: Path, target: Path) -> None:
    temporary = path.with_name(f".{path.name}.rebase-{uuid4().hex}")
    try:
        temporary.symlink_to(target)
        os.replace(temporary, path)
        _sync_directory(path.parent)
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
    if not sealed_links or all(
        entry.target == str(base) and entry.digest == base_digest
        for entry in sealed_links.values()
    ):
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
    _write_seal(generation, _Seal(runtime_digest=runtime_digest, entries=entries))
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
