"""Own native loader inodes without taking custody of retained result readers.

The service fence is inherited by the native process. A replacement supervisor
must acquire it before recovery; parent death alone never permits cleanup. The
requests tree has independent per-result receipts and is never walked or removed
here, including when model initialization fails or a loaded model is cancelled.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Literal, final

from pydantic import Field, model_validator

from skulk.shared.types.native_music import (
    NativeMusicContract,
    NativeMusicDirectoryIdentity,
    NativeMusicWeights,
)
from skulk.utils.runtime_files import RuntimeLock, write_private

MAX_MUSIC_LOADER_CACHE_BYTES = 256 * 1024**2
_MAX_ENTRIES = 4096
_MAX_DEPTH = 12
_MODEL_CODE = (
    "configuration_acestep_v15.py",
    "modeling_acestep_v15_base.py",
    "apg_guidance.py",
)
_ROLE_ROOTS = {
    "base": "acestep-v15-base",
    "text_encoder": "Qwen3-Embedding-0.6B",
    "autoencoder": "vae",
}
_CACHE_ROOTS = frozenset({"hf", "modules", "torch-cache"})
_CHECKPOINT_DIRECTORIES = frozenset(
    {"checkpoints"} | {"checkpoints/" + name for name in _ROLE_ROOTS.values()}
)


@final
class _LoaderFile(NativeMusicContract):
    """One permitted loader copy; partial writes may be smaller, never larger."""

    path: str = Field(min_length=1, max_length=128)
    size_bytes: int = Field(gt=0, le=8 * 1024**3)


@final
class _WorkspaceOwnership(NativeMusicContract):
    """Pre-spawn identities and cleanup limits, outside every native write root."""

    version: Literal[1] = 1
    directory: NativeMusicDirectoryIdentity
    requests: NativeMusicDirectoryIdentity
    loader: NativeMusicDirectoryIdentity
    files: tuple[_LoaderFile, ...] = Field(min_length=17, max_length=17)

    @model_validator(mode="after")
    def validate_inventory(self) -> _WorkspaceOwnership:
        """Keep recovery's accepted names identical to the pinned loader closure."""
        expected = (
            {
                "checkpoints/acestep-v15-base/" + name
                for name in (
                    "config.json",
                    "model.safetensors",
                    "silence_latent.pt",
                    *_MODEL_CODE,
                )
            }
            | {
                "checkpoints/Qwen3-Embedding-0.6B/" + name
                for name in (
                    "added_tokens.json",
                    "chat_template.jinja",
                    "config.json",
                    "merges.txt",
                    "model.safetensors",
                    "special_tokens_map.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "vocab.json",
                )
            }
            | {
                "checkpoints/vae/config.json",
                "checkpoints/vae/diffusion_pytorch_model.safetensors",
            }
        )
        if (
            len({item.path for item in self.files}) != 17
            or {item.path for item in self.files} != expected
        ):
            raise ValueError(
                "native music loader receipt has a foreign or incomplete inventory"
            )
        if any(
            item.size_bytes > 2 * 1024**2
            for item in self.files
            if item.path.endswith(".py")
        ):
            raise ValueError("native music code copy exceeds its sealed source bound")
        return self


def _loader_files(weights: NativeMusicWeights) -> tuple[_LoaderFile, ...]:
    files: list[_LoaderFile] = []
    for component in (weights.base, weights.text_encoder, weights.autoencoder):
        prefix = component.root + "/" if component.root is not None else ""
        for item in component.files:
            files.append(
                _LoaderFile(
                    path="checkpoints/"
                    + _ROLE_ROOTS[component.kind]
                    + "/"
                    + item.path.removeprefix(prefix),
                    size_bytes=item.size_bytes,
                )
            )
    # The native sealed model-code contract caps each reviewed module at 2 MiB.
    # Reserve that explicit upper bound until the final package is resolved;
    # the model's tensor/disk precision never substitutes for these copies.
    files.extend(
        _LoaderFile(path="checkpoints/acestep-v15-base/" + name, size_bytes=2 * 1024**2)
        for name in _MODEL_CODE
    )
    return tuple(files)


def music_loader_storage_budget(weights: NativeMusicWeights) -> int:
    """Bound the additional loader copy and dynamic caches, excluding originals.

    This is a disk reservation, not a resident/peak inference memory estimate.
    Request conditioning/results have separate bounded ownership and budgets.
    """
    return (
        sum(item.size_bytes for item in _loader_files(weights))
        + MAX_MUSIC_LOADER_CACHE_BYTES
    )


def _identity(info: os.stat_result) -> NativeMusicDirectoryIdentity:
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise ValueError("native music workspace directory must be owner-only")
    return NativeMusicDirectoryIdentity(device=info.st_dev, inode=info.st_ino)


def _open_directory(path: Path) -> int:
    descriptor = os.open(
        path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK
    )
    try:
        _identity(os.fstat(descriptor))
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _clear_loader(root: int, files: tuple[_LoaderFile, ...]) -> None:
    """Validate the complete bounded view before unlinking any known owned copy."""
    limits = {item.path: item.size_bytes for item in files}
    directories: dict[str, os.stat_result] = {"": os.fstat(root)}
    ordinary: list[tuple[str, os.stat_result]] = []
    entries = cache_bytes = 0

    def scan(descriptor: int, prefix: str, depth: int) -> None:
        nonlocal entries, cache_bytes
        if depth > _MAX_DEPTH:
            raise ValueError("native music loader cache depth exceeds its bound")
        with os.scandir(descriptor) as children:
            for child in children:
                entries += 1
                if entries > _MAX_ENTRIES:
                    raise ValueError("native music loader inventory exceeds its bound")
                path = prefix + "/" + child.name if prefix else child.name
                top = path.partition("/")[0]
                info = child.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    _identity(info)
                    if path not in _CHECKPOINT_DIRECTORIES and top not in _CACHE_ROOTS:
                        raise ValueError("native music loader has an unknown directory")
                    nested = os.open(
                        child.name,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=descriptor,
                    )
                    try:
                        if os.fstat(nested) != info:
                            raise ValueError(
                                "native music loader directory changed during recovery"
                            )
                        directories[path] = info
                        scan(nested, path, depth + 1)
                    finally:
                        os.close(nested)
                    continue
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_uid != os.getuid()
                    or info.st_mode & 0o077
                    or info.st_nlink != 1
                ):
                    raise ValueError(
                        "native music loader contains a foreign file or link"
                    )
                if top in _CACHE_ROOTS and "/" in path:
                    cache_bytes += info.st_size
                    if cache_bytes > MAX_MUSIC_LOADER_CACHE_BYTES:
                        raise ValueError(
                            "native music loader cache exceeds its byte budget"
                        )
                elif path not in limits or info.st_size > limits[path]:
                    raise ValueError(
                        "native music loader contains an unknown or oversized copy"
                    )
                ordinary.append((path, info))

    def parent(path: str) -> tuple[int, str]:
        descriptor = os.dup(root)
        parts = path.split("/")
        prefix = ""
        try:
            for part in parts[:-1]:
                prefix = prefix + "/" + part if prefix else part
                nested = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
                os.close(descriptor)
                descriptor = nested
                current = os.fstat(descriptor)
                recorded = directories[prefix]
                if (current.st_dev, current.st_ino) != (
                    recorded.st_dev,
                    recorded.st_ino,
                ):
                    raise ValueError(
                        "native music loader parent was replaced during cleanup"
                    )
            return descriptor, parts[-1]
        except BaseException:
            os.close(descriptor)
            raise

    scan(root, "", 0)
    for path, observed in ordinary:
        descriptor, name = parent(path)
        try:
            if os.stat(name, dir_fd=descriptor, follow_symlinks=False) != observed:
                raise ValueError("native music loader file changed during cleanup")
            os.unlink(name, dir_fd=descriptor)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    for path, observed in reversed(tuple(directories.items())):
        if not path:
            continue
        descriptor, name = parent(path)
        try:
            current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if _identity(current) != _identity(observed):
                raise ValueError("native music loader directory changed during cleanup")
            os.rmdir(name, dir_fd=descriptor)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    os.fsync(root)


@final
class NativeMusicWorkspace:
    """Create pre-spawn ownership and recover only a drained model loader view.

    The supervisor must inherit ``descriptor`` into its native child and retire
    that entire process group before ``retire_loading`` or ``close``. Completed
    and partial request directories remain owned by the separate result journal.
    """

    def __init__(self, root: Path, weights: NativeMusicWeights) -> None:
        """Select a private per-instance reader root without filesystem effects."""
        self.root, self.weights = root, weights
        self.directory = root / "workspace"
        self._lock: RuntimeLock | None = None
        self._record: _WorkspaceOwnership | None = None

    @property
    def descriptor(self) -> int:
        """Return the held service fence for inheritance, refusing an unopened owner."""
        if self._lock is None:
            raise RuntimeError("native music workspace is not open")
        return self._lock.descriptor

    @property
    def directory_identity(self) -> NativeMusicDirectoryIdentity:
        """Return the durably recorded workspace inode after successful open."""
        if self._record is None:
            raise RuntimeError("native music workspace is not prepared")
        return self._record.directory

    @property
    def requests_identity(self) -> NativeMusicDirectoryIdentity:
        """Return the retained request-parent identity, independent of loader restart."""
        if self._record is None:
            raise RuntimeError("native music workspace is not prepared")
        return self._record.requests

    @property
    def loader_identity(self) -> NativeMusicDirectoryIdentity:
        """Return the empty loader's identity for native pre-write verification."""
        if self._record is None:
            raise RuntimeError("native music workspace is not prepared")
        return self._record.loader

    def _validate(self, record: _WorkspaceOwnership) -> None:
        for path, identity in (
            (self.directory, record.directory),
            (self.directory / "requests", record.requests),
            (self.directory / "loader", record.loader),
        ):
            descriptor = _open_directory(path)
            try:
                if _identity(os.fstat(descriptor)) != identity:
                    raise ValueError("native music workspace changed its owned inode")
                if path == self.directory and set(os.listdir(descriptor)) != {
                    "requests",
                    "loader",
                }:
                    raise ValueError("native music workspace contains a foreign entry")
            finally:
                os.close(descriptor)

    def _read(self) -> _WorkspaceOwnership | None:
        try:
            descriptor = os.open(
                self.root / "workspace-owner.json",
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            )
        except FileNotFoundError:
            return None
        try:
            info = os.fstat(descriptor)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or info.st_mode & 0o077
                or info.st_nlink != 1
                or info.st_size > 8192
            ):
                raise ValueError(
                    "native music workspace receipt is not bounded private metadata"
                )
            return _WorkspaceOwnership.model_validate_json(os.read(descriptor, 8193))
        finally:
            os.close(descriptor)

    def verify(self) -> None:
        """Recheck held workspace identities without requiring an empty loader.

        The resident owner calls this before admission and output verification.
        A replaced directory or foreign top-level entry withdraws this lifetime;
        it never grants recovery authority over the replacement.
        """
        if self._lock is None or self._record is None:
            raise RuntimeError("native music workspace is not prepared")
        self._validate(self._record)

    def open(self) -> None:
        """Acquire the actual reader fence, recover loading and sync pre-spawn facts.

        Known failed copies are reclaimed before another load. Unknown contents,
        replaced inodes or incomplete deletion refuse preparation without touching
        retained request output. A missing receipt permits only empty create gaps.
        """
        if self._lock is not None:
            raise RuntimeError("native music workspace is already open")
        self._lock = RuntimeLock(self.root, name="service.lock")
        try:
            record = self._read()
            if record is not None:
                self._validate(record)
                loader = _open_directory(self.directory / "loader")
                try:
                    _clear_loader(loader, record.files)
                finally:
                    os.close(loader)
            else:
                self.directory.mkdir(mode=0o700, exist_ok=True)
                descriptor = _open_directory(self.directory)
                try:
                    if set(os.listdir(descriptor)) - {"requests", "loader"}:
                        raise ValueError("unproven native music workspace is not empty")
                    for name in ("requests", "loader"):
                        path = self.directory / name
                        path.mkdir(mode=0o700, exist_ok=True)
                        child = _open_directory(path)
                        try:
                            if os.listdir(child):
                                raise ValueError(
                                    "unproven native music workspace is not empty"
                                )
                            os.fsync(child)
                        finally:
                            os.close(child)
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            updated = _WorkspaceOwnership(
                directory=_identity(self.directory.lstat()),
                requests=_identity((self.directory / "requests").lstat()),
                loader=_identity((self.directory / "loader").lstat()),
                files=_loader_files(self.weights),
            )
            write_private(
                self.root / "workspace-owner.json", updated.model_dump_json().encode()
            )
            self._record = updated
        except BaseException:
            self.close()
            raise

    def retire_loading(self) -> bool:
        """After actual group drain, reclaim only known loader files; retain on error.

        A False result requires the supervisor to keep ownership and retry. This
        method never interprets task completion as process exit or result receipt.
        """
        if self._lock is None or self._record is None:
            raise RuntimeError("native music workspace is not prepared")
        try:
            self._validate(self._record)
            descriptor = _open_directory(self.directory / "loader")
            try:
                _clear_loader(descriptor, self._record.files)
            finally:
                os.close(descriptor)
            return True
        except (OSError, ValueError):
            return False

    def close(self) -> None:
        """Release this fence copy only after the supervisor has drained its child.

        Child copies retain the same OS lock after an abrupt supervisor exit.
        Releasing this object never recursively deletes a workspace or a result.
        """
        if self._lock is not None:
            self._lock.close()
            self._lock = None
        self._record = None
