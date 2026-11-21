"""Observe Core-created native directories before any model or output writes."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from .contracts import (
    NativeMusicDirectoryIdentity,
    NativeMusicRequest,
    NativeMusicServeConfig,
)


def verify_directory(
    path: Path, identity: NativeMusicDirectoryIdentity, *, empty: bool = False
) -> None:
    """Require the exact private directory without following its final symlink.

    Core creates and journals this inode before the child receives its identity.
    The native process neither invents ownership from an existing pathname nor
    removes a mismatched directory. An ancestor alias, such as macOS temporary
    directory aliases, is valid when it still resolves to the admitted inode.
    """
    descriptor = os.open(
        path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK
    )
    try:
        facts = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(facts.st_mode)
            or stat.S_IMODE(facts.st_mode) != 0o700
            or facts.st_uid != os.getuid()
            or (facts.st_dev, facts.st_ino) != (identity.device, identity.inode)
            or empty
            and os.listdir(descriptor)
        ):
            raise ValueError("native music directory differs from Core-owned custody")
    finally:
        os.close(descriptor)


def verify_lifetime(configuration: NativeMusicServeConfig, *, initial: bool) -> None:
    """Check owned inodes and require a fresh loader before initialization.

    Core may reserve result directories while the model loads. Their individual
    inode and emptiness checks belong immediately before each native admission.
    """
    root = Path(configuration.directory)
    verify_directory(root, configuration.directory_identity)
    verify_directory(root / "requests", configuration.requests_identity)
    verify_directory(root / "loader", configuration.loader_identity, empty=initial)
    if set(os.listdir(root)) != {"requests", "loader"}:
        raise ValueError("native music lifetime was not empty before initialization")


def verify_request_directory(
    configuration: NativeMusicServeConfig,
    request: NativeMusicRequest,
    *,
    empty: bool,
) -> Path:
    """Require an admitted result inode beneath the exact running lifetime.

    The parent must observe the result before submission so cancellation during
    conditioning or native inference still has independent cleanup authority.
    Native completion never grants ownership retroactively.
    """
    verify_lifetime(configuration, initial=False)
    expected = Path(configuration.directory) / "requests" / request.request_id
    if Path(request.output_directory) != expected:
        raise ValueError(
            "native music output must use its exclusive lifetime request directory"
        )
    verify_directory(expected, request.output_directory_identity, empty=empty)
    return expected
