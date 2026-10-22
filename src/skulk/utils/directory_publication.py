"""Atomic directory publication which never replaces an existing destination."""

import ctypes
import errno
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast


def publish_directory(source: str, source_parent: int, name: str, parent: int) -> None:
    """Move a complete directory into a pinned parent without a reservation.

    Args:
        source: Verified staging basename on the destination filesystem.
        source_parent: Already pinned directory containing the verified stage.
        name: Single destination basename in the already opened parent.
        parent: Caller-owned destination directory descriptor.

    Existing entries, including empty directories and symlinks, are never
    replaced. A process exit leaves either the staging directory or the complete
    destination, without an empty placeholder. Unsupported platforms/filesystems
    fail closed; ordinary rename is not a safe fallback. The caller owns durable
    intent, verification, cancellation and completion acknowledgement.
    Both parent descriptors must stay pinned across verification and publication;
    this function performs no pathname-based directory lookup.
    """
    if any(
        value in {"", ".", ".."} or Path(value).name != value
        for value in (source, name)
    ):
        raise ValueError("directory publication requires a single basename")
    if sys.platform == "darwin":
        symbol, flags = "renameatx_np", 0x00000004  # RENAME_EXCL
    elif sys.platform == "linux":
        symbol, flags = "renameat2", 1  # RENAME_NOREPLACE
    else:
        raise OSError(errno.ENOTSUP, "exclusive directory publication unavailable")
    library = ctypes.CDLL(None, use_errno=True)
    prototype = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
        use_errno=True,
    )
    try:
        rename = cast(
            Callable[[int, bytes, int, bytes, int], int], prototype((symbol, library))
        )
    except AttributeError as error:
        raise OSError(
            errno.ENOTSUP, "exclusive directory publication unavailable"
        ) from error
    if rename(source_parent, os.fsencode(source), parent, os.fsencode(name), flags):
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), name)
    os.fsync(source_parent)
    os.fsync(parent)
