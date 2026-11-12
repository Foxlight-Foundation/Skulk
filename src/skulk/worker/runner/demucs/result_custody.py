"""Record native output directory identity before sending any inference request.

The complete reader lifetime fence is acquired separately. These private receipts
let a replacement Worker reclaim only known result directories after that fence
proves reader exit; path spelling or parent-process death alone is insufficient.
"""

import os
import re
import stat
from pathlib import Path
from typing import Literal, final

from pydantic import Field

from skulk.shared.types.audio_separation import (
    MAX_SEPARATION_STEM_BYTES,
    SEPARATION_STEMS,
)
from skulk.utils.pydantic_ext import FrozenModel
from skulk.utils.runtime_files import read_private, remove_private, write_private


@final
class _ResultOwnership(FrozenModel):
    """One observed directory inode, persisted before native output can exist."""

    version: Literal[1] = 1
    directory: str = Field(pattern=r"^[a-f0-9]{32}$")
    filesystem_device: int = Field(ge=0)
    inode: int = Field(ge=0)


def _receipt(directory: Path) -> Path:
    if (
        directory.parent.name != "jobs"
        or re.fullmatch(r"[a-f0-9]{32}", directory.name) is None
    ):
        raise ValueError("native result custody needs its fixed private job path")
    return directory.with_name("." + directory.name + ".owner.json")


def _private_directory(directory: Path) -> os.stat_result:
    info = directory.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_mode & 0o077
    ):
        raise ValueError("native result directory must be private")
    return info


def remember_result_directory(directory: Path) -> None:
    """Durably record an exclusively created job before the native request is sent."""
    receipt = _receipt(directory)
    _private_directory(directory.parent)
    info = _private_directory(directory)
    try:
        receipt.lstat()
    except FileNotFoundError:
        pass
    else:
        raise ValueError("native result has an unresolved previous ownership receipt")
    record = _ResultOwnership(
        directory=directory.name, filesystem_device=info.st_dev, inode=info.st_ino
    )
    write_private(receipt, record.model_dump_json().encode())


def discard_result_receipt(directory: Path) -> None:
    """Forget a validated receipt only after its owned result directory is gone."""
    receipt = _receipt(directory)
    try:
        record = _ResultOwnership.model_validate_json(read_private(receipt, 1024))
    except FileNotFoundError:
        return
    if record.directory != directory.name:
        raise ValueError("native result receipt belongs to another directory")
    try:
        directory.lstat()
    except FileNotFoundError:
        remove_private(receipt)
        return
    raise ValueError("native result directory still needs cleanup")


def reclaim_result_directory(directory: Path) -> bool:
    """Remove only receipt-bound known files after the caller proves reader drain.

    Changed inodes, symlinks, foreign names and malformed receipts are preserved.
    A crash before writing the receipt cannot have started native work, so only
    an empty private directory is reclaimable in that narrow gap. Return False
    when the original and its admission slot must remain reserved.
    """
    receipt = _receipt(directory)
    try:
        record = _ResultOwnership.model_validate_json(read_private(receipt, 1024))
    except FileNotFoundError:
        record = None
    except (OSError, ValueError):
        return False
    try:
        info = _private_directory(directory)
    except FileNotFoundError:
        if record is not None:
            discard_result_receipt(directory)
        return True
    except (OSError, ValueError):
        return False
    if record is not None and (
        record.directory,
        record.filesystem_device,
        record.inode,
    ) != (directory.name, info.st_dev, info.st_ino):
        return False
    bounds = {f"{name}.wav": MAX_SEPARATION_STEM_BYTES for name in SEPARATION_STEMS} | {
        "result.json": 16 * 1024
    }
    owned: list[Path] = []
    with os.scandir(directory) as children:
        for entry in children:
            observed = entry.stat(follow_symlinks=False)
            if (
                record is None
                or entry.name not in bounds
                or not stat.S_ISREG(observed.st_mode)
                or observed.st_uid != os.getuid()
                or observed.st_nlink != 1
                or observed.st_mode & 0o077
                or observed.st_size > bounds[entry.name]
            ):
                return False
            owned.append(directory / entry.name)
    for path in owned:
        path.unlink()
    directory.rmdir()
    parent = os.open(directory.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
    discard_result_receipt(directory)
    return True
