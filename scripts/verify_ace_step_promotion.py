"""Select the complete fixed native ACE inventory for exact-byte publication.

Build success and hardware qualification do not select publication bytes. Core's
reviewed inventory does. GitHub provenance and successful source-run verification
are separate mandatory workflow checks before invoking the append-only publisher.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import stat
from pathlib import Path
from typing import final

from skulk.provisioning.ace_step_inventory import ACE_STEP_PACKAGES
from skulk.provisioning.ace_step_runtime import AceStepWheel


@final
class _Arguments(argparse.Namespace):
    directory: Path = Path()


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def verify_artifacts(
    directory: Path, *, packages: tuple[AceStepWheel, ...] = ACE_STEP_PACKAGES
) -> tuple[Path, ...]:
    """Verify every selected profile before returning any publishable wheel path.

    ``directory`` contains separate downloaded artifact directories, or flat
    producer output. Every wheel must match the trusted ``packages`` inventory
    by basename, exact size and SHA-256. Missing, duplicate, extra, symlinked or
    changed objects fail the entire selection before the caller uploads anything.
    The package argument allows small isolated refusal tests; the CLI always uses
    the complete built-in inventory and accepts no digest map or profile override.
    """
    expected = {package.filename: package for package in packages}
    if not expected or len(expected) != len(packages):
        raise ValueError("native ACE inventory is empty or contains duplicate names")
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("native ACE artifact directory must be a real directory")
    found: dict[str, Path] = {}
    # Do not follow artifact-directory links or special files. The bounded
    # traversal also refuses archive contents unrelated to the six fixed wheels.
    count = 0
    for root, directories, filenames in os.walk(directory, followlinks=False):
        count += len(directories) + len(filenames)
        if count > 256:
            raise ValueError("native ACE artifact inventory exceeds its file bound")
        for name in directories:
            if (Path(root) / name).is_symlink():
                raise ValueError("native ACE artifact directory contains a symlink")
        for name in filenames:
            path = Path(root) / name
            if not stat.S_ISREG(path.lstat().st_mode):
                raise ValueError("native ACE artifacts must be regular files")
            if not name.endswith(".whl"):
                continue
            if name not in expected or name in found:
                raise ValueError("native ACE artifact has an extra or duplicate wheel")
            found[name] = path
    if found.keys() != expected.keys():
        raise ValueError(
            "native ACE publication requires every fixed inventory profile"
        )
    selected: list[Path] = []
    for name, package in sorted(expected.items()):
        path = found[name]
        with path.open("rb") as source:
            before = os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size != package.size_bytes:
                raise ValueError(
                    "native ACE artifact size differs from the fixed inventory"
                )
            digest = hashlib.sha256()
            measured = 0
            while chunk := source.read(1024 * 1024):
                measured += len(chunk)
                if measured > package.size_bytes:
                    raise ValueError("native ACE artifact grew during verification")
                digest.update(chunk)
            after = os.fstat(source.fileno())
        current = path.lstat()
        if (
            measured != package.size_bytes
            or digest.hexdigest() != package.sha256
            or _file_identity(before) != _file_identity(after)
            or _file_identity(after) != _file_identity(current)
        ):
            raise ValueError(
                "native ACE artifact changed or differs from the fixed inventory"
            )
        selected.append(path)
    return tuple(selected)


def main() -> None:
    """Print only the complete verified inventory; expose no pin override option."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    arguments = _Arguments()
    _ = parser.parse_args(namespace=arguments)
    for wheel in verify_artifacts(arguments.directory):
        print(wheel)


if __name__ == "__main__":
    main()
