"""Fetch a closed ACE archive profile without index resolution or installation."""

from __future__ import annotations

import argparse
import hashlib
import json
from http.client import HTTPResponse
from pathlib import Path
from typing import cast
from urllib.request import Request, urlopen

from build_ace_step import ArchiveInventory
from package_inputs import (
    PROFILE_TARGETS,
    NativePackageVariant,
    PythonArchivePin,
    file_digest,
    read_python_catalog,
)


def fetch_object(url: str | None, destination: Path, size_bytes: int, sha256: str) -> None:
    """Fetch a bounded immutable object or verify an existing ordinary cache file.

    Exclusive staging and link publication prevent overwriting another owner.
    Neither version labels nor a transport success can replace the selected size
    and complete digest. Failed work removes only its own partial file.
    """
    if destination.is_symlink():
        raise ValueError("native archive cache cannot be a symbolic link")
    if not destination.exists():
        if url is None:
            raise ValueError(
                f"source-built native object {destination.name} must be reproduced "
                "from its selected source/tool pins before archive collection"
            )
        partial = destination.with_name(destination.name + ".part")
        output = partial.open("xb")
        try:
            digest, size = hashlib.sha256(), 0
            with (
                output,
                cast(
                    HTTPResponse,
                    cast(
                        object,
                        urlopen(
                            Request(
                                url,
                                headers={"User-Agent": "Foxlight-native-audio-intake"},
                            ),
                            timeout=120,
                        ),
                    ),
                ) as response,
            ):
                while block := response.read(1024 * 1024):
                    size += len(block)
                    if size > size_bytes:
                        raise ValueError(
                            "selected native object exceeds its admitted size"
                        )
                    digest.update(block)
                    output.write(block)
            if size != size_bytes or digest.hexdigest() != sha256:
                raise ValueError(
                    "downloaded native object differs from its selected identity"
                )
            destination.hardlink_to(partial)
        finally:
            partial.unlink(missing_ok=True)
    if (
        not destination.is_file()
        or destination.stat().st_size != size_bytes
        or file_digest(destination, limit=size_bytes) != sha256
    ):
        raise ValueError("cached native archive differs from its selected identity")


def _profile_inputs(
    root: Path, profile: str, variant: NativePackageVariant
) -> tuple[ArchiveInventory, PythonArchivePin]:
    interpreter_target = PROFILE_TARGETS.get(profile)
    if interpreter_target is None:
        raise ValueError("unknown native music archive profile")
    if (
        variant == "rocm" and profile != "linux-amd64"
        or variant == "cuda" and profile == "macos-arm64"
    ):
        raise ValueError("native dependency variant does not provide this architecture profile")
    target = "linux_x86_64" if variant == "rocm" else interpreter_target
    with (root / f"archives/{profile}-{variant}-py312.json").open("rb") as stream:
        inventory_data = stream.read(2 * 1024**2 + 1)
    if len(inventory_data) > 2 * 1024**2:
        raise ValueError("native music archive inventory exceeds its bound")
    inventory = ArchiveInventory.model_validate_json(inventory_data)
    pin = read_python_catalog(root).for_target(interpreter_target)
    with (root / f"locks/{profile}-{variant}-py312.lock").open("rb") as stream:
        lock = stream.read(2 * 1024**2 + 1)
    if len(lock) > 2 * 1024**2:
        raise ValueError("native music intake lock exceeds its bound")
    if (
        inventory.target != target
        or inventory.variant != variant
        or inventory.python_archive != pin
        or hashlib.sha256(lock).hexdigest() != inventory.lock_sha256
    ):
        raise ValueError(
            "native music intake does not match its selected lock/interpreter"
        )
    return inventory, pin


def fetch_profile(
    root: Path, profile: str, output: Path, *, variant: NativePackageVariant = "cpu"
) -> None:
    """Collect the exact committed objects for one closed architecture profile."""
    inventory, pin = _profile_inputs(root, profile, variant)
    wheels = output / "wheels"
    wheels.mkdir(parents=True, exist_ok=True)
    expected = {item.filename for item in inventory.objects}
    if any(path.name not in expected for path in wheels.iterdir()):
        raise ValueError("native music wheel cache contains an unselected object")
    for item in inventory.objects:
        fetch_object(
            item.source_url, wheels / item.filename, item.size_bytes, item.sha256
        )
        print(f"{profile}/{variant}: verified {item.filename}", flush=True)
    python = output / "python"
    python.mkdir(exist_ok=True)
    fetch_object(pin.url, python / pin.filename, pin.size_bytes, pin.sha256)
    with (output / "wheel-receipt.json").open("w") as receipt:
        receipt.write(inventory.model_dump_json(indent=2) + "\n")


def fetch_metapackage_inputs(root: Path, builder_profile: str, output: Path) -> None:
    """Fetch the pure ROCm source-build prerequisites without executing them.

    The fixed native target remains Linux x86_64. Its pure metapackage may be
    reproduced with any pinned builder interpreter, keeping its explicitly
    selected dependency family independent from the build host.
    """
    inventory, _ = _profile_inputs(root, "linux-amd64", "rocm")
    selected = tuple(item for item in inventory.objects if item.source_build is not None)
    if len(selected) != 1:
        raise ValueError("ROCm source build needs one exact metapackage")
    package = selected[0]
    build = package.source_build
    assert build is not None
    target = PROFILE_TARGETS.get(builder_profile)
    if target is None:
        raise ValueError("unknown standalone native builder profile")
    python = read_python_catalog(root).for_target(target)
    output.mkdir(parents=True, exist_ok=True)
    source_name = f"rocm-{package.version}.tar.gz"
    fetch_object(
        build.source_url, output / source_name,
        build.source_size_bytes, build.source_sha256,
    )
    # Native Triton's runtime Setuptools need not match the producer's tool.
    # Reproducing the metapackage uses its separately captured tool identity.
    tool_name = f"setuptools-{build.setuptools}-py3-none-any.whl"
    fetch_object(
        build.setuptools_source_url, output / tool_name,
        build.setuptools_size_bytes, build.setuptools_archive_sha256,
    )
    fetch_object(python.url, output / python.filename, python.size_bytes, python.sha256)
    (output / "build-inputs.json").write_text(json.dumps(dict(
        builder_profile=builder_profile,
        python_archive=python.filename,
        source_archive=source_name,
        setuptools_wheel=tool_name,
        source_sha256=build.source_sha256,
        setuptools_sha256=build.setuptools_archive_sha256,
        python_sha256=python.sha256,
    ), indent=2) + "\n")


class Arguments(argparse.Namespace):
    """One local selected profile and an explicitly owned intake directory."""

    profile: str
    output: Path
    variant: NativePackageVariant
    metapackage_inputs: bool


def main() -> None:
    """Collect pinned public archives without publication, installation or model downloads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=tuple(PROFILE_TARGETS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("cpu", "cuda", "rocm"), default="cpu")
    parser.add_argument("--metapackage-inputs", action="store_true")
    arguments = Arguments()
    parser.parse_args(namespace=arguments)
    if arguments.metapackage_inputs:
        if arguments.variant != "rocm":
            raise ValueError("source-build intake is implemented only for the ROCm metapackage")
        fetch_metapackage_inputs(Path(__file__).parent, arguments.profile, arguments.output)
    else:
        fetch_profile(
            Path(__file__).parent, arguments.profile, arguments.output,
            variant=arguments.variant,
        )


if __name__ == "__main__":
    main()
