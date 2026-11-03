"""Fetch only the committed ACE CPU archive inventory, without index resolution."""

from __future__ import annotations

import argparse
import hashlib
from http.client import HTTPResponse
from pathlib import Path
from typing import cast
from urllib.request import Request, urlopen

from build_ace_step import ArchiveInventory
from package_inputs import PROFILE_TARGETS, file_digest, read_python_catalog


def fetch_object(url: str, destination: Path, size_bytes: int, sha256: str) -> None:
    """Fetch a bounded immutable object or verify an existing ordinary cache file.

    Exclusive staging and link publication prevent overwriting another owner.
    Neither version labels nor a transport success can replace the selected size
    and complete digest. Failed work removes only its own partial file.
    """
    if destination.is_symlink():
        raise ValueError("native archive cache cannot be a symbolic link")
    if not destination.exists():
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


def fetch_profile(root: Path, profile: str, output: Path) -> None:
    """Collect the exact committed objects for one closed architecture profile."""
    target = PROFILE_TARGETS.get(profile)
    if target is None:
        raise ValueError("unknown native music archive profile")
    with (root / f"archives/{profile}-cpu-py312.json").open("rb") as stream:
        inventory_data = stream.read(2 * 1024**2 + 1)
    if len(inventory_data) > 2 * 1024**2:
        raise ValueError("native music archive inventory exceeds its bound")
    inventory = ArchiveInventory.model_validate_json(inventory_data)
    pin = read_python_catalog(root).for_target(target)
    with (root / f"locks/{profile}-cpu-py312.lock").open("rb") as stream:
        lock = stream.read(2 * 1024**2 + 1)
    if len(lock) > 2 * 1024**2:
        raise ValueError("native music intake lock exceeds its bound")
    if (
        inventory.target != target
        or inventory.python_archive != pin
        or hashlib.sha256(lock).hexdigest() != inventory.lock_sha256
    ):
        raise ValueError(
            "native music intake does not match its selected lock/interpreter"
        )
    wheels = output / "wheels"
    wheels.mkdir(parents=True, exist_ok=True)
    expected = {item.filename for item in inventory.objects}
    if any(path.name not in expected for path in wheels.iterdir()):
        raise ValueError("native music wheel cache contains an unselected object")
    for item in inventory.objects:
        fetch_object(
            item.source_url, wheels / item.filename, item.size_bytes, item.sha256
        )
        print(f"{profile}: verified {item.filename}", flush=True)
    python = output / "python"
    python.mkdir(exist_ok=True)
    fetch_object(pin.url, python / pin.filename, pin.size_bytes, pin.sha256)
    with (output / "wheel-receipt.json").open("w") as receipt:
        receipt.write(inventory.model_dump_json(indent=2) + "\n")


class Arguments(argparse.Namespace):
    """One local selected profile and an explicitly owned intake directory."""

    profile: str
    output: Path


def main() -> None:
    """Collect pinned public archives without publication, installation or model downloads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=tuple(PROFILE_TARGETS))
    parser.add_argument("--output", type=Path, required=True)
    arguments = Arguments()
    parser.parse_args(namespace=arguments)
    fetch_profile(Path(__file__).parent, arguments.profile, arguments.output)


if __name__ == "__main__":
    main()
