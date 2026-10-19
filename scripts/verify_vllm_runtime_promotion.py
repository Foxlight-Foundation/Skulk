"""Verify a complete six-wheel vLLM promotion against trusted manifest hashes."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import cast


def verify_promotion(directory: Path, expected: dict[str, str]) -> tuple[Path, ...]:
    """Verify every recipe/mirror digest and reject missing, extra or changed bytes.

    The source manifest independently binds all expected artifacts. Returns only
    fully verified paths for the publisher; this function never writes anything.
    """
    source = (
        Path(__file__).resolve().parents[1]
        / "src/skulk/provisioning/vllm_manifest.json"
    )
    recipes = cast(list[dict[str, str]], json.loads(source.read_text()))
    trusted = {
        recipe[filename]: recipe[digest]
        for recipe in recipes
        for filename, digest in (
            ("filename", "sha256"),
            ("upstream_filename", "upstream_sha256"),
        )
    }
    if expected != trusted:
        raise ValueError(
            "promotion selection does not match the complete pinned manifest"
        )
    wheels = {path.name: path for path in directory.glob("*.whl")}
    if set(wheels) != set(trusted):
        raise ValueError("promotion contains missing or unexpected wheels")
    for name, digest in trusted.items():
        builder = hashlib.sha256()
        with wheels[name].open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                builder.update(chunk)
        if builder.hexdigest() != digest:
            raise ValueError(f"promotion checksum mismatch: {name}")
    return tuple(wheels[name] for name in sorted(wheels))


def main() -> None:
    """Print only verified paths, failing before any publication on mismatch."""
    if len(sys.argv) != 3:
        raise ValueError(
            "usage: verify_vllm_runtime_promotion.py DIRECTORY EXPECTED_JSON"
        )
    expected = cast(dict[str, str], json.loads(sys.argv[2]))
    for wheel in verify_promotion(Path(sys.argv[1]), expected):
        print(wheel)


if __name__ == "__main__":
    main()
