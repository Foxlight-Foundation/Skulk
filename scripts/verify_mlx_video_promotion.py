"""Verify explicitly selected, attested MLX video candidate bytes before promotion."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from scripts.build_mlx_video_wheel import FILENAME, SOURCE_REVISION


def verify_artifact(directory: Path, expected_sha256: str) -> Path:
    """Require the exact source, package identity and explicit complete wheel digest."""
    if len(expected_sha256) != 64 or any(character not in "0123456789abcdef" for character in expected_sha256):
        raise ValueError("promotion requires one complete lowercase SHA-256 digest")
    wheel = directory / FILENAME
    digest = hashlib.sha256()
    with wheel.open("rb") as stream:
        while chunk := stream.read(1024**2):
            digest.update(chunk)
    descriptor = json.loads(wheel.with_suffix(".json").read_text())
    if digest.hexdigest() != expected_sha256 or descriptor.get("sha256") != expected_sha256 or descriptor.get("filename") != FILENAME or descriptor.get("source_revision") != SOURCE_REVISION or descriptor.get("size_bytes") != wheel.stat().st_size:
        raise ValueError("candidate bytes or source differ from the selected descriptor")
    if not 0 < wheel.stat().st_size <= 512 * 1024**2:
        raise ValueError("candidate exceeds the installation bound")
    return wheel


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: verify_mlx_video_promotion DIRECTORY SHA256")
    print(verify_artifact(Path(sys.argv[1]), sys.argv[2]))
