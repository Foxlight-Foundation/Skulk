"""Verify exact attested Comfy extension bytes before append-only publication."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import cast

from scripts.build_comfy_ltx_wheel import (
    COMFY_REVISION,
    FEATURES,
    FILENAME,
    LICENSE_FILES,
    LTX_REVISION,
    PACKAGE,
    SOURCE_FILES,
    VERSION,
)


def _object(data: bytes) -> dict[str, object]:
    value = cast("object", json.loads(data))
    if not isinstance(value, dict):
        raise ValueError("candidate metadata must be a JSON object")
    return cast("dict[str, object]", value)


def verify_artifact(directory: Path, expected_sha256: str) -> Path:
    """Require one explicit complete digest, reviewed source and authenticated payload.

    Attestation verification remains a separate workflow step bound to the
    successful candidate run's source digest. This function never publishes.
    """
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("promotion requires one complete lowercase SHA-256 digest")
    wheel = directory / FILENAME
    if set(directory.glob("*.whl")) != {wheel} or wheel.is_symlink() or not 0 < wheel.stat().st_size <= 4 * 1024**2:
        raise ValueError("candidate requires one bounded regular selected wheel")
    descriptor_path = wheel.with_suffix(".json")
    if descriptor_path.is_symlink() or descriptor_path.stat().st_size > 16384:
        raise ValueError("candidate descriptor is not a bounded regular file")
    descriptor = _object(descriptor_path.read_bytes())
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != expected_sha256 or descriptor.get("sha256") != expected_sha256 or descriptor.get("filename") != FILENAME or descriptor.get("size_bytes") != wheel.stat().st_size:
        raise ValueError("candidate bytes differ from the selected descriptor")
    if descriptor.get("comfy_revision") != COMFY_REVISION or descriptor.get("ltx_revision") != LTX_REVISION or descriptor.get("features") != list(FEATURES):
        raise ValueError("candidate source or features differ from the reviewed build")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names) or len(names) > 160 or sum(member.file_size for member in archive.infolist()) > 16 * 1024**2:
            raise ValueError("candidate archive exceeds its unique-member or byte bound")
        expected = {f"{PACKAGE}/{name}" for name in (*SOURCE_FILES, *LICENSE_FILES)}
        package_names = {name for name in names if name.startswith(f"{PACKAGE}/")}
        if package_names != {*expected, f"{PACKAGE}/manifest.json"}:
            raise ValueError("candidate package differs from the reviewed file set")
        members = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in sorted(expected)}
        payload_hash = hashlib.sha256(json.dumps(members, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        manifest = _object(archive.read(f"{PACKAGE}/manifest.json"))
        if descriptor.get("payload_sha256") != payload_hash or manifest != {"protocol": 1, "version": VERSION, "comfy_revision": COMFY_REVISION, "ltx_revision": LTX_REVISION, "features": list(FEATURES), "payload_sha256": payload_hash, "members": members}:
            raise ValueError("candidate manifest differs from its authenticated source payload")
    return wheel


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: verify_comfy_ltx_promotion DIRECTORY SHA256")
    print(verify_artifact(Path(sys.argv[1]), sys.argv[2]))
