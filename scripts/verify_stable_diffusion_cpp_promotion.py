"""Select only exact hardware-qualified native image wheel bytes for publication."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import cast

from scripts.build_stable_diffusion_cpp_wheel import PATCH_SHA256, SOURCE_REVISION

_WHEEL = re.compile(
    r"skulk_stable_diffusion_cpp_(?P<variant>metal|cuda|vulkan|rocm)-"
    r"[0-9]+(?:\.[0-9]+)*(?:\.post[0-9]+)?-py3-none-"
    r"(?P<platform>macosx_15_0_arm64|manylinux_2_35_x86_64|manylinux_2_35_aarch64)\.whl"
)
_TARGETS = frozenset({
    ("metal", "macosx_15_0_arm64"),
    ("cuda", "manylinux_2_35_x86_64"),
    ("cuda", "manylinux_2_35_aarch64"),
    ("vulkan", "manylinux_2_35_x86_64"),
    ("rocm", "manylinux_2_35_x86_64"),
})
_DIGEST = re.compile(r"[0-9a-f]{64}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate publication input: {key}")
        result[key] = value
    return result


def verify_artifacts(directory: Path, expected_sha256: str) -> tuple[Path, ...]:
    """Return only explicit digest-matching targets from a qualified build run.

    The expected filename/digest map is the promotion decision. Extra downloaded
    targets remain unpublished. Each selected wheel also needs its matching
    immutable source/payload/size descriptor; build success alone grants no model
    or device claim. GitHub attestation verification is performed by the caller.
    """
    decoded = cast("object", json.loads(expected_sha256, object_pairs_hook=_unique_object))
    if not isinstance(decoded, dict) or not 1 <= len(decoded) <= len(_TARGETS):
        raise ValueError("promotion requires one to five filename-to-digest entries")
    selected: list[Path] = []
    for name, digest in cast("dict[object, object]", decoded).items():
        if not isinstance(name, str) or not isinstance(digest, str):
            raise ValueError("wheel filenames and digests must be strings")
        match = _WHEEL.fullmatch(name)
        if match is None or (match.group("variant"), match.group("platform")) not in _TARGETS:
            raise ValueError("unexpected native image wheel target")
        if _DIGEST.fullmatch(digest) is None:
            raise ValueError("expected SHA-256 must be a complete lowercase digest")
        wheel = directory / name
        actual = hashlib.sha256()
        with wheel.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                actual.update(chunk)
        if actual.hexdigest() != digest:
            raise ValueError("selected native image wheel bytes differ from qualification")
        descriptor = cast("dict[str, object]", json.loads(wheel.with_suffix(".json").read_text(), object_pairs_hook=_unique_object))
        if (
            descriptor.get("filename") != name
            or descriptor.get("sha256") != digest
            or descriptor.get("compute") != match.group("variant")
            or descriptor.get("source_revision") != SOURCE_REVISION
            or descriptor.get("server_patch_sha256") != PATCH_SHA256
            or not isinstance(descriptor.get("payload_sha256"), str)
            or _DIGEST.fullmatch(cast("str", descriptor["payload_sha256"])) is None
            or descriptor.get("size_bytes") != wheel.stat().st_size
        ):
            raise ValueError("selected wheel descriptor differs from its qualified identity")
        expanded = descriptor.get("expanded_size_bytes")
        if (
            not 0 < wheel.stat().st_size <= 3 * 1024**3
            or type(expanded) is not int
            or not 0 < cast("int", expanded) <= 8 * 1024**3
        ):
            raise ValueError("selected wheel exceeds the bounded installation contract")
        selected.append(wheel)
    return tuple(sorted(selected))


def main() -> None:
    """Print the verified publication paths; reject incomplete promotion arguments."""
    if len(sys.argv) != 3:
        raise SystemExit("usage: verify_stable_diffusion_cpp_promotion.py DIRECTORY SHA256_MAP")
    for wheel in verify_artifacts(Path(sys.argv[1]), sys.argv[2]):
        print(wheel)


if __name__ == "__main__":
    main()
