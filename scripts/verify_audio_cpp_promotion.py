"""Verify the complete digest-pinned audio.cpp artifact set before promotion."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Literal, cast

AudioCppVariant = Literal["cpu", "vulkan", "cuda"]
_DIGEST = re.compile(r"[0-9a-f]{64}")
_CPU_WHEEL = re.compile(
    r"skulk_audio_cpp_cpu-(?P<version>[0-9]+(?:\.[0-9]+)*(?:\.post[0-9]+)?)-"
    r"py3-none-(?P<platform>macosx_15_0_arm64|manylinux_2_35_x86_64|"
    r"manylinux_2_35_aarch64)\.whl"
)
_CPU_PLATFORMS = frozenset(
    {"macosx_15_0_arm64", "manylinux_2_35_x86_64", "manylinux_2_35_aarch64"}
)
_GPU_WHEEL = re.compile(
    r"skulk_audio_cpp_(?P<variant>vulkan|cuda)-"
    r"(?P<version>[0-9]+(?:\.[0-9]+)*(?:\.post[0-9]+)?)-"
    r"py3-none-(?P<platform>manylinux_2_35_x86_64|manylinux_2_35_aarch64)\.whl"
)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate expected wheel: {key}")
        result[key] = value
    return result


def verify_artifacts(
    directory: Path, variant: AudioCppVariant, expected_sha256: str,
) -> tuple[Path, ...]:
    """Check selected wheel bytes and return only the verified publication set.

    CPU promotion requires a JSON filename-to-digest map covering all three
    supported platforms at one package version. GPU promotion accepts a
    one-entry filename-to-digest map to select one qualified platform, or the
    legacy digest when exactly one wheel exists. Only returned paths may be
    published; downloading another platform does not authorize its promotion.
    """
    wheels = tuple(sorted(directory.glob(f"skulk_audio_cpp_{variant}-*.whl")))
    if variant == "cpu":
        decoded = cast("object", json.loads(expected_sha256, object_pairs_hook=_unique_object))
        if not isinstance(decoded, dict):
            raise ValueError("CPU promotion requires a filename-to-SHA-256 JSON object")
        expected: dict[str, str] = {}
        for name, digest in cast("dict[object, object]", decoded).items():
            if not isinstance(name, str) or not isinstance(digest, str):
                raise ValueError("CPU wheel names and digests must be strings")
            expected[name] = digest
        if len(wheels) != 3 or set(expected) != {wheel.name for wheel in wheels}:
            raise ValueError("CPU promotion requires exactly the three selected wheels")
        versions: set[str] = set()
        platforms: set[str] = set()
        for wheel in wheels:
            match = _CPU_WHEEL.fullmatch(wheel.name)
            if match is None:
                raise ValueError(f"unexpected CPU wheel platform or filename: {wheel.name}")
            versions.add(match.group("version"))
            platforms.add(match.group("platform"))
        if len(versions) != 1 or frozenset(platforms) != _CPU_PLATFORMS:
            raise ValueError("CPU promotion requires one version across all supported platforms")
    elif expected_sha256.lstrip().startswith("{"):
        decoded = cast("object", json.loads(expected_sha256, object_pairs_hook=_unique_object))
        if not isinstance(decoded, dict) or len(decoded) != 1:
            raise ValueError("GPU promotion requires exactly one filename-to-SHA-256 entry")
        name, expected_digest = next(iter(cast("dict[object, object]", decoded).items()))
        if not isinstance(name, str) or not isinstance(expected_digest, str):
            raise ValueError("GPU wheel name and digest must be strings")
        selected = _GPU_WHEEL.fullmatch(name)
        if selected is None or selected.group("variant") != variant:
            raise ValueError("unexpected selected GPU wheel filename")
        platforms: set[str] = set()
        for wheel in wheels:
            match = _GPU_WHEEL.fullmatch(wheel.name)
            if (
                match is None
                or match.group("variant") != variant
                or match.group("version") != selected.group("version")
                or (variant == "vulkan" and match.group("platform") != "manylinux_2_35_x86_64")
                or match.group("platform") in platforms
            ):
                raise ValueError(f"unexpected additional GPU wheel: {wheel.name}")
            platforms.add(match.group("platform"))
        if name not in {wheel.name for wheel in wheels}:
            raise ValueError("selected GPU wheel is missing")
        expected = {name: expected_digest}
        wheels = (directory / name,)
    else:
        if len(wheels) != 1:
            raise ValueError("GPU promotion requires exactly one selected wheel")
        expected = {wheels[0].name: expected_sha256}
    for wheel in wheels:
        expected_digest = expected[wheel.name]
        if _DIGEST.fullmatch(expected_digest) is None:
            raise ValueError(f"invalid SHA-256 for {wheel.name}")
        digest = hashlib.sha256()
        with wheel.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected_digest:
            raise ValueError(f"SHA-256 mismatch for {wheel.name}")
    return wheels


def main() -> None:
    """Print the only paths authorized for attestation and publication."""
    if len(sys.argv) != 4 or sys.argv[2] not in {"cpu", "vulkan", "cuda"}:
        raise SystemExit("usage: verify_audio_cpp_promotion.py DIRECTORY VARIANT EXPECTED_SHA256")
    variant = cast(AudioCppVariant, sys.argv[2])
    for wheel in verify_artifacts(Path(sys.argv[1]), variant, sys.argv[3]):
        print(wheel)


if __name__ == "__main__":
    main()
