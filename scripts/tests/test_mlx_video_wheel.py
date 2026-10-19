"""Offline dependency ABI and exact-byte promotion checks."""

from pathlib import Path

import pytest

from scripts.build_mlx_video_wheel import _dependency_target_matches
from scripts.verify_mlx_video_promotion import verify_artifact


@pytest.mark.parametrize("filename,compatible", [
    ("mlx-0.32.2-cp313-cp313-macosx_15_0_arm64.whl", True),
    ("psutil-7.2.1-cp36-abi3-macosx_11_0_arm64.whl", True),
    ("certifi-1.0-py3-none-any.whl", True),
    ("lib-1.0-cp313-cp313-manylinux_2_35_aarch64.whl", False),
    ("lib-1.0-cp312-cp312-macosx_15_0_arm64.whl", False),
    ("lib-1.0-cp313-cp313-macosx_16_0_arm64.whl", False),
    ("lib-1.0-cp313-cp313-macosx_15_0_x86_64.whl", False),
])
def test_lock_digest_does_not_override_wheel_abi(filename: str, compatible: bool) -> None:
    assert _dependency_target_matches(filename) == compatible


@pytest.mark.parametrize("digest", ["", "../other", "a" * 63, "Z" * 64])
def test_promotion_requires_an_explicit_complete_digest(tmp_path: Path, digest: str) -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        verify_artifact(tmp_path, digest)
