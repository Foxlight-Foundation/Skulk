"""Promotion grants only explicit qualified native image artifact bytes."""

import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_stable_diffusion_cpp_wheel import PATCH_SHA256, SOURCE_REVISION
from scripts.verify_stable_diffusion_cpp_promotion import verify_artifacts


def _candidate(root: Path, *, compute: str = "metal", platform: str = "macosx_15_0_arm64") -> tuple[Path, str]:
    wheel = root / f"skulk_stable_diffusion_cpp_{compute}-0.1.0.post1-py3-none-{platform}.whl"
    wheel.write_bytes(b"qualified wheel bytes")
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    wheel.with_suffix(".json").write_text(json.dumps({
        "filename": wheel.name, "sha256": digest, "compute": compute,
        "source_revision": SOURCE_REVISION, "server_patch_sha256": PATCH_SHA256,
        "payload_sha256": "b" * 64, "size_bytes": wheel.stat().st_size,
        "expanded_size_bytes": 42,
    }))
    return wheel, digest


def test_promotion_never_includes_an_unselected_architecture(tmp_path: Path) -> None:
    wheel, digest = _candidate(tmp_path)
    _candidate(tmp_path, compute="cuda", platform="manylinux_2_35_aarch64")
    assert verify_artifacts(tmp_path, json.dumps({wheel.name: digest})) == (wheel,)


def test_promotion_rejects_changed_qualified_bytes(tmp_path: Path) -> None:
    wheel, digest = _candidate(tmp_path)
    wheel.write_bytes(b"replacement bytes")
    with pytest.raises(ValueError, match="bytes differ"):
        verify_artifacts(tmp_path, json.dumps({wheel.name: digest}))


@pytest.mark.parametrize("change", ["source_revision", "server_patch_sha256", "payload_sha256", "size_bytes", "expanded_size_bytes"])
def test_promotion_rejects_descriptor_drift(tmp_path: Path, change: str) -> None:
    wheel, digest = _candidate(tmp_path)
    descriptor = wheel.with_suffix(".json")
    body = json.loads(descriptor.read_text())
    body[change] = 9 * 1024**3 if change.endswith("bytes") else "wrong"
    descriptor.write_text(json.dumps(body))
    with pytest.raises(ValueError):
        verify_artifacts(tmp_path, json.dumps({wheel.name: digest}))


def test_promotion_rejects_duplicate_and_unsupported_targets(tmp_path: Path) -> None:
    wheel, digest = _candidate(tmp_path, compute="vulkan", platform="manylinux_2_35_aarch64")
    with pytest.raises(ValueError, match="target"):
        verify_artifacts(tmp_path, json.dumps({wheel.name: digest}))
    with pytest.raises(ValueError, match="duplicate"):
        verify_artifacts(tmp_path, '{"a":"b","a":"c"}')
