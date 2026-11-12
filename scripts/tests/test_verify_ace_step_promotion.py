"""Publication selection cannot silently omit or substitute a native platform."""

import hashlib
from pathlib import Path

import pytest

from scripts.verify_ace_step_promotion import verify_artifacts
from skulk.provisioning.ace_step_inventory import ACE_STEP_PACKAGES
from skulk.provisioning.ace_step_runtime import AceStepWheel


def candidates(directory: Path) -> tuple[AceStepWheel, ...]:
    """Retain all six real profile names with tiny, independently pinned test bytes."""
    selected: list[AceStepWheel] = []
    for index, package in enumerate(ACE_STEP_PACKAGES):
        parent = directory / f"artifact-{index}"
        parent.mkdir(parents=True)
        content = f"independently pinned fixture {index}".encode()
        (parent / package.filename).write_bytes(content)
        selected.append(
            AceStepWheel(
                variant=package.variant,
                platform_tag=package.platform_tag,
                filename=package.filename,
                size_bytes=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
                payload_sha256=package.payload_sha256,
            )
        )
    return tuple(selected)


def test_all_six_fixed_profiles_are_verified_before_selection(tmp_path: Path) -> None:
    """Nested CI artifacts keep their actual names and deterministic complete order."""
    packages = candidates(tmp_path)
    result = verify_artifacts(tmp_path, packages=packages)
    assert tuple(path.name for path in result) == tuple(
        sorted(p.filename for p in packages)
    )
    assert len(result) == 6


@pytest.mark.parametrize("index", range(6))
def test_each_missing_architecture_or_lane_refuses_the_whole_publish(
    tmp_path: Path, index: int
) -> None:
    """CPU availability never hides an omitted NVIDIA or AMD archive."""
    packages = candidates(tmp_path)
    (tmp_path / f"artifact-{index}" / packages[index].filename).unlink()
    with pytest.raises(ValueError, match="every fixed"):
        verify_artifacts(tmp_path, packages=packages)


@pytest.mark.parametrize(
    "mutation", ["size", "digest", "duplicate", "extra", "symlink", "directory-symlink"]
)
def test_artifact_substitution_and_link_escapes_fail_before_any_upload(
    tmp_path: Path, mutation: str
) -> None:
    """Names alone, a matching size or a neighboring linked copy earn no trust."""
    packages = candidates(tmp_path)
    original = tmp_path / "artifact-0" / packages[0].filename
    if mutation == "size":
        original.write_bytes(b"short")
    elif mutation == "digest":
        original.write_bytes(b"x" * original.stat().st_size)
    elif mutation == "duplicate":
        (tmp_path / packages[0].filename).write_bytes(original.read_bytes())
    elif mutation == "extra":
        (tmp_path / "unreviewed.whl").write_bytes(b"extra")
    elif mutation == "symlink":
        content = original.read_bytes()
        original.unlink()
        elsewhere = tmp_path / "linked-original"
        elsewhere.write_bytes(content)
        original.symlink_to(elsewhere)
    else:
        (tmp_path / "linked-directory").symlink_to(
            tmp_path / "artifact-0", target_is_directory=True
        )
    with pytest.raises(ValueError):
        verify_artifacts(tmp_path, packages=packages)


def test_empty_and_duplicated_trusted_inventory_are_configuration_errors(
    tmp_path: Path,
) -> None:
    """An accidental empty selection never succeeds as a zero-object publication."""
    packages = candidates(tmp_path)
    for wrong in [(), (packages[0], packages[0])]:
        with pytest.raises(ValueError, match="inventory"):
            verify_artifacts(tmp_path, packages=wrong)


def test_empty_directory_forest_cannot_bypass_the_traversal_bound(
    tmp_path: Path,
) -> None:
    """Downloaded artifact expansion stays bounded even without regular members."""
    packages = candidates(tmp_path)
    for index in range(257):
        (tmp_path / f"empty-{index}").mkdir()
    with pytest.raises(ValueError, match="file bound"):
        verify_artifacts(tmp_path, packages=packages)
