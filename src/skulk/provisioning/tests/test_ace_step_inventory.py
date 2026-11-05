"""Fixed ABI/profile coverage is distinct from executed hardware/model support."""

import pytest

from skulk.provisioning.ace_step_inventory import (
    ACE_STEP_PACKAGES,
    select_ace_step_package,
)
from skulk.shared.types.native_music import NativeMusicDevice


def test_exact_inventory_covers_all_six_built_dependency_profiles() -> None:
    """Do not silently lose an architecture or its independent GPU dependency set."""
    assert {
        (package.platform_tag, package.variant) for package in ACE_STEP_PACKAGES
    } == {
        ("macosx_14_0_arm64", "cpu"),
        ("manylinux_2_28_x86_64", "cpu"),
        ("manylinux_2_28_aarch64", "cpu"),
        ("manylinux_2_28_x86_64", "cuda"),
        ("manylinux_2_28_aarch64", "cuda"),
        ("linux_x86_64", "rocm"),
    }
    assert len({package.filename for package in ACE_STEP_PACKAGES}) == 6
    assert len({package.sha256 for package in ACE_STEP_PACKAGES}) == 6
    assert len({package.payload_sha256 for package in ACE_STEP_PACKAGES}) == 6


@pytest.mark.parametrize(
    ("platforms", "device", "variant"),
    [
        ({"macosx_14_0_arm64"}, "cpu", "cpu"),
        ({"macosx_14_0_arm64"}, "mps", "cpu"),
        ({"manylinux_2_28_x86_64", "linux_x86_64"}, "cpu", "cpu"),
        ({"manylinux_2_28_x86_64", "linux_x86_64"}, "cuda", "cuda"),
        ({"manylinux_2_28_x86_64", "linux_x86_64"}, "rocm", "rocm"),
        ({"manylinux_2_28_aarch64"}, "cpu", "cpu"),
        ({"manylinux_2_28_aarch64"}, "cuda", "cuda"),
    ],
)
def test_implemented_package_selection_uses_abi_and_explicit_lane(
    platforms: set[str],
    device: NativeMusicDevice,
    variant: str,
) -> None:
    """Synthetic ABI selection has no physical machine identity or quality claim."""
    package = select_ace_step_package(platforms, device=device)
    assert package.variant == variant
    assert package.required_platform_tags.issubset(platforms)


@pytest.mark.parametrize(
    ("platforms", "device"),
    [
        ({"win_amd64"}, "cpu"),
        ({"macosx_13_0_arm64"}, "cpu"),
        ({"manylinux_2_27_x86_64"}, "cpu"),
        ({"linux_x86_64"}, "rocm"),
        ({"manylinux_2_28_aarch64"}, "rocm"),
        ({"macosx_14_0_arm64"}, "cuda"),
        ({"manylinux_2_28_x86_64"}, "mps"),
    ],
)
def test_missing_abi_or_gpu_profile_cannot_fall_back_to_another_package(
    platforms: set[str],
    device: NativeMusicDevice,
) -> None:
    """A GPU request never succeeds by installing a CPU-only package instead."""
    with pytest.raises(ValueError, match="host ABI and compute lane"):
        select_ace_step_package(platforms, device=device)
