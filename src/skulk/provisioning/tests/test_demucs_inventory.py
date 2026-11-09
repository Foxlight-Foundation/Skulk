"""Native candidate selection follows real ABI/compute requirements, not claims."""

import pytest

from skulk.provisioning.demucs_inventory import DEMUCS_PACKAGES, select_demucs_package
from skulk.provisioning.demucs_runtime import DemucsWheel
from skulk.shared.types.audio_separation import SeparationDevice


@pytest.mark.parametrize("package", DEMUCS_PACKAGES)
def test_each_cpu_target_selects_its_exact_bytes_without_qualification(
    package: DemucsWheel,
) -> None:
    """All supported OS/CPU targets have explicit independent original pins."""
    selected = select_demucs_package(frozenset({package.platform_tag}), device="cpu")
    assert selected == package
    assert selected.sha256 != selected.payload_sha256
    assert selected.url.endswith("/wheels/" + selected.filename)


def test_apple_cpu_package_can_be_probed_for_explicit_mps() -> None:
    """Mac native MPS library presence is eligible; selection is not readiness."""
    selected = select_demucs_package(frozenset({"macosx_14_0_arm64"}), device="mps")
    assert selected.platform_tag == "macosx_14_0_arm64"


@pytest.mark.parametrize("device", ["cuda", "rocm", "mps"])
def test_linux_cpu_package_cannot_pretend_to_provide_acceleration(
    device: SeparationDevice,
) -> None:
    """Visible hardware cannot manufacture missing compiled/runtime dependencies."""
    with pytest.raises(ValueError, match="host ABI and compute"):
        select_demucs_package(frozenset({"manylinux_2_28_x86_64"}), device=device)


def test_wrong_os_architecture_or_libc_is_not_an_eligible_package() -> None:
    """An older libc or unrelated CPU has a genuine package compatibility gap."""
    for tags in (
        frozenset({"manylinux_2_17_x86_64"}),
        frozenset({"macosx_13_0_arm64"}),
        frozenset({"manylinux_2_28_riscv64"}),
        frozenset({"win_amd64"}),
    ):
        with pytest.raises(ValueError, match="host ABI and compute"):
            select_demucs_package(tags, device="cpu")
