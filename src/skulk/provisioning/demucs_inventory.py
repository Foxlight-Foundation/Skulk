"""Exact native separator package selection; availability/readiness are separate.

These CPU candidate bytes passed actual sealed offline execution on each target.
The Mac CPU package also contains the native MPS libraries. Selection alone
does not publish a file, execute Python, download weights or advertise capacity.
The cache must independently verify the object and explicit device before use.
"""

from collections.abc import Set as AbstractSet
from typing import Final

from skulk.provisioning.demucs_runtime import DemucsWheel
from skulk.shared.types.audio_separation import SeparationDevice

DEMUCS_ADAPTER_REVISION: Final = "78c9fd8e808e71f504b6d05f747673033fbf8211"
DEMUCS_PACKAGES: Final[tuple[DemucsWheel, ...]] = (
    DemucsWheel(
        platform_tag="macosx_14_0_arm64",
        filename="skulk_demucs_cpu-0.1.0-cp312-cp312-macosx_14_0_arm64.whl",
        size_bytes=139356688,
        sha256="b85a2055bda726bc35bd6f4e030e620ce21266679f99da78428d5b429fe5f851",
        payload_sha256="7c9cb53127c5afb546b31117d97c9d66db1cca0958e9c9f7e53920e7d282ffc8",
    ),
    DemucsWheel(
        platform_tag="manylinux_2_28_x86_64",
        filename="skulk_demucs_cpu-0.1.0-cp312-cp312-manylinux_2_28_x86_64.whl",
        size_bytes=286237374,
        sha256="957c53394d6aa6c68d096e912733bb93f1d7c972d1b55669d767ec451058da60",
        payload_sha256="167d88232729f5a9cdb60fbbfed403857bd389b1bed7f1f9ae9fb5d8596e5586",
    ),
    DemucsWheel(
        platform_tag="manylinux_2_28_aarch64",
        filename="skulk_demucs_cpu-0.1.0-cp312-cp312-manylinux_2_28_aarch64.whl",
        size_bytes=194196337,
        sha256="ead9721c4a63bb28a121d80353bf096c02cf98bd77c5c984a99b87682f879f03",
        payload_sha256="f96f752d8b8011613bd6c7852ceae2003fbd80ae3addac591fadbc0501859494",
    ),
)


def select_demucs_package(
    compatible_platforms: AbstractSet[str], *, device: SeparationDevice
) -> DemucsWheel:
    """Choose exact bytes by host ABI and implemented compute, never test history.

    ``compatible_platforms`` comes from the selected node's packaging platform
    tags, which include its real OS/libc baseline and CPU architecture. CUDA and
    ROCm need their own dependency profiles; a CPU wheel cannot acquire them
    because the node happens to have a GPU. The native probe remains mandatory.
    """
    for package in DEMUCS_PACKAGES:
        if package.platform_tag not in compatible_platforms:
            continue
        if (
            device == "cpu"
            or device == "mps"
            and package.platform_tag.startswith("macosx_")
        ):
            return package
    raise ValueError(
        "no native separation package implements this host ABI and compute lane"
    )
