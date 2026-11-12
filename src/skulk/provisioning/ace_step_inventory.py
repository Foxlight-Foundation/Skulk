"""Exact native ACE package inventory; candidate availability is never readiness.

Each immutable object reproduces from its pinned adapter and upstream source.
Target installation, actual device probing, full model load and music quality
remain distinct evidence. Selection has no filesystem or preparation effects.
"""

from collections.abc import Set as AbstractSet
from typing import Final

from skulk.provisioning.ace_step_runtime import AceStepWheel
from skulk.shared.types.native_music import NativeMusicDevice

ACE_STEP_ADAPTER_REVISION: Final = "1c4c3efbf9a6b0711c9814023c9cb9a802293352"
ACE_STEP_ROCM_ADAPTER_REVISION: Final = "12a8335c271d2476e79b388a09e45a0629bb3fb4"
ACE_STEP_PACKAGES: Final[tuple[AceStepWheel, ...]] = (
    AceStepWheel(
        variant="cpu",
        platform_tag="macosx_14_0_arm64",
        filename="skulk_ace_step_cpu-0.1.0-3-cp312-cp312-macosx_14_0_arm64.whl",
        size_bytes=183449246,
        sha256="3a31cf953b6206400daf9a493a82fa3dc1d8cf8eb063a5a1913ea78ffdb390d2",
        payload_sha256="abd70c7f702231a3d0be5241d94b3e5b0548f2c9b8de1d7cf3f8c302cacc6f67",
    ),
    AceStepWheel(
        variant="cpu",
        platform_tag="manylinux_2_28_x86_64",
        filename="skulk_ace_step_cpu-0.1.0-3-cp312-cp312-manylinux_2_28_x86_64.whl",
        size_bytes=332877099,
        sha256="22449de57f29f6833e30aa62a397ce758499d190385542489b10d23d6a4baf51",
        payload_sha256="9a0edf4998eb3fe7d0fd86652c7ab9179a242a9d87715916a6f6628c6b90ced8",
    ),
    AceStepWheel(
        variant="cpu",
        platform_tag="manylinux_2_28_aarch64",
        filename="skulk_ace_step_cpu-0.1.0-3-cp312-cp312-manylinux_2_28_aarch64.whl",
        size_bytes=282019749,
        sha256="10deebb256f5e1e1e41fc8d4a9172a92b4164d7185aa32a3d7c6ab3453e992b2",
        payload_sha256="eacc5b98ef396b9ab6ed38590cf7945bc7e8cf7086e2db831375cbbbcaa8a97a",
    ),
    AceStepWheel(
        variant="cuda",
        platform_tag="manylinux_2_28_x86_64",
        filename="skulk_ace_step_cuda-0.1.0-3-cp312-cp312-manylinux_2_28_x86_64.whl",
        size_bytes=2954467734,
        sha256="33b306bb1364bfacb718108c90c041d23668740305eb9f00f8ea215f5b09919e",
        payload_sha256="74d1b69c076cbd46029531c9fa8b19f111a104f2d2272295c249a9d7ee344f9c",
    ),
    AceStepWheel(
        variant="cuda",
        platform_tag="manylinux_2_28_aarch64",
        filename="skulk_ace_step_cuda-0.1.0-3-cp312-cp312-manylinux_2_28_aarch64.whl",
        size_bytes=3047344279,
        sha256="4eb05c9977171d59bb7ff3126007ae553cb8b68650574e8f5e2e62b26fb8d971",
        payload_sha256="622f7fcb88fc38fb5e6c3ee83f9c89ef93e52903a7923593e29c67cc0403602f",
    ),
    AceStepWheel(
        variant="rocm",
        platform_tag="linux_x86_64",
        filename="skulk_ace_step_rocm-0.1.0-5-cp312-cp312-linux_x86_64.whl",
        size_bytes=2206561827,
        sha256="17ef6097626fd350e96b848ce79e9793fbe0bd7a7e4f2a627f6fa2d3988dd1b7",
        payload_sha256="25d72ddfc85964a3f042fc144baab1aacdbce49ce4844a712f0a6e91b086542d",
    ),
)


def ace_step_package_revision(package: AceStepWheel) -> str:
    """Return the fixed producer revision for one exact built-in package.

    ROCm may advance independently while already qualified CPU/Metal/CUDA
    objects retain their bytes and source provenance. Reject modified or unknown
    records rather than deriving trust from a profile name or caller digest.
    This lookup performs no preparation, publication or readiness operation.
    """
    if package not in ACE_STEP_PACKAGES:
        raise ValueError(
            "native ACE producer requires an exact fixed inventory package"
        )
    return (
        ACE_STEP_ROCM_ADAPTER_REVISION
        if package.variant == "rocm"
        else ACE_STEP_ADAPTER_REVISION
    )


def select_ace_step_package(
    compatible_platforms: AbstractSet[str], *, device: NativeMusicDevice
) -> AceStepWheel:
    """Select implemented bytes by observed host ABI and requested math lane.

    Hardware qualification is not a permission list. A real installed library
    probe still decides device/driver availability; a CPU package supplies
    Metal only on macOS, never CUDA or ROCm by inference from a GPU name.
    """
    for package in ACE_STEP_PACKAGES:
        if not package.required_platform_tags.issubset(compatible_platforms):
            continue
        if package.variant == device or (
            device == "mps"
            and package.variant == "cpu"
            and package.platform_tag.startswith("macosx_")
        ):
            return package
    raise ValueError("no native ACE package implements this host ABI and compute lane")
