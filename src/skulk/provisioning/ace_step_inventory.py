"""Exact native ACE package inventory; candidate availability is never readiness.

These immutable objects reproduce from one pinned adapter and upstream source.
Target installation, actual device probing, full model load and music quality
remain distinct evidence. Selection has no filesystem or preparation effects.
"""

from collections.abc import Set as AbstractSet
from typing import Final

from skulk.provisioning.ace_step_runtime import AceStepWheel
from skulk.shared.types.native_music import NativeMusicDevice

ACE_STEP_ADAPTER_REVISION: Final = "113facdc998c7dd34054d43f54270fbf5c9e09b8"
ACE_STEP_PACKAGES: Final[tuple[AceStepWheel, ...]] = (
    AceStepWheel(
        variant="cpu",
        platform_tag="macosx_14_0_arm64",
        filename="skulk_ace_step_cpu-0.1.0-2-cp312-cp312-macosx_14_0_arm64.whl",
        size_bytes=183442498,
        sha256="6f0ba9f6295a89ca1caff5f9c77d5a2ae9d87bfc608a68a969f19e1d0b2e7907",
        payload_sha256="2d65d5168d3fa88d1946f163f04ac87b54a58b1d6f4abe0a807c188cbfd839a1",
    ),
    AceStepWheel(
        variant="cpu",
        platform_tag="manylinux_2_28_x86_64",
        filename="skulk_ace_step_cpu-0.1.0-2-cp312-cp312-manylinux_2_28_x86_64.whl",
        size_bytes=332870351,
        sha256="10e8723a5892141ed7d7dd0a7eeeeffce68570e950c7273b46d56d59bef509c2",
        payload_sha256="55f17c8b7b1d61eccc989395e631e12edb931d1d536cc577d873bd1bdaeaa0c8",
    ),
    AceStepWheel(
        variant="cpu",
        platform_tag="manylinux_2_28_aarch64",
        filename="skulk_ace_step_cpu-0.1.0-2-cp312-cp312-manylinux_2_28_aarch64.whl",
        size_bytes=282013001,
        sha256="85957a90246bf9b7eb94c4514fd4a22c9b274ce3d713f5afe5f42db5c391db3b",
        payload_sha256="43ea5349f83af9e0266673678f300c0bbf56964f4c95148d920804d8fa32e340",
    ),
    AceStepWheel(
        variant="cuda",
        platform_tag="manylinux_2_28_x86_64",
        filename="skulk_ace_step_cuda-0.1.0-2-cp312-cp312-manylinux_2_28_x86_64.whl",
        size_bytes=2954460986,
        sha256="3af27f5b3d25e4d1b9dd559103f54613c2e1dab0adb220d82cc7332f4adc5416",
        payload_sha256="c18f007507299cc20bfd46b51652d5a8cabac9f32554edfd3a373a4d79c58148",
    ),
    AceStepWheel(
        variant="cuda",
        platform_tag="manylinux_2_28_aarch64",
        filename="skulk_ace_step_cuda-0.1.0-2-cp312-cp312-manylinux_2_28_aarch64.whl",
        size_bytes=3047337531,
        sha256="7f76f313e8c00a6d0e6934723ed722846954304010a1ca0d25934b1ca480bb2a",
        payload_sha256="6c1abb340f0fdbfda22c5c27db31a4657f39b4475ced7c6352331d574ba2fcfe",
    ),
    AceStepWheel(
        variant="rocm",
        platform_tag="linux_x86_64",
        filename="skulk_ace_step_rocm-0.1.0-2-cp312-cp312-linux_x86_64.whl",
        size_bytes=2206552798,
        sha256="044af5fbeb0e446ef573a1ab0fbc65c2824fa9fdb35ceafcbfce62757539f9bc",
        payload_sha256="23476787f667055c512e0a58693da6ffa6d27881eafa76874d0da7771c52691f",
    ),
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
