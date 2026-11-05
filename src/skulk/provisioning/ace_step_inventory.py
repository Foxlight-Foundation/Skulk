"""Exact native ACE package inventory; candidate availability is never readiness.

These immutable objects reproduce from one pinned adapter and upstream source.
Target installation, actual device probing, full model load and music quality
remain distinct evidence. Selection has no filesystem or preparation effects.
"""

from collections.abc import Set as AbstractSet
from typing import Final

from skulk.provisioning.ace_step_runtime import AceStepWheel
from skulk.shared.types.native_music import NativeMusicDevice

ACE_STEP_ADAPTER_REVISION: Final = "042d263010787ae8aa0140088d8b3dc0453d43f8"
ACE_STEP_PACKAGES: Final[tuple[AceStepWheel, ...]] = (
    AceStepWheel(
        variant="cpu",
        platform_tag="macosx_14_0_arm64",
        filename="skulk_ace_step_cpu-0.1.0-cp312-cp312-macosx_14_0_arm64.whl",
        size_bytes=183436900,
        sha256="ec50b161aa0953d5305bec31a75b1a6ce73b3df44d0b96ad1fc3bf07c75e538b",
        payload_sha256="d58924e0dca7c05f9b2cd40f9d2494500408443eb8a19b72c43940c31867736b",
    ),
    AceStepWheel(
        variant="cpu",
        platform_tag="manylinux_2_28_x86_64",
        filename="skulk_ace_step_cpu-0.1.0-cp312-cp312-manylinux_2_28_x86_64.whl",
        size_bytes=332864753,
        sha256="b5a3497f80fcdbfc602835ff5a257527004ea7e379fa88e796c27dd47bb7d16e",
        payload_sha256="24f920d42af3f1f67f3c69119be96153d2870da218b0a9859e58228053ab327b",
    ),
    AceStepWheel(
        variant="cpu",
        platform_tag="manylinux_2_28_aarch64",
        filename="skulk_ace_step_cpu-0.1.0-cp312-cp312-manylinux_2_28_aarch64.whl",
        size_bytes=282007403,
        sha256="d3d664ce12bc71710f9be381648a9dfd78f1f71232f0d38e7348ffe049693a5f",
        payload_sha256="d514d683b165f5a4619658af17df1255698706adb346c3b7e7a6dc4e9563821a",
    ),
    AceStepWheel(
        variant="cuda",
        platform_tag="manylinux_2_28_x86_64",
        filename="skulk_ace_step_cuda-0.1.0-cp312-cp312-manylinux_2_28_x86_64.whl",
        size_bytes=2954455388,
        sha256="bdf7118395caea119a78c7cd2ba5af29d93f38f718348d0ffb6ea6705984bde2",
        payload_sha256="f5c5d01dce9a59c507dff216d7c28a855a18575a2e95d8666a59c67b68af012e",
    ),
    AceStepWheel(
        variant="cuda",
        platform_tag="manylinux_2_28_aarch64",
        filename="skulk_ace_step_cuda-0.1.0-cp312-cp312-manylinux_2_28_aarch64.whl",
        size_bytes=3047331933,
        sha256="7c89ced0251dc44a17126bf257e28bbb18d90de00dee535a010c6695e19cb750",
        payload_sha256="9d5c577104731d4cd7ef2fa71fd4ee235c25524c73b6123c6ae4b83e58133f2c",
    ),
    AceStepWheel(
        variant="rocm",
        platform_tag="linux_x86_64",
        filename="skulk_ace_step_rocm-0.1.0-cp312-cp312-linux_x86_64.whl",
        size_bytes=2206547200,
        sha256="898a5bf9123eb98d098ce15d613012d35d1a5fe87179d4d77437181cca790254",
        payload_sha256="1949b71ef01c79a801d8a1d8715de59af8b7702f7a836afe86ccf44758289d38",
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
