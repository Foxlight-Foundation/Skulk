"""Exact native ACE package inventory; candidate availability is never readiness.

These immutable objects reproduce from one pinned adapter and upstream source.
Target installation, actual device probing, full model load and music quality
remain distinct evidence. Selection has no filesystem or preparation effects.
"""

from collections.abc import Set as AbstractSet
from typing import Final

from skulk.provisioning.ace_step_runtime import AceStepWheel
from skulk.shared.types.native_music import NativeMusicDevice

ACE_STEP_ADAPTER_REVISION: Final = "60c7910f253d32699cb53d48d6d6fb9c67c8fa06"
ACE_STEP_PACKAGES: Final[tuple[AceStepWheel, ...]] = (
    AceStepWheel(
        variant="cpu",
        platform_tag="macosx_14_0_arm64",
        filename="skulk_ace_step_cpu-0.1.0-1-cp312-cp312-macosx_14_0_arm64.whl",
        size_bytes=183439837,
        sha256="e3303deec4273760f5a4d03ceb2f34dad518dc91a7cb954a0880984586de4912",
        payload_sha256="3a68a3ce8aff0fd01d9cd03f3fec2734deeb18fefba084375512fadeca08d1d3",
    ),
    AceStepWheel(
        variant="cpu",
        platform_tag="manylinux_2_28_x86_64",
        filename="skulk_ace_step_cpu-0.1.0-1-cp312-cp312-manylinux_2_28_x86_64.whl",
        size_bytes=332867690,
        sha256="799542baffe7a9a65a3fe6f17024ad87639e3fe69d32a2718a66307f34c2d3ed",
        payload_sha256="e89fb007bcdfdef33c59eef5302744a65516f2580f160da2cbd3d7dd7bfa0345",
    ),
    AceStepWheel(
        variant="cpu",
        platform_tag="manylinux_2_28_aarch64",
        filename="skulk_ace_step_cpu-0.1.0-1-cp312-cp312-manylinux_2_28_aarch64.whl",
        size_bytes=282010340,
        sha256="0c02ff7e723ad637925281012f5155fb281bcce700bc5158605fb414801f023e",
        payload_sha256="fc703f4f7398370dcc1297e4a9714307c979dc04c7fcb46ae17e8b298a96976b",
    ),
    AceStepWheel(
        variant="cuda",
        platform_tag="manylinux_2_28_x86_64",
        filename="skulk_ace_step_cuda-0.1.0-1-cp312-cp312-manylinux_2_28_x86_64.whl",
        size_bytes=2954458325,
        sha256="926fb89ec9f7d60a29178d2522a9db89605ba9802c90ad7fcdf9c10f82a4097c",
        payload_sha256="d6e01697eb78d98757536b39925b47265c94dae9d3792abf7eb924b30b04580d",
    ),
    AceStepWheel(
        variant="cuda",
        platform_tag="manylinux_2_28_aarch64",
        filename="skulk_ace_step_cuda-0.1.0-1-cp312-cp312-manylinux_2_28_aarch64.whl",
        size_bytes=3047334870,
        sha256="ad1e86f3816ed85e6bfca6948ef87489810f36b5e72903515c28d4b98c3da37b",
        payload_sha256="86cf334e4e5398db5e8900004e201c2f9a32ab5ccca6e2f714c2e2e814f37943",
    ),
    AceStepWheel(
        variant="rocm",
        platform_tag="linux_x86_64",
        filename="skulk_ace_step_rocm-0.1.0-1-cp312-cp312-linux_x86_64.whl",
        size_bytes=2206550137,
        sha256="4c2e6fb651193644536cbe91a9ba8b44bc842b595167011390b630b647b02e9a",
        payload_sha256="9d9f7e342a6d8475eaf543df59dde1fdb3f6c7114bdb320336f738f1423f5751",
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
