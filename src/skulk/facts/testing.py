"""Synthetic Node Facts factories for tests.

The whole capability pipeline (derivation, doctor, health) is pure over
:class:`~skulk.shared.types.node_facts.NodeFacts`, so tests build records here
instead of touching hardware. Shipped as a non-test module because both the
facts and doctor test suites (and any future consumer's tests) share it.
"""

from __future__ import annotations

from skulk.shared.types.node_facts import (
    EngineBinaryFact,
    GpuDeviceFact,
    LlamaServerDeviceProbe,
    NodeFacts,
)

NVIDIA_A40 = GpuDeviceFact(
    vendor="nvidia",
    name="NVIDIA A40",
    detection_source="nvml",
    vram_total_bytes=48 * 2**30,
    compute_capability="8.6",
)
"""A fully detected discrete NVIDIA GPU (the NVML happy path)."""

NVIDIA_PRESENCE_ONLY = GpuDeviceFact(
    vendor="nvidia", detection_source="nvidia_device_node"
)
"""An NVIDIA device visible without NVML (the #612 degraded shape)."""

AMD_STRIX = GpuDeviceFact(
    vendor="amd",
    name="AMD GPU",
    detection_source="amdgpu_sysfs",
    vram_total_bytes=8 * 2**30,
    gtt_total_bytes=120 * 2**30,
)
"""A unified-memory AMD APU (Strix Halo shape: small carve, large GTT)."""

APPLE_GPU = GpuDeviceFact(vendor="apple", detection_source="apple_platform")
"""The implicit Apple Silicon GPU on Darwin."""


def make_facts(
    *,
    platform: str = "linux",
    gpus: tuple[GpuDeviceFact, ...] = (),
    llama_cpp_importable: bool = False,
    llama_cpp_gpu_offload: bool | None = None,
    mlx_audio_importable: bool = False,
    pynvml_importable: bool = False,
    llama_server_bin: EngineBinaryFact | None = None,
    vllm_bin: EngineBinaryFact | None = None,
    rpc_bin: EngineBinaryFact | None = None,
    device_probe: LlamaServerDeviceProbe | None = None,
    declared_llama_cpp: str | None = None,
    declared_llama_server: str | None = None,
    declared_vllm: str | None = None,
) -> NodeFacts:
    """Build a synthetic facts record with unconfigured-binary defaults."""
    return NodeFacts(
        platform=platform,
        gpus=gpus,
        pynvml_importable=pynvml_importable,
        llama_cpp_importable=llama_cpp_importable,
        llama_cpp_gpu_offload=llama_cpp_gpu_offload,
        mlx_audio_importable=mlx_audio_importable,
        llama_server_binary=llama_server_bin
        or EngineBinaryFact(env_var="SKULK_LLAMA_SERVER_BIN"),
        vllm_binary=vllm_bin or EngineBinaryFact(env_var="SKULK_VLLM_BIN"),
        rpc_server_binary=rpc_bin or EngineBinaryFact(env_var="SKULK_RPC_SERVER_BIN"),
        llama_server_device_probe=device_probe or LlamaServerDeviceProbe(),
        declared_llama_cpp_backends=declared_llama_cpp,
        declared_llama_server_backends=declared_llama_server,
        declared_vllm_backends=declared_vllm,
    )


def ok_bin(env_var: str) -> EngineBinaryFact:
    """A configured engine binary that resolves to an executable file."""
    return EngineBinaryFact(
        env_var=env_var, configured_path=f"/opt/{env_var}", state="ok"
    )


def bad_bin(env_var: str) -> EngineBinaryFact:
    """A configured engine binary whose path does not exist (#462 shape)."""
    return EngineBinaryFact(
        env_var=env_var, configured_path=f"/opt/{env_var}", state="missing"
    )
