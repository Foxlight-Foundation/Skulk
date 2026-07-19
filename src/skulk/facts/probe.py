"""Gathering side of Node Facts: observe hardware, software, and declarations.

Every probe here is passive and degradation-safe: a failing or absent source
yields "not observed" (never a fabricated value, never an exception out of
:func:`gather_node_facts`). Effects are injectable so the whole record can be
produced synthetically in tests without hardware.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import cast, final

from loguru import logger

from skulk.shared.types.node_facts import (
    EngineBinaryFact,
    EngineBinaryState,
    GpuDeviceFact,
    LlamaServerDeviceProbe,
    NodeFacts,
)

# Import lazily-safe helpers from the existing collectors rather than growing
# sibling probe logic (#614: nobody probes ad hoc; existing gatherers become
# sources for this one record).
from skulk.utils.info_gatherer.nvidia_gpu import NvmlLike, load_nvml


@final
class _MemoryInfoLike:
    """The slice of NVML's memory-info struct this probe reads."""

    total: int


_DRM_ROOT = Path("/sys/class/drm")

# Filesystem markers that an NVIDIA device is present even when NVML is not
# importable: the kernel driver's proc node, the device nodes udev creates, and
# the vendor CLI. Any one of them is positive evidence of hardware the node
# cannot fully detect (#612).
_NVIDIA_PRESENCE_PATHS = (
    Path("/proc/driver/nvidia/version"),
    Path("/dev/nvidia0"),
)

# How long `llama-server --list-devices` may take before we call the probe
# failed. CUDA context initialization on a cold driver can take several
# seconds; a hung binary must not stall node startup indefinitely.
_LIST_DEVICES_TIMEOUT_SECONDS = 20.0

# Device lines look like "  CUDA0: NVIDIA A100 80GB PCIe (81037 MiB, ...)" /
# "  Vulkan0: AMD Radeon Graphics (RADV GFX1151) ...". The backend prefix maps
# onto our compute vocabulary; HIP is ROCm's runtime name in some builds.
_DEVICE_LINE = re.compile(r"^\s*([A-Za-z]+)\d+\s*:")
_DEVICE_PREFIX_TO_COMPUTE = {
    "cuda": "cuda",
    "vulkan": "vulkan",
    "rocm": "rocm",
    "hip": "rocm",
}


def _nvidia_gpu_facts(nvml: NvmlLike | None) -> tuple[GpuDeviceFact, ...]:
    """Observe every NVIDIA device NVML can enumerate (full-detection path)."""
    if nvml is None:
        return ()
    try:
        count = nvml.nvmlDeviceGetCount()
    except Exception:  # noqa: BLE001 - NVML error means "no observed devices"
        return ()
    facts: list[GpuDeviceFact] = []
    for index in range(count):
        name = "Unknown"
        vram_total: int | None = None
        capability: str | None = None
        try:
            handle = nvml.nvmlDeviceGetHandleByIndex(index)
        except Exception:  # noqa: BLE001 - device-level degradation
            continue
        try:
            raw = nvml.nvmlDeviceGetName(handle)
            name = raw.decode() if isinstance(raw, bytes) else raw
        except Exception:  # noqa: BLE001 - per-field degradation
            pass
        try:
            memory = cast("_MemoryInfoLike", nvml.nvmlDeviceGetMemoryInfo(handle))
            vram_total = int(memory.total)
        except Exception:  # noqa: BLE001
            pass
        try:
            major, minor = nvml.nvmlDeviceGetCudaComputeCapability(handle)
            capability = f"{int(major)}.{int(minor)}"
        except Exception:  # noqa: BLE001
            pass
        facts.append(
            GpuDeviceFact(
                vendor="nvidia",
                name=name,
                index=index,
                detection_source="nvml",
                vram_total_bytes=vram_total,
                compute_capability=capability,
            )
        )
    return tuple(facts)


def nvidia_device_visibly_present() -> bool:
    """Whether an NVIDIA device is visible without NVML (presence-only evidence).

    The driver's proc node and device nodes are direct evidence. ``nvidia-smi``
    merely being on PATH is not (CUDA toolkit images run without a GPU, and
    CPU dev boxes carry the tools): it counts only when it actually runs and
    lists a device, so a GPU-less toolkit box cannot fabricate a presence fact
    and cascade into false conflicts or a GPU engine download.
    """
    if any(path.exists() for path in _NVIDIA_PRESENCE_PATHS):
        return True
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return False
    try:
        completed = subprocess.run(  # noqa: S603 - fixed, known-safe command
            [nvidia_smi, "-L"], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0 and "GPU" in completed.stdout


def _read_sysfs_int(path: Path) -> int | None:
    """Read a single integer from a sysfs file, or ``None`` if unavailable."""
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _amd_gpu_facts(drm_root: Path) -> tuple[GpuDeviceFact, ...]:
    """Observe every amdgpu render device under ``drm_root`` (passive sysfs)."""
    try:
        candidates = sorted(drm_root.glob("card[0-9]*/device"))
    except OSError:
        return ()
    facts: list[GpuDeviceFact] = []
    for index, device in enumerate(candidates):
        if not (device / "gpu_busy_percent").is_file():
            continue
        facts.append(
            GpuDeviceFact(
                vendor="amd",
                name="AMD GPU",
                index=index,
                detection_source="amdgpu_sysfs",
                vram_total_bytes=_read_sysfs_int(device / "mem_info_vram_total"),
                gtt_total_bytes=_read_sysfs_int(device / "mem_info_gtt_total"),
            )
        )
    return tuple(facts)


def _binary_fact(env_var: str, env: Mapping[str, str]) -> EngineBinaryFact:
    """Classify one engine-binary declaration into an :class:`EngineBinaryFact`."""
    configured = env.get(env_var, "").strip()
    if not configured:
        return EngineBinaryFact(env_var=env_var)
    state: EngineBinaryState
    if not os.path.isfile(configured):
        state = "missing"
    elif not os.access(configured, os.X_OK):
        state = "not_executable"
    else:
        state = "ok"
    return EngineBinaryFact(env_var=env_var, configured_path=configured, state=state)


def parse_list_devices_output(output: str) -> tuple[str, ...]:
    """Parse ``llama-server --list-devices`` output into compute backends.

    Returns the deduplicated compute backends (``cuda``/``vulkan``/``rocm``)
    that the binary reported at least one device for, in first-seen order.
    Metal and CPU device lines are deliberately not mapped: Metal serving is
    MLX's domain in our vocabulary, and CPU is the implicit floor.
    """
    computes: list[str] = []
    for line in output.splitlines():
        match = _DEVICE_LINE.match(line)
        if match is None:
            continue
        compute = _DEVICE_PREFIX_TO_COMPUTE.get(match.group(1).lower())
        if compute is not None and compute not in computes:
            computes.append(compute)
    return tuple(computes)


def probe_llama_server_devices(binary: str) -> LlamaServerDeviceProbe:
    """Ask a llama-server binary which devices its build can drive.

    Runs ``<binary> --list-devices`` under a hard timeout. The flag exits
    nonzero on builds that predate it (``unsupported``); crashes, timeouts, and
    launch failures are ``failed``. Both fall back to hardware-vendor
    derivation, so a probe failure can never make a node less capable than the
    pre-probe behavior.
    """
    try:
        completed = subprocess.run(  # noqa: S603 - operator-configured binary
            [binary, "--list-devices"],
            capture_output=True,
            text=True,
            timeout=_LIST_DEVICES_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return LlamaServerDeviceProbe(
            outcome="failed",
            detail=f"--list-devices timed out after {_LIST_DEVICES_TIMEOUT_SECONDS:.0f}s",
        )
    except OSError as error:
        return LlamaServerDeviceProbe(outcome="failed", detail=str(error))
    output = completed.stdout + "\n" + completed.stderr
    if completed.returncode != 0:
        # Old builds reject the flag with an "invalid argument" usage error;
        # anything else (driver fault, missing shared library) is a failure.
        if "--list-devices" in output or "invalid argument" in output.lower():
            return LlamaServerDeviceProbe(outcome="unsupported")
        detail = output.strip().splitlines()[-1] if output.strip() else ""
        return LlamaServerDeviceProbe(
            outcome="failed", detail=f"exit {completed.returncode}: {detail}"[:200]
        )
    return LlamaServerDeviceProbe(
        outcome="devices", computes=parse_list_devices_output(output)
    )


def gather_node_facts(
    *,
    env: Mapping[str, str] | None = None,
    nvml: NvmlLike | None = None,
    drm_root: Path = _DRM_ROOT,
    platform: str | None = None,
    nvidia_presence: bool | None = None,
) -> NodeFacts:
    """Gather one complete :class:`NodeFacts` record for this node.

    Args:
        env: Environment mapping to read declarations from (injected in tests;
            defaults to the process environment).
        nvml: An initialized NVML surface, or ``None`` to load the real one.
        drm_root: The drm sysfs root to scan for amdgpu devices.
        platform: ``sys.platform`` override for tests.
        nvidia_presence: Override for the NVML-less NVIDIA presence check
            (injected in tests; defaults to probing the real filesystem).

    Returns:
        The observed-and-declared record. Never raises: every failing source
        degrades to its "not observed" form.
    """
    # Env var names live in skulk.shared.backends (the vocabulary module);
    # imported here at call time because backends' probe entry point delegates
    # back into this package (function-level import on both sides keeps the
    # module graph acyclic).
    from skulk.shared.backends import (
        LLAMA_CPP_BACKENDS_ENV,
        LLAMA_SERVER_BACKENDS_ENV,
        LLAMA_SERVER_BIN_ENV,
        RPC_SERVER_BIN_ENV,
        VLLM_BACKENDS_ENV,
        VLLM_BIN_ENV,
    )

    env = env if env is not None else os.environ
    platform = platform if platform is not None else sys.platform

    pynvml_importable = False
    try:
        import pynvml  # noqa: F401  # pyright: ignore[reportMissingImports, reportUnusedImport]

        pynvml_importable = True
    except Exception as error:  # noqa: BLE001 - never-raises contract: a broken
        # install must degrade to "not importable", not crash the probe.
        logger.debug(f"pynvml not importable: {error}")

    gpus: list[GpuDeviceFact] = []
    if platform == "darwin":
        gpus.append(
            GpuDeviceFact(
                vendor="apple", name="Apple GPU", detection_source="apple_platform"
            )
        )
    else:
        nvml_surface = nvml if nvml is not None else load_nvml()
        nvidia = _nvidia_gpu_facts(nvml_surface)
        visibly_present = (
            nvidia_presence
            if nvidia_presence is not None
            else nvidia_device_visibly_present()
        )
        if not nvidia and visibly_present:
            # Presence-only evidence: the hardware is there but the node cannot
            # learn its name or VRAM (#612's degraded state, made visible).
            nvidia = (
                GpuDeviceFact(vendor="nvidia", detection_source="nvidia_device_node"),
            )
        gpus.extend(nvidia)
        gpus.extend(_amd_gpu_facts(drm_root))

    llama_cpp_importable = False
    llama_cpp_gpu_offload: bool | None = None
    try:
        import llama_cpp  # pyright: ignore[reportMissingImports]

        llama_cpp_importable = True
        try:
            # llama_cpp ships no type stubs, so the member + result are Unknown.
            llama_cpp_gpu_offload = bool(llama_cpp.llama_supports_gpu_offload())  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        except Exception:  # noqa: BLE001 - binding/ABI quirk means "unverifiable"
            llama_cpp_gpu_offload = None
    except Exception as error:  # noqa: BLE001 - native extension loads can fail
        # with OSError/ABI errors, not just ImportError; degrade, never raise.
        logger.debug(f"llama_cpp not importable: {error}")

    mlx_audio_importable = False
    if platform == "darwin":
        try:
            import mlx_audio  # noqa: F401  # pyright: ignore[reportMissingTypeStubs, reportUnusedImport]

            mlx_audio_importable = True
        except Exception as error:  # noqa: BLE001 - native wheels can fail with ABI errors
            logger.debug(f"mlx_audio not importable: {error}")

    llama_server_binary = _binary_fact(LLAMA_SERVER_BIN_ENV, env)
    vllm_binary = _binary_fact(VLLM_BIN_ENV, env)
    rpc_binary = _binary_fact(RPC_SERVER_BIN_ENV, env)

    declared_llama_cpp = env.get(LLAMA_CPP_BACKENDS_ENV)
    declared_llama_server = env.get(LLAMA_SERVER_BACKENDS_ENV)
    declared_vllm = env.get(VLLM_BACKENDS_ENV)

    # Probe the binary's own device list only when there is a usable binary
    # and no SERVER-specific declaration answers the question. A llama.cpp
    # declaration deliberately does NOT suppress the probe: it describes the
    # in-process binding's build, not this binary, and using it to skip the
    # probe would let e.g. a CUDA llama-cpp declaration mask a managed Vulkan
    # server build that cannot actually drive the GPU (PR #615 review).
    device_probe = LlamaServerDeviceProbe()
    declaration_answers = bool((declared_llama_server or "").strip())
    if llama_server_binary.state == "ok" and not declaration_answers:
        assert llama_server_binary.configured_path is not None
        device_probe = probe_llama_server_devices(llama_server_binary.configured_path)

    return NodeFacts(
        platform=platform,
        gpus=tuple(gpus),
        pynvml_importable=pynvml_importable,
        llama_cpp_importable=llama_cpp_importable,
        llama_cpp_gpu_offload=llama_cpp_gpu_offload,
        mlx_audio_importable=mlx_audio_importable,
        llama_server_binary=llama_server_binary,
        vllm_binary=vllm_binary,
        rpc_server_binary=rpc_binary,
        llama_server_device_probe=device_probe,
        declared_llama_cpp_backends=declared_llama_cpp,
        declared_llama_server_backends=declared_llama_server,
        declared_vllm_backends=declared_vllm,
    )
