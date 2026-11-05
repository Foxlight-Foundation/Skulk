"""Explicit native device execution and observed component precision policy."""

from __future__ import annotations

import importlib
import os
import platform
import subprocess
import time
from typing import Literal, Protocol, cast

import torch
from torch.version import hip

from .contracts import (
    NativeMusicDevice,
    NativeMusicExecution,
    NativeMusicLibraryProbe,
    NativeMusicMemoryCapacity,
    NativeMusicPrecision,
    NativeMusicServeConfig,
    NativeMusicTensorPrecision,
    native_music_host_available_bytes,
)
from .upstream_bindings import LoadedHandler


class _DeviceProperties(Protocol):
    """Pinned Torch 2.10 CUDA/HIP properties exposed by its native binding."""

    is_integrated: int


class _HostMemory(Protocol):
    """Read-only physical fields from psutil's platform-specific memory tuple."""

    total: int
    available: int


class _HostMemoryAPI(Protocol):
    """Narrow pinned psutil sampling signature where its return stub is untyped."""

    def virtual_memory(self) -> _HostMemory:
        """Return actual physical total and currently available host bytes."""
        ...


class _DeviceMemoryAPI(Protocol):
    """Pinned CUDA/HIP memory binding without its incomplete property stub."""

    def get_device_properties(self, device: Literal[0]) -> _DeviceProperties:
        """Return the driver's properties for exactly the selected device."""
        ...

    def mem_get_info(self, device: Literal[0]) -> tuple[int, int]:
        """Return current free and total bytes for exactly that selected device."""
        ...


# Fixed owned-module bindings keep the audited interfaces strict despite the
# libraries' incomplete tuple/property stubs. No module name is caller supplied.
psutil = cast(_HostMemoryAPI, cast(object, importlib.import_module("psutil")))


def sample_memory_capacity(lane: NativeMusicDevice) -> NativeMusicMemoryCapacity:
    """Observe host and exactly the device this private runtime will use.

    No GPU sum, largest-device selection or model loading occurs. Torch 2.10's
    CUDA/HIP binding exposes the actual driver's integrated flag for device zero:
    https://github.com/pytorch/pytorch/blob/v2.10.0/torch/csrc/cuda/Module.cpp
    A failed or incomplete query is a diagnostic, never invented capacity.
    """
    host = psutil.virtual_memory()
    text: str | None = None
    if platform.system() == "Darwin":
        try:
            snapshot = subprocess.run(
                ["/usr/bin/vm_stat"], capture_output=True, check=False, timeout=1,
            )
        except (OSError, subprocess.TimeoutExpired):
            snapshot = None
        if snapshot is not None and snapshot.returncode == 0:
            text = snapshot.stdout.decode("utf-8", errors="replace")
    available_host = native_music_host_available_bytes(host.total, host.available, text)
    sampled = time.time_ns() // 1_000_000
    if lane == "cpu":
        return NativeMusicMemoryCapacity(
            sampled_at_unix_ms=sampled,
            host_total_bytes=host.total,
            host_available_bytes=available_host,
        )
    if lane == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("native Metal memory is unavailable")
        ceiling = torch.mps.recommended_max_memory()
        available = max(0, ceiling - torch.mps.driver_allocated_memory())
        return NativeMusicMemoryCapacity(
            sampled_at_unix_ms=sampled,
            host_total_bytes=host.total,
            host_available_bytes=available_host,
            device_index=0,
            device_memory_kind="unified",
            device_total_bytes=ceiling,
            device_available_bytes=available,
            shared_total_bytes=host.total,
            shared_available_bytes=available_host,
        )
    if not torch.cuda.is_available() or (hip is not None) != (lane == "rocm"):
        raise ValueError("native accelerator memory is unavailable in this runtime")
    if torch.cuda.current_device() != 0:
        raise ValueError(
            "native music must use its admitted runtime-visible device zero"
        )
    device_memory = cast(
        _DeviceMemoryAPI, cast(object, importlib.import_module("torch.cuda"))
    )
    properties = device_memory.get_device_properties(0)
    if properties.is_integrated not in (0, 1):
        raise ValueError("native accelerator reported an unknown memory topology")
    available, ceiling = device_memory.mem_get_info(0)
    unified = properties.is_integrated == 1
    return NativeMusicMemoryCapacity(
        sampled_at_unix_ms=sampled,
        host_total_bytes=host.total,
        host_available_bytes=available_host,
        device_index=0,
        device_memory_kind="unified" if unified else "discrete",
        device_total_bytes=ceiling,
        device_available_bytes=available,
        shared_total_bytes=host.total if unified else None,
        shared_available_bytes=available_host if unified else None,
    )


def _library_policy(
    lane: NativeMusicDevice,
) -> tuple[NativeMusicPrecision, tuple[int, int] | None]:
    """Resolve the pinned initializer's policy from available native libraries."""
    precision: NativeMusicPrecision = "float32"
    capability: tuple[int, int] | None = None
    if lane == "mps":
        if (
            os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0"
            or not torch.backends.mps.is_available()
        ):
            raise ValueError(
                "native music Metal requires available MPS and disabled implicit CPU fallback"
            )
    elif lane in ("cuda", "rocm"):
        if not torch.cuda.is_available() or (hip is not None) != (lane == "rocm"):
            raise ValueError(
                "explicit native music accelerator is unavailable in this runtime"
            )
        if lane == "cuda":
            capability = torch.cuda.get_device_capability()
            # Torch may report emulated BF16 on SM7. The pinned initializer
            # selects native BF16 only from SM8; match it before model loading.
            precision = "bfloat16" if capability[0] >= 8 else "float16"
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return precision, capability


def _probe_math(lane: NativeMusicDevice, precision: NativeMusicPrecision) -> None:
    """Finish tiny native math before reporting a library execution observation."""
    native = "cuda" if lane == "rocm" else lane
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[precision]
    values = torch.ones((4, 4), dtype=dtype, device=native)
    measured = (values @ values).cpu()
    if not torch.equal(measured, torch.full((4, 4), 4.0, dtype=dtype)):
        raise ValueError(
            "native music device failed its requested-precision execution probe"
        )


def probe_libraries(
    lane: NativeMusicDevice, payload_sha256: str
) -> NativeMusicLibraryProbe:
    """Execute a weights-free lane probe inside the verified private runtime.

    No model path, configuration, download or ACE loader is needed. Successful
    native math establishes package/device availability; loading a particular
    model remains a separate supervised operation with an identity receipt.
    """
    precision, capability = _library_policy(lane)
    _probe_math(lane, precision)
    autoencoder_precision: NativeMusicTensorPrecision = (
        "float16" if lane == "mps" else precision
    )
    if autoencoder_precision != precision:
        _probe_math(lane, autoencoder_precision)
    return NativeMusicLibraryProbe(
        payload_sha256=payload_sha256,
        device=lane,
        precision=precision,
        autoencoder_precision=autoencoder_precision,
        cuda_compute_capability=capability,
        torch_version=str(torch.__version__),
        memory_capacity=sample_memory_capacity(lane),
    )


def probe_device(configuration: NativeMusicServeConfig) -> str:
    """Execute and bind the requested loaded-model precision to native capability.

    The pinned initializer uses FP32 on CPU/MPS/ROCm, native BF16 on SM8+ CUDA,
    and FP16 on earlier CUDA. Missing physical qualification is not a refusal.
    """
    precision, _ = _library_policy(configuration.device)
    if configuration.precision != precision:
        raise ValueError(
            "explicit native music precision differs from the pinned initializer policy"
        )
    _probe_math(configuration.device, precision)
    return "cuda" if configuration.device == "rocm" else configuration.device


def _component_policy(
    module: torch.nn.Module | None,
) -> tuple[NativeMusicDevice, NativeMusicTensorPrecision]:
    if module is None:
        raise ValueError("native music model component did not initialize")
    observed: tuple[NativeMusicDevice, NativeMusicTensorPrecision] | None = None
    for parameter in module.parameters():
        if not parameter.is_floating_point():
            continue
        device = parameter.device.type
        if device == "cuda":
            lane: NativeMusicDevice = "rocm" if hip is not None else "cuda"
        elif device in ("cpu", "mps"):
            lane = device
        else:
            raise ValueError("native music component uses an unexpected compute device")
        if parameter.dtype == torch.float32:
            precision: NativeMusicTensorPrecision = "float32"
        elif parameter.dtype == torch.bfloat16:
            precision = "bfloat16"
        elif parameter.dtype == torch.float16:
            precision = "float16"
        else:
            raise ValueError(
                "native music component uses an unexpected tensor precision"
            )
        current = (lane, precision)
        if observed is not None and current != observed:
            raise ValueError(
                "native music component has mixed or offloaded parameter policy"
            )
        observed = current
    if observed is None:
        raise ValueError("native music component has no observed floating parameters")
    return observed


def observe_execution(
    handler: LoadedHandler, configuration: NativeMusicServeConfig
) -> NativeMusicExecution:
    """Verify actual loaded parameter placement before readiness or completion.

    MPS's autoencoder is FP16 even when its diffusion model is FP32. Report each
    component separately. This observation cannot establish the precision of
    every operator or exclude temporary upstream CPU work during an operation.
    """
    base = _component_policy(handler.model)
    encoder = _component_policy(handler.text_encoder)
    vae = _component_policy(handler.vae)
    expected = (configuration.device, configuration.precision)
    vae_precision = (
        "float16" if configuration.device == "mps" else configuration.precision
    )
    if (
        base != expected
        or encoder != expected
        or vae != (configuration.device, vae_precision)
    ):
        raise ValueError(
            "loaded native music component policy differs from explicit admission"
        )
    native = "cuda" if configuration.device == "rocm" else configuration.device
    if (
        handler.device != native
        or handler.dtype
        != {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[configuration.precision]
        or handler.sample_rate != 48000
    ):
        raise ValueError(
            "native handler initialization disagrees with observed components"
        )
    return NativeMusicExecution(
        device=base[0],
        precision=base[1],
        text_encoder_device=encoder[0],
        text_encoder_precision=encoder[1],
        autoencoder_device=vae[0],
        autoencoder_precision=vae[1],
    )
