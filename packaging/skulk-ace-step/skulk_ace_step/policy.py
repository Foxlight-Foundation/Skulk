"""Explicit native device execution and observed component precision policy."""

from __future__ import annotations

import os

import torch
from torch.version import hip

from .contracts import (
    NativeMusicDevice,
    NativeMusicExecution,
    NativeMusicPrecision,
    NativeMusicServeConfig,
    NativeMusicTensorPrecision,
)
from .upstream_bindings import LoadedHandler


def probe_device(configuration: NativeMusicServeConfig) -> str:
    """Execute math on the requested lane and reject upstream silent fallback.

    The pinned initializer uses FP32 on CPU/MPS/ROCm, BF16 on capable CUDA and
    FP16 on earlier CUDA. Compatibility is based on the installed runtime and
    actual device functionality, not a list of previously tested GPU models.
    """
    lane = configuration.device
    native = "cpu"
    precision: NativeMusicPrecision = "float32"
    if lane == "mps":
        if (
            os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0"
            or not torch.backends.mps.is_available()
        ):
            raise ValueError(
                "native music Metal requires available MPS and disabled implicit CPU fallback"
            )
        native = "mps"
    elif lane in ("cuda", "rocm"):
        if not torch.cuda.is_available() or (hip is not None) != (lane == "rocm"):
            raise ValueError(
                "explicit native music accelerator is unavailable in this runtime"
            )
        native = "cuda"
        precision = (
            ("bfloat16" if torch.cuda.is_bf16_supported() else "float16")
            if lane == "cuda"
            else "float32"
        )
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if configuration.precision != precision:
        raise ValueError(
            "explicit native music precision differs from the pinned initializer policy"
        )
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
    return native


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
