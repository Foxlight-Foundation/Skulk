"""Patch GatedDeltaNet's ``compute_g`` to use float32 for the softplus gate.

The default bfloat16 softplus in ``gated_delta`` loses precision for large
inputs, producing NaN or clipped activations. Casting to float32 before
the exp/log1p and back afterwards fixes this with negligible overhead.
"""

import sys

import mlx.core as mx
from mlx_lm.models.gated_delta import compute_g


def _compute_g_f32(a_log: mx.array, a: mx.array, dt_bias: mx.array) -> mx.array:
    return mx.exp(
        -mx.exp(a_log.astype(mx.float32))
        * mx.where(
            (a + dt_bias).astype(mx.float32) > 20,
            (a + dt_bias).astype(mx.float32),
            mx.log1p(mx.exp((a + dt_bias).astype(mx.float32))),
        )
    ).astype(a.dtype)


def patch_gdn_softplus() -> None:
    """Replace ``gated_delta.compute_g`` with a float32-safe version globally."""
    from mlx_lm.models import gated_delta

    gated_delta.compute_g = _compute_g_f32

    # Only mlx-lm/mlx-vlm model modules can hold a `from gated_delta import
    # compute_g` copy. Probing arbitrary modules is dangerous: lazy-loading
    # packages run imports on attribute access — transformers 5.10 resolves
    # a top-level "compute_g" probe to an unrelated symbol in
    # models.aria.image_processing_aria, whose import requires torchvision
    # and crashed the runner at startup (ModuleNotFoundError escapes a
    # getattr default, which only covers AttributeError).
    for mod in list(sys.modules.values()):
        if mod is gated_delta:
            continue
        module_name = getattr(mod, "__name__", "")
        if not module_name.startswith(("mlx_lm", "mlx_vlm")):
            continue
        if getattr(mod, "compute_g", None) is compute_g:
            object.__setattr__(mod, "compute_g", _compute_g_f32)
