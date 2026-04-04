"""Tests for the MLX VLM compatibility wrapper."""

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from exo.worker.engines.mlx.utils_mlx import _VlmModelWrapper


class _FakeOutput:
    """Minimal logits container matching mlx-vlm output shape."""

    def __init__(self, logits: mx.array) -> None:
        self.logits = logits


class _FakeInner(nn.Module):
    """Inner model stub that records keyword arguments."""

    def __init__(self) -> None:
        super().__init__()
        self.last_kwargs: dict[str, Any] = {}

    def __call__(self, *_args: object, **kwargs: object) -> _FakeOutput:
        self.last_kwargs = dict(kwargs)
        return _FakeOutput(mx.array([1.0]))


def test_vlm_wrapper_tolerates_missing_pixel_values_attr() -> None:
    """Missing transient pixel values should behave like a text-only call."""
    inner = _FakeInner()
    wrapper = _VlmModelWrapper(inner)
    del wrapper.__dict__["_pixel_values"]

    result = wrapper(mx.array([1]))

    assert "pixel_values" not in inner.last_kwargs
    assert result.tolist() == [1.0]


def test_vlm_wrapper_injects_pixel_values_when_present() -> None:
    """Native vision pixel values should be forwarded exactly once per call."""
    inner = _FakeInner()
    wrapper = _VlmModelWrapper(inner)
    pixel_values = mx.array([2.0])
    wrapper.set_pixel_values(pixel_values)

    _ = wrapper(mx.array([1]))

    assert inner.last_kwargs["pixel_values"] is pixel_values
