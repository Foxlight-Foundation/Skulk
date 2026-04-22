"""Tests for the MLX VLM compatibility wrapper."""

from typing import Callable, cast

import mlx.core as mx

from exo.worker.engines.mlx import utils_mlx as utils_mlx_module


def _vlm_model_wrapper(inner: object) -> object:
    module_dict = cast(dict[str, object], utils_mlx_module.__dict__)
    wrapper_cls = cast(Callable[[object], object], module_dict["_VlmModelWrapper"])
    return wrapper_cls(inner)


class _FakeOutput:
    """Minimal logits container matching mlx-vlm output shape."""

    def __init__(self, logits: mx.array) -> None:
        self.logits = logits


class _FakeInner:
    """Inner model stub that records keyword arguments."""

    def __init__(self) -> None:
        self.last_kwargs: dict[str, object] = {}

    def __call__(self, *_args: object, **kwargs: object) -> _FakeOutput:
        self.last_kwargs = dict(kwargs)
        return _FakeOutput(mx.array([1.0]))


def test_vlm_wrapper_tolerates_missing_pixel_values_attr() -> None:
    """Missing transient pixel values should behave like a text-only call."""
    inner = _FakeInner()
    wrapper = _vlm_model_wrapper(inner)
    del cast(dict[str, object], object.__getattribute__(wrapper, "__dict__"))[
        "_pixel_values"
    ]

    result = cast(Callable[[mx.array], mx.array], wrapper)(mx.array([1]))

    assert "pixel_values" not in inner.last_kwargs
    assert result.tolist() == [1.0]


def test_vlm_wrapper_injects_pixel_values_when_present() -> None:
    """Native vision pixel values should be forwarded exactly once per call."""
    inner = _FakeInner()
    wrapper = _vlm_model_wrapper(inner)
    pixel_values = mx.array([2.0])
    cast(
        Callable[[mx.array | list[mx.array] | None], None],
        object.__getattribute__(wrapper, "set_pixel_values"),
    )(pixel_values)

    _ = cast(Callable[[mx.array], mx.array], wrapper)(mx.array([1]))

    assert inner.last_kwargs["pixel_values"] is pixel_values
