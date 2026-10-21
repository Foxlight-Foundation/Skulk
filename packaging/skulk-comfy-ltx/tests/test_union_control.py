"""Refuse silent native metadata fallbacks and malformed reference timelines."""

import pytest
import torch
from skulk_comfy_ltx.union_control import validate_union_inputs


@pytest.mark.parametrize("metadata", [None, {}, {"reference_downscale_factor": "1"}, {"reference_downscale_factor": "nan"}, {"reference_downscale_factor": True}, {"reference_downscale_factor": "two"}, {"reference_downscale_factor": "2", "other.reference_downscale_factor": "2"}])
def test_union_rejects_missing_ambiguous_or_wrong_actual_metadata(metadata: object) -> None:
    """The native helper's factor-one fallback must not admit the wrong adapter."""
    with pytest.raises(ValueError):
        validate_union_inputs(metadata, {"samples": torch.zeros(1, 128, 2, 2, 4)}, torch.zeros(9, 64, 128, 3), 9)


def test_union_preserves_valid_inputs_without_allocating_a_model() -> None:
    """Only bounded reference tensors participate in this CPU contract check."""
    samples = torch.ones(1, 128, 2, 2, 4)
    images = torch.ones(9, 64, 128, 3)
    validate_union_inputs({"reference_downscale_factor": "2"}, {"samples": samples}, images, 9)
    assert bool((samples == 1).all()) and bool((images == 1).all())


@pytest.mark.parametrize("frames,height,width,invalid_pixel", [(8, 64, 128, False), (9, 32, 128, False), (9, 64, 64, False), (9, 64, 128, True)])
def test_union_rejects_frame_geometry_and_nonfinite_pixels(frames: int, height: int, width: int, invalid_pixel: bool) -> None:
    """Guide alignment is checked before the native VAE consumes any pixels."""
    images = torch.zeros(frames, height, width, 3)
    if invalid_pixel:
        images[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        validate_union_inputs({"reference_downscale_factor": "2"}, {"samples": torch.zeros(1, 128, 2, 2, 4)}, images, 9)
