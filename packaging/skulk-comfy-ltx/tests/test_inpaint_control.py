"""Keep full-grid inpainting and half-grid Union adapter contracts distinct."""

import pytest
import torch
from skulk_comfy_ltx.inpaint_control import validate_inpaint_guide


@pytest.mark.parametrize("metadata", [None, {}, {"reference_downscale_factor": "2"}, {"reference_downscale_factor": "nan"}, {"reference_downscale_factor": True}, {"reference_downscale_factor": "one"}, {"reference_downscale_factor": "1", "other.reference_downscale_factor": "1"}])
def test_inpainting_refuses_missing_ambiguous_or_half_grid_metadata(metadata: object) -> None:
    """The native factor-one fallback cannot hide the wrong actually loaded LoRA."""
    with pytest.raises(ValueError):
        validate_inpaint_guide(metadata, {"samples": torch.zeros(1, 128, 2, 2, 4)}, torch.zeros(9, 64, 128, 3), 9, check_cancelled=lambda: None)


def test_inpainting_keeps_valid_full_grid_inputs_unchanged() -> None:
    """Bounded tensor validation never allocates model weights or changes pixels."""
    samples, images = torch.ones(1, 128, 2, 2, 4), torch.ones(9, 64, 128, 3)
    validate_inpaint_guide({"reference_downscale_factor": "1"}, {"samples": samples}, images, 9, check_cancelled=lambda: None)
    assert bool((samples == 1).all()) and bool((images == 1).all())


@pytest.mark.parametrize("frames,height,width,bad_pixel", [(8, 64, 128, False), (9, 32, 128, False), (9, 64, 64, False), (9, 64, 128, True)])
def test_inpainting_refuses_partial_timelines_wrong_stage_shapes_and_nonfinite_pixels(frames: int, height: int, width: int, bad_pixel: bool) -> None:
    """Every guide frame must match the actual current-stage latent geometry."""
    images = torch.zeros(frames, height, width, 3)
    if bad_pixel:
        images[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        validate_inpaint_guide({"reference_downscale_factor": "1"}, {"samples": torch.zeros(1, 128, 2, 2, 4)}, images, 9, check_cancelled=lambda: None)


def test_inpainting_guide_cancellation_precedes_native_encoding() -> None:
    """Cancellation propagates through validation rather than admitting guide work."""
    def cancel() -> None:
        raise InterruptedError("cancelled")

    with pytest.raises(InterruptedError):
        validate_inpaint_guide({"reference_downscale_factor": "1"}, {"samples": torch.zeros(1, 128, 2, 2, 4)}, torch.zeros(9, 64, 128, 3), 9, check_cancelled=cancel)
