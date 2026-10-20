"""Dense-video mask and short anchored-clip contracts without model rendering."""

import pytest
import torch
from skulk_comfy_ltx.dfr import detailing_frame_override, prepare_dfr_latent
from torch import Tensor


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_dense_upscaled_video_keeps_samples_and_gets_a_compact_sampling_mask(dtype: torch.dtype) -> None:
    samples = torch.full((2, 4, 7, 3, 5), 0.75, dtype=dtype)
    source: dict[str, object] = {"samples": samples, "metadata": ("keep",)}
    result = prepare_dfr_latent(source)
    mask = result["noise_mask"]
    assert isinstance(mask, Tensor)
    assert mask.shape == (2, 1, 7, 1, 1) and mask.dtype == torch.float32 and mask.device == samples.device
    assert bool((mask == 1).all()) and result["samples"] is samples
    assert source == {"samples": samples, "metadata": ("keep",)}


@pytest.mark.parametrize("spatial", [False, True])
@pytest.mark.parametrize("batch", [1, 2])
def test_existing_image_and_guide_masks_remain_exact_and_independent(spatial: bool, batch: int) -> None:
    samples = torch.full((2, 4, 9, 3, 5), 0.75)
    mask = torch.ones((batch, 1, 9, 3 if spatial else 1, 5 if spatial else 1))
    mask[:, :, 0] = 0
    mask[:, :, -1] = 0.05
    original = mask.clone()
    result = prepare_dfr_latent({"samples": samples, "noise_mask": mask})
    observed = result["noise_mask"]
    assert isinstance(observed, Tensor) and observed is not mask
    torch.testing.assert_close(observed, original, rtol=0, atol=0)
    observed[:, :, 1] = 0.25
    torch.testing.assert_close(mask, original, rtol=0, atol=0)
    assert bool((samples == 0.75).all())


@pytest.mark.parametrize("mask", [torch.ones((1, 1, 7, 1)), torch.ones((1, 4, 7, 1, 1)), torch.ones((1, 1, 6, 1, 1)), torch.ones((1, 1, 7, 2, 1)), torch.ones((1, 1, 7, 1, 1), dtype=torch.int64), torch.full((1, 1, 7, 1, 1), float("nan")), torch.full((1, 1, 7, 1, 1), -0.01), torch.full((1, 1, 7, 1, 1), 1.01)])
def test_invalid_masks_fail_before_keyframe_creation(mask: Tensor) -> None:
    with pytest.raises(ValueError, match="DFR sampling mask"):
        prepare_dfr_latent({"samples": torch.ones((1, 4, 7, 3, 5)), "noise_mask": mask})


@pytest.mark.parametrize("frames,occupied,candidates,expected", [
    (25, {0}, [24], ""),
    (25, {0, 24}, [24], "12"),
    (49, {0, 24, 48}, [24, 48], "12"),
    (49, {0, 8, 16, 24, 48}, [24, 48], "36"),
    (49, {0, 48}, [24, 48], ""),
])
def test_detailing_preserves_the_default_grid_or_uses_the_largest_free_gap(frames: int, occupied: set[int], candidates: list[int], expected: str) -> None:
    assert detailing_frame_override(frames, occupied, candidates) == expected
    if expected:
        assert 0 < int(expected) < frames and int(expected) not in occupied


def test_a_fully_occupied_canvas_cannot_silently_drop_detailing() -> None:
    with pytest.raises(ValueError, match="unoccupied pixel frame"):
        detailing_frame_override(9, set(range(9)), [8])
