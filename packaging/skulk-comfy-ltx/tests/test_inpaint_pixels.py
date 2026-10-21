"""Exercise inpaint pixel semantics and refusal before retained allocation."""

import pytest
import torch
from skulk_comfy_ltx.inpaint_pixels import dilate_inpaint_mask, green_inpaint_context

_BUDGET = 1024 * 1024
_CPU = torch.device("cpu")


def test_static_red_mask_broadcasts_without_changing_unmasked_source() -> None:
    """White regenerates, black retains source, and soft masks use the trained key."""
    source = torch.full((3, 4, 6, 3), 0.25)
    source[1].fill_(0.75)
    mask = torch.zeros(1, 4, 6)
    mask[0, 1, 2] = 1
    mask[0, 2, 3] = 0.5
    original_source, original_mask = source.clone(), mask.clone()
    result = green_inpaint_context(source, mask, maximum_bytes=_BUDGET, output_device=_CPU, check_cancelled=lambda: None)
    assert result.shape == source.shape and result.dtype == torch.float32
    torch.testing.assert_close(result[:, 0, 0], source[:, 0, 0], rtol=0, atol=0)
    torch.testing.assert_close(result[:, 1, 2], torch.tensor([0.4, 1.0, 0.0]).expand(3, -1), rtol=0, atol=0)
    torch.testing.assert_close(result[0, 2, 3], torch.tensor([0.325, 0.625, 0.125]), rtol=0, atol=0)
    torch.testing.assert_close(source, original_source, rtol=0, atol=0)
    torch.testing.assert_close(mask, original_mask, rtol=0, atol=0)


def test_video_mask_keeps_its_distinct_complete_timeline() -> None:
    """A moving mask must affect its own frame rather than silently broadcast frame zero."""
    source = torch.zeros(3, 4, 6, 3)
    mask = torch.zeros(3, 4, 6)
    mask[1, 1, 2] = 1
    result = green_inpaint_context(source, mask, maximum_bytes=_BUDGET, output_device=_CPU, check_cancelled=lambda: None)
    assert bool((result[0] == 0).all()) and bool((result[2] == 0).all())
    torch.testing.assert_close(result[1, 1, 2], torch.tensor([0.4, 1.0, 0.0]), rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(2, 4, 6), (1, 5, 6), (1, 4, 7)])
def test_mask_mismatch_is_refused_instead_of_trimming_or_resizing(shape: tuple[int, int, int]) -> None:
    """The upstream min-frame truncation is deliberately excluded from this contract."""
    with pytest.raises(ValueError, match="geometry and aligned timelines"):
        green_inpaint_context(torch.zeros(3, 4, 6, 3), torch.zeros(shape), maximum_bytes=_BUDGET, output_device=_CPU, check_cancelled=lambda: None)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -0.01, 1.01])
def test_invalid_pixels_never_reach_compositing(invalid: float) -> None:
    """Corrupt floating media cannot be converted into a seemingly valid guide."""
    mask = torch.zeros(1, 4, 6)
    mask[0, 0, 0] = invalid
    with pytest.raises(ValueError, match="finite values"):
        green_inpaint_context(torch.zeros(3, 4, 6, 3), mask, maximum_bytes=_BUDGET, output_device=_CPU, check_cancelled=lambda: None)


def test_budget_refusal_precedes_retained_tensor_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    """The private runner budget counts both retained outputs and frame workspace."""
    source, mask = torch.zeros(3, 4, 6, 3), torch.zeros(1, 4, 6)

    def no_allocation(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("over-budget input allocated a retained output")

    monkeypatch.setattr(torch, "empty", no_allocation)
    with pytest.raises(ValueError, match="allocation budget"):
        green_inpaint_context(source, mask, maximum_bytes=1, output_device=_CPU, check_cancelled=lambda: None)


def test_dilation_expands_spatially_without_leaking_into_adjacent_frames() -> None:
    """The official recipe dilates each frame spatially, with no temporal radius."""
    mask = torch.zeros(3, 5, 7)
    mask[1, 2, 3] = 1
    result = dilate_inpaint_mask(mask, 1, maximum_bytes=_BUDGET, check_cancelled=lambda: None)
    assert result.sum().item() == 9 and mask.sum().item() == 1
    assert bool((result[0] == 0).all()) and bool((result[2] == 0).all())
    assert bool((result[1, 1:4, 2:5] == 1).all())


def test_zero_radius_still_applies_native_strict_threshold() -> None:
    """Exactly one-half is black; a value just above one-half is white."""
    mask = torch.tensor([[[0.0, 0.5, 0.5001, 1.0]]])
    result = dilate_inpaint_mask(mask, 0, maximum_bytes=_BUDGET, check_cancelled=lambda: None)
    torch.testing.assert_close(result, torch.tensor([[[0.0, 0.0, 1.0, 1.0]]]), rtol=0, atol=0)


def test_cancellation_stops_before_output_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validation yields to ordinary cancellation without a background writer."""
    source, mask = torch.zeros(3, 4, 6, 3), torch.zeros(1, 4, 6)

    def cancel() -> None:
        raise InterruptedError("cancelled")

    def no_allocation(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("cancelled input allocated an output")

    monkeypatch.setattr(torch, "empty", no_allocation)
    with pytest.raises(InterruptedError, match="cancelled"):
        green_inpaint_context(source, mask, maximum_bytes=_BUDGET, output_device=_CPU, check_cancelled=cancel)
