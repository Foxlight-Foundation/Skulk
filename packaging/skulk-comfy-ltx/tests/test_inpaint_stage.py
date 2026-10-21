"""Check native resize-before-dilation order and complete stage mask lifetimes."""

import pytest
import torch
import torch.nn.functional as functional
from skulk_comfy_ltx.inpaint_pixels import prepare_inpaint_stage

_CPU = torch.device("cpu")
_BUDGET = 16 * 1024 * 1024


@pytest.mark.parametrize("radius", [0, 1, 15, 30])
def test_stage_uses_area_resize_before_spatial_dilation_and_green_fill(radius: int) -> None:
    """The two official stages consume their resized masks, not a dilated full mask."""
    images = torch.full((2, 4, 6, 3), 0.25)
    mask = torch.zeros(2, 8, 12)
    mask[0, 2:4, 2:4] = 0.9
    mask[1, 6, 10] = 0.9
    before = mask.clone()
    green, stage_mask = prepare_inpaint_stage(images, mask, radius, maximum_bytes=_BUDGET, output_device=_CPU, check_cancelled=lambda: None)
    resized = functional.interpolate(mask.unsqueeze(1), size=(4, 6), mode="area")
    expanded = functional.max_pool2d(resized, 2 * radius + 1, stride=1, padding=radius) if radius else resized
    expected = (expanded[:, 0] > 0.5).float()
    assert torch.equal(stage_mask, expected)
    assert not bool(stage_mask[1].any())
    color = torch.tensor((102 / 255, 1, 0))
    torch.testing.assert_close(green, images * (1 - expected.unsqueeze(-1)) + color * expected.unsqueeze(-1), rtol=0, atol=0)
    assert torch.equal(mask, before)


def test_stage_refuses_a_partial_mask_and_insufficient_budget_before_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """A static mask must already have been explicitly expanded at source placement."""
    images, mask = torch.zeros(2, 4, 6, 3), torch.zeros(2, 8, 12)

    def no_allocation(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("invalid inpainting stage allocated a mask")

    monkeypatch.setattr(torch, "empty", no_allocation)
    with pytest.raises(ValueError, match="aligned"):
        prepare_inpaint_stage(images, mask[:1], 15, maximum_bytes=_BUDGET, output_device=_CPU, check_cancelled=lambda: None)
    with pytest.raises(ValueError, match="allocation budget"):
        prepare_inpaint_stage(images, mask, 15, maximum_bytes=1, output_device=_CPU, check_cancelled=lambda: None)
