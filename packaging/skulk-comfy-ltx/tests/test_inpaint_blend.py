"""Check blend preservation, timeline refusal and bounded cancellation."""

import importlib.util

import pytest
import torch

if importlib.util.find_spec("kornia") is None:
    pytest.skip("Native pyramid checks require the Comfy or macOS oracle dependencies", allow_module_level=True)

from skulk_comfy_ltx.inpaint_blend import blend_inpaint_images

_BUDGET = 256 * 1024 * 1024
_CPU = torch.device("cpu")


@pytest.mark.parametrize("white", [False, True])
def test_whole_mask_preserves_the_selected_context_on_a_rectangular_canvas(white: bool) -> None:
    """Power-of-two padding and reconstruction must not change the chosen plate."""
    generated, context = torch.full((2, 65, 97, 3), 0.75), torch.full((2, 65, 97, 3), 0.25)
    mask = torch.full((2, 65, 97), float(white))
    result = blend_inpaint_images(generated, context, mask, 5, maximum_bytes=_BUDGET, device=_CPU, output_device=_CPU, check_cancelled=lambda: None)
    torch.testing.assert_close(result, generated if white else context, rtol=0, atol=0.000001)
    assert result.shape == generated.shape
    assert bool((generated == 0.75).all()) and bool((context == 0.25).all())


def test_blend_refuses_an_unexpanded_static_mask_and_budget_before_allocating(monkeypatch: pytest.MonkeyPatch) -> None:
    """Broadcast is explicit upstream; the blend never chooses a shortest clip."""
    images, mask = torch.zeros(2, 32, 48, 3), torch.zeros(2, 32, 48)

    def no_allocation(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("invalid blend allocated its output")

    monkeypatch.setattr(torch, "empty", no_allocation)
    with pytest.raises(ValueError, match="timelines"):
        blend_inpaint_images(images, images, mask[:1], 5, maximum_bytes=_BUDGET, device=_CPU, output_device=_CPU, check_cancelled=lambda: None)
    with pytest.raises(ValueError, match="allocation budget"):
        blend_inpaint_images(images, images, mask, 5, maximum_bytes=1, device=_CPU, output_device=_CPU, check_cancelled=lambda: None)


def test_blend_cancellation_precedes_output_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    """An interrupted blend cannot retain another decoded frame batch."""
    images, mask = torch.zeros(2, 32, 48, 3), torch.zeros(2, 32, 48)

    def cancel() -> None:
        raise InterruptedError("cancelled")

    def no_allocation(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("cancelled blend allocated its output")

    monkeypatch.setattr(torch, "empty", no_allocation)
    with pytest.raises(InterruptedError):
        blend_inpaint_images(images, images, mask, 6, maximum_bytes=_BUDGET, device=_CPU, output_device=_CPU, check_cancelled=cancel)
