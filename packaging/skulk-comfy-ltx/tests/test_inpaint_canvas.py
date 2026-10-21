"""Verify exact outpaint placement and refusal before full-canvas allocation."""

import pytest
import torch
from skulk_comfy_ltx.inpaint_pixels import place_inpaint_source

_BUDGET = 64 * 1024 * 1024
_CPU = torch.device("cpu")


@pytest.mark.parametrize("static", [False, True])
def test_outpaint_places_complete_source_and_regenerates_only_padding_and_mask(static: bool) -> None:
    """Static masks expand explicitly; moving masks retain the complete timeline."""
    source = torch.rand(3, 65, 97, 3)
    mask = torch.zeros(1 if static else 3, 65, 97)
    mask[:, 16:32, 20:36] = 1
    before = source.clone()
    canvas, expanded = place_inpaint_source(source, mask, width=192, height=128, left=11, top=7, maximum_bytes=_BUDGET, output_device=_CPU, check_cancelled=lambda: None)
    assert canvas.shape == (3, 128, 192, 3)
    torch.testing.assert_close(canvas[:, 7:72, 11:108], source, rtol=0, atol=0)
    torch.testing.assert_close(expanded[:, 7:72, 11:108], mask.expand(3, -1, -1), rtol=0, atol=0)
    padding = torch.ones(128, 192, dtype=torch.bool)
    padding[7:72, 11:108] = False
    assert bool((expanded[:, padding] == 1).all())
    assert bool((canvas[:, padding] == 0).all())
    assert torch.equal(source, before)


@pytest.mark.parametrize("width,height,left,top", [(128, 128, -1, 0), (128, 128, 32, 0), (128, 128, 0, 64), (129, 128, 0, 0), (64, 128, 0, 0)])
def test_invalid_outpaint_geometry_refuses_before_canvas_allocation(monkeypatch: pytest.MonkeyPatch, width: int, height: int, left: int, top: int) -> None:
    """Offsets never crop a source and canvas dimensions never silently round."""
    source, mask = torch.zeros(2, 65, 97, 3), torch.zeros(2, 65, 97)

    def no_allocation(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("invalid outpaint allocated a canvas")

    monkeypatch.setattr(torch, "empty", no_allocation)
    with pytest.raises(ValueError):
        place_inpaint_source(source, mask, width=width, height=height, left=left, top=top, maximum_bytes=_BUDGET, output_device=_CPU, check_cancelled=lambda: None)


def test_outpaint_counts_full_expanded_timeline_before_allocating(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tiny source and still mask cannot evade the full output reservation."""
    source, mask = torch.zeros(9, 4, 4, 3), torch.zeros(1, 4, 4)

    def no_allocation(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("over-budget outpaint allocated a canvas")

    monkeypatch.setattr(torch, "empty", no_allocation)
    with pytest.raises(ValueError, match="allocation budget"):
        place_inpaint_source(source, mask, width=4096, height=4096, left=0, top=0, maximum_bytes=_BUDGET, output_device=_CPU, check_cancelled=lambda: None)


def test_outpaint_cancellation_propagates_between_frames() -> None:
    """Cancellation stops placement before another source frame is copied."""
    source, mask = torch.zeros(3, 65, 97, 3), torch.zeros(1, 65, 97)
    calls = 0

    def cancel() -> None:
        nonlocal calls
        calls += 1
        if calls == 6:
            raise InterruptedError("cancelled")

    with pytest.raises(InterruptedError):
        place_inpaint_source(source, mask, width=192, height=128, left=11, top=7, maximum_bytes=_BUDGET, output_device=_CPU, check_cancelled=cancel)
    assert calls == 6
