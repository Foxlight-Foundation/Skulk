"""Check strict public placement and measured full-source canvas geometry."""

import pytest
from pydantic import ValidationError

from skulk.shared.types.video_inpainting import VideoInpainting


@pytest.mark.parametrize("value", [{"left": True}, {"left": 0.5}, {"top": "1"}, {"left": -1}, {"top": 4096}, {"resize": True}])
def test_inpainting_placement_is_strict_and_rejects_unknown_execution_controls(value: dict[str, object]) -> None:
    """Preserved descriptive metadata cannot grant a new rendering operation."""
    with pytest.raises(ValidationError):
        VideoInpainting.model_validate(value)


def test_inpainting_placement_is_frozen() -> None:
    """The replicated request cannot change its source offset in place."""
    placement = VideoInpainting(left=11, top=7)
    with pytest.raises(ValidationError, match="frozen"):
        placement.left = 12


@pytest.mark.parametrize("canvas", [(128, 128), (192, 128), (4096, 4096)])
def test_inpainting_accepts_measured_odd_source_dimensions_without_resizing(canvas: tuple[int, int]) -> None:
    """Only the output follows the trained grid; source pixels retain their shape."""
    VideoInpainting(left=11, top=7).validate_source_canvas(*canvas, source_width=97, source_height=65)


@pytest.mark.parametrize("width,height,source_width,source_height", [(129, 128, 97, 65), (64, 128, 32, 32), (4160, 128, 97, 65), (128, 128, 118, 65), (128, 128, 97, 122), (128, 128, 0, 65)])
def test_inpainting_never_rounds_canvas_or_crops_the_complete_source(width: int, height: int, source_width: int, source_height: int) -> None:
    """A larger placement is explicit rather than an implicit fit or crop."""
    with pytest.raises(ValueError):
        VideoInpainting(left=11, top=7).validate_source_canvas(width, height, source_width=source_width, source_height=source_height)
