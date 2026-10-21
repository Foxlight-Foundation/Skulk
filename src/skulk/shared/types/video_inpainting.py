"""Explicit source placement for a complete masked LTX inpainting take.

Source and mask dimensions are decoded worker facts. Callers specify the output
canvas on the normal video request and only its source offset here; declaring
source dimensions would not establish that the uploaded media actually matches.
"""

from typing import final

from pydantic import BaseModel, ConfigDict, Field


@final
class VideoInpainting(BaseModel):
    """Regenerate a source mask and any canvas padding, keeping its full timeline."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    left: int = Field(default=0, ge=0, le=4095, description="Horizontal source position on the explicit output canvas; no source resizing or cropping is performed.")
    top: int = Field(default=0, ge=0, le=4095, description="Vertical source position on the explicit output canvas; all pixels outside the source are regenerated.")

    def validate_source_canvas(self, width: int, height: int, *, source_width: int, source_height: int) -> None:
        """Check measured source geometry before creating any padded frame batch.

        Args:
            width: Exact requested output width, on the native 64-pixel grid.
            height: Exact requested output height, on the native 64-pixel grid.
            source_width: Width measured by decoding every source frame.
            source_height: Height measured by decoding every source frame.

        Raises:
            ValueError: The canvas is unbounded, off-grid or would crop the source.
        """
        if min(width, height) < 128 or max(width, height) > 4096 or width % 64 or height % 64:
            raise ValueError("Inpainting requires an explicit 64-pixel output grid between 128 and 4096 pixels")
        if min(source_width, source_height) < 1 or self.left + source_width > width or self.top + source_height > height:
            raise ValueError("The complete inpainting source must fit inside the output canvas without cropping")
