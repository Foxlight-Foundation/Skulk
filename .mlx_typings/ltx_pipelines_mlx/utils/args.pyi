from typing import NamedTuple

class ImageConditioningInput(NamedTuple):
    path: str
    frame_idx: int
    strength: float
    crf: int | None = None
