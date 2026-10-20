from collections.abc import Sequence

from torch import Tensor

def get_keyframe_idxs(conditioning: list[list[object]], latent_shape: Sequence[int]) -> tuple[Tensor | None, int]: ...
