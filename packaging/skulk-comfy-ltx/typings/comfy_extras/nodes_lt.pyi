from collections.abc import Mapping, Sequence

from comfy_api.latest import io
from torch import Tensor

class LTXVAddGuide:
    @classmethod
    def execute(cls, positive: list[list[object]], negative: list[list[object]], vae: object, latent: Mapping[str, object], image: Tensor, frame_idx: int, strength: float, *, iclora_parameters: Mapping[str, int]) -> io.NodeOutput: ...

def get_keyframe_idxs(conditioning: list[list[object]], latent_shape: Sequence[int]) -> tuple[Tensor | None, int]: ...
