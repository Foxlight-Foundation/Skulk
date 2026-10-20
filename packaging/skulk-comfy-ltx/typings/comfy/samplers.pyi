from collections.abc import Callable

from torch import Tensor

class KSAMPLER:
    def __init__(self, sampler_function: Callable[..., Tensor]) -> None: ...
