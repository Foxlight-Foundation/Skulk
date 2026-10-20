from typing import Protocol

from torch import Tensor

class AudioFunctional(Protocol):
    def resample(self, waveform: Tensor, orig_freq: int, new_freq: int) -> Tensor: ...

class AudioLibrary(Protocol):
    functional: AudioFunctional

torchaudio: AudioLibrary
