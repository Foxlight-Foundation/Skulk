from typing import Protocol

from skulk_comfy_ltx.editing import AudioVae, AudioWave
from torch import Tensor

class EncodedLatentOutput(Protocol):
    @property
    def result(self) -> tuple[dict[str, Tensor], ...]: ...

class LTXVAudioVAEEncode:
    @classmethod
    def execute(cls, audio: AudioWave, audio_vae: AudioVae) -> EncodedLatentOutput: ...
