from .distilled import DistilledPipeline
from .utils.args import ImageConditioningInput

class A2VidDistilledPipeline(DistilledPipeline):
    def generate_and_save(self, prompt: str, output_path: str, audio_path: str, height: int, width: int, num_frames: int, *, frame_rate: float, seed: int, images: list[ImageConditioningInput] | None = None) -> str: ...
