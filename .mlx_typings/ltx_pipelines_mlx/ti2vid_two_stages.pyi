from ._base import BasePipeline

class TI2VidTwoStagesPipeline(BasePipeline):
    def __init__(self, model_dir: str, gemma_model_id: str, low_memory: bool, low_ram_streaming: bool, distilled_lora_strength: float = 1.0) -> None: ...
