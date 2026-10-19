from ._base import BasePipeline

class DistilledPipeline(BasePipeline):
    def __init__(self, model_dir: str, gemma_model_id: str, low_memory: bool, low_ram_streaming: bool) -> None: ...
