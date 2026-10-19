from dataclasses import dataclass, field

@dataclass(frozen=True)
class MultiModalGuiderParams:
    cfg_scale: float = 1.0
    stg_scale: float = 0.0
    stg_blocks: list[int] | None = field(default_factory=list)
    rescale_scale: float = 0.0
    modality_scale: float = 1.0
    skip_step: int = 0
