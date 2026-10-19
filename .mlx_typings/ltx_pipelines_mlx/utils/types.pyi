from dataclasses import dataclass

@dataclass(frozen=True)
class AutoDuration:
    min_seconds: float = 1.0
    max_seconds: float = 20.0
