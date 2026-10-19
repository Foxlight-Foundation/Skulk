from dataclasses import dataclass

@dataclass
class PromptRelayInput:
    local_prompts: list[str]
    segment_lengths: list[int | None] | None = None
    epsilon: float = 1e-3
    strength: float = 1.0
