"""Keep measured text ranges separate from padding and encoder side effects."""

import json
from typing import cast

import pytest
import torch
from skulk_comfy_ltx.relay_encoding import (
    RELAY_CONDITION_KEY,
    RelayLayout,
    encode_prompt_relay,
)


class _Clip:
    def __init__(self, *, truncate: bool = False, unprocessed: bool = True, unstable: bool = False) -> None:
        self.calls: list[str] = []
        self.encodes = 0
        self.metadata: dict[str, object] = {"unprocessed_ltxav_embeds": unprocessed, "retained_metadata": "original"}
        self.truncate = truncate
        self.unstable = unstable

    def tokenize(self, text: str) -> dict[str, object]:
        """Expose a transparent padded/start-token fixture with stable boundaries."""
        self.calls.append(text)
        tokens = [(2, 1.0), *((ord(character) + 5, 1.0) for character in text)]
        if self.unstable and len(self.calls) > 1:
            tokens[1] = (999, 1.0)
        return {"gemma4": [[*([(0, 1.0)] * max(0, 32 - len(tokens))), *tokens]]}

    def encode_from_tokens_scheduled(self, tokens: dict[str, object]) -> list[list[object]]:
        """Record exactly one encoding and expose any simulated truncation."""
        self.encodes += 1
        blocks = cast("list[list[tuple[int, float]]]", tokens["gemma4"])
        length = sum(identifier != 0 for identifier, _ in blocks[0])
        return [[torch.zeros(1, length - int(self.truncate), 4), self.metadata]]


def _scenes() -> str:
    return json.dumps([{"prompt": "red", "length": 2}, {"prompt": "blue", "length": 1}])


def test_combined_encoding_measures_real_prefixes_and_preserves_metadata() -> None:
    """Tokenize prefixes, encode once and leave the original metadata unmodified."""
    clip = _Clip()
    result = encode_prompt_relay(clip, "global", _scenes())
    assert clip.calls == ["global", "global red", "global red blue"]
    assert clip.encodes == 1
    metadata = cast("dict[str, object]", result[0][1])
    assert metadata[RELAY_CONDITION_KEY] == RelayLayout(((7, 11), (11, 16)), (2, 1))
    assert metadata["retained_metadata"] == "original"
    assert RELAY_CONDITION_KEY not in clip.metadata


@pytest.mark.parametrize("settings", [{"truncate": True}, {"unprocessed": False}])
def test_encoder_layout_changes_are_refused(settings: dict[str, bool]) -> None:
    """Do not sample with ranges that point at truncated or already connected text."""
    with pytest.raises(ValueError, match="changed or truncated"):
        encode_prompt_relay(_Clip(**settings), "global", _scenes())


def test_unstable_token_prefixes_fail_before_encoding() -> None:
    """Reject a tokenizer boundary change instead of inventing an offset."""
    clip = _Clip(unstable=True)
    with pytest.raises(ValueError, match="scene boundaries"):
        encode_prompt_relay(clip, "global", _scenes())
    assert clip.encodes == 0


@pytest.mark.parametrize("scenes", [[], [{"prompt": "red", "length": 1}], [{"prompt": "red", "length": True}, {"prompt": "blue", "length": 1}], [{"prompt": "red", "length": 0}, {"prompt": "blue", "length": 1}], [{"prompt": " ", "length": 1}, {"prompt": "blue", "length": 1}], [{"prompt": "red", "length": 1, "unknown": 2}, {"prompt": "blue", "length": 1}]])
def test_invalid_scene_envelopes_fail_before_encoder_work(scenes: list[dict[str, object]]) -> None:
    """Keep invalid direct-node data bounded without GPU model work."""
    clip = _Clip()
    with pytest.raises(ValueError):
        encode_prompt_relay(clip, "global", json.dumps(scenes))
    assert not clip.calls and clip.encodes == 0
