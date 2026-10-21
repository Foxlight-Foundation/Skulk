"""Measure relay text columns with the placed LTX Gemma4 encoder itself."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol, cast

from torch import Tensor

RELAY_CONDITION_KEY = "skulk_ltx_relay_layout"
"""Positive-only conditioning marker, retained through the normal Comfy graph."""


@dataclass(frozen=True, slots=True)
class RelayLayout:
    """Immutable text ranges and relative lengths; never retains model tensors."""

    token_ranges: tuple[tuple[int, int], ...]
    weights: tuple[int, ...]


class RelayClip(Protocol):
    """The exact pinned CLIP tokenization and scheduled-encoding boundary."""

    def tokenize(self, text: str) -> dict[str, object]:
        """Return padded token-weight blocks for this actual loaded encoder."""
        ...

    def encode_from_tokens_scheduled(self, tokens: dict[str, object]) -> list[list[object]]:
        """Encode the combined text once, retaining normal conditioning metadata."""
        ...


def _valid_token_ids(tokens: dict[str, object]) -> tuple[int, ...]:
    if set(tokens) != {"gemma4"}:
        raise ValueError("LTX relay requires the selected Gemma4 text encoder")
    raw = tokens["gemma4"]
    if not isinstance(raw, list):
        raise ValueError("relay requires the pinned text-token block contract")
    blocks = cast("list[object]", raw)
    if len(blocks) != 1 or not isinstance(blocks[0], list):
        raise ValueError("relay requires one complete, untruncated text sequence")
    identifiers: list[int] = []
    began = False
    for item in cast("list[object]", blocks[0]):
        if not isinstance(item, tuple):
            raise ValueError("relay accepts plain text tokens without media embeddings")
        pair = cast("tuple[object, ...]", item)
        if len(pair) != 2:
            raise ValueError("relay accepts plain text tokens without media embeddings")
        identifier, weight = pair
        if type(identifier) is not int or type(weight) not in (int, float) or weight != 1:
            raise ValueError("relay requires the pinned unweighted Gemma4 token sequence")
        if identifier == 0:
            if began:
                raise ValueError("relay does not permit padding tokens inside scene text")
            continue
        if identifier < 0:
            raise ValueError("relay text contains an invalid token identifier")
        began = True
        identifiers.append(identifier)
    if not identifiers or identifiers[0] != 2 or len(identifiers) > 32768:
        raise ValueError("relay requires the pinned Gemma4 start token and bounded text context")
    return tuple(identifiers)


def encode_prompt_relay(clip: RelayClip, global_prompt: str, segments_json: str) -> list[list[object]]:
    """Encode global and local text together and stamp measured positive ranges.

    Padding and start tokens are measured through the renderer's own tokenizer.
    Prefix instability, truncation, unexpected weighting and preprocessed context
    are rejected before sampling. The existing connector processes the combined
    sequence once; this function never encodes or concatenates individual scenes.
    """
    if len(segments_json.encode("utf-8")) > 65536:
        raise ValueError("relay scene data exceeds its bounded text envelope")
    parsed = cast(object, json.loads(segments_json))
    if not isinstance(parsed, list):
        raise ValueError("relay scenes must be an ordered JSON array")
    items = cast("list[object]", parsed)
    if not 2 <= len(items) <= 8:
        raise ValueError("relay requires two to eight local prompts")
    prompts: list[str] = []
    weights: list[int] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("relay scene entries must contain prompt and length")
        fields = cast("dict[object, object]", item)
        if set(fields) != {"prompt", "length"}:
            raise ValueError("relay scene entries accept only prompt and length")
        prompt, length = fields["prompt"], fields["length"]
        if not isinstance(prompt, str) or not prompt.strip() or type(length) is not int or not 1 <= length <= 120:
            raise ValueError("relay requires nonempty local text and integer relative lengths")
        prompts.append(prompt.strip())
        weights.append(length)
    if len(global_prompt) + sum(len(prompt) for prompt in prompts) > 8000:
        raise ValueError("relay global and local prompts exceed the video text budget")
    combined = global_prompt.strip()
    tokens = clip.tokenize(combined)
    previous = _valid_token_ids(tokens)
    ranges: list[tuple[int, int]] = []
    for prompt in prompts:
        combined = combined + " " + prompt
        tokens = clip.tokenize(combined)
        identifiers = _valid_token_ids(tokens)
        if len(identifiers) <= len(previous) or identifiers[:len(previous)] != previous:
            raise ValueError("relay tokenizer cannot preserve the measured scene boundaries")
        ranges.append((len(previous), len(identifiers)))
        previous = identifiers
    layout = RelayLayout(tuple(ranges), tuple(weights))
    encoded = clip.encode_from_tokens_scheduled(tokens)
    if not 1 <= len(encoded) <= 32:
        raise ValueError("relay encoder returned invalid scheduled conditioning")
    result: list[list[object]] = []
    for entry in encoded:
        if len(entry) != 2 or not isinstance(entry[0], Tensor) or not isinstance(entry[1], dict):
            raise ValueError("relay encoder returned an unsupported conditioning entry")
        context = entry[0]
        metadata = cast("dict[object, object]", entry[1])
        if context.ndim != 3 or context.shape[0] != 1 or context.shape[1] != len(previous) or metadata.get("unprocessed_ltxav_embeds") is not True or any(not isinstance(key, str) for key in metadata):
            raise ValueError("relay encoder changed or truncated the measured text columns")
        copied = {cast(str, key): value for key, value in metadata.items()}
        copied[RELAY_CONDITION_KEY] = layout
        result.append([context, copied])
    return result
