"""Bundled cards for one base model must not contradict each other on tools.

Whether a model can call tools is a property of the model, so two cards for the
same `base_model` cannot both be right when one says it can and the other says
it cannot. A contradiction is not academic: it decides whether the API
advertises the capability, and on the served engines whether a request carrying
tools is rejected outright. It also reaches callers through `/v1/models`, so a
client picking a quantization can be told the same model does and does not
support tools depending which one it picked.

An unstated value is not a contradiction. A card with no `[tooling]` section
resolves through conservative family defaults, so silence next to an explicit
claim is under-specification rather than disagreement, and that is not what
this guards.
"""

from __future__ import annotations

import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

CARD_DIRECTORY = (
    Path(__file__).resolve().parents[5] / "resources" / "inference_model_cards"
)

# Contradictions that exist today and are tracked for the signed registry
# rather than fixed here. The vLLM-only cards under-declare when a model does
# call tools but no `runtime.vllm_tool_call_parser` pin has been validated on
# GPU hardware yet: the vLLM runner rejects a tools request without a pin, so
# flipping the flag alone would advertise a capability that fails at request
# time. Anything NOT listed here is a new contradiction and fails. The Qwen3.6
# FP8 entries left this list when their `qwen3_xml` pin was validated live on
# an A100-class pod.
KNOWN_CONTRADICTIONS: frozenset[str] = frozenset()


def load_cards() -> dict[str, dict[str, Any]]:
    """Return every bundled card keyed by file name."""

    cards: dict[str, dict[str, Any]] = {}
    for path in sorted(CARD_DIRECTORY.glob("*.toml")):
        cards[path.name] = tomllib.loads(path.read_text())
    return cards


def group_by_base_model() -> dict[str, list[tuple[str, dict[str, Any]]]]:
    """Group bundled cards by their declared base model."""

    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for name, card in load_cards().items():
        base = card.get("base_model")
        if isinstance(base, str) and base:
            grouped[base].append((name, card))
    return grouped


def tooling_of(card: dict[str, Any]) -> dict[str, Any]:
    section = cast("object", card.get("tooling"))
    if not isinstance(section, dict):
        return {}
    return cast("dict[str, Any]", section)


class TestToolingAgreement:
    def test_the_card_directory_was_found(self) -> None:
        # A wrong path would make every assertion below vacuous.
        assert CARD_DIRECTORY.is_dir()
        assert len(load_cards()) > 50

    def test_no_two_cards_disagree_on_whether_tools_are_supported(self) -> None:
        offenders: dict[str, dict[bool, list[str]]] = {}
        for base, group in group_by_base_model().items():
            stated: dict[bool, list[str]] = defaultdict(list)
            for name, card in group:
                supports = tooling_of(card).get("supports_tool_calling")
                if isinstance(supports, bool):
                    stated[supports].append(name)
            if len(stated) > 1 and base not in KNOWN_CONTRADICTIONS:
                offenders[base] = dict(stated)
        assert not offenders, (
            "cards for one base model disagree on tool support: "
            f"{offenders}. Whether a model can call tools is a property of the "
            "model, so one of these is wrong."
        )

    def test_no_two_cards_disagree_on_the_tool_call_format(self) -> None:
        offenders: dict[str, dict[str, list[str]]] = {}
        for base, group in group_by_base_model().items():
            stated: dict[str, list[str]] = defaultdict(list)
            for name, card in group:
                fmt = tooling_of(card).get("tool_call_format")
                if isinstance(fmt, str):
                    stated[fmt].append(name)
            if len(stated) > 1 and base not in KNOWN_CONTRADICTIONS:
                offenders[base] = dict(stated)
        assert not offenders, (
            "cards for one base model disagree on the tool-call dialect: "
            f"{offenders}. The dialect a model writes is a property of the "
            "model, and wiring the wrong markers makes its calls unparseable."
        )

    def test_every_known_contradiction_still_exists(self) -> None:
        # The exception list is debt, not decoration: once the registry fixes
        # one, this fails so the entry is removed rather than quietly widening
        # what the guard above allows.
        grouped = group_by_base_model()
        still_wrong: set[str] = set()
        for base in KNOWN_CONTRADICTIONS:
            stated: set[bool] = set()
            for _, card in grouped.get(base, []):
                supports = tooling_of(card).get("supports_tool_calling")
                if isinstance(supports, bool):
                    stated.add(supports)
            if len(stated) > 1:
                still_wrong.add(base)
        assert still_wrong == set(KNOWN_CONTRADICTIONS), (
            "KNOWN_CONTRADICTIONS is stale; remove the entries that now agree: "
            f"{set(KNOWN_CONTRADICTIONS) - still_wrong}"
        )
