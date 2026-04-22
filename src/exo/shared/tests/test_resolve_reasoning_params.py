import pytest

from exo.shared.models.capabilities import ResolvedCapabilityProfile
from exo.shared.types.text_generation import resolve_reasoning_params


def test_both_none_returns_none_none() -> None:
    assert resolve_reasoning_params(None, None) == (None, None)


def test_both_set_passes_through_unchanged_except_explicit_none() -> None:
    assert resolve_reasoning_params("high", True) == ("high", True)
    assert resolve_reasoning_params("low", False) == ("none", False)


def test_explicit_none_disables_thinking_even_with_explicit_flag() -> None:
    assert resolve_reasoning_params("none", True) == ("none", False)
    assert resolve_reasoning_params("none", False) == ("none", False)


def test_non_toggleable_thinking_models_preserve_explicit_effort_only() -> None:
    profile = ResolvedCapabilityProfile(
        supports_thinking=True,
        supports_thinking_toggle=False,
        default_reasoning_effort="high",
        disabled_reasoning_effort="none",
    )

    assert resolve_reasoning_params(None, True, profile) == (None, None)
    assert resolve_reasoning_params("none", None, profile) == (None, None)
    assert resolve_reasoning_params("high", True, profile) == ("high", None)
    assert resolve_reasoning_params("low", False, profile) == ("low", None)


def test_models_without_thinking_support_drop_reasoning_controls() -> None:
    profile = ResolvedCapabilityProfile(
        supports_thinking=False,
        supports_thinking_toggle=False,
    )

    assert resolve_reasoning_params(None, True, profile) == (None, None)
    assert resolve_reasoning_params("high", None, profile) == (None, None)


def test_enable_thinking_true_derives_medium() -> None:
    assert resolve_reasoning_params(None, True) == ("medium", True)


def test_enable_thinking_false_derives_none() -> None:
    assert resolve_reasoning_params(None, False) == ("none", False)


def test_reasoning_effort_none_derives_thinking_false() -> None:
    assert resolve_reasoning_params("none", None) == ("none", False)


@pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high", "xhigh"])
def test_non_none_effort_derives_thinking_true(effort: str) -> None:
    assert resolve_reasoning_params(effort, None) == (effort, True)  # pyright: ignore[reportArgumentType]
