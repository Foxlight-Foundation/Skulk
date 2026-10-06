"""The fleet served context default in ``skulk.yaml``."""

import pytest
from pydantic import ValidationError

from skulk.shared.models.memory_estimate import SERVED_CONTEXT_DEFAULT_TOKENS
from skulk.store.config import InferenceConfig, SkulkConfig, served_context_default


def test_default_applies_when_the_section_is_absent() -> None:
    assert served_context_default(None) == SERVED_CONTEXT_DEFAULT_TOKENS
    assert served_context_default(SkulkConfig()) == SERVED_CONTEXT_DEFAULT_TOKENS
    assert SERVED_CONTEXT_DEFAULT_TOKENS == 32768


def test_the_setting_overrides_the_default() -> None:
    config = SkulkConfig(inference=InferenceConfig(served_context_tokens=65536))
    assert served_context_default(config) == 65536


@pytest.mark.parametrize("value", [0, 255, 1048577])
def test_out_of_range_values_are_rejected(value: int) -> None:
    with pytest.raises(ValidationError):
        InferenceConfig(served_context_tokens=value)
