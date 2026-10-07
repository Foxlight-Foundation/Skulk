"""Tests for environment switch parsing in ``skulk.shared.constants``."""

import pytest

from skulk.shared.constants import env_switch_enabled


@pytest.mark.parametrize("value", [None, "", "   "])
@pytest.mark.parametrize("default", [True, False])
def test_unset_or_blank_switch_keeps_default(value: str | None, default: bool) -> None:
    assert env_switch_enabled(value, default=default) is default


@pytest.mark.parametrize("value", ["false", "FALSE", "0", "no", "Off", " off "])
def test_disabling_values_turn_a_default_on_switch_off(value: str) -> None:
    assert env_switch_enabled(value, default=True) is False


@pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes", "on"])
def test_other_values_turn_a_default_off_switch_on(value: str) -> None:
    assert env_switch_enabled(value, default=False) is True
