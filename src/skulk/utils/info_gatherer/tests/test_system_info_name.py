"""Tests for the node friendly-name resolution (#555)."""

import pytest

from skulk.utils.info_gatherer.system_info import get_friendly_name


async def test_node_name_env_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    # Containers and rented pods carry runtime-random hostnames and cannot
    # sethostname; the env override is their only path to a real identity.
    monkeypatch.setenv("SKULK_NODE_NAME", "skulk-a100")
    assert await get_friendly_name() == "skulk-a100"


async def test_blank_override_falls_back_to_hostname(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SKULK_NODE_NAME", "   ")
    name = await get_friendly_name()
    assert name
    assert name != "   "
