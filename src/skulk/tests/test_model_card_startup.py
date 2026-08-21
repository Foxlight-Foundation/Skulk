"""Tests for model-card truth ordering during node startup."""

import pytest

from skulk import main as skulk_main
from skulk.main import Node


@pytest.mark.asyncio
async def test_node_loads_cards_before_starting_without_store_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-store node initializes durable ownership before serving commands."""
    catalog_loaded = False

    async def load_catalog() -> list[object]:
        nonlocal catalog_loaded
        catalog_loaded = True
        raise RuntimeError("stop before runtime tasks")

    monkeypatch.setattr(skulk_main, "get_all_model_cards", load_catalog)
    node = object.__new__(Node)
    node.store_server = None

    with pytest.raises(RuntimeError, match="stop before runtime tasks"):
        await node.run()

    assert catalog_loaded
