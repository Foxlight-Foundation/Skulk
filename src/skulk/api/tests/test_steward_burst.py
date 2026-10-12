"""The steward reads remote capacity only under separate plugin read access."""

import json
from typing import TYPE_CHECKING, cast

from skulk.api.steward import StewardHarness, steward_tool_definitions

if TYPE_CHECKING:
    from skulk.api.main import API


async def test_remote_capacity_needs_its_own_reader() -> None:
    """A forged tool call cannot read providers or rentals with general access."""
    assert "get_burst_capacity" not in json.dumps(steward_tool_definitions())
    assert "get_burst_capacity" in json.dumps(
        steward_tool_definitions(include_burst=True)
    )
    general = StewardHarness(cast("API", object()))
    assert "plugins:read" in await general.execute_tool("get_burst_capacity", {})


async def test_remote_capacity_is_reported_and_failures_reveal_nothing() -> None:
    reads = 0

    async def reader() -> dict[str, object]:
        nonlocal reads
        reads += 1
        return {
            "scope": "local_api_host",
            "providers": [{"pluginId": "managed.runpod", "state": "available"}],
            "rentals": [{"operationId": "1" * 32, "phase": "ready"}],
        }

    allowed = StewardHarness(cast("API", object()), burst_reader=reader)
    result = cast(
        "dict[str, list[dict[str, str]]]",
        json.loads(await allowed.execute_tool("get_burst_capacity", {})),
    )
    assert result["providers"][0]["state"] == "available"
    assert result["rentals"][0]["phase"] == "ready" and reads == 1

    async def failing() -> dict[str, object]:
        raise OSError("private-path-never-disclose")

    refused = await StewardHarness(
        cast("API", object()), burst_reader=failing
    ).execute_tool("get_burst_capacity", {})
    assert "unavailable" in refused and "private" not in refused
