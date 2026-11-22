"""The plugin manager starts, answers and stops on every platform Skulk runs on."""

from pathlib import Path

import pytest

from skulk.extensions import runtime_manager
from skulk.extensions.runtime_attachment import HostSettings
from skulk.extensions.runtime_manager import (
    InventoryRequest,
    RuntimeManager,
    manager_request,
)
from skulk.utils.runtime_files import private_directory, write_private


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A provisioned, owner-only service root, as local setup leaves it."""
    # Test runners on some platforms run as an administrator; the rule that
    # refuses that has its own tests.
    monkeypatch.setattr(runtime_manager, "has_root_authority", lambda: False)
    path = tmp_path / "service"
    private_directory(path)
    write_private(
        path / "host.json",
        HostSettings(transport_node_id="node-a", profile_id="a" * 32)
        .model_dump_json()
        .encode(),
    )
    return path


async def test_a_manager_serves_its_inventory_and_stops(root: Path) -> None:
    manager = RuntimeManager(root)
    await manager.start()
    try:
        assert manager.boot is not None
        await manager.boot
        assert manager.errors == {}
        for _ in range(3):
            reply = await manager_request(root, InventoryRequest())
            result = reply["result"]
            assert isinstance(result, dict)
            assert result["installations"] == []
    finally:
        await manager.close()

    with pytest.raises(OSError):
        await manager_request(root, InventoryRequest())


async def test_one_root_has_one_manager(root: Path) -> None:
    manager = RuntimeManager(root)
    await manager.start()
    try:
        second = RuntimeManager(root)
        with pytest.raises(BlockingIOError):
            await second.start()
        # The refused one did not disturb the first.
        assert "result" in await manager_request(root, InventoryRequest())
    finally:
        await manager.close()

    again = RuntimeManager(root)
    await again.start()
    try:
        assert "result" in await manager_request(root, InventoryRequest())
    finally:
        await again.close()


def test_a_manager_refuses_root_authority(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime_manager, "has_root_authority", lambda: True)
    with pytest.raises(ValueError, match="without root"):
        RuntimeManager(root)
