"""What a store host shares about staging, and what each node then stages into.

The store host sends its settings to every node and every node writes them to
its own file. A staging folder nobody wrote is a node's own default, so it
must not travel: two nodes whose homes differ would otherwise end up staging
into the host's folder.
"""

from __future__ import annotations

import socket
from pathlib import Path
from typing import cast

import pytest
import yaml

import skulk.store.config as store_config
from skulk import main as main_module
from skulk.download import coordinator as coordinator_module
from skulk.download.tests.test_progress_throttle import (
    _make_coordinator,  # pyright: ignore[reportPrivateUsage]
)
from skulk.main import Node
from skulk.shared.types.commands import ForwarderDownloadCommand, SyncConfig
from skulk.store.config import load_skulk_config, resolve_node_staging


def _use_home(monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
    """Make this process a node whose home was moved to ``home``."""
    monkeypatch.setattr(store_config, "SKULK_HOME_IS_SET", True)
    monkeypatch.setattr(store_config, "SKULK_DATA_HOME", home)


async def _broadcast(monkeypatch: pytest.MonkeyPatch, host_file: Path) -> str:
    """Run the store host's broadcast against ``host_file`` and return what it sent."""
    loaded = load_skulk_config(host_file)
    assert loaded is not None
    monkeypatch.setattr(main_module, "load_skulk_config", lambda: loaded)
    sent: list[ForwarderDownloadCommand] = []

    class _Sender:
        async def send(self, command: ForwarderDownloadCommand) -> None:
            sent.append(command)

    class _Router:
        def sender(self, _topic: object) -> _Sender:
            return _Sender()

    node = object.__new__(Node)
    object.__setattr__(node, "skulk_config", loaded)
    object.__setattr__(node, "router", _Router())
    object.__setattr__(node, "node_id", "not-the-store-host-by-id")

    await Node._broadcast_config_if_store_host(node)  # pyright: ignore[reportPrivateUsage]

    assert len(sent) == 1
    command = sent[0].command
    assert isinstance(command, SyncConfig)
    return command.config_yaml


async def _receive(
    monkeypatch: pytest.MonkeyPatch, node_file: Path, config_yaml: str
) -> None:
    """Have a node write a received broadcast to its own settings file."""
    monkeypatch.setattr(coordinator_module, "resolve_config_path", lambda: node_file)
    await _make_coordinator()._sync_config(config_yaml)  # pyright: ignore[reportPrivateUsage]


def _store_section(tmp_path: Path, **extra: object) -> str:
    return yaml.safe_dump(
        {
            "model_store": {
                "store_host": socket.gethostname(),
                "store_http_host": "192.0.2.10",
                "store_path": str(tmp_path / "store"),
                **extra,
            }
        }
    )


async def test_a_staging_folder_nobody_wrote_stays_each_nodes_own(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two nodes with different homes keep two staging folders through a broadcast."""
    host_home, other_home = tmp_path / "host-home", tmp_path / "other-home"
    host_file, other_file = tmp_path / "host.yaml", tmp_path / "other.yaml"
    host_file.write_text(_store_section(tmp_path))

    _use_home(monkeypatch, host_home)
    config_yaml = await _broadcast(monkeypatch, host_file)
    assert "node_cache_path" not in config_yaml
    # The host writes its own broadcast too, and still stages in its own home.
    await _receive(monkeypatch, host_file, config_yaml)
    host = load_skulk_config(host_file)
    assert host is not None and host.model_store is not None
    assert host.model_store.staging.node_cache_path == str(host_home / "staging")

    _use_home(monkeypatch, other_home)
    await _receive(monkeypatch, other_file, config_yaml)
    other = load_skulk_config(other_file)
    assert other is not None and other.model_store is not None
    assert other.model_store.staging.node_cache_path == str(other_home / "staging")
    assert "node_cache_path" not in other_file.read_text()

    # A second broadcast from the host's rewritten file is the same again.
    _use_home(monkeypatch, host_home)
    assert "node_cache_path" not in await _broadcast(monkeypatch, host_file)


async def test_a_staging_folder_a_person_wrote_is_shared_as_written(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    host_file, other_file = tmp_path / "host.yaml", tmp_path / "other.yaml"
    host_file.write_text(
        _store_section(tmp_path, staging={"node_cache_path": "~/.skulk/staging"})
    )

    _use_home(monkeypatch, tmp_path / "host-home")
    config_yaml = await _broadcast(monkeypatch, host_file)
    _use_home(monkeypatch, tmp_path / "other-home")
    await _receive(monkeypatch, other_file, config_yaml)

    other = load_skulk_config(other_file)
    assert other is not None and other.model_store is not None
    assert other.model_store.staging.node_cache_path == "~/.skulk/staging"


async def test_a_partial_node_override_stays_partial_through_a_broadcast(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An override that names one setting must go on inheriting the others.

    Sent with every default filled in, it would stop inheriting the base
    staging folder after one broadcast, and carry the host's own default
    folder to the node it names.
    """
    host_file, other_file = tmp_path / "host.yaml", tmp_path / "other.yaml"
    host_file.write_text(
        _store_section(
            tmp_path,
            staging={"node_cache_path": "/fast/staging"},
            node_overrides={
                "node-b": {"staging": {"cleanup_on_deactivate": False}},
                "node-c": {"staging": {"node_cache_path": "/other/staging"}},
            },
        )
    )

    _use_home(monkeypatch, tmp_path / "host-home")
    config_yaml = await _broadcast(monkeypatch, host_file)
    shared = cast(dict[str, object], yaml.safe_load(config_yaml))
    overrides = cast(
        dict[str, dict[str, object]],
        cast(dict[str, object], shared["model_store"])["node_overrides"],
    )
    assert overrides["node-b"] == {"staging": {"cleanup_on_deactivate": False}}
    assert overrides["node-c"] == {"staging": {"node_cache_path": "/other/staging"}}

    _use_home(monkeypatch, tmp_path / "other-home")
    await _receive(monkeypatch, other_file, config_yaml)
    other = load_skulk_config(other_file)
    assert other is not None and other.model_store is not None
    on_b = resolve_node_staging(other.model_store, "node-b")
    assert (on_b.node_cache_path, on_b.cleanup_on_deactivate) == ("/fast/staging", False)
    on_c = resolve_node_staging(other.model_store, "node-c")
    assert (on_c.node_cache_path, on_c.cleanup_on_deactivate) == ("/other/staging", True)
