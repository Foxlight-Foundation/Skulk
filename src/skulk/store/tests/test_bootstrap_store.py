"""A node that starts without skulk.yaml gets a single-node model store (#629)."""

from pathlib import Path

import pytest

import skulk.store.config as store_config
from skulk.store.config import (
    fill_model_store_defaults,
    load_skulk_config,
    write_bootstrap_config_if_absent,
)


@pytest.fixture(autouse=True)
def _this_host(  # pyright: ignore[reportUnusedFunction] - autouse
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(store_config.socket, "gethostname", lambda: "Kite-Dev.local")
    monkeypatch.setattr(store_config, "SKULK_DATA_HOME", tmp_path / "data")


def test_a_fresh_node_writes_the_single_node_store(tmp_path: Path) -> None:
    path = tmp_path / "skulk.yaml"

    written = write_bootstrap_config_if_absent(path)

    assert written is not None
    config = load_skulk_config(path)
    assert config is not None and config.model_store is not None
    store = config.model_store
    assert store.enabled
    assert store.store_host == "Kite-Dev"
    assert store.store_http_host == "127.0.0.1"
    assert store.store_port == 12415
    assert store.store_path == str(tmp_path / "data" / "model-store")
    assert Path(store.store_path).is_dir()
    assert path.read_text().startswith("# Written by Skulk at first start")


def test_an_existing_config_is_never_touched(tmp_path: Path) -> None:
    path = tmp_path / "skulk.yaml"
    path.write_text("inference:\n  kv_cache_backend: default\n")

    assert write_bootstrap_config_if_absent(path) is None
    assert path.read_text() == "inference:\n  kv_cache_backend: default\n"
    assert not (tmp_path / "data").exists()


def test_a_legacy_exo_yaml_is_left_for_the_loud_rename(tmp_path: Path) -> None:
    (tmp_path / "exo.yaml").write_text("model_store: {}\n")
    path = tmp_path / "skulk.yaml"

    assert write_bootstrap_config_if_absent(path) is None
    assert not path.exists()


def test_a_blank_enabled_store_takes_this_node_and_the_default_path(
    tmp_path: Path,
) -> None:
    filled = fill_model_store_defaults(
        {"enabled": True, "store_host": " ", "store_http_host": "", "store_path": ""}
    )

    assert filled == {
        "enabled": True,
        "store_host": "Kite-Dev",
        "store_http_host": "127.0.0.1",
        "store_path": str(tmp_path / "data" / "model-store"),
    }


def test_an_omitted_enabled_flag_still_counts_as_enabled(tmp_path: Path) -> None:
    filled = fill_model_store_defaults({})

    assert filled["store_host"] == "Kite-Dev"
    assert filled["store_path"] == str(tmp_path / "data" / "model-store")


def test_named_values_are_kept() -> None:
    store = {"store_host": "mac-studio", "store_path": "/Volumes/Models"}

    assert fill_model_store_defaults(store) == store


def test_a_named_host_keeps_its_own_http_host(tmp_path: Path) -> None:
    filled = fill_model_store_defaults({"store_host": "mac-studio", "store_path": ""})

    assert filled == {
        "store_host": "mac-studio",
        "store_path": str(tmp_path / "data" / "model-store"),
    }


def test_a_disabled_store_is_left_alone() -> None:
    store = {"enabled": False, "store_host": "", "store_path": ""}

    assert fill_model_store_defaults(store) == store
