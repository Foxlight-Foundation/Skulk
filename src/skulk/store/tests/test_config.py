from pathlib import Path

import pytest

from skulk.store.config import (
    ModelStoreConfig,
    NodeOverrideConfig,
    StagingNodeConfig,
    hostname_aliases,
    node_matches_store_host,
    resolve_node_staging,
)


def test_hostname_aliases_include_short_and_local_variants() -> None:
    aliases = hostname_aliases("kite3")

    assert aliases == {"kite3", "kite3.local"}


def test_node_matches_store_host_accepts_local_suffix_variant() -> None:
    assert node_matches_store_host(
        store_host="kite3.local",
        node_id="12D3KooExample",
        hostname="kite3",
    )


def test_node_matches_store_host_keeps_node_id_matching_exact() -> None:
    assert node_matches_store_host(
        store_host="12D3KooExactNodeId",
        node_id="12D3KooExactNodeId",
        hostname="kite3",
    )
    assert not node_matches_store_host(
        store_host="12d3kooexactnodeid",
        node_id="12D3KooExactNodeId",
        hostname="kite3",
    )


def test_staging_cleanup_defaults_to_budgeted_eviction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Eviction-on-deactivate with a recent-use grace budget is the default:
    staged copies are cheap to recreate from the LAN store, local disk is
    the scarce resource (two nodes filled to 58-70 GB in the launch smoke),
    and the grace budget keeps crashes/restarts/repeat placements from
    re-paying the staging copy (deliberate product decision, 2026-06-06).

    HOME is isolated so the legacy-staging fallback validator (below) cannot
    fire on developer machines that still carry a populated ~/.exo/staging —
    the unisolated form of this test passes or fails depending on host state.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    config = StagingNodeConfig()

    assert config.enabled
    assert config.node_cache_path == "~/.skulk/staging"
    assert config.cleanup_on_deactivate
    assert config.staging_keep_recent_gb == 40.0


def test_staging_default_prefers_populated_legacy_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A populated pre-rename ~/.exo/staging keeps being used when staging is
    unconfigured, so the 2026-06 exo->skulk rename does not silently orphan
    staged copies and re-stage everything. An explicit path is respected
    verbatim, and an empty legacy dir does not trigger the fallback."""
    monkeypatch.setenv("HOME", str(tmp_path))
    legacy = tmp_path / ".exo" / "staging"
    legacy.mkdir(parents=True)
    (legacy / "mlx-community--some-model").mkdir()

    assert StagingNodeConfig().node_cache_path == "~/.exo/staging"
    # Explicit configuration is never rewritten.
    explicit = StagingNodeConfig(node_cache_path="/Volumes/foxlight/models")
    assert explicit.node_cache_path == "/Volumes/foxlight/models"
    # Once the new dir exists, the default sticks.
    (tmp_path / ".skulk" / "staging").mkdir(parents=True)
    assert StagingNodeConfig().node_cache_path == "~/.skulk/staging"


def test_resolve_node_staging_matches_local_hostname_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("skulk.store.config.socket.gethostname", lambda: "kite3")
    config = ModelStoreConfig(
        store_host="kite3.local",
        store_path="/Volumes/foxlight/models",
        staging=StagingNodeConfig(node_cache_path="~/.exo/staging"),
        node_overrides={
            "kite3.local": NodeOverrideConfig(
                staging=StagingNodeConfig(
                    node_cache_path="/Volumes/foxlight/models",
                    cleanup_on_deactivate=False,
                )
            )
        },
    )

    resolved = resolve_node_staging(config, "12D3KooNodeId")

    assert resolved.node_cache_path == "/Volumes/foxlight/models"
    assert not resolved.cleanup_on_deactivate
