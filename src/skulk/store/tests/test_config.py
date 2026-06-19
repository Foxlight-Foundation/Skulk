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

    HOME is isolated for consistency with the other staging tests so host
    state cannot influence the result.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    config = StagingNodeConfig()

    assert config.enabled
    assert config.node_cache_path == "~/.skulk/staging"
    assert config.cleanup_on_deactivate
    assert config.staging_keep_recent_gb == 40.0


def test_staging_default_ignores_legacy_exo_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default staging path is always ~/.skulk/staging. The EXO_ deprecation
    runway is gone (#324): a populated legacy ~/.exo/staging is no longer
    migrated, and an explicit path is respected verbatim."""
    monkeypatch.setenv("HOME", str(tmp_path))
    legacy = tmp_path / ".exo" / "staging"
    legacy.mkdir(parents=True)
    (legacy / "mlx-community--some-model").mkdir()

    # The populated legacy dir is ignored; the default stays ~/.skulk/staging.
    assert StagingNodeConfig().node_cache_path == "~/.skulk/staging"
    # Explicit configuration is never rewritten.
    explicit = StagingNodeConfig(node_cache_path="/Volumes/foxlight/models")
    assert explicit.node_cache_path == "/Volumes/foxlight/models"


def test_resolve_node_staging_matches_local_hostname_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("skulk.store.config.socket.gethostname", lambda: "kite3")
    config = ModelStoreConfig(
        store_host="kite3.local",
        store_path="/Volumes/foxlight/models",
        staging=StagingNodeConfig(node_cache_path="~/.skulk/staging"),
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
