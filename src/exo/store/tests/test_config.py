import pytest

from exo.store.config import (
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


def test_staging_cleanup_defaults_to_warm_cache() -> None:
    config = StagingNodeConfig()

    assert config.enabled
    assert config.node_cache_path == "~/.exo/staging"
    assert not config.cleanup_on_deactivate


def test_resolve_node_staging_matches_local_hostname_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("exo.store.config.socket.gethostname", lambda: "kite3")
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
