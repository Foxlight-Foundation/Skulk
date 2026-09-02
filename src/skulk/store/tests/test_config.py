import os
from pathlib import Path

import pytest

from skulk.store.config import (
    DEFAULT_MODEL_STORE_PORT,
    ExperimentsConfig,
    ModelStoreConfig,
    NodeOverrideConfig,
    SkulkConfig,
    StagingNodeConfig,
    hostname_aliases,
    load_skulk_config,
    node_matches_store_host,
    persist_model_trust_config,
    resolve_node_staging,
)


def test_model_store_default_port_avoids_dynamic_client_range() -> None:
    """Fresh listeners must not race ordinary outbound client connections."""

    config = ModelStoreConfig(store_host="store.local", store_path="/models")

    assert DEFAULT_MODEL_STORE_PORT == 12415
    assert config.store_port == DEFAULT_MODEL_STORE_PORT


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


def test_load_skulk_config_absent_returns_none(tmp_path: Path) -> None:
    """A missing skulk.yaml with no legacy file boots zero-config (returns None)."""
    assert load_skulk_config(tmp_path / "skulk.yaml") is None


def test_load_skulk_config_fails_loud_on_legacy_exo_yaml(tmp_path: Path) -> None:
    """A leftover exo.yaml without skulk.yaml fails loudly with the rename, not
    a silent zero-config boot that would drop store/logging/auth settings (#324)."""
    (tmp_path / "exo.yaml").write_text("model_store:\n  enabled: true\n")
    target = tmp_path / "skulk.yaml"
    with pytest.raises(FileNotFoundError, match="exo.yaml is no longer read"):
        load_skulk_config(target)


def test_persist_model_trust_updates_only_authoritative_section(
    tmp_path: Path,
) -> None:
    """Indexed trust updates retain node-local secrets and unrelated settings."""
    target = tmp_path / "skulk.yaml"
    target.write_text(
        "hf_token: keep-secret\nlogging:\n  enabled: true\n  ingest_url: https://logs.invalid\n"
    )
    card_id = f"card_{'a' * 52}"

    config = persist_model_trust_config(target, [card_id])

    assert config.hf_token == "keep-secret"
    assert config.logging is not None and config.logging.enabled
    assert config.model_trust is not None
    assert config.model_trust.approved_remote_code_identities == [card_id]
    assert target.stat().st_mode & 0o777 == 0o600


def test_persist_model_trust_without_descriptor_chmod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Platforms without os.fchmod still atomically persist cluster trust."""

    target = tmp_path / "skulk.yaml"
    target.write_text("{}\n")
    monkeypatch.delattr(os, "fchmod")
    card_id = f"card_{'b' * 52}"

    config = persist_model_trust_config(target, [card_id])

    assert config.model_trust is not None
    assert config.model_trust.approved_remote_code_identities == [card_id]


def test_experiments_config_defaults_speech_streaming_off() -> None:
    """Experimental feature toggles default off until explicitly opted in."""

    config = SkulkConfig(experiments=ExperimentsConfig())

    assert config.experiments is not None
    assert config.experiments.tts_streaming is False
    assert config.experiments.stt_realtime is False
    assert config.experiments.speech_translation is False


def test_load_skulk_config_parses_speech_streaming_experiments(
    tmp_path: Path,
) -> None:
    """The config file can opt into each experimental speech transport."""

    target = tmp_path / "skulk.yaml"
    target.write_text(
        "experiments:\n  tts_streaming: true\n  stt_realtime: true\n"
        "  speech_translation: true\n"
    )

    config = load_skulk_config(target)

    assert config is not None
    assert config.experiments is not None
    assert config.experiments.tts_streaming is True
    assert config.experiments.stt_realtime is True
    assert config.experiments.speech_translation is True


def test_intelligent_fabric_config_defaults() -> None:
    """The intelligent-fabric section parses, defaults off, and carries the
    benched steward model preference order."""
    from skulk.store.config import IntelligentFabricConfig, SkulkConfig

    config = SkulkConfig.model_validate({})
    assert config.intelligent_fabric is None

    parsed = SkulkConfig.model_validate({"intelligent_fabric": {"enabled": True}})
    assert parsed.intelligent_fabric is not None
    assert parsed.intelligent_fabric.enabled
    assert parsed.intelligent_fabric.steward_models == [
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "mlx-community/Qwen3.6-35B-A3B-4bit",
        "mlx-community/Qwen3.5-4B-MLX-4bit",
        "unsloth/Qwen3.5-4B-GGUF",
        "unsloth/Qwen3.5-0.8B-GGUF",
    ]

    # The documented YAML-sequence override must load (list, not tuple:
    # strict validation rejects coercion).
    override = SkulkConfig.model_validate(
        {"intelligent_fabric": {"enabled": True, "steward_models": ["a/b"]}}
    )
    assert override.intelligent_fabric is not None
    assert override.intelligent_fabric.steward_models == ["a/b"]

    default_section = IntelligentFabricConfig()
    assert not default_section.enabled


def test_enabled_store_refuses_blank_host() -> None:
    """The installer-shaped brokenness fails loudly at validation (#888).

    ``store_host: ''`` matches no node, so no store server ever starts and
    every client builds ``http://:12415`` URLs; a fleet shipped in this shape
    could not place any model that was not already staged.
    """
    import pytest
    from pydantic import ValidationError

    from skulk.store.config import ModelStoreConfig

    with pytest.raises(ValidationError, match="store_host"):
        ModelStoreConfig(enabled=True, store_host="", store_path="/models")
    with pytest.raises(ValidationError, match="store_path"):
        ModelStoreConfig(enabled=True, store_host="kite", store_path="  ")


def test_disabled_store_permits_blank_identity() -> None:
    """Running without a store is spelled enabled: false, and stays valid."""
    from skulk.store.config import ModelStoreConfig

    config = ModelStoreConfig(enabled=False, store_host="", store_path="")
    assert config.enabled is False


def test_enabled_store_with_identity_is_valid() -> None:
    from skulk.store.config import ModelStoreConfig

    config = ModelStoreConfig(enabled=True, store_host="kite", store_path="/models")
    assert config.store_host == "kite"


def test_normalized_hf_token_treats_whitespace_as_absent() -> None:
    """Whitespace is truthy but not a credential (#922 review)."""
    from skulk.store.config import normalized_hf_token

    assert normalized_hf_token(None) is None
    assert normalized_hf_token("") is None
    assert normalized_hf_token("   ") is None
    assert normalized_hf_token("\t\n") is None
    assert normalized_hf_token(123) is None
    assert normalized_hf_token(" real-token ") == "real-token"


class TestPromoteHfToken:
    """Rotation converges; operator launch values are never replaced (#922)."""

    def _clear(self, monkeypatch: object) -> None:
        from skulk.store.config import HF_TOKEN_USER_SET_MARKER

        monkeypatch.delenv("HF_TOKEN", raising=False)  # pyright: ignore[reportAttributeAccessIssue]
        monkeypatch.delenv(HF_TOKEN_USER_SET_MARKER, raising=False)  # pyright: ignore[reportAttributeAccessIssue]

    def test_promotes_into_an_empty_environment(
        self, monkeypatch: "pytest.MonkeyPatch"
    ) -> None:
        import os

        from skulk.store.config import promote_hf_token

        self._clear(monkeypatch)
        assert promote_hf_token("token-a", source="test") is True
        assert os.environ["HF_TOKEN"] == "token-a"

    def test_replaces_a_config_derived_value_so_rotation_converges(
        self, monkeypatch: "pytest.MonkeyPatch"
    ) -> None:
        import os

        from skulk.store.config import promote_hf_token

        self._clear(monkeypatch)
        monkeypatch.setenv("HF_TOKEN", "token-a")  # promoted from config earlier
        assert promote_hf_token("token-b", source="test") is True
        assert os.environ["HF_TOKEN"] == "token-b"

    def test_never_replaces_an_operator_supplied_launch_value(
        self, monkeypatch: "pytest.MonkeyPatch"
    ) -> None:
        import os

        from skulk.store.config import HF_TOKEN_USER_SET_MARKER, promote_hf_token

        self._clear(monkeypatch)
        monkeypatch.setenv("HF_TOKEN", "operator-token")
        monkeypatch.setenv(HF_TOKEN_USER_SET_MARKER, "1")
        assert promote_hf_token("fleet-token", source="test") is False
        assert os.environ["HF_TOKEN"] == "operator-token"

    def test_whitespace_never_lands_in_the_environment(
        self, monkeypatch: "pytest.MonkeyPatch"
    ) -> None:
        import os

        from skulk.store.config import promote_hf_token

        self._clear(monkeypatch)
        assert promote_hf_token("   ", source="test") is False
        assert "HF_TOKEN" not in os.environ

    def test_noop_when_the_value_is_already_current(
        self, monkeypatch: "pytest.MonkeyPatch"
    ) -> None:
        from skulk.store.config import promote_hf_token

        self._clear(monkeypatch)
        monkeypatch.setenv("HF_TOKEN", "token-a")
        assert promote_hf_token("token-a", source="test") is False
