"""Tests for XDG Base Directory Specification compliance."""

import os
import sys
from pathlib import Path
from unittest import mock


def _safe_env_without(*prefixes: str) -> dict[str, str]:
    """Build a clean env dict, stripping vars with the given prefixes.

    Always preserves SKULK_DASHBOARD_DIR / EXO_DASHBOARD_DIR and
    SKULK_RESOURCES_DIR / EXO_RESOURCES_DIR so that `importlib.reload`
    on constants doesn't fail looking for dashboard assets in CI.
    """
    keep = {
        "SKULK_DASHBOARD_DIR",
        "EXO_DASHBOARD_DIR",
        "SKULK_RESOURCES_DIR",
        "EXO_RESOURCES_DIR",
    }
    return {
        k: v
        for k, v in os.environ.items()
        if k in keep or not any(k.startswith(p) for p in prefixes)
    }


def test_xdg_paths_on_linux():
    """Test that XDG paths are used on Linux when XDG env vars are set."""
    env = _safe_env_without("SKULK_", "EXO_", "XDG_")
    env.update(
        {
            "XDG_CONFIG_HOME": "/tmp/test-config",
            "XDG_DATA_HOME": "/tmp/test-data",
            "XDG_CACHE_HOME": "/tmp/test-cache",
        }
    )
    with (
        mock.patch.dict(os.environ, env, clear=True),
        mock.patch.object(sys, "platform", "linux"),
    ):
        import importlib

        import exo.shared.constants as constants

        importlib.reload(constants)

        assert Path("/tmp/test-config/skulk") == constants.SKULK_CONFIG_HOME
        assert Path("/tmp/test-data/skulk") == constants.SKULK_DATA_HOME
        assert Path("/tmp/test-cache/skulk") == constants.SKULK_CACHE_HOME
        # Deprecated aliases still work
        assert constants.SKULK_CONFIG_HOME == constants.EXO_CONFIG_HOME


def test_xdg_default_paths_on_linux():
    """Test that XDG default paths are used on Linux when env vars are not set."""
    env = _safe_env_without("XDG_", "SKULK_", "EXO_")
    with (
        mock.patch.dict(os.environ, env, clear=True),
        mock.patch.object(sys, "platform", "linux"),
    ):
        import importlib

        import exo.shared.constants as constants

        importlib.reload(constants)

        home = Path.home()
        assert home / ".config" / "skulk" == constants.SKULK_CONFIG_HOME
        assert home / ".local/share" / "skulk" == constants.SKULK_DATA_HOME
        assert home / ".cache" / "skulk" == constants.SKULK_CACHE_HOME


def test_skulk_home_takes_precedence():
    """Test that SKULK_HOME environment variable takes precedence."""
    env = _safe_env_without("SKULK_", "EXO_")
    env["SKULK_HOME"] = ".custom-skulk"
    env["XDG_CONFIG_HOME"] = "/tmp/test-config"
    with (
        mock.patch.dict(os.environ, env, clear=True),
    ):
        import importlib

        import exo.shared.constants as constants

        importlib.reload(constants)

        home = Path.home()
        assert home / ".custom-skulk" == constants.SKULK_CONFIG_HOME
        assert home / ".custom-skulk" == constants.SKULK_DATA_HOME


def test_legacy_exo_home_fallback():
    """Test that EXO_HOME still works as a fallback when SKULK_HOME is not set."""
    env = _safe_env_without("SKULK_", "EXO_")
    env["EXO_HOME"] = ".custom-exo"
    with mock.patch.dict(os.environ, env, clear=True):
        import importlib

        import exo.shared.constants as constants

        importlib.reload(constants)

        home = Path.home()
        assert home / ".custom-exo" == constants.SKULK_CONFIG_HOME


def test_macos_uses_skulk_directory():
    """Test that macOS uses ~/.skulk directory by default."""
    env = _safe_env_without("SKULK_", "EXO_")
    with (
        mock.patch.dict(os.environ, env, clear=True),
        mock.patch.object(sys, "platform", "darwin"),
    ):
        import importlib

        import exo.shared.constants as constants

        importlib.reload(constants)

        home = Path.home()
        # On a fresh install, .skulk is used. If .exo exists and .skulk
        # doesn't, the fallback logic picks .exo — but we can't easily
        # test filesystem state here, so just check it's one of the two.
        assert constants.SKULK_CONFIG_HOME in (home / ".skulk", home / ".exo")


def test_node_id_in_config_dir():
    """Test that node ID keypair is in the config directory."""
    # Reload to get a clean state after previous tests may have changed env
    env = _safe_env_without("SKULK_", "EXO_")
    with mock.patch.dict(os.environ, env, clear=True):
        import importlib

        import exo.shared.constants as constants

        importlib.reload(constants)

        assert constants.SKULK_NODE_ID_KEYPAIR.parent == constants.SKULK_CONFIG_HOME


def test_models_in_data_dir():
    """Test that models directory is in the data directory."""
    env = _safe_env_without("SKULK_MODELS", "EXO_MODELS", "SKULK_HOME", "EXO_HOME")
    with mock.patch.dict(os.environ, env, clear=True):
        import importlib

        import exo.shared.constants as constants

        importlib.reload(constants)

        assert constants.SKULK_MODELS_DIR.parent == constants.SKULK_DATA_HOME
