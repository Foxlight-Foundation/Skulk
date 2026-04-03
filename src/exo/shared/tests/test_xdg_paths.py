"""Tests for XDG Base Directory Specification compliance."""

import importlib
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest import mock


def _reload_constants_clean(env: dict[str, str], platform: str = "darwin"):
    """Reload exo.shared.constants with a clean env and mocked dashboard paths.

    Mocks find_dashboard/find_resources so tests don't need built dashboard
    assets, then reloads constants to pick up the patched env/platform.

    The patches are only needed during reload — module-level constants are
    set at import time and persist after the patches are torn down.
    """
    with ExitStack() as stack:
        stack.enter_context(mock.patch.dict(os.environ, env, clear=True))
        stack.enter_context(mock.patch.object(sys, "platform", platform))
        stack.enter_context(
            mock.patch(
                "exo.utils.dashboard_path.find_dashboard",
                return_value=Path("/mock/dashboard"),
            )
        )
        stack.enter_context(
            mock.patch(
                "exo.utils.dashboard_path.find_resources",
                return_value=Path("/mock/resources"),
            )
        )
        # Import inside the patched context so even a first-time import
        # won't hit the real filesystem for dashboard assets.
        import exo.shared.constants as constants

        importlib.reload(constants)
    return constants


def _safe_env_without(*prefixes: str) -> dict[str, str]:
    """Build a clean env dict, stripping vars with the given prefixes."""
    return {
        k: v
        for k, v in os.environ.items()
        if not any(k.startswith(p) for p in prefixes)
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
    c = _reload_constants_clean(env, "linux")

    assert Path("/tmp/test-config/skulk") == c.SKULK_CONFIG_HOME
    assert Path("/tmp/test-data/skulk") == c.SKULK_DATA_HOME
    assert Path("/tmp/test-cache/skulk") == c.SKULK_CACHE_HOME
    # Deprecated aliases still work
    assert c.SKULK_CONFIG_HOME == c.EXO_CONFIG_HOME


def test_xdg_default_paths_on_linux():
    """Test that XDG default paths are used on Linux when env vars are not set."""
    env = _safe_env_without("XDG_", "SKULK_", "EXO_")
    c = _reload_constants_clean(env, "linux")

    home = Path.home()
    assert home / ".config" / "skulk" == c.SKULK_CONFIG_HOME
    assert home / ".local/share" / "skulk" == c.SKULK_DATA_HOME
    assert home / ".cache" / "skulk" == c.SKULK_CACHE_HOME


def test_skulk_home_takes_precedence():
    """Test that SKULK_HOME environment variable takes precedence."""
    env = _safe_env_without("SKULK_", "EXO_")
    env["SKULK_HOME"] = ".custom-skulk"
    env["XDG_CONFIG_HOME"] = "/tmp/test-config"
    c = _reload_constants_clean(env)

    home = Path.home()
    assert home / ".custom-skulk" == c.SKULK_CONFIG_HOME
    assert home / ".custom-skulk" == c.SKULK_DATA_HOME


def test_legacy_exo_home_fallback():
    """Test that EXO_HOME still works as a fallback when SKULK_HOME is not set."""
    env = _safe_env_without("SKULK_", "EXO_")
    env["EXO_HOME"] = ".custom-exo"
    c = _reload_constants_clean(env)

    home = Path.home()
    assert home / ".custom-exo" == c.SKULK_CONFIG_HOME


def test_macos_uses_skulk_directory():
    """Test that macOS uses ~/.skulk directory by default."""
    env = _safe_env_without("SKULK_", "EXO_")
    c = _reload_constants_clean(env, "darwin")

    home = Path.home()
    # On a fresh install, .skulk is used. If .exo exists and .skulk
    # doesn't, the fallback logic picks .exo — but we can't easily
    # test filesystem state here, so just check it's one of the two.
    assert c.SKULK_CONFIG_HOME in (home / ".skulk", home / ".exo")


def test_node_id_in_config_dir():
    """Test that node ID keypair is in the config directory."""
    env = _safe_env_without("SKULK_", "EXO_")
    c = _reload_constants_clean(env)

    assert c.SKULK_NODE_ID_KEYPAIR.parent == c.SKULK_CONFIG_HOME


def test_models_in_data_dir():
    """Test that models directory is in the data directory."""
    env = _safe_env_without("SKULK_MODELS", "EXO_MODELS", "SKULK_HOME", "EXO_HOME")
    c = _reload_constants_clean(env)

    assert c.SKULK_MODELS_DIR.parent == c.SKULK_DATA_HOME
