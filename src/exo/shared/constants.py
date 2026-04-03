import os
import sys
from pathlib import Path

from exo.utils.dashboard_path import find_dashboard, find_resources


def _env(skulk_key: str, exo_key: str, default: str | None = None) -> str | None:
    """Read an env var, preferring SKULK_* and falling back to legacy EXO_*.

    This lets existing deployments keep working with EXO_* env vars while
    new deployments use SKULK_*.  SKULK_* always wins if both are set.
    """
    return os.environ.get(skulk_key, os.environ.get(exo_key, default))


_SKULK_HOME_ENV = _env("SKULK_HOME", "EXO_HOME")


def _get_home_dir() -> Path:
    """Determine the home directory for Skulk data.

    Priority: SKULK_HOME env var > EXO_HOME env var > ~/.skulk (if exists
    or ~/.exo doesn't) > ~/.exo (legacy fallback).  On Linux, respects
    XDG directories when no home override is set.
    """
    if _SKULK_HOME_ENV is not None:
        return Path.home() / _SKULK_HOME_ENV

    if sys.platform != "linux":
        skulk_path = Path.home() / ".skulk"
        exo_path = Path.home() / ".exo"
        # Prefer .skulk; fall back to .exo only if it exists and .skulk doesn't
        if exo_path.exists() and not skulk_path.exists():
            return exo_path
        return skulk_path

    return Path.home() / ".skulk"


def _get_xdg_dir(env_var: str, fallback: str) -> Path:
    """Get XDG directory, prioritising SKULK_HOME/EXO_HOME if set. On non-Linux platforms, default to ~/.skulk."""

    if _SKULK_HOME_ENV is not None:
        return Path.home() / _SKULK_HOME_ENV

    if sys.platform != "linux":
        return _get_home_dir()

    xdg_value = os.environ.get(env_var, None)
    if xdg_value is not None:
        return Path(xdg_value) / "skulk"
    return Path.home() / fallback / "skulk"


SKULK_CONFIG_HOME = _get_xdg_dir("XDG_CONFIG_HOME", ".config")
SKULK_DATA_HOME = _get_xdg_dir("XDG_DATA_HOME", ".local/share")
SKULK_CACHE_HOME = _get_xdg_dir("XDG_CACHE_HOME", ".cache")

# Models directory (data)
_SKULK_MODELS_DIR_ENV = _env("SKULK_MODELS_DIR", "EXO_MODELS_DIR")
SKULK_MODELS_DIR = (
    SKULK_DATA_HOME / "models"
    if _SKULK_MODELS_DIR_ENV is None
    else Path.home() / _SKULK_MODELS_DIR_ENV
)

# Read-only search path for pre-downloaded models (colon-separated directories)
_SKULK_MODELS_PATH_ENV = _env("SKULK_MODELS_PATH", "EXO_MODELS_PATH")
_extra_model_paths: list[Path] = []
SKULK_MODELS_PATH: tuple[Path, ...] | None = (
    tuple(Path(p).expanduser() for p in _SKULK_MODELS_PATH_ENV.split(":") if p)
    if _SKULK_MODELS_PATH_ENV is not None
    else None
)


# Alias for upstream compatibility — upstream reworked models dir into
# a multi-directory tuple. Our model store handles directory management,
# so this is just a single-element tuple wrapping the existing path.
SKULK_MODELS_DIRS: tuple[Path, ...] = (SKULK_MODELS_DIR,)
SKULK_MODELS_READ_ONLY_DIRS: tuple[Path, ...] = (
    tuple(Path(p).expanduser() for p in _SKULK_MODELS_PATH_ENV.split(":") if p)
    if _SKULK_MODELS_PATH_ENV is not None
    else ()
)


def add_model_search_path(path: Path) -> None:
    """Add a directory to the model search path at runtime.

    Used by the model store to make the staging directory searchable
    by ``build_model_path`` / ``resolve_model_in_path``.
    """
    global SKULK_MODELS_PATH, EXO_MODELS_PATH  # noqa: PLW0603
    expanded = path.expanduser()
    if expanded not in _extra_model_paths:
        _extra_model_paths.append(expanded)
    existing = list(SKULK_MODELS_PATH) if SKULK_MODELS_PATH else []
    SKULK_MODELS_PATH = tuple(existing + _extra_model_paths)  # pyright: ignore[reportConstantRedefinition]
    EXO_MODELS_PATH = SKULK_MODELS_PATH  # pyright: ignore[reportConstantRedefinition] — keep alias in sync


_RESOURCES_DIR_ENV = _env("SKULK_RESOURCES_DIR", "EXO_RESOURCES_DIR")
RESOURCES_DIR = (
    find_resources() if _RESOURCES_DIR_ENV is None else Path.home() / _RESOURCES_DIR_ENV
)
_DASHBOARD_DIR_ENV = _env("SKULK_DASHBOARD_DIR", "EXO_DASHBOARD_DIR")
DASHBOARD_DIR = (
    find_dashboard() if _DASHBOARD_DIR_ENV is None else Path.home() / _DASHBOARD_DIR_ENV
)

# Log files (data/logs or cache)
SKULK_LOG_DIR = SKULK_CACHE_HOME / "logs"
SKULK_LOG = SKULK_LOG_DIR / "skulk.log"
SKULK_TEST_LOG = SKULK_CACHE_HOME / "skulk_test.log"

# Identity (config)
SKULK_NODE_ID_KEYPAIR = SKULK_CONFIG_HOME / "node_id.keypair"
SKULK_CONFIG_FILE = SKULK_CONFIG_HOME / "config.toml"

# libp2p topics for event forwarding
LIBP2P_LOCAL_EVENTS_TOPIC = "worker_events"
LIBP2P_GLOBAL_EVENTS_TOPIC = "global_events"
LIBP2P_ELECTION_MESSAGES_TOPIC = "election_message"
LIBP2P_COMMANDS_TOPIC = "commands"

SKULK_MAX_CHUNK_SIZE = 512 * 1024

SKULK_CUSTOM_MODEL_CARDS_DIR = SKULK_DATA_HOME / "custom_model_cards"

SKULK_EVENT_LOG_DIR = SKULK_DATA_HOME / "event_log"
SKULK_IMAGE_CACHE_DIR = SKULK_CACHE_HOME / "images"
SKULK_TRACING_CACHE_DIR = SKULK_CACHE_HOME / "traces"

SKULK_ENABLE_IMAGE_MODELS = (
    _env("SKULK_ENABLE_IMAGE_MODELS", "EXO_ENABLE_IMAGE_MODELS", "false") or "false"
).lower() == "true"

SKULK_OFFLINE = (
    _env("SKULK_OFFLINE", "EXO_OFFLINE", "false") or "false"
).lower() == "true"

SKULK_TRACING_ENABLED = (
    _env("SKULK_TRACING_ENABLED", "EXO_TRACING_ENABLED", "false") or "false"
).lower() == "true"

SKULK_MAX_CONCURRENT_REQUESTS = int(
    _env("SKULK_MAX_CONCURRENT_REQUESTS", "EXO_MAX_CONCURRENT_REQUESTS", "8") or "8"
)


# ── Deprecated aliases ──────────────────────────────────────────────────
# Keep these so existing code that imports the old names still works
# during the transition. These will be removed in a future release.
EXO_CONFIG_HOME = SKULK_CONFIG_HOME
EXO_DATA_HOME = SKULK_DATA_HOME
EXO_CACHE_HOME = SKULK_CACHE_HOME
EXO_MODELS_DIR = SKULK_MODELS_DIR
EXO_MODELS_PATH = SKULK_MODELS_PATH
EXO_MODELS_DIRS = SKULK_MODELS_DIRS
EXO_MODELS_READ_ONLY_DIRS = SKULK_MODELS_READ_ONLY_DIRS
EXO_LOG_DIR = SKULK_LOG_DIR
EXO_LOG = SKULK_LOG
EXO_TEST_LOG = SKULK_TEST_LOG
EXO_NODE_ID_KEYPAIR = SKULK_NODE_ID_KEYPAIR
EXO_CONFIG_FILE = SKULK_CONFIG_FILE
EXO_MAX_CHUNK_SIZE = SKULK_MAX_CHUNK_SIZE
EXO_CUSTOM_MODEL_CARDS_DIR = SKULK_CUSTOM_MODEL_CARDS_DIR
EXO_EVENT_LOG_DIR = SKULK_EVENT_LOG_DIR
EXO_IMAGE_CACHE_DIR = SKULK_IMAGE_CACHE_DIR
EXO_TRACING_CACHE_DIR = SKULK_TRACING_CACHE_DIR
EXO_ENABLE_IMAGE_MODELS = SKULK_ENABLE_IMAGE_MODELS
EXO_OFFLINE = SKULK_OFFLINE
EXO_TRACING_ENABLED = SKULK_TRACING_ENABLED
EXO_MAX_CONCURRENT_REQUESTS = SKULK_MAX_CONCURRENT_REQUESTS
