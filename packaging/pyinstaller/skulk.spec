# -*- mode: python ; coding: utf-8 -*-

import importlib.util
import os
import shutil
import tomllib
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules, copy_metadata

# PyInstaller exposes SPECPATH while evaluating a spec file. Resolving from the
# spec instead of the caller's working directory keeps release builds stable in
# CI and when invoked from a desktop-packaging repository.
PROJECT_ROOT = Path(SPECPATH).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
ENTRYPOINT = SOURCE_ROOT / "skulk" / "__main__.py"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard-react" / "dist"
RESOURCES_DIR = PROJECT_ROOT / "resources"
APP_ICON = PROJECT_ROOT / "packaging" / "macos" / "Skulk.icns"
SKULK_SHARED_MODELS_DIR = SOURCE_ROOT / "skulk" / "shared" / "models"

with (PROJECT_ROOT / "pyproject.toml").open("rb") as pyproject_file:
    PROJECT_VERSION = tomllib.load(pyproject_file)["project"]["version"]

BUNDLE_IDENTIFIER = os.environ.get(
    "SKULK_MACOS_BUNDLE_IDENTIFIER",
    "foundation.foxlight.skulk.desktop.probe",
)
BUNDLE_VERSION = os.environ.get("SKULK_MACOS_BUNDLE_VERSION", "1")
CODESIGN_IDENTITY = os.environ.get("SKULK_CODESIGN_IDENTITY") or None

if not ENTRYPOINT.is_file():
    raise SystemExit(f"Unable to locate Skulk entrypoint: {ENTRYPOINT}")

if not DASHBOARD_DIR.is_dir():
    raise SystemExit(f"Dashboard assets are missing: {DASHBOARD_DIR}")

if not RESOURCES_DIR.is_dir():
    raise SystemExit(f"Resource assets are missing: {RESOURCES_DIR}")

if not APP_ICON.is_file():
    raise SystemExit(f"macOS app icon is missing: {APP_ICON}")

if not SKULK_SHARED_MODELS_DIR.is_dir():
    raise SystemExit(f"Shared model assets are missing: {SKULK_SHARED_MODELS_DIR}")

block_cipher = None


def _module_directory(module_name: str) -> Path:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        raise SystemExit(f"Module '{module_name}' is not available in the current environment.")
    if spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations))).resolve()
    if spec.origin:
        return Path(spec.origin).resolve().parent
    raise SystemExit(f"Unable to determine installation directory for '{module_name}'.")


MLX_PACKAGE_DIR = _module_directory("mlx")
MLX_LIB_DIR = MLX_PACKAGE_DIR / "lib"
if not MLX_LIB_DIR.is_dir():
    raise SystemExit(f"mlx Metal libraries are missing: {MLX_LIB_DIR}")


def _safe_collect(package_name: str) -> list[str]:
    try:
        return collect_submodules(package_name)
    except ImportError:
        return []


HIDDEN_IMPORTS = sorted(
    set(
        collect_submodules("mlx")
        + _safe_collect("mlx_lm")
        + _safe_collect("transformers")
    )
)

DATAS: list[tuple[str, str]] = [
    (str(DASHBOARD_DIR), "dashboard"),
    (str(RESOURCES_DIR), "resources"),
    (str(MLX_LIB_DIR), "mlx/lib"),
    (str(SKULK_SHARED_MODELS_DIR), "skulk/shared/models"),
]

# Skulk and its inference engines inspect installed distribution versions at
# runtime. PyInstaller does not retain package metadata unless the spec asks
# for it explicitly.
for distribution_name in (
    "skulk",
    "mlx",
    "mlx-lm",
    "mlx-vlm",
    "mlx-audio",
    "transformers",
):
    DATAS.extend(copy_metadata(distribution_name))

MACTOP_PATH = shutil.which("mactop")
if MACTOP_PATH is None:
    raise SystemExit(
        "mactop binary not found in PATH. "
        "Install it via: brew install mactop"
    )

BINARIES: list[tuple[str, str]] = [
    (MACTOP_PATH, "."),
]

a = Analysis(
    [str(ENTRYPOINT)],
    pathex=[str(SOURCE_ROOT)],
    binaries=BINARIES,
    datas=DATAS,
    hiddenimports=HIDDEN_IMPORTS,
    hookspath=[str(PROJECT_ROOT / "packaging" / "pyinstaller" / "hooks")],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Skulk",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch="arm64",
    codesign_identity=CODESIGN_IDENTITY,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Skulk",
)

app = BUNDLE(
    coll,
    name="Skulk.app",
    icon=str(APP_ICON),
    bundle_identifier=BUNDLE_IDENTIFIER,
    version=PROJECT_VERSION,
    info_plist={
        "CFBundleDisplayName": "Skulk",
        "CFBundleName": "Skulk",
        "CFBundleShortVersionString": PROJECT_VERSION,
        "CFBundleVersion": BUNDLE_VERSION,
        "LSMinimumSystemVersion": "15.0",
        "NSHighResolutionCapable": True,
        "NSLocalNetworkUsageDescription": (
            "Skulk uses the local network to discover and connect to nearby "
            "Skulk nodes, including Thunderbolt-connected peers."
        ),
    },
)
