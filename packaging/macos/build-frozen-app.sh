#!/usr/bin/env bash
# Build the disposable frozen Skulk macOS application bundle.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DIST_ROOT="${PROJECT_ROOT}/dist/macos"
WORK_ROOT="${PROJECT_ROOT}/dist/pyinstaller-work"
BUILD_DASHBOARD=1

if [[ "${1:-}" == "--skip-dashboard" ]]; then
    BUILD_DASHBOARD=0
elif [[ $# -gt 0 ]]; then
    echo "Usage: $0 [--skip-dashboard]" >&2
    exit 2
fi

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "The initial Skulk frozen-runtime probe requires Apple Silicon macOS." >&2
    exit 2
fi

cd "${PROJECT_ROOT}"

if [[ $BUILD_DASHBOARD -eq 1 ]]; then
    echo "==> Building the Skulk dashboard"
    npm --prefix dashboard-react ci
    npm --prefix dashboard-react run build
elif [[ ! -f dashboard-react/dist/index.html ]]; then
    echo "Dashboard assets are missing; omit --skip-dashboard for a complete build." >&2
    exit 1
fi

echo "==> Building the frozen Skulk application"
uv run pyinstaller \
    --noconfirm \
    --clean \
    --distpath "${DIST_ROOT}" \
    --workpath "${WORK_ROOT}" \
    packaging/pyinstaller/skulk.spec

APP_PATH="${DIST_ROOT}/Skulk.app"
if [[ ! -x "${APP_PATH}/Contents/MacOS/Skulk" ]]; then
    echo "Frozen application executable is missing: ${APP_PATH}/Contents/MacOS/Skulk" >&2
    exit 1
fi

echo "==> Validating bundle structure and code signature"
plutil -lint "${APP_PATH}/Contents/Info.plist"
codesign --verify --deep --strict "${APP_PATH}"

echo "==> Frozen application ready: ${APP_PATH}"
