#!/usr/bin/env bash
# Install the Skulk LaunchAgent on macOS.
#
# LaunchAgents (not LaunchDaemons) are used because:
#   - Metal access requires a graphical user session
#   - LaunchAgents inherit the user's shell PATH via login-shell wrapper
#   - the daemon variant would need root + a separate user-context shim

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEMPLATE="$REPO_ROOT/deployment/launchd/foundation.foxlight.skulk.plist"
TARGET_DIR="$HOME/Library/LaunchAgents"
TARGET="$TARGET_DIR/foundation.foxlight.skulk.plist"
LABEL="foundation.foxlight.skulk"

if [[ "$OSTYPE" != darwin* ]]; then
    echo "error: install-launchd.sh is for macOS. On Linux use install-systemd.sh." >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "error: 'uv' not found on PATH. Install uv first (https://docs.astral.sh/uv/) and re-run." >&2
    exit 1
fi

# Capture the user's PATH from a login shell so the same `uv` resolution
# Skulk uses interactively is what the agent gets.
USER_PATH="$(/bin/bash -lc 'echo -n $PATH')"
# On macOS, _get_xdg_dir() in src/exo/shared/constants.py returns ~/.skulk
# for all XDG dirs (sys.platform != "linux" branch), so SKULK_LOG_DIR is
# always ~/.skulk/logs on macOS regardless of XDG_CACHE_HOME.
LOG_DIR="$HOME/.skulk/logs"
mkdir -p "$LOG_DIR"
mkdir -p "$TARGET_DIR"

sed -e "s|__SKULK_REPO__|$REPO_ROOT|g" \
    -e "s|__SKULK_LOG_DIR__|$LOG_DIR|g" \
    -e "s|__USER_PATH__|$USER_PATH|g" \
    "$TEMPLATE" > "$TARGET"

echo "Installed agent: $TARGET"

# bootstrap is the modern equivalent of `load`. We unload first so a re-run
# of this installer cleanly replaces an existing agent.
if launchctl print "gui/$(id -u)/$LABEL" >/dev/null 2>&1; then
    launchctl bootout "gui/$(id -u)" "$TARGET" 2>/dev/null || true
fi
launchctl bootstrap "gui/$(id -u)" "$TARGET"
launchctl enable "gui/$(id -u)/$LABEL"
launchctl kickstart -k "gui/$(id -u)/$LABEL"

echo
echo "Skulk LaunchAgent is installed and running."
echo "  status: launchctl print gui/$(id -u)/$LABEL"
echo "  logs:   tail -f $LOG_DIR/skulk.stderr.log"
echo "  stop:   launchctl bootout gui/$(id -u) $TARGET"
echo "  remove: launchctl bootout gui/$(id -u) $TARGET && rm $TARGET"
