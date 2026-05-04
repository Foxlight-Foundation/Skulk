#!/usr/bin/env bash
# Install the Skulk LaunchAgent(s) on macOS.
#
# Two agents get installed by default:
#   1. foundation.foxlight.skulk         — runs skulk via the wrapper
#   2. foundation.foxlight.skulk-vector  — runs vector log shipper
#
# LaunchAgents (not LaunchDaemons) are used because:
#   - Metal access requires a graphical user session
#   - LaunchAgents inherit the user's shell PATH via login-shell wrapper
#   - the daemon variant would need root + a separate user-context shim
#
# Flags:
#   --no-vector    skip installing the Vector log-shipper agent
#                  (use this if you don't run centralized logging)
#   --uninstall    remove both agents and exit

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAUNCHD_DIR="$REPO_ROOT/deployment/launchd"
SKULK_TEMPLATE="$LAUNCHD_DIR/foundation.foxlight.skulk.plist"
VECTOR_TEMPLATE="$LAUNCHD_DIR/foundation.foxlight.skulk-vector.plist"
ENV_TEMPLATE="$REPO_ROOT/deployment/install/skulk.env.example"

TARGET_DIR="$HOME/Library/LaunchAgents"
SKULK_LABEL="foundation.foxlight.skulk"
VECTOR_LABEL="foundation.foxlight.skulk-vector"
SKULK_TARGET="$TARGET_DIR/$SKULK_LABEL.plist"
VECTOR_TARGET="$TARGET_DIR/$VECTOR_LABEL.plist"

ENV_TARGET_DIR="$HOME/.skulk"
ENV_TARGET="$ENV_TARGET_DIR/skulk.env"
LOG_DIR="$ENV_TARGET_DIR/logs"

INSTALL_VECTOR=1
ACTION=install

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-vector) INSTALL_VECTOR=0; shift ;;
        --uninstall) ACTION=uninstall; shift ;;
        -h|--help)
            sed -n '2,20p' "$0"
            exit 0
            ;;
        *)
            echo "error: unknown flag '$1'" >&2
            exit 1
            ;;
    esac
done

if [[ "$OSTYPE" != darwin* ]]; then
    echo "error: install-launchd.sh is for macOS. On Linux use install-systemd.sh." >&2
    exit 1
fi

uid="$(id -u)"

bootout_if_loaded() {
    local label="$1" target="$2"
    if launchctl print "gui/$uid/$label" >/dev/null 2>&1; then
        launchctl bootout "gui/$uid" "$target" 2>/dev/null || true
    fi
}

if [[ "$ACTION" == "uninstall" ]]; then
    bootout_if_loaded "$SKULK_LABEL" "$SKULK_TARGET"
    bootout_if_loaded "$VECTOR_LABEL" "$VECTOR_TARGET"
    rm -f "$SKULK_TARGET" "$VECTOR_TARGET"
    echo "Removed Skulk LaunchAgent(s). Your config and models under ~/.skulk are untouched."
    exit 0
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "error: 'uv' not found on PATH. Install uv first (https://docs.astral.sh/uv/) and re-run." >&2
    exit 1
fi

if [[ "$INSTALL_VECTOR" == "1" ]] && ! command -v vector >/dev/null 2>&1; then
    echo "warning: 'vector' not found on PATH." >&2
    echo "         The Vector agent will be installed but will fail to start until you" >&2
    echo "         install Vector (https://vector.dev/docs/setup/installation/)." >&2
    echo "         Re-run with --no-vector if you don't want centralized logging." >&2
fi

# Login-shell PATH so the agent gets the same `uv` / `npm` resolution
# the operator uses interactively.
USER_PATH="$(/bin/bash -lc 'echo -n $PATH')"
mkdir -p "$LOG_DIR" "$ENV_TARGET_DIR" "$TARGET_DIR"

# First-install env file. Never overwrite the operator's edits on a
# re-run; they can diff against the template if they want new defaults.
if [[ ! -f "$ENV_TARGET" ]]; then
    cp "$ENV_TEMPLATE" "$ENV_TARGET"
    echo "Created env file: $ENV_TARGET"
    echo "  (edit this to customize cluster namespace, debug flags, ingest URL, etc.)"
else
    echo "Env file already exists: $ENV_TARGET (left untouched)"
    echo "  (compare with $ENV_TEMPLATE if you want to pick up new defaults)"
fi

render_plist() {
    local template="$1" target="$2"
    sed -e "s|__SKULK_REPO__|$REPO_ROOT|g" \
        -e "s|__SKULK_LOG_DIR__|$LOG_DIR|g" \
        -e "s|__USER_PATH__|$USER_PATH|g" \
        "$template" > "$target"
}

reload_agent() {
    local label="$1" target="$2"
    bootout_if_loaded "$label" "$target"
    launchctl bootstrap "gui/$uid" "$target"
    launchctl enable "gui/$uid/$label"
    launchctl kickstart -k "gui/$uid/$label"
}

render_plist "$SKULK_TEMPLATE" "$SKULK_TARGET"
echo "Installed agent: $SKULK_TARGET"
reload_agent "$SKULK_LABEL" "$SKULK_TARGET"

if [[ "$INSTALL_VECTOR" == "1" ]]; then
    render_plist "$VECTOR_TEMPLATE" "$VECTOR_TARGET"
    echo "Installed agent: $VECTOR_TARGET"
    reload_agent "$VECTOR_LABEL" "$VECTOR_TARGET"
else
    # Clean up an existing vector agent if --no-vector is passed on a
    # second run; otherwise an old one would silently keep running.
    if [[ -f "$VECTOR_TARGET" ]]; then
        bootout_if_loaded "$VECTOR_LABEL" "$VECTOR_TARGET"
        rm -f "$VECTOR_TARGET"
        echo "Removed previously installed Vector agent (--no-vector specified)."
    fi
fi

echo
echo "Skulk LaunchAgent is installed and running."
echo "  status:    launchctl print gui/$uid/$SKULK_LABEL | grep 'state ='"
echo "  logs:      tail -f $LOG_DIR/skulk.stderr.log"
echo "  prep log:  tail -f $LOG_DIR/skulk.prep.log"
echo "  env file:  $ENV_TARGET"
echo "  restart:   launchctl kickstart -k gui/$uid/$SKULK_LABEL"
echo "  stop:      launchctl bootout gui/$uid $SKULK_TARGET"
echo "  uninstall: $0 --uninstall"
if [[ "$INSTALL_VECTOR" == "1" ]]; then
    echo
    echo "Vector log-shipper agent:"
    echo "  status: launchctl print gui/$uid/$VECTOR_LABEL | grep 'state ='"
    echo "  logs:   tail -f $LOG_DIR/vector.stderr.log"
fi
