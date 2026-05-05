#!/usr/bin/env bash
# Install the Skulk user-level systemd service.
#
# User-level (not system-level) is the default because:
#   - it matches the macOS LaunchAgent model
#   - GPU/Metal-equivalent compute typically requires a logged-in user session
#   - linger (`loginctl enable-linger`) gives autostart on boot for headless
#     machines without the broader blast radius of a system unit
#
# For true server-style deployments where Skulk should run regardless of any
# user session, see website/docs/deployment/headless-resilience.md for the
# system-unit variant.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEMPLATE="$REPO_ROOT/deployment/systemd/skulk.service"
ENV_TEMPLATE="$REPO_ROOT/deployment/install/skulk.env.example"
TARGET_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
TARGET="$TARGET_DIR/skulk.service"

ENV_TARGET_DIR="$HOME/.skulk"
ENV_TARGET="$ENV_TARGET_DIR/skulk.env"

if [[ "$OSTYPE" != linux-gnu* ]]; then
    echo "error: install-systemd.sh is for Linux. On macOS use install-launchd.sh." >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "error: 'uv' not found on PATH. Install uv first (https://docs.astral.sh/uv/) and re-run." >&2
    exit 1
fi

if ! command -v systemctl >/dev/null 2>&1; then
    echo "error: 'systemctl' not found. This installer requires systemd." >&2
    exit 1
fi

mkdir -p "$TARGET_DIR" "$ENV_TARGET_DIR"

# Substitute the repo path placeholder. We use sed with a non-/ delimiter so
# paths containing / don't need escaping.
sed "s|__SKULK_REPO__|$REPO_ROOT|g" "$TEMPLATE" > "$TARGET"

echo "Installed unit: $TARGET"

# First-install env file. Linux has no separate Vector LaunchAgent in
# this release, so default to SKULK_LOGGING_EXTERNAL=0 (in-process Vector
# subprocess). Operators who run an external shipper can flip it to 1.
# Re-runs never overwrite operator edits.
if [[ ! -f "$ENV_TARGET" ]]; then
    cp "$ENV_TEMPLATE" "$ENV_TARGET"
    sed -i 's/^SKULK_LOGGING_EXTERNAL=1$/SKULK_LOGGING_EXTERNAL=0/' "$ENV_TARGET"
    echo "Created env file: $ENV_TARGET (SKULK_LOGGING_EXTERNAL=0 — using in-process Vector subprocess)"
    echo "  (edit this to customize cluster namespace, debug flags, ingest URL, etc.)"
else
    echo "Env file already exists: $ENV_TARGET (left untouched)"
fi

systemctl --user daemon-reload

# Enable linger so the user manager keeps running across logout/reboot. Without
# this, user services stop when the user logs out — defeating the headless use
# case. Already-enabled lingering is a no-op.
if command -v loginctl >/dev/null 2>&1; then
    if ! loginctl show-user "$USER" 2>/dev/null | grep -q "Linger=yes"; then
        echo "Enabling user lingering so Skulk runs without an active login session..."
        if ! loginctl enable-linger "$USER" 2>/dev/null; then
            echo "warn: failed to enable linger (may need sudo). Run manually:" >&2
            echo "       sudo loginctl enable-linger $USER" >&2
        fi
    fi
fi

systemctl --user enable --now skulk.service

echo
echo "Skulk service is enabled and running."
echo "  status:    systemctl --user status skulk"
echo "  logs:      journalctl --user -u skulk -f"
echo "  prep log:  tail -f $HOME/.skulk/logs/skulk.prep.log"
echo "  env file:  $ENV_TARGET"
echo "  restart:   systemctl --user restart skulk"
echo "  stop:      systemctl --user stop skulk"
echo "  remove:    systemctl --user disable --now skulk && rm $TARGET"
