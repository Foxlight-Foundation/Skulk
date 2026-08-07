#!/usr/bin/env bash
# Vector log-shipper entrypoint.
#
# Runs as a separate LaunchAgent / systemd unit from skulk so a Vector
# crash, slow downstream sink, or config error can't backpressure or
# kill the inference process. Vector tails Skulk's stdout log file
# (see deployment/logging/vector.yaml) and ships JSON to VictoriaLogs.
#
# Operators customize behavior by editing ~/.skulk/skulk.env (the same
# file Skulk uses; SKULK_LOGGING_INGEST_URL and SKULK_VECTOR_DATA_DIR are
# the relevant knobs).

set -u
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${SKULK_ENV_FILE:-$HOME/.skulk/skulk.env}"
CONFIG="$REPO_ROOT/deployment/logging/vector-external.yaml"

if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

# Vector does not recursively expand environment-variable defaults. Leaving
# these unset makes `${VAR:-${HOME}/...}` resolve to a literal `${HOME}` path,
# and launchd's repository working directory then becomes a dirty checkout.
# Resolve the user-relative defaults in the shell before Vector reads YAML.
SKULK_VECTOR_DATA_DIR="${SKULK_VECTOR_DATA_DIR:-$HOME/.skulk/vector}"
SKULK_LOG_FILE="${SKULK_LOG_FILE:-$HOME/.skulk/logs/skulk.stdout.log}"
export SKULK_VECTOR_DATA_DIR SKULK_LOG_FILE

if ! command -v vector >/dev/null 2>&1; then
    echo "error: 'vector' not found on PATH. Install Vector (https://vector.dev/docs/setup/installation/) and re-run." >&2
    exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "error: Vector config not found at $CONFIG" >&2
    exit 1
fi

exec vector --config "$CONFIG"
