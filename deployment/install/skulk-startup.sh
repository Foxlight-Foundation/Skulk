#!/usr/bin/env bash
# Skulk service entrypoint.
#
# Invoked by the LaunchAgent (macOS) and systemd unit (Linux). Performs
# best-effort boot-time updates, then execs skulk. Designed for the
# "middle option" failure policy:
#
#   - `git pull`, `uv sync`            -> non-fatal (warn and continue)
#   - `npm install`, `npm run build`   -> fatal only if dashboard-react/dist
#                                         is missing afterwards (no UI = no
#                                         service)
#
# Operators customize behavior by editing ~/.skulk/skulk.env. See
# deployment/install/skulk.env.example for the supported knobs.

set -u
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${SKULK_ENV_FILE:-$HOME/.skulk/skulk.env}"
PREP_LOG="$HOME/.skulk/logs/skulk.prep.log"

mkdir -p "$(dirname "$PREP_LOG")"

# Timestamped operator-facing log of what the prep phase did. Distinct
# from the captured stdout/stderr launchd writes for the skulk process
# itself so operators can audit boot-time updates separately.
log() {
    printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$PREP_LOG" >&2
}

# Source the operator env file if present. `set -a` exports every
# assignment so child processes (uv, npm, skulk) inherit them.
if [[ -f "$ENV_FILE" ]]; then
    log "sourcing env file: $ENV_FILE"
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
else
    log "no env file at $ENV_FILE — using defaults"
fi

cd "$REPO_ROOT"

AUTO_UPDATE="${SKULK_AUTO_UPDATE:-1}"
VERBOSITY="${SKULK_VERBOSITY:--v}"

run_prep() {
    # `git pull` — non-fatal. Common failure modes (offline at boot,
    # auth prompt, dirty tree) shouldn't block service start. Log the
    # exit code so an operator can spot a long-running silent failure.
    if [[ -d .git ]]; then
        log "git pull (non-fatal)"
        if ! git pull --ff-only 2>&1 | tee -a "$PREP_LOG" >&2; then
            log "warning: git pull failed (continuing with on-disk revision)"
        fi
    else
        log "not a git checkout — skipping git pull"
    fi

    # `uv sync` — non-fatal. If the lockfile is unchanged, this is a
    # no-op; if PyPI is down or wheels can't build, fall back to the
    # currently installed environment.
    log "uv sync (non-fatal)"
    if ! uv sync 2>&1 | tee -a "$PREP_LOG" >&2; then
        log "warning: uv sync failed (continuing with current venv)"
    fi

    # Dashboard build — non-fatal on success path (we boot with the
    # previously built dist/), fatal only if dist/ ends up missing.
    if [[ -d dashboard-react ]]; then
        log "npm install + build (non-fatal unless dist/ is missing)"
        (
            cd dashboard-react
            npm install 2>&1 | tee -a "$PREP_LOG" >&2 || \
                log "warning: npm install failed"
            npm run build 2>&1 | tee -a "$PREP_LOG" >&2 || \
                log "warning: npm run build failed"
        )
    fi

    if [[ ! -d dashboard-react/dist ]]; then
        log "ERROR: dashboard-react/dist is missing — cannot start without a built dashboard."
        log "fix: run 'cd dashboard-react && npm install && npm run build' manually, then restart the service."
        exit 1
    fi
}

if [[ "$AUTO_UPDATE" == "1" ]]; then
    run_prep
else
    log "SKULK_AUTO_UPDATE=$AUTO_UPDATE — skipping boot-time update"
    if [[ ! -d dashboard-react/dist ]]; then
        log "ERROR: dashboard-react/dist is missing and auto-update is off."
        log "fix: build the dashboard once, or set SKULK_AUTO_UPDATE=1 in $ENV_FILE."
        exit 1
    fi
fi

log "exec: uv run skulk ${VERBOSITY}"

# `exec` so launchd / systemd track the skulk process directly rather
# than this wrapper. Quoted ${VERBOSITY} preserves empty-string semantics.
if [[ -n "$VERBOSITY" ]]; then
    exec uv run skulk "$VERBOSITY"
else
    exec uv run skulk
fi
