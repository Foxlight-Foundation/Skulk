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

LOG_DIR="$HOME/.skulk/logs"
# Tail of the previous run kept across a restart for crash diagnosis before the
# captured file is truncated. The authoritative, size-rotated record is
# ~/.skulk/logs/skulk.log; these launchd/systemd capture files are a boot- and
# crash-time safety net, not the durable log.
CAPTURE_KEEP_BYTES="${SKULK_CAPTURE_KEEP_BYTES:-5242880}"  # 5 MB

# Bound the launchd/systemd-captured stdout/stderr so they cannot accumulate
# across restarts (#382). launchd holds these fds open for this process, so the
# file must be truncated in place (same inode) rather than renamed: a renamed
# file would still receive this run's output. We snapshot the tail to ".1"
# first so the previous run's final output survives one restart.
rotate_capture() {
    local f="$1"
    [[ -s "$f" ]] || return 0
    tail -c "$CAPTURE_KEEP_BYTES" "$f" > "${f}.1" 2>/dev/null || true
    : > "$f"
}
rotate_capture "$LOG_DIR/skulk.stdout.log"
rotate_capture "$LOG_DIR/skulk.stderr.log"

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

# Augment PATH with common user-space tool locations so uv, git, and npm
# are findable when the script is invoked from systemd — which starts with
# a minimal PATH that excludes ~/.local/bin, ~/.cargo/bin, and Homebrew.
# On macOS the launchd agent injects __USER_PATH__ at install time, so
# those directories are already present and this loop is a no-op for them.
for _d in "$HOME/.local/bin" "$HOME/.cargo/bin" /opt/homebrew/bin /usr/local/bin; do
    [[ -d "$_d" && ":$PATH:" != *":$_d:"* ]] && PATH="$_d:$PATH"
done
export PATH
unset _d

AUTO_UPDATE="${SKULK_AUTO_UPDATE:-1}"
# Default to INFO. DEBUG (-v) is opt-in via SKULK_VERBOSITY=-v because at DEBUG
# the libp2p transport logs a per-dial firehose that grew skulk.stderr.log to
# tens of GB on long-lived nodes (#382). The durable, size-rotated record lives
# in ~/.skulk/logs/skulk.log regardless of this setting.
VERBOSITY="${SKULK_VERBOSITY:-}"
# Headless nodes serve the API without the web UI (no dashboard build).
HEADLESS="${SKULK_HEADLESS:-0}"

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
    #
    # On a node that declares a GPU llama.cpp backend, the GPU wheel is built
    # from source out-of-band (CMAKE_ARGS=...; see deployment/rocm) and is NOT in
    # uv's locked resolution (llama-cpp-python is an optional extra). A plain
    # `uv sync` would PRUNE that wheel as extraneous, dropping the node to
    # CPU-only (or off the llama.cpp roster entirely) until a manual rebuild.
    # `--inexact` tells uv to leave packages outside the resolution in place, so
    # the source-built GPU wheel survives the sync. Macs / CPU nodes keep an
    # exact sync. (SC2086: SYNC_FLAGS is a controlled "--inexact" or empty.)
    # Strip spaces first: probe_node_backends accepts "vulkan, rocm" (it strips
    # each token), so the comma-pattern match below must too or a GPU token with a
    # leading space (e.g. "cpu, vulkan") would miss and the wheel get pruned.
    SYNC_FLAGS=""
    DECLARED_BACKENDS="${SKULK_LLAMA_CPP_BACKENDS:-}"
    DECLARED_BACKENDS="${DECLARED_BACKENDS// /}"
    case ",${DECLARED_BACKENDS}," in
    *,vulkan,* | *,rocm,* | *,cuda,*)
        SYNC_FLAGS="--inexact"
        log "GPU llama.cpp node: 'uv sync --inexact' to preserve the source-built wheel"
        ;;
    esac
    log "uv sync (non-fatal)"
    # shellcheck disable=SC2086
    if ! uv sync $SYNC_FLAGS 2>&1 | tee -a "$PREP_LOG" >&2; then
        log "warning: uv sync failed (continuing with current venv)"
    fi

    # Headless nodes (e.g. a non-Mac worker like a Strix Halo / ROCm box)
    # intentionally serve the API without the web UI: the node sets
    # DASHBOARD_DIR=None and skips the mount when assets are absent (#333).
    # For those, skip the dashboard build and its fatal dist/ check entirely
    # so the service can run without Node/npm installed.
    if [[ "$HEADLESS" == "1" ]]; then
        log "SKULK_HEADLESS=1: skipping dashboard build; API serves without the web UI"
        return
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
        log "     (headless nodes that serve the API without the UI: set SKULK_HEADLESS=1 in $ENV_FILE)"
        exit 1
    fi
}

if [[ "$AUTO_UPDATE" == "1" ]]; then
    run_prep
else
    log "SKULK_AUTO_UPDATE=$AUTO_UPDATE — skipping boot-time update"
    if [[ "$HEADLESS" != "1" && ! -d dashboard-react/dist ]]; then
        log "ERROR: dashboard-react/dist is missing and auto-update is off."
        log "fix: build the dashboard once, or set SKULK_AUTO_UPDATE=1 in $ENV_FILE."
        log "     (headless nodes that serve the API without the UI: set SKULK_HEADLESS=1 in $ENV_FILE)"
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
