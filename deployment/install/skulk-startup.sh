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
# Explicitly headless nodes serve the API without the web UI. A normal Linux
# install is not implicitly headless: Skulk's bundled Node.js runtime builds
# the dashboard even when the host has no npm.
HEADLESS="${SKULK_HEADLESS:-0}"

run_bundled_npm() {
    uv run --project "$REPO_ROOT" python \
        "$REPO_ROOT/scripts/run_bundled_npm.py" "$@"
}

run_prep() {
    # `git pull` — non-fatal. Common failure modes (offline at boot,
    # auth prompt, dirty tree) shouldn't block service start. Log the
    # exit code so an operator can spot a long-running silent failure.
    if [[ -d .git ]]; then
        log "git pull (non-fatal)"
        PRE_PULL_HEAD="$(git rev-parse HEAD 2>/dev/null || true)"
        if ! git pull --ff-only 2>&1 | tee -a "$PREP_LOG" >&2; then
            log "warning: git pull failed (continuing with on-disk revision)"
        fi
        # If the pull updated THIS script, the running shell still executes
        # the old body it already read, so new prep steps (e.g. the bindings
        # rebuild below) would not run until a second restart. Re-exec the
        # freshly pulled script once so prep changes take effect on the same
        # restart that delivered them. Single-shot via the env guard: the
        # re-exec'd script pulls again (a no-op) and proceeds normally.
        POST_PULL_HEAD="$(git rev-parse HEAD 2>/dev/null || true)"
        if [[ -z "${SKULK_STARTUP_REEXECED:-}" \
            && -n "$PRE_PULL_HEAD" && -n "$POST_PULL_HEAD" \
            && "$PRE_PULL_HEAD" != "$POST_PULL_HEAD" ]] \
            && ! git diff --quiet "$PRE_PULL_HEAD" "$POST_PULL_HEAD" \
                -- deployment/install/skulk-startup.sh 2>/dev/null; then
            log "startup script changed by git pull; re-executing the updated script"
            export SKULK_STARTUP_REEXECED=1
            exec bash "$REPO_ROOT/deployment/install/skulk-startup.sh"
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

    # Rust bindings rebuild (#659). `uv sync` reuses the cached
    # skulk_pyo3_bindings wheel unless the PROJECT VERSION changes, so a git
    # pull that changes rust/ source leaves the venv running the OLD wire
    # code indefinitely: the live fleet ran pre-telemetry-isolation bindings
    # for eight days while reporting itself up to date, and a fresh build
    # joining it became a fully-synced node invisible to membership. Track
    # the last commit that touched rust/ and force a bindings reinstall when
    # it moves. Non-fatal: a failed rebuild keeps the current (stale)
    # bindings and the node still starts; NETWORK_VERSION bumps make a truly
    # wire-incompatible stale build refuse connections loudly.
    # Root Cargo.toml/Cargo.lock are workspace inputs: a dependency bump
    # changes the built bindings without touching rust/ source.
    RUST_TREE_COMMIT="$(git log -1 --format=%H -- rust/ Cargo.toml Cargo.lock 2>/dev/null || true)"
    RUST_TREE_MARKER=".venv/.skulk-rust-tree-commit"
    # The marker lives inside the venv (the artifact it describes); no venv
    # yet (first boot, or a fully failed sync) means nothing to mark and the
    # reinstall below would fail the same way the sync just did.
    if [ -n "$RUST_TREE_COMMIT" ] && [ -d .venv ]; then
        if [ "$(cat "$RUST_TREE_MARKER" 2>/dev/null || true)" != "$RUST_TREE_COMMIT" ]; then
            log "rust/ tree moved to ${RUST_TREE_COMMIT}; rebuilding skulk_pyo3_bindings (non-fatal)"
            # shellcheck disable=SC2086
            if uv sync $SYNC_FLAGS --reinstall-package skulk_pyo3_bindings 2>&1 | tee -a "$PREP_LOG" >&2; then
                printf '%s\n' "$RUST_TREE_COMMIT" > "$RUST_TREE_MARKER"
            else
                log "warning: bindings rebuild failed (continuing with current bindings)"
            fi
        else
            log "rust bindings current (rust/ tree at ${RUST_TREE_COMMIT})"
        fi
    fi

    # GPU llama.cpp wheel self-heal (#568). --inexact above PRESERVES a
    # present source-built wheel, but cannot RESTORE one that was already
    # pruned (a plain `uv sync` run by hand or by another tool drops it). A
    # GPU node that declares a llama.cpp backend but cannot import llama_cpp
    # with GPU offload would otherwise come up silently degraded -- advertising
    # no llama.cpp backend and dropping out of all GGUF/served-MTP placement
    # with no error. So on such a node, verify the wheel and rebuild it from
    # source once if it is missing or CPU-only. Non-fatal and single-shot: the
    # node still serves (just without the in-process GGUF engine) if the
    # rebuild fails, and the common case (wheel present) skips the rebuild.
    if [ -n "$SYNC_FLAGS" ]; then
        WHEEL_PROBE='import sys; import llama_cpp; sys.exit(0 if llama_cpp.llama_supports_gpu_offload() else 3)'
        if uv run --no-sync python -c "$WHEEL_PROBE" >/dev/null 2>&1; then
            log "GPU llama.cpp wheel: present with GPU offload"
        else
            # Match the backend to the canonical install scripts: NVIDIA builds
            # with CUDA, AMD Strix (vulkan/rocm) builds with Vulkan/RADV.
            case ",${DECLARED_BACKENDS}," in
            *,cuda,*) WHEEL_CMAKE="-DGGML_CUDA=ON" ;;
            *)        WHEEL_CMAKE="-DGGML_VULKAN=on" ;;
            esac
            # Pin to the version in uv.lock so a self-healed node cannot drift
            # onto whatever llama-cpp-python PyPI serves at boot; a mixed
            # binding version across the fleet is the anti-pattern (#568 review).
            LOCKED_LLAMA_CPP="$(sed -n '/^name = "llama-cpp-python"$/{n;s/^version = "\(.*\)"$/\1/p;}' uv.lock | head -1)"
            if [ -n "$LOCKED_LLAMA_CPP" ]; then
                WHEEL_SPEC="llama-cpp-python==${LOCKED_LLAMA_CPP}"
            else
                # uv.lock parse failed (format change): rebuild unpinned rather
                # than skip the self-heal, so the node still regains GGUF.
                WHEEL_SPEC="llama-cpp-python"
                log "warning: could not read llama-cpp-python version from uv.lock; rebuilding unpinned"
            fi
            log "GPU llama.cpp wheel MISSING or CPU-only; rebuilding ${WHEEL_SPEC} from source with CMAKE_ARGS=${WHEEL_CMAKE} (self-heal, #568)"
            # llama-cpp-python's own dependencies (e.g. diskcache) live only in
            # the llama-cpp optional extra, so a plain `uv sync` prunes them
            # alongside the wheel. Restore just those deps at their LOCKED
            # versions first (uv export of the extra, minus the package itself),
            # so the rebuilt wheel can import; then the --no-deps source build
            # swaps in the GPU wheel without letting pip re-resolve anything off
            # uv.lock (#569 review: --no-deps alone left llama_cpp
            # importable-but-broken when diskcache was also pruned).
            LLAMA_DEPS_REQ="$(mktemp)"
            if uv export --frozen --extra llama-cpp --no-hashes \
                --no-emit-project --no-emit-workspace \
                --no-emit-package llama-cpp-python -o "$LLAMA_DEPS_REQ" 2>>"$PREP_LOG"; then
                uv pip install --python .venv/bin/python -r "$LLAMA_DEPS_REQ" 2>&1 \
                    | tee -a "$PREP_LOG" >&2 \
                    || log "warning: could not restore llama-cpp extra deps; wheel may still fail to import"
            else
                log "warning: uv export of llama-cpp deps failed; rebuilding wheel without restoring its deps"
            fi
            rm -f "$LLAMA_DEPS_REQ"
            # --no-deps: the extra's deps are handled above at locked versions;
            # this step only swaps the wheel artifact and must not re-resolve.
            if CMAKE_ARGS="$WHEEL_CMAKE" uv pip install --force-reinstall \
                --no-deps --no-cache-dir --no-binary llama-cpp-python \
                --python .venv/bin/python "$WHEEL_SPEC" 2>&1 \
                | tee -a "$PREP_LOG" >&2; then
                if uv run --no-sync python -c "$WHEEL_PROBE" >/dev/null 2>&1; then
                    log "GPU llama.cpp wheel: rebuilt, GPU offload OK"
                else
                    log "warning: llama.cpp wheel rebuilt but still lacks GPU offload; node runs without the in-process GGUF engine"
                fi
            else
                log "warning: llama.cpp wheel rebuild failed; node runs without the in-process GGUF engine"
            fi
        fi
    fi

    # Explicitly headless nodes intentionally serve the API without the web UI:
    # the node sets DASHBOARD_DIR=None and skips the mount when assets are absent
    # (#333). A normal Linux install still builds the UI through the bundled
    # runtime below, so the absence of a host npm executable is not headlessness.
    if [[ "$HEADLESS" == "1" ]]; then
        log "SKULK_HEADLESS=1: skipping dashboard build; API serves without the web UI"
        return
    fi

    # Dashboard build — non-fatal on success path (we boot with the previously
    # built dist/), fatal only if dist/ ends up missing. Prefer the same pinned
    # bundled runtime as install.sh so a supervised Linux node without system
    # Node/npm can refresh its UI after every git update. The system fallback is
    # retained for recovery when the bundled dependency itself cannot launch.
    if [[ -d dashboard-react ]]; then
        log "dashboard install + build (non-fatal unless dist/ is missing)"
        (
            cd dashboard-react
            if run_bundled_npm --version 2>&1 | tee -a "$PREP_LOG" >&2; then
                log "using Skulk's bundled Node.js runtime"
                run_bundled_npm install --no-fund --no-audit 2>&1 \
                    | tee -a "$PREP_LOG" >&2 \
                    || log "warning: bundled npm install failed"
                run_bundled_npm run build 2>&1 | tee -a "$PREP_LOG" >&2 \
                    || log "warning: bundled npm run build failed"
            elif command -v node >/dev/null 2>&1 \
                && command -v npm >/dev/null 2>&1; then
                log "warning: bundled Node.js runtime unavailable; using system npm"
                npm install --no-fund --no-audit 2>&1 | tee -a "$PREP_LOG" >&2 \
                    || log "warning: system npm install failed"
                npm run build 2>&1 | tee -a "$PREP_LOG" >&2 \
                    || log "warning: system npm run build failed"
            else
                log "warning: bundled Node.js runtime unavailable and system npm is absent"
            fi
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
