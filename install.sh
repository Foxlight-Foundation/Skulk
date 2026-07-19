#!/usr/bin/env bash
# Skulk one-command installer (#614 Phase 4).
#
# From a fresh macOS or Linux machine to a working node:
#
#   curl -fsSL https://raw.githubusercontent.com/Foxlight-Foundation/Skulk/main/install.sh | bash
#
# The installer is deliberately thin: it fetches prerequisites (uv, rustup, a
# C toolchain), clones the repo, syncs the environment, builds the dashboard
# when npm is available, and then hands off to `skulk doctor --fix`, which
# owns all environment intelligence (GPU detection, engine provisioning,
# remediation). Anything the doctor cannot fix is printed with its exact
# consequence and remediation.
#
# Flags / environment:
#   --dir <path>       install location            (default: ~/skulk, or SKULK_INSTALL_DIR)
#   --ref <git-ref>    branch or tag to install    (default: main, or SKULK_INSTALL_REF)
#   --headless         skip the dashboard build even if npm is present
#   --with-vllm        NVIDIA Linux only: create a dedicated vLLM venv with
#                      Skulk's validated dependency matrix (several GB of
#                      wheels; the concurrency-serving fast path on CUDA)
#
# Re-running is safe: every step is idempotent.

set -euo pipefail

INSTALL_DIR="${SKULK_INSTALL_DIR:-$HOME/skulk}"
INSTALL_REF="${SKULK_INSTALL_REF:-main}"
HEADLESS=0
WITH_VLLM=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dir) INSTALL_DIR="$2"; shift 2 ;;
        --ref) INSTALL_REF="$2"; shift 2 ;;
        --headless) HEADLESS=1; shift ;;
        --with-vllm) WITH_VLLM=1; shift ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

log()  { printf '\033[1;36m[skulk-install]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[skulk-install]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[skulk-install]\033[0m %s\n' "$*" >&2; exit 1; }

OS="$(uname -s)"
case "$OS" in
    Darwin|Linux) ;;
    *) die "unsupported platform: $OS (Skulk supports macOS and Linux)" ;;
esac

# --- prerequisites ---------------------------------------------------------

need_apt() {
    # Return 0 when we can install system packages non-interactively.
    [[ "$OS" == "Linux" ]] && command -v apt-get >/dev/null 2>&1 \
        && { [[ "$(id -u)" == "0" ]] || command -v sudo >/dev/null 2>&1; }
}

apt_install() {
    if [[ "$(id -u)" == "0" ]]; then
        apt-get update -qq && apt-get install -y -qq "$@"
    else
        sudo apt-get update -qq && sudo apt-get install -y -qq "$@"
    fi
}

if ! command -v git >/dev/null 2>&1; then
    if need_apt; then
        log "installing git"
        apt_install git
    else
        die "git is required; install it and re-run (macOS: xcode-select --install)"
    fi
fi

if ! command -v curl >/dev/null 2>&1; then
    if need_apt; then
        log "installing curl"
        apt_install curl ca-certificates
    else
        die "curl is required; install it and re-run"
    fi
fi

if ! command -v cc >/dev/null 2>&1; then
    # The Rust networking bindings compile from source; that needs a linker.
    if need_apt; then
        log "installing C toolchain (build-essential) for the Rust bindings"
        apt_install build-essential
    elif [[ "$OS" == "Darwin" ]]; then
        die "no C compiler found; run: xcode-select --install, then re-run"
    else
        die "no C compiler found; install your distro's build tools and re-run"
    fi
fi

# A GPU Linux node serves GGUF through the managed Vulkan llama-server build,
# which needs the Vulkan loader; minimal CUDA/ROCm container images often lack
# it while the driver's ICD is present.
if [[ "$OS" == "Linux" ]] && command -v ldconfig >/dev/null 2>&1 \
    && ! ldconfig -p 2>/dev/null | grep -q libvulkan.so.1; then
    if { command -v nvidia-smi >/dev/null 2>&1 || [[ -d /sys/class/drm ]]; } && need_apt; then
        log "installing Vulkan loader (libvulkan1) for GPU serving"
        apt_install libvulkan1 || warn "libvulkan1 install failed; GPU GGUF serving may be unavailable (skulk doctor will report it)"
    fi
fi

if ! command -v cargo >/dev/null 2>&1 && [[ ! -x "$HOME/.cargo/bin/cargo" ]]; then
    log "installing Rust (rustup) for the skulk networking bindings"
    curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile minimal
fi
# The rustup installer puts cargo here; make it visible to uv's build backend.
export PATH="$HOME/.cargo/bin:$PATH"

if ! command -v uv >/dev/null 2>&1 && [[ ! -x "$HOME/.local/bin/uv" ]]; then
    log "installing uv"
    curl -fsSL https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || die "uv installation failed; see https://docs.astral.sh/uv/"

# --- fetch -----------------------------------------------------------------

if [[ -d "$INSTALL_DIR/.git" ]]; then
    log "updating existing checkout at $INSTALL_DIR (ref: $INSTALL_REF)"
    git -C "$INSTALL_DIR" fetch origin "$INSTALL_REF"
    # A tag or remote-only ref may not be checkout-able by name after a bare
    # fetch; FETCH_HEAD always is, keeping re-runs idempotent for any ref.
    git -C "$INSTALL_DIR" checkout "$INSTALL_REF" 2>/dev/null         || git -C "$INSTALL_DIR" checkout --detach FETCH_HEAD
    git -C "$INSTALL_DIR" pull --ff-only origin "$INSTALL_REF" 2>/dev/null || true
else
    log "cloning Skulk into $INSTALL_DIR (ref: $INSTALL_REF)"
    git clone --branch "$INSTALL_REF" \
        https://github.com/Foxlight-Foundation/Skulk.git "$INSTALL_DIR"
fi
cd "$INSTALL_DIR"

# --- environment -----------------------------------------------------------

# An existing-but-old uv rejects this project's configuration; upgrade it to
# the repo's declared minimum before syncing so re-runs stay idempotent on
# hosts that already had uv.
UV_MIN="$(grep -oE 'required-version = ">=([0-9.]+)"' pyproject.toml | grep -oE '[0-9.]+' | head -1 || true)"
if [[ -n "$UV_MIN" ]]; then
    UV_HAVE="$(uv --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)"
    if [[ -n "$UV_HAVE" ]] && [[ "$(printf '%s\n%s\n' "$UV_MIN" "$UV_HAVE" | sort -V | head -1)" != "$UV_MIN" ]]; then
        log "upgrading uv ($UV_HAVE -> >=$UV_MIN)"
        uv self update >/dev/null 2>&1 || curl -fsSL https://astral.sh/uv/install.sh | sh
        UV_HAVE="$(uv --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)"
        if [[ -n "$UV_HAVE" ]] && [[ "$(printf '%s\n%s\n' "$UV_MIN" "$UV_HAVE" | sort -V | head -1)" != "$UV_MIN" ]]; then
            die "uv $UV_HAVE is below the required >=$UV_MIN and could not be upgraded (package-manager installs need a manual upgrade)"
        fi
    fi
fi

log "syncing the Python environment (first run compiles the Rust bindings; this can take a few minutes)"
uv sync

# --- engine wheels (Linux GPU) ----------------------------------------------

# The wheel version derives from the CHECKED-OUT ref's engine pin, so
# installing --ref dev after a pin advance pulls the matching wheel instead
# of a hardcoded one the runtime would then ignore as a pin mismatch.
ENGINE_BUILD="$(grep -oE 'LLAMA_SERVER_PIN: Final = "b[0-9]+"' src/skulk/provisioning/manifest.py | grep -oE '[0-9]+' || true)"

# The Foxlight wheel index is the source of truth for engine wheels (the
# CUDA wheel exceeds PyPI's per-file limit); wheels carry sigstore build
# provenance (gh attestation verify <wheel> --owner Foxlight-Foundation).
FOXLIGHT_WHEEL_INDEX="https://wheels.foxlight.ai/simple/"
# UV SEMANTICS, verified live: uv consults --extra-index-url indexes BEFORE
# the --index-url default (the opposite of pip), and its default
# first-index-wins strategy is the dependency-confusion defense. So the
# Foxlight index is passed as the extra index (consulted first, wins for the
# packages it carries) while PyPI stays the default for the NVIDIA runtime
# dependencies. Making Foxlight the --index-url instead DEMOTES it under uv:
# resolution then finds the empty PyPI project first and fails.
# The default index is pinned to PyPI explicitly: a host exporting
# UV_INDEX_URL / UV_DEFAULT_INDEX would otherwise silently replace the
# fallback that supplies the NVIDIA runtime dependencies.
ENGINE_INDEX_FLAGS=(--extra-index-url "$FOXLIGHT_WHEEL_INDEX" --index-url "https://pypi.org/simple/")

if [[ "$OS" == "Linux" ]] && [[ -z "$ENGINE_BUILD" ]]; then
    warn "could not read the engine pin from the checkout; skipping engine wheel install (skulk doctor will report the outcome)"
elif [[ "$OS" == "Linux" ]] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q GPU; then
    log "installing the CUDA llama-server engine wheel (engine build b$ENGINE_BUILD)"
    if ! uv pip install "${ENGINE_INDEX_FLAGS[@]}" "skulk-llama-server-cuda==0.${ENGINE_BUILD}.*"; then
        warn "the CUDA engine wheel is unavailable (index not yet live, no network, or unsupported platform);"
        warn "trying the Vulkan engine wheel (NVIDIA GPUs run the Vulkan build on bare metal)"
        # Mirrors runtime preference: cuda wheel, then vulkan wheel, then the
        # managed tarball path that skulk itself provisions.
        uv pip install "${ENGINE_INDEX_FLAGS[@]}" "skulk-llama-server-vulkan==0.${ENGINE_BUILD}.*" \
            || warn "vulkan wheel also unavailable; falling back to the managed tarball build (skulk doctor will report the outcome)"
    fi
elif [[ "$OS" == "Linux" ]] \
    && compgen -G "/sys/class/drm/card*/device/gpu_busy_percent" > /dev/null 2>&1; then
    log "installing the Vulkan llama-server engine wheel (engine build b$ENGINE_BUILD)"
    if ! uv pip install "${ENGINE_INDEX_FLAGS[@]}" "skulk-llama-server-vulkan==0.${ENGINE_BUILD}.*"; then
        warn "skulk-llama-server-vulkan unavailable (index not yet live or no network);"
        warn "falling back to the managed tarball build; skulk doctor will report the outcome"
    fi
fi

# --- dashboard (optional) --------------------------------------------------

if [[ "$HEADLESS" == "1" ]]; then
    log "skipping dashboard build (--headless); the API serves without the web UI"
elif command -v npm >/dev/null 2>&1; then
    log "building the dashboard"
    (cd dashboard-react && npm install --no-fund --no-audit && npm run build)
else
    warn "npm not found: skipping the dashboard build. The API serves without"
    warn "the web UI; install Node.js and re-run to add it."
fi

# --- vLLM (optional, NVIDIA Linux) ----------------------------------------

if [[ "$WITH_VLLM" == "1" ]]; then
    if [[ "$OS" != "Linux" ]] || ! command -v nvidia-smi >/dev/null 2>&1 \
        || ! nvidia-smi -L 2>/dev/null | grep -q GPU; then
        warn "--with-vllm requested but no NVIDIA GPU is visible on this Linux node; skipping"
    else
        VLLM_ENV="$HOME/.skulk/vllm-env"
        log "installing vLLM into $VLLM_ENV (validated matrix; several GB)"
        uv venv "$VLLM_ENV" --python 3.12
        # Skulk's own env pins transformers>=5, which vLLM cannot use yet, so
        # vLLM lives in its own venv and Skulk drives its CLI as an external
        # served engine (SKULK_VLLM_BIN).
        uv pip install --python "$VLLM_ENV/bin/python" \
            "vllm==0.11.0" "transformers<5" --torch-backend=cu128
        mkdir -p "$HOME/.skulk"
        if ! grep -q "SKULK_VLLM_BIN" "$HOME/.skulk/skulk.env" 2>/dev/null; then
            echo "SKULK_VLLM_BIN=$VLLM_ENV/bin/vllm" >> "$HOME/.skulk/skulk.env"
        fi
        export SKULK_VLLM_BIN="$VLLM_ENV/bin/vllm"
        log "vLLM installed; SKULK_VLLM_BIN recorded in ~/.skulk/skulk.env"
        log "The service wrappers (deployment/install) source that file; an"
        log "interactive shell does not, so launch interactively with:"
        log "    SKULK_VLLM_BIN=$VLLM_ENV/bin/vllm uv run skulk"
    fi
fi

# --- doctor ----------------------------------------------------------------

log "auditing the node (skulk doctor --fix)"
set +e
uv run skulk doctor --fix
DOCTOR_EXIT=$?
set -e

echo
case "$DOCTOR_EXIT" in
    0) log "node is fully healthy." ;;
    2) warn "node works but has DEGRADED findings above; each lists its consequence and fix." ;;
    *) warn "node has FAILED findings above; serving will not work correctly until they are fixed." ;;
esac

log "install complete. Start a node with:"
log "    cd $INSTALL_DIR && uv run skulk"
log "Dashboard (when built): http://localhost:52415"
log "Run as a service: deployment/install/install-systemd.sh (Linux) or install-launchd.sh (macOS)"

# DEGRADED (exit 2) still means a working node; only FAIL blocks the install.
if [[ "$DOCTOR_EXIT" != "0" && "$DOCTOR_EXIT" != "2" ]]; then
    exit 1
fi
exit 0
