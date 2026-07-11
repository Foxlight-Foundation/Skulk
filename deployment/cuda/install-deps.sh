#!/usr/bin/env bash
# Copyright 2026 Foxlight Foundation
#
# One-shot dependency installer for a Skulk NVIDIA / CUDA worker node,
# designed for rented GPU pods (RunPod-style CUDA base images) where the
# NVIDIA driver and CUDA toolkit are ALREADY present, as well as bare Ubuntu
# boxes with the vendor driver installed. It installs everything the node
# needs BELOW the Skulk repo: the build toolchain, `uv`, the NVML telemetry
# binding, the CUDA `llama-cpp-python` source build (the in-process GGUF
# engine), and optionally the CUDA `llama-server` build (native MTP +
# ggml-rpc-server for multi-node GGUF pooling).
#
# The script never installs or touches the NVIDIA driver itself: on rented
# pods the image owns it, and on owned boxes the operator does. `--check`
# verifies the stack without installing anything.
#
# Idempotent: safe to re-run. It installs missing pieces, skips present ones.
#
# Usage:
#   deployment/cuda/install-deps.sh [options]
#
# Options:
#   --with-llama-server   Also clone+build llama.cpp with CUDA + RPC, producing
#                         llama-server (native MTP) and ggml-rpc-server (the
#                         multi-node GGUF pooling donor daemon). Prints the
#                         SKULK_LLAMA_SERVER_BIN path to set in the node env.
#   --with-skulk-env      If run from a Skulk checkout, also run `uv sync` and
#                         the CUDA `llama-cpp-python` source build. NOTE: any
#                         later bare `uv sync` restores the CPU-only PyPI wheel
#                         over the CUDA build (the kite lesson); re-run this
#                         script (or the build step below) after dependency
#                         changes.
#   --llama-cpp-dir DIR   Where to clone/build llama.cpp (default: $HOME/llama.cpp).
#   --check               Verify the stack only; install nothing.
#   -h | --help           Show this help.
#
# After install, the node advertises its CUDA engines via the node env:
#   SKULK_LLAMA_CPP_BACKENDS=cuda
# (backends.py cross-checks the wheel actually has GPU offload compiled in
# and falls back to llama_cpp-cpu with a warning when it does not.)
set -euo pipefail

# Pods typically run as root (no sudo installed or needed); bare boxes run as
# the login user and escalate per-command, mirroring the ROCm installer.
if [ "$(id -u)" -eq 0 ]; then SUDO=""; else SUDO="sudo"; fi

# The astral installer drops uv at ~/.local/bin, often missing from
# non-login-shell PATHs; make both --check and install see it.
export PATH="$HOME/.local/bin:$PATH"

log() { printf '\033[1;36m[deps]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[deps] WARNING:\033[0m %s\n' "$*" >&2; }
fail() { printf '\033[1;31m[deps] FAIL:\033[0m %s\n' "$*" >&2; exit 1; }

WITH_LLAMA_SERVER=0
WITH_SKULK_ENV=0
CHECK_ONLY=0
LLAMA_CPP_DIR="${HOME}/llama.cpp"

while [ $# -gt 0 ]; do
  case "$1" in
    --with-llama-server) WITH_LLAMA_SERVER=1 ;;
    --with-skulk-env) WITH_SKULK_ENV=1 ;;
    --llama-cpp-dir)
      [ $# -ge 2 ] || fail "--llama-cpp-dir requires a directory argument"
      LLAMA_CPP_DIR="$2"; shift ;;
    --check) CHECK_ONLY=1 ;;
    -h|--help) sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) fail "unknown option: $1" ;;
  esac
  shift
done

# ---- 1. Driver + CUDA toolkit presence (never installed by this script) ----
if ! command -v nvidia-smi >/dev/null 2>&1; then
  fail "nvidia-smi not found: this script expects the NVIDIA driver to exist already (rented CUDA images ship it; owned boxes install the vendor driver first)."
fi
log "driver: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"

if ! command -v nvcc >/dev/null 2>&1; then
  # Many runtime-only pod images lack nvcc; llama.cpp CUDA builds need it.
  warn "nvcc not found. Runtime-only image? CUDA builds below will fail; use a devel image (e.g. nvidia/cuda:*-devel) or install cuda-toolkit."
fi

if [ "$CHECK_ONLY" -eq 1 ]; then
  log "check mode: verifying stack"
  MISSING=0
  command -v uv >/dev/null 2>&1 && log "uv: $(uv --version)" || { warn "uv missing"; MISSING=1; }
  python3 -c "import pynvml, sys; pynvml.nvmlInit(); print('NVML OK')" 2>/dev/null \
    && log "NVML binding OK" || { warn "pynvml missing or NVML init failed"; MISSING=1; }
  # The documented install puts llama-cpp-python in the Skulk venv; check
  # there when run from a checkout, falling back to the system python.
  # Verify GPU offload is actually compiled in (the same probe backends.py
  # trusts), not merely that the module imports: a bare `uv sync` restoring
  # the CPU-only PyPI wheel is exactly the failure --check exists to catch.
  LLAMA_PROBE='import llama_cpp, sys; sys.exit(0 if llama_cpp.llama_supports_gpu_offload() else 3)'
  if [ -f "pyproject.toml" ] && command -v uv >/dev/null 2>&1; then
    # --no-sync: a plain `uv run` syncs the env first, which could ITSELF
    # restore the CPU-only wheel that this probe exists to detect.
    LLAMA_CHECK_CMD=(uv run --no-sync python -c "$LLAMA_PROBE")
  else
    LLAMA_CHECK_CMD=(python3 -c "$LLAMA_PROBE")
  fi
  set +e; "${LLAMA_CHECK_CMD[@]}" >/dev/null 2>&1; LLAMA_RC=$?; set -e
  case "$LLAMA_RC" in
    0) log "llama-cpp-python present with GPU offload" ;;
    3) warn "llama-cpp-python is a CPU-only build (uv sync restored the PyPI wheel?); rebuild with CMAKE_ARGS=-DGGML_CUDA=ON"; MISSING=1 ;;
    *) warn "llama-cpp-python missing"; MISSING=1 ;;
  esac
  [ -x "${LLAMA_CPP_DIR}/build/bin/llama-server" ] \
    && log "llama-server present at ${LLAMA_CPP_DIR}/build/bin/llama-server" \
    || warn "llama-server not built (only needed for native MTP / RPC pooling; not an error)"
  # Required pieces missing = nonzero exit so automation can gate on --check.
  [ "$MISSING" -eq 0 ] || fail "required pieces missing (see warnings above)"
  log "check passed"
  exit 0
fi

# ---- 2. Build toolchain + basics -------------------------------------------
export DEBIAN_FRONTEND=noninteractive
log "installing build toolchain"
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq build-essential cmake git curl pkg-config python3-dev python3-pip >/dev/null

# ---- 3. uv -------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  log "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
log "uv: $(uv --version)"

# ---- 4. NVML telemetry binding ----------------------------------------------
# Installed into the Skulk venv when --with-skulk-env runs below; also into
# the system python so `--check` and ad-hoc diagnostics work either way.
log "installing NVML binding (nvidia-ml-py)"
python3 -m pip install --quiet --break-system-packages nvidia-ml-py 2>/dev/null \
  || $SUDO python3 -m pip install --quiet --break-system-packages nvidia-ml-py 2>/dev/null \
  || python3 -m pip install --quiet --user nvidia-ml-py

# ---- 5. Optional: CUDA llama-server (native MTP + RPC donor) -----------------
if [ "$WITH_LLAMA_SERVER" -eq 1 ]; then
  log "building llama.cpp (CUDA + RPC) in ${LLAMA_CPP_DIR}"
  if [ ! -d "$LLAMA_CPP_DIR/.git" ]; then
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_DIR"
  fi
  cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" \
    -DGGML_CUDA=ON -DGGML_RPC=ON -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build "$LLAMA_CPP_DIR/build" --target llama-server ggml-rpc-server -j"$(nproc)" >/dev/null
  log "llama-server: ${LLAMA_CPP_DIR}/build/bin/llama-server"
  log "ggml-rpc-server (pooling donor): ${LLAMA_CPP_DIR}/build/bin/ggml-rpc-server"
  log "set in the node env: SKULK_LLAMA_SERVER_BIN=${LLAMA_CPP_DIR}/build/bin/llama-server"
fi

# ---- 6. Optional: Skulk env + CUDA llama-cpp-python ---------------------------
if [ "$WITH_SKULK_ENV" -eq 1 ]; then
  [ -f "pyproject.toml" ] || fail "--with-skulk-env must run from a Skulk checkout"
  log "uv sync (Skulk env)"
  uv sync >/dev/null
  log "building CUDA llama-cpp-python (source build; this takes a while)"
  CMAKE_ARGS="-DGGML_CUDA=ON" uv pip install --no-cache-dir --force-reinstall \
    --no-binary llama-cpp-python llama-cpp-python >/dev/null
  uv pip install --quiet nvidia-ml-py
  log "verify: uv run python -c 'import llama_cpp' && nvidia-smi"
  log "remember the node env: SKULK_LLAMA_CPP_BACKENDS=cuda"
fi

log "done"
