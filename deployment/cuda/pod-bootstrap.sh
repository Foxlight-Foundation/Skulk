#!/usr/bin/env bash
# Per-session Skulk setup on the prebaked CUDA pod image.
#
# The image carries everything slow (CUDA llama-cpp-python wheel at
# /opt/wheels, llama-server + ggml-rpc-server at /opt/llama.cpp/build/bin,
# uv, the Rust toolchain); this script does only the per-session parts:
# clone the requested Skulk ref, sync its environment, and swap in the
# prebaked CUDA wheel. Minutes instead of the recipe's ~1 hour.
#
# Usage: /opt/skulk/pod-bootstrap.sh [git-ref]   (default: dev)
set -euo pipefail

REF="${1:-dev}"
SKULK_DIR="${SKULK_DIR:-/root/skulk}"

log() { printf '[bootstrap] %s\n' "$*"; }

if [ ! -d "${SKULK_DIR}/.git" ]; then
  log "cloning Skulk (${REF})"
  git clone --branch "${REF}" --depth 1 \
    https://github.com/Foxlight-Foundation/Skulk "${SKULK_DIR}"
else
  log "reusing checkout at ${SKULK_DIR} (fetching ${REF})"
  git -C "${SKULK_DIR}" fetch --depth 1 origin "${REF}"
  git -C "${SKULK_DIR}" checkout FETCH_HEAD
fi

cd "${SKULK_DIR}"
log "uv sync (builds the Rust bindings)"
uv sync

# AFTER uv sync: a plain sync restores the CPU-only PyPI llama-cpp-python
# wheel, so the CUDA wheel must be (re)installed last, exactly like
# install-deps.sh --with-skulk-env does.
log "installing prebaked CUDA llama-cpp-python wheel"
uv pip install --force-reinstall /opt/wheels/llama_cpp_python-*.whl
uv pip install --quiet nvidia-ml-py

log "verifying GPU offload support in the installed binding"
uv run --no-sync python - <<'PY'
import llama_cpp

if not llama_cpp.llama_supports_gpu_offload():
    raise SystemExit("llama-cpp-python build has NO GPU offload; wheel/arch mismatch?")
print("[bootstrap] llama-cpp-python: GPU offload OK")
PY

log "done. Launch example:"
log "  cd ${SKULK_DIR} && SKULK_LLAMA_CPP_BACKENDS=cuda \\"
log "  SKULK_LLAMA_SERVER_BIN=/opt/llama.cpp/build/bin/llama-server uv run skulk"
