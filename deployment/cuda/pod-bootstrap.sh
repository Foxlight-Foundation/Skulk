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

# One ref-resolution path for fresh and reused checkouts: clone the default
# branch shallow, then fetch/checkout the requested ref. `git clone --branch`
# cannot take a commit SHA, and pinning a session to an exact commit (a PR
# head, a workflow SHA) is a primary use of this script.
if [ ! -d "${SKULK_DIR}/.git" ]; then
  log "cloning Skulk"
  git clone --depth 1 https://github.com/Foxlight-Foundation/Skulk "${SKULK_DIR}"
fi
log "fetching ${REF}"
# Branches and SHAs fetch bare; tags need the qualified refspec.
git -C "${SKULK_DIR}" fetch --depth 1 origin "${REF}" \
  || git -C "${SKULK_DIR}" fetch --depth 1 origin "refs/tags/${REF}:refs/tags/${REF}"
git -C "${SKULK_DIR}" checkout FETCH_HEAD

cd "${SKULK_DIR}"
# --extra llama-cpp: llama-cpp-python lives in an optional extra, so a plain
# sync would install neither it nor its locked dependencies (diskcache and
# friends), and the --no-deps CUDA wheel swap below would leave the import
# broken. The extra installs the locked CPU build + deps; the swap then only
# replaces the artifact.
log "uv sync --extra llama-cpp (builds the Rust bindings)"
uv sync --extra llama-cpp

# AFTER uv sync: a plain sync restores the CPU-only PyPI llama-cpp-python
# wheel, so the CUDA wheel must be (re)installed last, exactly like
# install-deps.sh --with-skulk-env does.
log "installing prebaked CUDA llama-cpp-python wheel"
# Exactly one wheel must match: zero means the image build silently failed,
# more than one means an ambiguous install; both should stop the session here.
WHEELS=(/opt/wheels/llama_cpp_python-*.whl)
if [ "${#WHEELS[@]}" -ne 1 ] || [ ! -f "${WHEELS[0]}" ]; then
  echo "[bootstrap] ERROR: expected exactly one prebaked wheel, found: ${WHEELS[*]}" >&2
  exit 1
fi
# The image is reusable across refs, but the wheel it carries is pinned to
# the pins of the commit that built it. A ref with a different locked
# llama-cpp-python would otherwise end up silently un-locked, and a
# different .python-version fails later with a confusing wheel-tag error;
# both mismatches stop here with instructions instead.
WHEEL_NAME="$(basename "${WHEELS[0]}")"
WHEEL_VERSION="$(printf '%s' "${WHEEL_NAME}" | cut -d- -f2)"
WHEEL_PY_TAG="$(printf '%s' "${WHEEL_NAME}" | cut -d- -f3)"
LOCKED_VERSION="$(sed -n '/^name = "llama-cpp-python"$/{n;s/^version = "\(.*\)"$/\1/p;}' uv.lock | head -1)"
EXPECTED_PY_TAG="cp$(tr -d '[:space:].' < .python-version | cut -c1-3)"
if [ "${WHEEL_VERSION}" != "${LOCKED_VERSION}" ] || [ "${WHEEL_PY_TAG}" != "${EXPECTED_PY_TAG}" ]; then
  echo "[bootstrap] ERROR: prebaked wheel ${WHEEL_NAME} does not match this ref's pins" >&2
  echo "[bootstrap]   ref locks llama-cpp-python==${LOCKED_VERSION} on ${EXPECTED_PY_TAG}" >&2
  echo "[bootstrap]   use the image built for this ref, or rebuild via the cuda-image workflow" >&2
  exit 1
fi
# --no-deps: uv sync already installed llama-cpp-python's dependencies at
# their locked versions; this step only swaps the wheel artifact and must
# not let pip drift the rest of the environment away from uv.lock.
uv pip install --force-reinstall --no-deps "${WHEELS[0]}"
# Optional NVML telemetry binding: best-effort, because a transient PyPI
# failure must not abort a session Skulk can run without it.
uv pip install --quiet --no-deps nvidia-ml-py \
  || log "WARNING: nvidia-ml-py install failed; NVML telemetry disabled this session"

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
