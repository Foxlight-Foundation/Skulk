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

# ---- verify this ref matches the prebaked wheel BEFORE any sync ------------
# The checks run against the ref's COMMITTED pins: syncing first (and without
# --locked) could regenerate uv.lock in the pod and the checks would then
# validate against bootstrap-generated state instead of the ref.
# Exactly one wheel must match: zero means the image build silently failed,
# more than one means an ambiguous install; both should stop the session here.
WHEELS=(/opt/wheels/llama_cpp_python-*.whl)
if [ "${#WHEELS[@]}" -ne 1 ] || [ ! -f "${WHEELS[0]}" ]; then
  echo "[bootstrap] ERROR: expected exactly one prebaked wheel, found: ${WHEELS[*]}" >&2
  exit 1
fi
for pin_file in uv.lock .python-version; do
  if [ ! -f "${pin_file}" ]; then
    echo "[bootstrap] ERROR: ${pin_file} missing from this checkout; the ref predates (or is incompatible with) the prebaked-image flow. Provision with deployment/cuda/install-deps.sh instead." >&2
    exit 1
  fi
done
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
# llama-cpp-python loads its native library via ctypes, so its wheel builds
# as version-independent py3-none; that tag is valid for any synced Python.
if [ -z "${LOCKED_VERSION}" ] || [ "${EXPECTED_PY_TAG}" = "cp" ]; then
  echo "[bootstrap] ERROR: could not parse llama-cpp-python pin (got version '${LOCKED_VERSION:-}', tag '${EXPECTED_PY_TAG}') from this ref's uv.lock/.python-version." >&2
  exit 1
fi
if [ "${WHEEL_VERSION}" != "${LOCKED_VERSION}" ] || { [ "${WHEEL_PY_TAG}" != "${EXPECTED_PY_TAG}" ] && [ "${WHEEL_PY_TAG}" != "py3" ]; }; then
  echo "[bootstrap] ERROR: prebaked wheel ${WHEEL_NAME} does not match this ref's pins" >&2
  echo "[bootstrap]   ref locks llama-cpp-python==${LOCKED_VERSION} on ${EXPECTED_PY_TAG}" >&2
  echo "[bootstrap]   use the image built for this ref, or rebuild via the cuda-image workflow" >&2
  exit 1
fi

# ---- sync and swap ----------------------------------------------------------
# --locked: the sync must install the ref's committed lock verbatim, never
# re-resolve it on the pod. The llama-cpp extra is NOT synced: its
# llama-cpp-python entry is sdist-only, so syncing it would compile a CPU
# build on the pod - the exact rental-time cost this image exists to avoid.
# Instead the extra's locked DEPENDENCIES (diskcache and friends) install
# from an export below, and the prebaked CUDA wheel supplies the package.
log "uv sync --locked (builds the Rust bindings)"
uv sync --locked

log "installing the llama-cpp extra's locked dependencies (without the sdist)"
uv export --frozen --extra llama-cpp --no-hashes --no-emit-project --no-emit-workspace \
  --no-emit-package llama-cpp-python -o /tmp/llama-extra-deps.txt
uv pip install --requirement /tmp/llama-extra-deps.txt

log "installing prebaked CUDA llama-cpp-python wheel"
# --no-deps: uv sync already installed llama-cpp-python's dependencies at
# their locked versions; this step only swaps the wheel artifact and must
# not let pip drift the rest of the environment away from uv.lock.
# --force-reinstall keeps re-runs honest; --no-deps because the locked
# dependency set was installed above and must not drift.
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
