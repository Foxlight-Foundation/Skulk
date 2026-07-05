#!/usr/bin/env bash
# Copyright 2025 Foxlight Foundation
#
# One-shot dependency installer for a Skulk AMD / Strix Halo (gfx1151) worker
# node on Ubuntu. It installs everything the node needs BELOW the Skulk repo:
# the GPU compute stack (Mesa RADV Vulkan driver + loader + tools), the build
# toolchain that compiles the Rust bindings and the Vulkan llama.cpp, `uv`, and
# the GPU device-group membership. Optionally it also builds the `llama-server`
# binary that native MTP (speculative decoding) needs.
#
# Why this exists: Skulk's inference on gfx1151 runs through Vulkan (Mesa RADV),
# NOT ROCm/HIP -- `llama-server` links `libvulkan`, not `libamdhip`. On Ubuntu
# 26.04 the entire Vulkan path is distro-native (main + universe), so this script
# needs no third-party AMD repository. ROCm is optional here, installed only for
# the `rocminfo` diagnostic (also in Ubuntu universe), never for inference.
#
# Idempotent: safe to re-run. It installs missing packages, skips present ones.
#
# Usage:
#   deployment/rocm/install-deps.sh [options]
#
# Options:
#   --with-llama-server   Also clone+build llama.cpp with Vulkan and produce the
#                         llama-server binary (needed for native MTP). Prints the
#                         SKULK_LLAMA_SERVER_BIN path to set in ~/.skulk/skulk.env.
#   --with-skulk-env      If run from a Skulk checkout, also run `uv sync` and the
#                         Vulkan `llama-cpp-python` source build (the in-process
#                         GGUF engine). Equivalent to the docs' Quick-start steps.
#   --no-rocminfo         Skip the optional rocminfo diagnostic package.
#   --llama-cpp-dir DIR   Where to clone/build llama.cpp (default: $HOME/llama.cpp).
#   --check               Verify the stack only; install nothing.
#   -h | --help           Show this help.
#
# Validated on: Ubuntu 26.04 LTS (kernel 7.x), Ryzen AI Max+ 395 / Radeon 8060S
# (gfx1151), Mesa 26 RADV, Vulkan 1.4, llama.cpp Vulkan build b9820.
set -euo pipefail

log() { printf '\033[1;36m[deps]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[deps] WARNING:\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m[deps] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

WITH_LLAMA_SERVER=0
WITH_SKULK_ENV=0
WANT_ROCMINFO=1
CHECK_ONLY=0
LLAMA_CPP_DIR="${HOME}/llama.cpp"

while [ $# -gt 0 ]; do
  case "$1" in
    --with-llama-server) WITH_LLAMA_SERVER=1 ;;
    --with-skulk-env) WITH_SKULK_ENV=1 ;;
    --no-rocminfo) WANT_ROCMINFO=0 ;;
    --llama-cpp-dir) [ $# -ge 2 ] || die "--llama-cpp-dir requires a directory argument"; LLAMA_CPP_DIR="$2"; shift ;;
    --check) CHECK_ONLY=1 ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

# Counts every gap --check would have fixed; a verify run that found gaps must
# not exit 0, or automation trusting the exit code treats a broken box as ready.
CHECK_GAPS=0

check_gap() { CHECK_GAPS=$((CHECK_GAPS + 1)); warn "$@"; }

# --- Preflight: platform + GPU sanity ---------------------------------------
preflight() {
  [ "$(uname -s)" = "Linux" ] || die "this installer is for Linux AMD nodes; on macOS use the LaunchAgent path."
  # Run as the login user that will run Skulk, never as root: the script
  # escalates with sudo only where needed, and user-scoped steps (uv, rust,
  # the env build) would otherwise land under /root and leave a root-owned
  # .venv in the user checkout.
  [ "$(id -u)" -ne 0 ] || die "do not run as root/sudo; run as the login user that will run Skulk (the script uses sudo internally where needed)."
  if [ -r /etc/os-release ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    log "OS: ${PRETTY_NAME:-unknown} (kernel $(uname -r))"
    case "${ID:-}" in
      ubuntu|debian) : ;;
      *) warn "not Ubuntu/Debian; the apt package names below may not match your distro." ;;
    esac
  fi
  if command -v lspci >/dev/null 2>&1; then
    # Match against a captured copy, not a live `lspci | grep -q` pipeline: under
    # `set -o pipefail`, grep -q closes the pipe on first match and SIGPIPEs
    # lspci, which pipefail would then report as a (spurious) failure.
    local pci_list; pci_list="$(lspci -nn 2>/dev/null || true)"
    if grep -qiE '1002:1586|Strix Halo|Radeon 80[0-9]0S' <<<"$pci_list"; then
      log "GPU: Strix Halo Radeon detected."
    else
      warn "no Strix Halo Radeon found via lspci; continuing, but this node may not have the target GPU."
    fi
  fi
  if [ -e /dev/dri/renderD128 ]; then
    log "render node present: /dev/dri/renderD128"
  else
    warn "/dev/dri/renderD128 missing; the amdgpu kernel driver may not have bound the GPU (check firmware / kernel)."
  fi
}

# --- APT packages ------------------------------------------------------------
# The hard set (Vulkan inference + building the Rust bindings and Vulkan
# llama.cpp). All from Ubuntu main/universe on 26.04 -- no AMD third-party repo.
apt_packages() {
  local pkgs=(
    # build toolchain (Rust bindings via uv, and the Vulkan llama.cpp/llama-server)
    build-essential cmake git curl ca-certificates
    # Vulkan runtime: Mesa RADV driver + loader + vulkaninfo
    mesa-vulkan-drivers libvulkan1 vulkan-tools
    # Vulkan + shader build deps for -DGGML_VULKAN=ON. glslc alone is not enough:
    # llama.cpp's ggml-vulkan CMake needs the SPIRV-Headers cmake config package
    # and glslangValidator, which a clean Ubuntu 26.04 does not carry. Without
    # these the llama-server build fails at configure ("Could not find
    # SPIRV-Headers" / "missing components: glslangValidator").
    libvulkan-dev glslc glslang-tools spirv-headers spirv-tools
    # AMD GPU firmware for gfx1151 (usually present, listed so a minimal image gets it)
    linux-firmware
  )
  [ "$WANT_ROCMINFO" -eq 1 ] && pkgs+=(rocminfo)  # optional diagnostic (Ubuntu universe)
  printf '%s\n' "${pkgs[@]}"
}

install_apt() {
  local missing=()
  while IFS= read -r p; do
    dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p")
  done < <(apt_packages)
  if [ "${#missing[@]}" -eq 0 ]; then
    log "all apt packages already installed."
    return 0
  fi
  log "installing apt packages: ${missing[*]}"
  if [ "$CHECK_ONLY" -eq 1 ]; then
    check_gap "--check: missing apt packages: ${missing[*]}"
    return 0
  fi
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "${missing[@]}"
}

# --- GPU device group membership --------------------------------------------
# /dev/dri/renderD128 is owned by group `render`, card by `video`. A Skulk
# worker must be in both to reach the GPU. Newly-added groups need a fresh login.
ensure_gpu_groups() {
  # Target the real login user, not root: under `sudo ./install-deps.sh`,
  # $USER is root, and adding root to render/video leaves the actual Skulk
  # user unable to open /dev/dri/*. $USER can also be unset in
  # non-interactive shells (set -u would abort).
  local target_user; target_user="${SUDO_USER:-$(id -un)}"
  local added=0
  for grp in render video; do
    if getent group "$grp" >/dev/null 2>&1 && ! id -nG "$target_user" | tr ' ' '\n' | grep -qx "$grp"; then
      log "adding $target_user to group $grp"
      [ "$CHECK_ONLY" -eq 0 ] && sudo usermod -aG "$grp" "$target_user" && added=1
    fi
  done
  [ "$added" -eq 1 ] && warn "group membership changed: log out and back in (or reboot) before starting Skulk."
  return 0
}

# --- uv ----------------------------------------------------------------------
ensure_uv() {
  # The astral installer drops uv at ~/.local/bin/uv, which is not on a
  # non-login shell's PATH, so `command -v uv` alone can miss an installed uv.
  # Make it visible for the rest of this run, then detect either way.
  [ -x "${HOME}/.local/bin/uv" ] && export PATH="${HOME}/.local/bin:${PATH}"
  if command -v uv >/dev/null 2>&1; then
    log "uv present: $(uv --version 2>/dev/null || echo installed)"
    return 0
  fi
  log "installing uv (astral installer -> ~/.local/bin)"
  [ "$CHECK_ONLY" -eq 1 ] && { check_gap "--check: uv is not installed"; return 0; }
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Make uv visible for the rest of this script run.
  export PATH="${HOME}/.local/bin:${PATH}"
  command -v uv >/dev/null 2>&1 || die "uv install did not land on PATH (~/.local/bin); open a new shell and re-run."
}

# --- Rust toolchain (only needed for --with-skulk-env) ------------------------
ensure_rust() {
  # `uv sync` builds the editable rust/skulk_pyo3_bindings workspace via
  # maturin, which needs cargo/rustc. Ubuntu's packaged rustc lags what the
  # bindings need, so install via rustup (same pattern as ensure_uv).
  [ -x "${HOME}/.cargo/bin/cargo" ] && export PATH="${HOME}/.cargo/bin:${PATH}"
  if command -v cargo >/dev/null 2>&1; then
    log "rust present: $(cargo --version 2>/dev/null || echo installed)"
    return 0
  fi
  log "installing rust (rustup -> ~/.cargo/bin); the Skulk env build needs cargo"
  [ "$CHECK_ONLY" -eq 1 ] && { check_gap "--check: rust (cargo) is not installed"; return 0; }
  curl --proto '=https' --tlsv1.2 -LsSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
  export PATH="${HOME}/.cargo/bin:${PATH}"
  command -v cargo >/dev/null 2>&1 || die "rust install did not land on PATH (~/.cargo/bin); open a new shell and re-run."
}

# --- Optional: build llama-server (native MTP) -------------------------------
build_llama_server() {
  local bin="${LLAMA_CPP_DIR}/build/bin/llama-server"
  local rpc_bin="${LLAMA_CPP_DIR}/build/bin/ggml-rpc-server"
  # Skip only when BOTH binaries exist: a tree built under the earlier
  # Vulkan-only instructions has llama-server but no ggml-rpc-server (it was
  # configured without GGML_RPC), and multi-node GGUF pooling needs the
  # sibling donor daemon. Rebuilding such a tree picks the RPC target up.
  if [ -x "$bin" ] && [ -x "$rpc_bin" ]; then
    log "llama-server already built: $bin ($("$bin" --version 2>&1 | head -1))"
    log "ggml-rpc-server (pooling donor daemon) present: $rpc_bin"
  else
    if [ -x "$bin" ]; then
      log "llama-server exists but ggml-rpc-server is missing; rebuilding with -DGGML_RPC=ON for multi-node pooling"
    else
      log "building llama-server with Vulkan at ${LLAMA_CPP_DIR}"
    fi
    [ "$CHECK_ONLY" -eq 1 ] && { check_gap "--check: llama-server / ggml-rpc-server not built"; return 0; }
    if [ ! -d "${LLAMA_CPP_DIR}/.git" ]; then
      git clone https://github.com/ggml-org/llama.cpp.git "${LLAMA_CPP_DIR}"
    fi
    ( cd "${LLAMA_CPP_DIR}"
      # GGML_RPC=ON also produces ggml-rpc-server, the memory-donor daemon
      # that multi-node GGUF pooling launches on donor nodes; Skulk finds it
      # as a sibling of SKULK_LLAMA_SERVER_BIN (or via SKULK_RPC_SERVER_BIN).
      cmake -B build -DGGML_VULKAN=ON -DGGML_RPC=ON -DCMAKE_BUILD_TYPE=Release
      cmake --build build --target llama-server ggml-rpc-server -j"$(nproc)" )
    [ -x "$bin" ] || die "llama-server did not build at $bin"
    [ -x "$rpc_bin" ] || die "ggml-rpc-server did not build at $rpc_bin (multi-node pooling needs it)"
  fi
  log "MTP-capable server ready. Add this to ~/.skulk/skulk.env:"
  printf '    SKULK_LLAMA_SERVER_BIN=%s\n' "$bin"
}

# --- Optional: skulk repo env (uv sync + Vulkan llama-cpp-python) -------------
build_skulk_env() {
  if [ ! -f "pyproject.toml" ] || ! grep -q '^name *= *"skulk"' pyproject.toml 2>/dev/null; then
    warn "--with-skulk-env: run this from a Skulk checkout root; skipping."
    return 0
  fi
  ensure_rust
  log "uv sync (builds Rust bindings; --inexact preserves a Vulkan llama-cpp-python wheel)"
  [ "$CHECK_ONLY" -eq 1 ] && { warn "--check: not running uv sync + llama-cpp-python build (build step, not a verifiable state)"; return 0; }
  uv sync --inexact
  log "building llama-cpp-python from source with Vulkan (in-process GGUF engine)"
  # --no-binary forces the source build; without it uv installs a CPU-only wheel
  # and CMAKE_ARGS is ignored. Rebuild by hand only on a llama.cpp version bump.
  CMAKE_ARGS="-DGGML_VULKAN=on" uv pip install --force-reinstall --no-cache-dir \
    --no-binary llama-cpp-python --python .venv/bin/python llama-cpp-python
}

# --- Verify ------------------------------------------------------------------
# All extractions below capture the tool's full output first, then match on the
# copy. A live `tool | grep -m1 | head` pipeline would SIGPIPE the tool on early
# close and, under pipefail + set -e, abort the whole script (which silently ate
# the rocm and uv lines in an earlier version).
verify() {
  log "verifying GPU stack:"
  if command -v vulkaninfo >/dev/null 2>&1; then
    local vk dev
    vk="$(vulkaninfo 2>/dev/null || true)"
    dev="$(sed -n 's/.*deviceName[[:space:]]*=[[:space:]]*//p' <<<"$vk" | head -1 || true)"
    # A GPU Vulkan cannot see is a --check failure, not a footnote: the box
    # would advertise a backend it cannot serve (broken driver binding, or
    # group membership not yet active in this session). A SOFTWARE Vulkan
    # device (Mesa llvmpipe) also fails: it exists on CPU-only or
    # permission-broken boxes and would let --check pass while the Radeon is
    # unusable.
    if [ -z "$dev" ]; then
      check_gap "vulkaninfo returned no device (driver/permissions?)"
    elif grep -qiE 'llvmpipe|swiftshader|software' <<<"$dev"; then
      check_gap "Vulkan device is a software rasterizer (${dev}); the Radeon GPU is not usable via RADV"
    else
      log "  vulkan: ${dev}"
    fi
  else
    check_gap "vulkaninfo missing"
  fi
  if command -v rocminfo >/dev/null 2>&1; then
    local roc gfx
    roc="$(rocminfo 2>/dev/null || true)"
    gfx="$(awk '/Name:[[:space:]]*gfx/{print $2; exit}' <<<"$roc" || true)"
    [ -n "$gfx" ] && log "  rocm: ${gfx}" || warn "  rocminfo found no gfx target"
  fi
  command -v uv >/dev/null 2>&1 && log "  uv: $(uv --version 2>/dev/null)"
}

main() {
  preflight
  install_apt
  ensure_gpu_groups
  ensure_uv
  [ "$WITH_LLAMA_SERVER" -eq 1 ] && build_llama_server
  [ "$WITH_SKULK_ENV" -eq 1 ] && build_skulk_env
  verify
  if [ "$CHECK_ONLY" -eq 1 ] && [ "$CHECK_GAPS" -gt 0 ]; then
    die "--check found ${CHECK_GAPS} gap(s) above; this node is not ready."
  fi
  log "done. Next: clone Skulk (if not already), then see deployment/rocm/README.md"
  log "for launching the node (launch-skulk.sh) or the managed systemd service."
}

main "$@"
