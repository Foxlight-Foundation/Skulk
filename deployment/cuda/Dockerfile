# syntax=docker/dockerfile:1
# Prebaked CUDA pod image for rented-GPU Skulk sessions (RunPod and similar).
#
# Motivation: provisioning a fresh pod through deployment/cuda/install-deps.sh
# takes about an hour, dominated by two CUDA compiles (llama.cpp's llama-server
# and the llama-cpp-python wheel). Baking those artifacts into an image cuts a
# session's setup to cloning Skulk and running /opt/skulk/pod-bootstrap.sh
# (minutes). Built by .github/workflows/cuda-image.yml on standard amd64
# runners: nvcc compiles CUDA kernels without a GPU present.
#
# CUDA_ARCHS pins the compiled GPU architectures. Default covers the rental
# fleet we actually use: 86 (A40/A5000/3090) and 89 (4090/L40S). Add 80 (A100)
# or 90 (H100) when renting those tiers; every extra arch lengthens the build.

ARG CUDA_VERSION=12.4.1
ARG UBUNTU_VERSION=22.04

# ---- build stage: the two expensive CUDA compiles ---------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build
ARG CUDA_ARCHS="86;89"
# pipefail: a curl|sh installer whose download dies must fail the build, not
# publish an image that is missing its toolchain.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git curl pkg-config python3-dev python3-pip \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# llama-server + ggml-rpc-server: the served-backend engine binaries (native
# MTP via --spec-type, RPC pooling). Mirrors install-deps.sh's cmake flags,
# plus two portability requirements of a prebuilt image:
# - LLAMA_CPP_REF pins the upstream revision so re-running the workflow for
#   the same Skulk commit republishes identical engine behavior; bump
#   deliberately.
# - GGML_NATIVE=OFF: default builds tune for the connected (builder) CPU, and
#   AVX-512/AMX code from a GitHub runner would die with illegal-instruction
#   on a rented pod with a different CPU.
# - --allow-shlib-undefined: libggml-cuda references CUDA Driver API symbols
#   (cuMemMap, ...) that only the driver's libcuda.so provides, and a GPU-less
#   runner has no driver. Upstream's own CI images defer resolution to runtime
#   this way (.devops/cuda.Dockerfile); the pod's real driver satisfies them.
ARG LLAMA_CPP_REF=b9969
RUN git clone --branch "${LLAMA_CPP_REF}" --depth 1 \
         https://github.com/ggml-org/llama.cpp /opt/llama.cpp \
    && cmake -S /opt/llama.cpp -B /opt/llama.cpp/build \
         -DGGML_CUDA=ON -DGGML_RPC=ON -DGGML_NATIVE=OFF \
         -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
         -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
    && cmake --build /opt/llama.cpp/build \
         --target llama-server ggml-rpc-server -j"$(nproc)"

# CUDA llama-cpp-python wheel for the in-process llama_cpp engine. Built as a
# wheel (not installed) so the session's `uv sync`-managed venv can install it
# after syncing, which is what defeats the uv-sync-restores-CPU-wheel trap.
# Built with a uv-provisioned Python at the version pinned in the repo's
# .python-version (Ubuntu 22.04's system python is 3.10, whose wheel tag the
# synced environment would reject).
# --no-binary pins the CUDA source build even if PyPI ever starts shipping
# prebuilt (CPU-only) wheels for this package. Both the package version and
# the Python version come from the repo's own pins (uv.lock and
# .python-version), so the wheel can never drift from what pod-bootstrap.sh's
# `uv sync` will demand (PR #538 review rounds 3-4).
COPY uv.lock /tmp/uv.lock
COPY .python-version /tmp/python-version
# UV_VERSION pins the installer for reproducible rebuilds of the same commit;
# bump deliberately alongside fleet uv upgrades.
ARG UV_VERSION=0.11.6
RUN curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh \
    && LLAMA_CPP_PYTHON_VERSION="$(sed -n '/^name = "llama-cpp-python"$/{n;s/^version = "\(.*\)"$/\1/p;}' /tmp/uv.lock | head -1)" \
    && test -n "${LLAMA_CPP_PYTHON_VERSION}" \
    && PYTHON_VERSION="$(tr -d '[:space:]' < /tmp/python-version)" \
    && test -n "${PYTHON_VERSION}" \
    && echo "building llama-cpp-python==${LLAMA_CPP_PYTHON_VERSION} for python ${PYTHON_VERSION}" \
    && /root/.local/bin/uv venv --seed --python "${PYTHON_VERSION}" /opt/wheelbuild \
    && CMAKE_ARGS="-DGGML_CUDA=ON -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS} -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined" \
      /opt/wheelbuild/bin/python -m pip wheel --no-cache-dir --no-deps \
        --no-binary llama-cpp-python --wheel-dir /opt/wheels \
        "llama-cpp-python==${LLAMA_CPP_PYTHON_VERSION}"

# ---- runtime stage: everything a session needs except Skulk itself ----------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}
# SHELL does not survive FROM; re-assert pipefail for this stage's curl|sh
# installers (rustup, uv).
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# build-essential + cmake stay: `uv sync` compiles Skulk's Rust bindings at
# session time (the Skulk checkout is per-session, so it cannot be prebaked).
# The host keys openssh-server generates at install time are deleted: this
# image is public, so baked-in keys would be shared by every pod pulled from
# it (and readable by anyone), enabling host impersonation. pod-start.sh
# regenerates per-pod keys before starting sshd.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git curl pkg-config python3-dev \
      ca-certificates openssh-server rsync tmux \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /run/sshd \
    && rm -f /etc/ssh/ssh_host_* \
    && printf '%s\n' \
         'PasswordAuthentication no' \
         'KbdInteractiveAuthentication no' \
         'PermitRootLogin prohibit-password' \
         > /etc/ssh/sshd_config.d/90-skulk-pod.conf
# Key-only auth is pinned explicitly rather than trusted from base-image
# defaults: this image is public and runs sshd as its entrypoint, so a future
# base change must not be able to ship password auth silently.

# Rust toolchain for the skulk-pyo3-bindings build during `uv sync`. Pinned
# (not `stable`) so rebuilding this published image later cannot silently
# change the compiler under older Skulk refs; bump deliberately.
ARG RUST_VERSION=1.94.1
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --profile minimal --default-toolchain "${RUST_VERSION}"
ENV PATH="/root/.cargo/bin:/root/.local/bin:/usr/local/cuda/bin:${PATH}"
# ENV only reaches the container CMD; SSH sessions build their environment
# from login profiles, so bake the same PATH there or every remote command
# hits the uv/nvcc-not-found ambush.
RUN printf 'export PATH="/root/.cargo/bin:/root/.local/bin:/usr/local/cuda/bin:$PATH"\n' \
      > /etc/profile.d/skulk-pod.sh \
    && printf '%s\n' 'export PATH="/root/.cargo/bin:/root/.local/bin:/usr/local/cuda/bin:$PATH"' \
      >> /root/.bashrc

# uv: the canonical Skulk runtime path. Same pin rationale as the build stage.
ARG UV_VERSION=0.11.6
RUN curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh

# The whole bin directory, not just the two executables: upstream llama.cpp
# builds shared on Linux (BUILD_SHARED_LIBS defaults on), so llama-server and
# ggml-rpc-server need their sibling .so files or they die at loader time
# (PR #538 review round 3, P1).
COPY --from=build /opt/llama.cpp/build/bin /opt/llama.cpp/build/bin
COPY --from=build /opt/wheels /opt/wheels

COPY deployment/cuda/pod-bootstrap.sh /opt/skulk/pod-bootstrap.sh
COPY deployment/cuda/pod-start.sh /opt/skulk/pod-start.sh
RUN chmod +x /opt/skulk/pod-bootstrap.sh /opt/skulk/pod-start.sh

# RunPod runs the image command as pid 1; pod-start.sh installs PUBLIC_KEY for
# sshd and then idles so the pod stays up for SSH sessions.
CMD ["/opt/skulk/pod-start.sh"]
