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

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git curl pkg-config python3-dev python3-pip \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# llama-server + ggml-rpc-server: the served-backend engine binaries (native
# MTP via --spec-type, RPC pooling). Mirrors install-deps.sh's cmake flags.
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp /opt/llama.cpp \
    && cmake -S /opt/llama.cpp -B /opt/llama.cpp/build \
         -DGGML_CUDA=ON -DGGML_RPC=ON -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
    && cmake --build /opt/llama.cpp/build \
         --target llama-server ggml-rpc-server -j"$(nproc)"

# CUDA llama-cpp-python wheel for the in-process llama_cpp engine. Built as a
# wheel (not installed) so the session's `uv sync`-managed venv can install it
# after syncing, which is what defeats the uv-sync-restores-CPU-wheel trap.
# Built with uv-provisioned Python 3.13 to match Skulk's .python-version:
# Ubuntu 22.04's system python is 3.10, and a cp310 wheel would be rejected by
# the 3.13 environment pod-bootstrap.sh syncs (PR #538 review, P1).
# --no-binary pins the CUDA source build even if PyPI ever starts shipping
# prebuilt (CPU-only) wheels for this package.
ARG PYTHON_VERSION=3.13
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && /root/.local/bin/uv venv --seed --python "${PYTHON_VERSION}" /opt/wheelbuild \
    && CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS}" \
      /opt/wheelbuild/bin/python -m pip wheel --no-cache-dir \
        --no-binary llama-cpp-python --wheel-dir /opt/wheels llama-cpp-python

# ---- runtime stage: everything a session needs except Skulk itself ----------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

# build-essential + cmake stay: `uv sync` compiles Skulk's Rust bindings at
# session time (the Skulk checkout is per-session, so it cannot be prebaked).
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git curl pkg-config python3-dev \
      ca-certificates openssh-server rsync tmux \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /run/sshd

# Rust toolchain for the skulk-pyo3-bindings build during `uv sync`.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --profile minimal --default-toolchain stable
ENV PATH="/root/.cargo/bin:/root/.local/bin:/usr/local/cuda/bin:${PATH}"

# uv: the canonical Skulk runtime path.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

COPY --from=build /opt/llama.cpp/build/bin/llama-server /opt/llama.cpp/build/bin/llama-server
COPY --from=build /opt/llama.cpp/build/bin/ggml-rpc-server /opt/llama.cpp/build/bin/ggml-rpc-server
COPY --from=build /opt/wheels /opt/wheels

COPY deployment/cuda/pod-bootstrap.sh /opt/skulk/pod-bootstrap.sh
COPY deployment/cuda/pod-start.sh /opt/skulk/pod-start.sh
RUN chmod +x /opt/skulk/pod-bootstrap.sh /opt/skulk/pod-start.sh

# RunPod runs the image command as pid 1; pod-start.sh installs PUBLIC_KEY for
# sshd and then idles so the pod stays up for SSH sessions.
CMD ["/opt/skulk/pod-start.sh"]
