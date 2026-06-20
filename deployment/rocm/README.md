<!-- Copyright 2025 Foxlight Foundation -->

# AMD / ROCm worker node setup (Strix Halo, gfx1151)

This directory holds the deployment assets for running a Skulk **worker node**
on an AMD Ryzen AI Max (Strix Halo / gfx1151) box under Linux. Such a node
joins a Skulk cluster as a llama.cpp engine running GGUF models on the integrated
Radeon GPU, alongside Apple Silicon nodes serving MLX models.

This setup is validated on:

| Component         | Validated version                                   |
| ----------------- | --------------------------------------------------- |
| Hardware          | AMD Ryzen AI Max+ 395 w/ Radeon 8060S (gfx1151)     |
| OS                | Ubuntu 26.04 LTS (kernel 7.0)                       |
| ROCm              | 6.4.0                                                |
| Vulkan driver     | RADV (Mesa), `STRIX_HALO`                            |
| Python / uv       | Python 3.13, uv 0.11                                 |
| llama-cpp-python  | 0.3.30, built with the Vulkan backend               |

> The GPU compute path here is **Vulkan (RADV)**, not the ROCm/HIP backend.
> On gfx1151 the Vulkan backend is the reliable, well-supported route for
> llama.cpp today; ROCm is installed for the runtime/driver stack. See
> `website/docs/amd-strix-halo-nodes.md` for the rationale and benchmarks.

## What a Skulk node needs on this box

1. **A working GPU compute stack**: ROCm runtime + a Vulkan driver (RADV).
2. **The Skulk repo + its `uv` environment** (the Rust bindings build via
   `uv sync`; no MLX is required or used on a non-Mac node).
3. **`llama-cpp-python` built with Vulkan**: `uv sync` installs the CPU wheel,
   so the Vulkan build must be (re)installed after any `uv sync`.
4. **A launcher** that exports the node's cluster env and starts skulk detached
   so it survives an SSH disconnect (Linux has no launchd; see
   `launch-skulk.sh.example`).

## Quick start

```bash
# 1. Clone and build the uv environment (Rust bindings compile here).
git clone https://github.com/Foxlight-Foundation/Skulk.git
cd Skulk
uv sync

# 2. Build llama-cpp-python from source with the Vulkan backend. --no-binary
#    forces the source build; without it uv installs the prebuilt CPU wheel and
#    CMAKE_ARGS is ignored. Re-run after any `uv sync`, which restores the wheel.
CMAKE_ARGS="-DGGML_VULKAN=on" uv pip install --force-reinstall --no-cache-dir \
  --no-binary llama-cpp-python --python .venv/bin/python llama-cpp-python

# 3. Tell Skulk this node serves a Vulkan llama.cpp backend, then launch.
cp deployment/rocm/launch-skulk.sh.example ~/launch-skulk.sh
# Edit ~/launch-skulk.sh only if your checkout is not at ~/projects/foxlight/Skulk
# (set SKULK_DIR), or if your cluster uses a custom libp2p namespace or the Zenoh
# DATA plane (then match the rest of the fleet; a stock cluster needs neither).
chmod +x ~/launch-skulk.sh
~/launch-skulk.sh           # foreground, to watch the first boot

# 4. Once it joins cleanly, run it detached so it survives the SSH session.
#    nohup + disown matter: a bare `setsid ... &` inside an SSH command can be
#    SIGHUP'd during session teardown before it fully detaches.
nohup setsid bash ~/launch-skulk.sh > ~/skulk.log 2>&1 < /dev/null & disown
```

The detached launcher above is fine for first-boot watching, but it does **not**
restart skulk after a crash or reboot. For a permanent node, install the managed
service instead (next section).

## Running as a managed service (recommended)

For a node that should rejoin the cluster automatically after a crash or reboot,
run Skulk under the same systemd user service the rest of the fleet uses, rather
than the detached launcher. This mirrors the macOS LaunchAgent on Apple Silicon
nodes: start-on-boot (via linger), restart-on-failure, and boot-time `git pull` /
`uv sync` through `deployment/install/skulk-startup.sh`.

```bash
# 1. Put this node's cluster env in the service env file. Headless is required
#    so the boot-time prep skips the (absent) dashboard build. Add the same
#    backend/namespace/Zenoh knobs you'd otherwise set in launch-skulk.sh.
mkdir -p ~/.skulk
cat >> ~/.skulk/skulk.env <<'ENV'
SKULK_HEADLESS=1
SKULK_LLAMA_CPP_BACKENDS=vulkan
# Skip boot-time `uv sync`: it reinstalls the CPU llama-cpp-python wheel and
# clobbers the Vulkan build below. Pin the revision and update manually instead.
SKULK_AUTO_UPDATE=0
# SKULK_LIBP2P_NAMESPACE=...   # match the rest of the fleet
# SKULK_ZENOH_DATA_PLANE / SKULK_ZENOH_LISTEN / SKULK_ZENOH_CONNECT as needed
ENV

# 2. Install + enable the user service (enables linger for boot autostart).
deployment/install/install-systemd.sh
```

The Vulkan `llama-cpp-python` build from Quick-start step 2 must already be in the
`.venv` before you start the service. With `SKULK_AUTO_UPDATE=0` the service runs
whatever is on disk, so the Vulkan wheel survives restarts; to update the node,
`git pull` and re-run the `uv sync` + `--no-binary` Vulkan install by hand, then
`systemctl --user restart skulk`. Manage the service with
`systemctl --user {status,restart,stop} skulk` and follow logs via
`journalctl --user -u skulk -f`.

The node advertises `llama_cpp` + `llama_cpp-vulkan` backends (because
`SKULK_LLAMA_CPP_BACKENDS=vulkan` is set and `llama_cpp` imports), and the
master places GGUF models whose `compatible_backends` include `llama_cpp-vulkan`
onto it. No dashboard build is needed on a headless node.

## Verifying the GPU stack

```bash
rocminfo | grep -iE "Name:.*gfx"            # -> gfx1151
vulkaninfo | grep -iE "deviceName"          # -> Radeon 8060S Graphics (RADV STRIX_HALO)
.venv/bin/python -c "import llama_cpp; print(llama_cpp.__version__)"
```

A quick standalone llama.cpp decode (outside Skulk) confirms the GPU path before
joining a cluster; a 7B Q4_K_M model should offload all layers to the Radeon iGPU.

## Files

- `launch-skulk.sh.example`: the node launcher template (cluster env + detached
  start). Copy to `~/launch-skulk.sh` and set the peer IPs.
