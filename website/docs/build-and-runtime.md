---
id: build-and-runtime
title: Build And Runtime Paths
sidebar_position: 3
---

<!-- Copyright 2025 Foxlight Foundation -->

Skulk supports both `uv` and Nix in development, but they do not have the same
job.

## One-Command Install

The fastest path from a fresh macOS or Linux machine to a working node:

```bash
curl -fsSL https://raw.githubusercontent.com/Foxlight-Foundation/Skulk/main/install.sh | bash
```

The installer targets the stable branch (`main`) regardless of which docs
channel you are reading. To install the development branch instead (matching
the `/next/` docs), pass a ref:

```bash
curl -fsSL https://raw.githubusercontent.com/Foxlight-Foundation/Skulk/main/install.sh | bash -s -- --ref dev
```

`--ref` also accepts a full 40-character commit ID. This is the deterministic
path used by release-candidate qualification: the installer fetches that exact
object and checks it out detached, so a moving branch cannot change the code
between approval and installation.

The installer fetches prerequisites (git, a C toolchain, rustup, uv), clones
the repo into `~/skulk`, syncs the environment, and builds the dashboard with
Skulk's bundled cross-platform Node.js runtime (falling back to a compatible
system toolchain if necessary). It finishes with `skulk doctor --fix`, which audits the node
(GPU detection, engine availability, storage headroom) and applies safe
remediations, printing the consequence and fix for anything it cannot repair.
Re-running the installer is safe: every step is idempotent. Pass `--headless`
only for an intentionally API-only node; a normal fresh install always
includes the dashboard.

Skulk releases qualify this same path on clean Apple Silicon, AMD Linux, and
NVIDIA Linux environments. A candidate run pins the proposed commit; after
promotion, the shipping run executes the literal `main` command above. Tests
that attach to an already-configured fleet remain valuable regression coverage,
but do not substitute for fresh-install qualification.

At any later point, `uv run skulk doctor` re-audits the node; see
[Node doctor](node-doctor) for the check list and verdicts.

### Engine provisioning

The installer wires an inference engine matched to the hardware it detects:

- **NVIDIA Linux**: installs the `skulk-llama-server-cuda` wheel from the
  Foxlight wheel index (`wheels.foxlight.ai`), a pinned `llama-server` build
  compiled from upstream llama.cpp source with sigstore build-provenance
  attestations. If that wheel is unavailable, it falls back to the
  `skulk-llama-server-vulkan` wheel (NVIDIA drives Vulkan fine on bare metal),
  and finally to the managed tarball build Skulk provisions itself at startup.
- **AMD Linux**: installs the `skulk-llama-server-vulkan` wheel, with the same
  managed-tarball fallback.
- **macOS**: needs nothing; Apple Silicon serves through in-process MLX.

Skulk also auto-provisions a pinned, checksum-verified `llama-server` at node
startup on Linux when no engine is configured. Setting
`SKULK_LLAMA_SERVER_BIN` to your own build always overrides, and
`SKULK_NO_ENGINE_AUTOPROVISION=1` opts a node out of auto-provisioning
entirely.

On an NVIDIA Linux node, the installer's `--with-vllm` flag additionally
installs vLLM: the concurrent-serving fast path on CUDA, where continuous
batching holds latency flat under many simultaneous clients. It lives in its
own virtual environment (vLLM's dependency matrix conflicts with Skulk's), and
the installer records `SKULK_VLLM_BIN` in `~/.skulk/skulk.env` so the served
vLLM engine is available to the node.

The manual paths below are for development: use them when you work on Skulk
itself or want control over each step.

## Recommended Contract

- `uv` is the canonical runtime path for Skulk on macOS.
- Nix is the canonical tooling and validation path for formatter, dev shell,
  and `flake`-based checks.
- Nix should match the `uv` runtime contract rather than silently swapping in a
  different MLX runtime.

Today that means both paths align on the same macOS MLX dependency contract:
the official `mlx` and `mlx-metal` wheel stack pinned by Skulk.

## What `uv` Does

Use `uv` when you want to run Skulk itself:

```bash
uv sync
uv run skulk
```

On macOS, this path uses the official `mlx` and `mlx-metal` wheel stack that
the project pins in [pyproject.toml](https://github.com/Foxlight-Foundation/Skulk/blob/main/pyproject.toml).
On the first run from a local interactive terminal, Skulk opens the dashboard
in the default browser. SSH, redirected, and service launches still print the
dashboard URL but do not open a GUI browser.

That means the runtime path is the one most users and nodes should follow.

## What Nix Does

Use Nix when you want reproducible development tooling:

```bash
nix develop
nix fmt
nix flake check
```

Nix gives us:

- a reproducible dev shell
- a consistent formatter entrypoint
- hermetic lint and typecheck checks
- a single place to express CI-oriented tooling

## Why This Matters

Some tooling historically carried a macOS Nix path that also changed how MLX
and Metal were built. That made Nix behave like a hidden "real" runtime path,
even though other docs implied that source installs and Nix installs were
equivalent.

Skulk intentionally avoids that ambiguity:

- the runtime contract lives with the `uv` environment
- the Nix shell exists to support development and validation around that same
  runtime contract

## Current macOS Guidance

For local development on Apple Silicon:

1. Install [`uv`](https://docs.astral.sh/uv/) (the Python toolchain manager) and
   clone the Skulk repo. `uv` provides the pinned Python and all dependencies, so
   you do not install Python separately.
2. From the repo root, run Skulk with `uv`:

   ```bash
   uv sync
   uv run skulk
   ```

   The API comes up at `http://localhost:52415`. To also serve the web dashboard,
   build it once: `cd dashboard-react && npm install && npm run build` (a headless
   node can skip this and serve the API without the UI).
3. Use Nix for `nix fmt`, `nix develop`, and `nix flake check`.

Once a node is up, the [API guide](api-guide) walks from placing a model to your
first token. If you are standing up nodes, treat `uv` as the path that must work first.
Treat Nix as a developer convenience and CI reproducibility layer unless the
project explicitly documents otherwise in a future release note.
