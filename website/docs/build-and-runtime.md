---
id: build-and-runtime
title: Build And Runtime Paths
sidebar_position: 3
---

<!-- Copyright 2025 Foxlight Foundation -->

Skulk supports both `uv` and Nix in development, but they do not have the same
job.

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

Upstream exo historically carried a macOS Nix path that also changed how MLX
and Metal were built. That made Nix behave like a hidden "real" runtime path,
even though other docs implied that source installs and Nix installs were
equivalent.

Skulk intentionally avoids that ambiguity:

- the runtime contract lives with the `uv` environment
- the Nix shell exists to support development and validation around that same
  runtime contract

## Current macOS Guidance

For local development on Apple Silicon:

1. Install the normal runtime prerequisites.
2. Run Skulk with `uv`.
3. Use Nix for `nix fmt`, `nix develop`, and `nix flake check`.

If you are standing up nodes, treat `uv` as the path that must work first.
Treat Nix as a developer convenience and CI reproducibility layer unless the
project explicitly documents otherwise in a future release note.
