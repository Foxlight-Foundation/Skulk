---
id: speculative-decoding
title: Speculative Decoding (MTP)
sidebar_position: 6
---

<!-- Copyright 2025 Foxlight Foundation -->

This guide explains Skulk's speculative decoding feature from an operator's
point of view: what it does, which models use it, what speedups to expect,
when it turns itself off, and how to confirm it is working.

The short version:

- speculative decoding makes supported models generate faster, for free
- it is **automatic** — there is nothing to configure
- it pays off the most for dense models sharded across multiple nodes
- it deliberately turns itself off in a few honest cases (see below)
- you can verify it is active from the runner logs

## What It Is

Speculative decoding (we also call it **MTP**, for multi-token prediction) is
a way to generate several tokens per model step instead of one. A small,
cheap "drafter" proposes a short run of likely next tokens, and the full
model verifies all of them in a single forward pass, keeping the longest
correct prefix. When the drafter guesses well, you get multiple tokens for
roughly the cost of one — and because the verify step accepts or rejects
against the real model, **the output quality is the model's own**: greedy
requests produce a valid greedy continuation, and sampled requests preserve
the model's output distribution exactly. It is a pure speedup, not a
quality trade-off. (On hybrid-state Qwen models the greedy text can differ
token-for-token from a non-speculative run while remaining equally greedy —
see Known Limitations.)

Skulk runs speculative decoding on single-node, tensor-parallel, and
pipeline (sharded) placements through one shared decode loop. You do not
enable it, size it, or tune it for normal use: if a model ships with a
drafter and the placement supports it, it activates on its own.

## Two Engines: MLX and Served (llama.cpp)

MTP runs on both of Skulk's speculative-capable engines, and placement routes
each model to the right one from its card:

- **MLX** (Apple Silicon, in-process): the drafters and models in the table
  below. Skulk owns the generation loop, the multi-node ring, and the
  speculative decode across sharded placements. This is the engine the speedup
  numbers and multi-node discussion on this page describe.
- **Served / `llama_server`** (GPU nodes, including AMD): llama.cpp's **native**
  MTP, reached by launching `llama-server --spec-type draft-mtp` and proxying it
  (native MTP lives in the server app, not the in-process binding). This is how
  speculative decoding works on an AMD node. It is single-node and applies to
  GGUF models whose card declares `served_spec_type = draft_mtp`, in two shapes:
  baked-in MTP heads (Qwen3.5 / Qwen3.6 MTP GGUFs) and a base plus a separate
  `--model-draft` GGUF (Gemma 4 31B). See
  [AMD Strix Halo nodes](amd-strix-halo-nodes.md) for enabling it
  (`SKULK_LLAMA_SERVER_BIN`) and the served MTP cards.

Everything below (the MLX drafter table, the multi-node speedups, and the
turn-itself-off cases) is about the MLX engine. The served engine's speculation
is configured on the model card (`served_spec_type`, `served_spec_n_max`) and,
like MLX MTP, is carded off per model when a pairing does not pay.

### Served-engine speedups (AMD / llama.cpp)

Measured on an AMD Ryzen AI Max+ 395 (Radeon 8060S, `gfx1151`) serving GGUF
models through the Vulkan `llama-server`, native MTP on (`--spec-type draft-mtp`)
versus off. Both arms run through Skulk's production API with the same protocol
as the MLX table: greedy decoding, 200-token completions, median of 3 runs, with
throughput in decode tokens per second; the off arm is the identical GGUF served
in plain decode (the node's `SKULK_LLAMA_SERVER_FORCE_NO_SPEC` benchmarking knob),
so the gain is attributable to speculation alone.

| Model | Class | Plain | With MTP | Gain |
|---|---|---:|---:|---:|
| `Qwen3.5-9B-MTP` | dense, small | 55.6 | 76.2 | **+37%** |
| `Qwen3.6-27B-MTP` | dense, mid | 20.0 | 35.6 | **+78%** |
| `Qwen3.6-35B-A3B-MTP` | MoE (A3B) | 90.7 | 95.8 | **+6%** |
| `gemma-4-31B` (+ draft) | dense, draft-model | 17.4 | 25.2 | **+45%** |

The shape mirrors the MLX results: the dense mid-size model gains the most
(+78%), because its slower base decode gives speculation the most to amortize;
the MoE model gains the least (+6%), because its small active-parameter count
already makes decode memory-bound-fast, so the per-round draft and verify
overhead nets little. The Gemma row uses the other MTP shape (a separate
`--model-draft` GGUF rather than baked-in heads) and still pays (+45%),
confirming both served MTP shapes work on the Radeon backend.

## Which Models Ship With It

These models carry a drafter in their model card and use speculative
decoding automatically. "Sidecar" drafters are MTP heads trained alongside
the model; "assistant" drafters are a small companion model that
cross-attends the target's cache.

| Model | Drafter | Type | Depth |
|---|---|---|---|
| `mlx-community/gemma-4-e2b-it-8bit` | `gemma-4-E2B-it-assistant-bf16` | assistant | 2 |
| `mlx-community/gemma-4-e4b-it-8bit` | `gemma-4-E4B-it-assistant-bf16` | assistant | 2 |
| `mlx-community/gemma-4-12B-it-4bit` | `gemma-4-12B-it-assistant-bf16` | assistant | 2 |
| `mlx-community/gemma-4-31b-it-4bit` | `gemma-4-31B-it-assistant-bf16` | assistant | 2 |
| `mlx-community/gemma-4-26b-a4b-it-4bit` | `gemma-4-26B-A4B-it-assistant-bf16` | assistant | 1 (single-node only) |
| `mlx-community/Qwen3.5-9B-MLX-4bit` | `FoxlightAI/qwen3-5-9b-base-mtp` | sidecar | 1 |
| `mlx-community/Qwen3.5-27B-4bit` | `FoxlightAI/qwen3-5-27b-mtp` | sidecar | 1 |
| `mlx-community/Qwen3.6-27B-4bit` | `FoxlightAI/qwen3-6-27b-mtp` | sidecar | 1 |
| `mlx-community/Qwen3.5-2B-4bit` | `FoxlightAI/qwen3-5-2b-base-mtp` | sidecar | 1 |

The drafter weights are companion repos. Skulk fetches and stages them
alongside the target model — you do not download or reference them directly.

## What Speedups To Expect

The numbers below were measured on **M4-base nodes** (the kites), which are
the *lowest-bandwidth Apple Silicon currently being manufactured* and so are
a deliberately worst-case platform for absolute throughput. Read the
**ratios** as the portable result: absolute tok/s scales almost linearly
with memory bandwidth, so the same build on an M4 Pro/Max prints 2–4.5× the
absolute numbers below with no code changes, while the speedup ratio stays
roughly the same.

Protocol: production API, greedy decoding, 200-token completions, median of
3 runs per arm on the same live instance.

| Configuration | Hardware | Plain | With MTP | Gain |
|---|---|---|---|---|
| gemma-4-E2B-8bit, single node | M4 24GB | 37.7 | 54.0 | **+43%** |
| gemma-4-E4B-8bit, single node | M4 24GB | 19.5 | 25.4 | **+30%** |
| Qwen3.5-9B-MLX-4bit, single node | M4 24GB | 21.3 | 28.8 | **+35%** |
| gemma-4-12B-4bit, 2-node pipeline | 2× M4 16GB | 8.4 | 15.1 | **+81%** |
| gemma-4-31B-4bit dense, 2-node pipeline | 2× M4 16GB | 5.3 | 7.35 | **+38%** |
| Qwen3.5-27B-4bit dense, 2-node pipeline | 2× M4 16GB | 6.3 | 10.5 | **+67%** |
| Qwen3.5-9B-MLX-4bit, 2-node tensor-parallel | 2× M4 16GB | 16.7 | 21.8 | **+31%** |

These ratios hold up under longer generations and sampling. At 1000 tokens
the 12B 2-node pipeline still measures +60% (8.3 → 13.3) and Qwen 9B single
still +28% (21.4 → 27.4); at temperature 0.7 the 12B pipeline is +54% and
Qwen 9B is +21%. The 200-token greedy table is not flattering the feature by
much.

For external context: production native-MTP serving on datacenter GPUs
lands in the 1.3–1.8× band; Skulk measures 1.35× single-node and 1.81×
on a 2-node pipeline — at the top of that band, on far slower hardware, and
the pipeline figure beats published distributed-speculation results on
comparable clusters.

## Where It Shines: Dense Models Sharded Across Nodes

The biggest wins are dense models split across a pipeline (the +67% to +81%
rows above). When a model is sharded, every decoded token has to cross the
inter-node links, and that hop latency is what makes pipelined decode slow.
A speculative round crosses those hops **once regardless of how many tokens
it verifies**, so every accepted draft amortizes exactly the latency that
sharding adds. This is also the favourable case in general: the bigger the
target model relative to the nodes it runs on, the more speculation pays —
which is precisely Skulk's cluster pitch of running a big model across
several smaller machines.

A practical corollary: shard to the **smallest** node count that fits the
model. Over-sharding costs MTP headroom because each verify round then pays
an extra network traversal — the 31B drops from +38% on 2 nodes to +17% on
3 nodes. Skulk's placement already prefers the smallest cycle that fits, so
the default does the right thing.

## Where It Turns Itself Off (And Why)

Speculative decoding is honest about when it does not help. In these cases
Skulk falls back to plain decode rather than slowing you down:

- **Multi-node MoE placements.** Sparse (mixture-of-experts) models like
  `gemma-4-26b-a4b-it-4bit` already decode fast when sharded, because
  sharding halves the active-parameter bandwidth bottleneck. At that point
  the per-round draft+verify overhead nets slightly negative — measured
  −7% (30.2 → 28.2 tok/s) on a 2-node pipeline. The card gates this with
  `speculative_multi_node = false`, so these models run plain decode when
  sharded but **keep speculation on a single node**, where the same model
  measures ~2.2× (16 → 35.1 tok/s).
- **Sampled requests at higher depth.** Any request with `temperature > 0`
  forces draft depth to 1. Acceptance under sampling still preserves the
  output distribution exactly, but deeper chains stop paying, so the loop
  caps depth automatically.
- **Requests with repetition penalties.** A request that sets a repetition
  penalty disables speculation for that request. This only affects the
  individual request that asked for the penalty.

## How To Verify It Is Active

The simplest signal is the runner log. While a supported model generates,
the drafting rank periodically emits an acceptance line:

```
MTP acceptance so far: 137/180 (76%)
```

A non-zero, healthy acceptance rate (typically ~50–97% on the shipped
models) means speculation is running and paying off. The other signal is
throughput: compare the runner's `generated N tokens @ X tok/s` figure for a
supported model against the plain-decode numbers in the table above.

## Tuning

Draft **depth** — how many tokens the drafter proposes per round — is a
per-model field on the model card (`mtp_max_depth`), set from direct
measurement on each carded artifact. The shipped defaults are measured
optima:

- Gemma assistant cards use **depth 2**
- Qwen sidecar (GDN/SSM) cards use **depth 1**

You *can* override depth with a custom model card, but the defaults are not
guesses — they are the measured best for each model. Deeper is not better:
on this hardware, verifying up to 2 candidates per step is effectively free,
but each additional candidate beyond that costs a meaningful fraction of a
full forward pass (the "verify-width cliff"). Past depth 2 the extra width
costs more than the declining odds of the deeper guesses being accepted pay
back — so a larger depth can be measurably *slower*, not faster. Trust the
shipped values unless you are running your own depth sweep on your own
hardware.

## Known Limitations

- **Speculative decoding currently runs one generation at a time.** Models
  with an active drafter use a sequential generator, so concurrent requests
  to that model queue and run strictly first-in-first-out. (Gemma 4 models
  use the sequential path regardless of speculation, so this applies to
  every model in the table above; models outside these constraints batch
  concurrent requests.) The queueing is correct and stable — a 4-way
  concurrent test completed cleanly in FIFO order with no failures and no
  interleaving — but it means throughput on these models does not currently
  scale with concurrent callers.
- **Non-streaming errors return a truncated body.** A non-streaming request
  that fails part-way through generation terminates promptly but returns an
  empty or truncated body under a `200` status (the status line is already
  on the wire when the failure lands) rather than a clean error document.
  Treat an unparseable non-streaming body as a failure and retry — or use
  streaming requests, which surface first-class error events.
- **Greedy MTP output on Qwen GDN models is semantically greedy but not
  byte-identical** to the same model decoding without speculation. The text
  is a valid greedy generation; it may differ token-for-token from the
  non-MTP path.
