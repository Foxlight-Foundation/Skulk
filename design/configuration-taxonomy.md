---
id: configuration-taxonomy
title: Configuration Taxonomy and Re-homing
sidebar_position: 96
---

<!-- Copyright 2025 Foxlight Foundation -->

# Configuration Taxonomy and Re-homing

This is a design record for reorganizing Skulk's configuration surface. Today
that surface is roughly sixty `SKULK_*` environment variables, most inherited
from the exo era and accreted during debugging. They are read at ~116 sites
across ~21 files, with no shared parsing rules and no categorization.

The original framing (issue: "typed env config migration") was to wrap the env
reads in a typed object. That is the wrong fix: it would put a clean coat of
paint on a grab bag without addressing why the grab bag exists. The real problem
is that three independent properties of every knob were collapsed into one
mechanism (a launch-time environment variable):

- **Owner / scope**: who legitimately sets it: the node, the cluster operator,
  the model, or the request.
- **Lifecycle**: whether it is fixed at launch or should be changeable at
  runtime and retained.
- **Home**: where it should actually live: a model card, cluster settings
  (already synced over gossipsub and surfaced in the dashboard), node-local
  launch config, an API request parameter, or nowhere (dev-only / dead).

When you classify the real knobs by *home*, only about a third are genuinely
node configuration. The rest are model behavior or cluster policy that escaped
into env vars because env vars were the only mechanism that existed when the
code was written.

## What is and is not a config knob

A few `SKULK_`-prefixed names are not environment variables and are out of scope:

- `SKULK_RUNNER_MUST_FAIL` / `SKULK_RUNNER_MUST_OOM` /
  `SKULK_RUNNER_MUST_TIMEOUT` are magic strings matched in prompt **content**
  for fault injection (`batch_generator.py`), not env vars.
- `SKULK_MAX_CHUNK_SIZE`, `SKULK_MODELS_DIRS`, `SKULK_MODELS_READ_ONLY_DIRS` are
  internal Python constants derived in `constants.py`, not env reads.

## The six homes

### A. Filesystem layout (keep as node env; already centralized)

Deployment layout, not user-facing knobs. Already resolved in one place
(`shared/constants.py`) with XDG support. Leave as env; the only cleanup is to
make sure new code reads the `constants.py` value rather than calling
`os.environ` again.

`SKULK_HOME`, `SKULK_CONFIG_HOME`, `SKULK_DATA_HOME`, `SKULK_CACHE_HOME`,
`SKULK_CONFIG_FILE`, `SKULK_NODE_ID_KEYPAIR`, `SKULK_MODELS_DIR`,
`SKULK_MODELS_PATH`, `SKULK_CUSTOM_MODEL_CARDS_DIR`, `SKULK_EVENT_LOG_DIR`,
`SKULK_IMAGE_CACHE_DIR`, `SKULK_TRACING_CACHE_DIR`, `SKULK_LOG`, `SKULK_LOG_DIR`,
`SKULK_RESOURCES_DIR`, `SKULK_DASHBOARD_DIR`, `SKULK_RUNTIME_DIR`,
`SKULK_VECTOR_DATA_DIR`.

### B. Per-model behavior (move to configurable model cards)

These depend on the *model*, not the node. Several already have model-card
fields; the env var is a debugging override that leaked out. This is the
largest mis-homing.

| Var | Controls | Card home today |
|---|---|---|
| `SKULK_KV_CACHE_BACKEND` | KV cache backend selection | `runtime` (partial; env overrides) |
| `SKULK_KV_CACHE_BITS` | mlx_quantized bit width | none yet |
| `SKULK_TQ_K_BITS` / `SKULK_TQ_V_BITS` / `SKULK_TQ_FP16_LAYERS` | TurboQuant params | none yet |
| `SKULK_OPTIQ_BITS` / `SKULK_OPTIQ_FP16_LAYERS` | OptiQ params | none yet |
| `SKULK_FAST_SYNCH` | Metal fast-synch pin | `runtime.metal_fast_synch` exists |
| `SKULK_MODEL_LOAD_TIMEOUT` | per-load timeout | none yet |
| `SKULK_GEMMA4_MAX_SOFT_TOKENS` / `SKULK_GEMMA4_IMAGE_ONLY_MAX_SOFT_TOKENS` | Gemma 4 vision token budget | none yet |
| `SKULK_PIPELINE_EVAL_TIMEOUT_SECONDS` | pipeline eval timeout (model-shape dependent) | none yet |

### C. Cluster policy (move to runtime Settings: gossipsub-synced, UI-exposed, retained)

Fleet-wide policy that an operator should change at runtime without restarting.
The mechanism already exists: logging settings are configured this way today
(synced over gossipsub, editable in dashboard Settings). These never got moved.

`SKULK_TRACING_ENABLED`, `SKULK_LOGGING_EXTERNAL`, `SKULK_LOGGING_INGEST_URL`,
`SKULK_OFFLINE`, `SKULK_ENABLE_IMAGE_MODELS`, `SKULK_NO_BATCH`,
`SKULK_MAX_CONCURRENT_REQUESTS`, `SKULK_GROUP_CONNECT_DEADLINE_SECONDS`,
`SKULK_WARMUP_DEADLINE_SECONDS`.

### D. Node-local launch config (keep as env; this is where a typed spine belongs)

Genuinely machine-specific: this box's network identity, its built backends, its
hardware limits. Fixed at launch. Type and validate these centrally (the
salvageable core of the original typed-config idea), but they stay env.

`SKULK_LIBP2P_NAMESPACE`, `SKULK_LIBP2P_PORT`, `SKULK_BOOTSTRAP_PEERS`,
`SKULK_ZENOH_DATA_PLANE`, `SKULK_ZENOH_LISTEN`, `SKULK_ZENOH_CONNECT`,
`SKULK_LLAMA_CPP_BACKENDS` (what this node was *built* with), `SKULK_MEMORY_THRESHOLD`,
`SKULK_NODE_PARTICIPATION` (candidate for a per-node UI setting later).

### E. Per-request (API parameter / cluster default)

Properties of a single request. `SKULK_MAX_OUTPUT_TOKENS` is really a cluster
*default* (home C) with a per-request override (already an API field). The rest
are request-scoped behavior currently smuggled through env.

`SKULK_MAX_OUTPUT_TOKENS` / `SKULK_MAX_TOKENS` (default in C, override per
request), `SKULK_TRACE_THINKING_STREAM`, `SKULK_TEXT_IMAGE_HASH_CACHE`.

### F. Dev / test only (quarantine, never a user knob)

Fault injection, debug tracing, and test scaffolding. Keep as env but namespace
them clearly as developer-only (e.g. a `SKULK_DEV_*` prefix or a documented
"unsupported" section) so they never appear in operator-facing config.

`SKULK_MLX_HANG_DEBUG`, `SKULK_MLX_HANG_DEBUG_INTERVAL_SECONDS`,
`SKULK_SKIP_LLM_WARMUP`, `SKULK_DEBUG_WARMUP_REPEAT_COUNT`,
`SKULK_DEBUG_WARMUP_INCLUDE_INSTRUCTIONS`, `SKULK_TRACE_REQUEST_SHAPES`,
`SKULK_VISION_DEBUG_SAVE_DIR`, `SKULK_NATIVE_VISION_REFERENCE_PATH`,
`SKULK_IMAGE_TRANSPORT_DEBUG`, `SKULK_DATA_REORDER_BUFFER` (transport
belt-and-suspenders override), `SKULK_TEST_DISTRIBUTED_MODEL`, `SKULK_TEST_LOG`.

## Summary of the shape

| Home | Count (approx) | Action |
|---|---|---|
| A. Filesystem layout | ~18 | Keep (already centralized); read constants, not env |
| B. Per-model behavior | ~12 | Move to model-card fields |
| C. Cluster policy | ~9 | Move to runtime Settings (gossipsub + UI, retained) |
| D. Node-local launch | ~9 | Keep as env; type + validate centrally |
| E. Per-request | ~4 | API parameter (+ cluster default for max tokens) |
| F. Dev / test only | ~12 | Quarantine under a developer-only namespace |

The headline: only homes A + D (~27) are truly node configuration. Homes B and C
(~21) are model behavior and cluster policy that should never have been env vars,
and home F (~12) should be invisible to operators.

## Re-homing roadmap

Sequenced so each step is independently shippable and the cluster keeps running
between steps. No mixed-version clusters, so each step is a coordinated upgrade.

1. **Card-ize per-model knobs (home B).** Add the missing model-card fields,
   make the resolver prefer card value, keep the env var as a temporary,
   logged-as-deprecated override. Per-model is the largest and highest-value
   bucket and it builds on the existing capability spine.
2. **Promote cluster policy to Settings (home C).** Extend the existing
   gossipsub-synced Settings object + dashboard Settings UI to cover these,
   with persistence. Env becomes a bootstrap default only.
3. **Type the node-local remainder (home D).** A small frozen settings object
   (the salvageable core of the original typed-config idea) reads and validates
   only the ~9 genuinely node-local vars, fail-loud on bad values.
4. **Quarantine dev/test (home F).** Move behind a clearly developer-only
   namespace; drop from operator docs.
5. **Generate the env reference.** The architecture reference env table is
   hand-maintained and drifts; once homes D and F are typed/centralized, generate
   that table from the definitions.

Each step removes env-var read sites; the count is the progress metric. The
original "type all the env reads" task collapses into step 3 once B and C have
moved the knobs that did not belong in env in the first place.
