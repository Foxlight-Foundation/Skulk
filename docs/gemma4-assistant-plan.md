# Plan: Gemma 4 Assistant (Speculative Decoding) Support in Skulk

> Status: PROPOSED — research complete, no code written yet.
> Companion work in SWP (catalog + GUI) is already merged on `feature/ui-v1`.
> Last updated after confirming upstream mlx-vlm already implements the drafter.

---

## 1. Context

Gemma 4 uses a fundamentally different speculative-decoding mechanism than the
Qwen3 / DeepSeek models Skulk supports today. Those embed projection-only
`mtp.*` heads inside the checkpoint; SWP extracts them to an `mtp.safetensors`
sidecar and Skulk's `MTPHead` (Phase 1) consumes them.

Gemma 4 instead pairs each model with a **separate 4-layer transformer drafter**,
published by Google as `{model}-assistant`. It is NOT a projection head, so
Skulk's existing Phase-1 MTP path cannot consume it directly.

---

## 2. The assistant models (verified on HF)

All four exist, each a single `model.safetensors` + `config.json`,
`model_type: gemma4_assistant`, `dtype: bfloat16`:

| Target (bf16)                          | Assistant (bf16)                                | LM head           |
| -------------------------------------- | ----------------------------------------------- | ----------------- |
| `mlx-community/gemma-4-E2B-it-bf16`     | `mlx-community/gemma-4-E2B-it-assistant-bf16`     | centroid (sparse) |
| `mlx-community/gemma-4-E4B-it-bf16`     | `mlx-community/gemma-4-E4B-it-assistant-bf16`     | centroid (sparse) |
| `mlx-community/gemma-4-26B-A4B-it-bf16` | `mlx-community/gemma-4-26B-A4B-it-assistant-bf16` | tied dense        |
| `mlx-community/gemma-4-31B-it-bf16`     | `mlx-community/gemma-4-31B-it-assistant-bf16`     | tied dense        |

(Google's originals live at `google/gemma-4-*-it-assistant`; the
`mlx-community/*-bf16` conversions are the MLX-ready ones.)

### Architecture (from the real `config.json`)

- `architectures: ["Gemma4AssistantForCausalLM"]`, `dtype: bfloat16`
- `backbone_hidden_size: 2816` (the target's hidden dim, consumed as input)
- `text_config`: 4 layers, `hidden_size: 1024`,
  `layer_types: [sliding, sliding, sliding, full]`,
  `head_dim: 256` / `global_head_dim: 512`, `attention_k_eq_v: true`
- **`num_kv_shared_layers: 4` over 4 layers** — every layer is KV-shared. The
  drafter computes NO K/V of its own; it attends over the *target model's* KV
  cache (last full-attention layer + last sliding-attention layer). Its only
  recurrent state is the target's last hidden, projected via `post_projection`.
- Drafter input each step:
  `concat([target_embed(last_token), last_hidden], dim=-1)` shape
  `[B, 1, 2 * backbone_hidden_size]`, projected to drafter hidden by
  `pre_projection`.
- Queries RoPE-rotated at the bonus token's absolute position, held constant
  across draft steps within a block.
- E2B/E4B use a **centroid-routed sparse softmax** LM head: score 2048 clusters,
  materialise top-K (32) clusters (~4096 of 262144 tokens), scatter back into a
  full-vocab tensor. 26B/31B use a tied dense head.

---

## 3. KEY FINDING — upstream mlx-vlm already implements this

`Blaizzy/mlx-vlm` **v0.5.0** (released 2026-05-06) ships a complete, tested MLX
port of the Gemma 4 assistant drafter under `mlx_vlm/speculative/`:

```
mlx_vlm/speculative/
├── common.py              # shared draft/verify utilities
├── dflash.py
├── ddtree.py
├── eagle3.py              # (also has EAGLE3 drafters)
├── mtp.py
├── utils.py
└── drafters/
    ├── __init__.py        # load_drafter(), auto-discovery by model_type
    ├── eagle3/
    ├── qwen3_5_mtp/
    └── gemma4_assistant/
        ├── config.py            # Gemma4AssistantConfig (HF-compatible)
        ├── gemma4_assistant.py  # Gemma4AssistantDraftModel:
        │                        #   forward, bind, set_shared_kv,
        │                        #   draft_block, sanitize
        ├── masked_embedder.py   # centroid sparse LM head (E2B/E4B)
        ├── masks.py             # bidirectional full/SWA masks
        └── parity_check.py      # fake-target smoke test
```

- Auto-discovered by `model_type == "gemma4_assistant"`.
- Reference CLI:
  `python -m mlx_vlm.generate --model <target-bf16> --draft-model <assistant-bf16>
   --draft-kind mtp --draft-block-size 4 --temp 0`.
- `--draft-block-size` = `num_assistant_tokens` (first token is the accepted
  bonus, so it drafts `block_size - 1` candidates/round).
- Claims byte-identical greedy output vs the target at temp 0.

**Skulk is pinned to `mlx-vlm==0.4.4`** (`pyproject.toml`), whose installed wheel
has no `speculative/` module. So the drafter code exists upstream but is not yet
available to Skulk.

This collapses the original "Phase B greenfield drafter" risk: the hard math
(KV-shared attention, sparse centroid head, masks, RoPE-at-bonus-position) is
already written and parity-checked upstream. The Skulk job becomes **adopt +
integrate**, not **implement**.

---

## 4. Skulk today — what exists to build on

(Exact paths from codebase exploration.)

- **MTP framework (projection-only, Phase 1):**
  - `src/exo/worker/engines/mlx/mtp.py` — `MTPHead` (DeepSeek + Qwen3.5 layouts).
  - `src/exo/worker/engines/mlx/generator/generate.py:1167`
    `_stream_generate_with_mtp` — draft → batched verify → accept/reject →
    cache-trim. Single-node, greedy (temp 0.0) only.
  - `src/exo/worker/runner/llm_inference/runner.py:743` — `force_sequential_for_mtp`.
- **Sidecar loading:** `src/exo/worker/engines/mlx/utils_mlx.py:694` — downloads
  `mtp_sidecar_repo`, loads `mtp.safetensors` via `mx.load`.
- **Model-card runtime config:** `src/exo/shared/models/model_cards.py:232`
  `RuntimeCapabilityCardConfig` — `mtp_heads`, `mtp_sidecar_repo`, `mtp_max_depth`.
- **Gemma 4 base inference (works):** loaded via mlx-vlm
  (`utils_mlx.py:508 load_model`, mlx-vlm fallback). `mlx_vlm/models/gemma4/
  language.py` already implements KV-shared attention (`is_kv_shared_layer`
  reads `cache.state`), QK-norm, sliding/full layer types — the exact primitives
  the drafter depends on.
- **Gemma 4 detection:** `src/exo/shared/models/capabilities.py:82`
  `is_gemma4_family`.
- **No `gemma4_assistant`** in the installed mlx-vlm 0.4.4.

---

## 5. The gap, precisely

1. **Get the upstream drafter into Skulk.** Either bump `mlx-vlm` 0.4.4 → 0.5.0,
   or vendor `mlx_vlm/speculative/drafters/gemma4_assistant/` (+ its `masks.py`,
   `masked_embedder.py`, `common.py` deps).
2. **Drive the drafter from Skulk's own generation loop.** Skulk does NOT use
   mlx-vlm's generate loop — it has its own distributed `generate.py` with the
   Phase-1 accept/reject machinery. The drafter must be wired into that loop,
   feeding it the target's per-layer KV cache + last hidden each step (the one
   genuinely new data flow — today the cache only flows INTO the model).
3. **Model-card + catalog field agreement.** Add `assistant_model_repo` to the
   runtime config; SWP already emits `assistant_model_repo` on catalog entries.

---

## 6. Proposed approach

### Phase A — Model-card + loader plumbing (low risk, ~0.5 day)
- Add `assistant_model_repo: str | None` to `RuntimeCapabilityCardConfig`
  (`model_cards.py`), mutually exclusive with `mtp_sidecar_repo`.
- In `utils_mlx.py::load_mlx_items` (~694): when set, `build_model_path` +
  download the assistant alongside the target, load it, keep the handle on the
  bound instance next to the target model.
- Author Gemma 4 model cards pointing at the `mlx-community/*-assistant-bf16`
  repos. Reuse the SWP field name verbatim.

### Phase B — Adopt the upstream drafter via the mlx-vlm 0.5.0 bump (~1 day)

> **Decision (current, matches [`gemma4-mtp-initiative.md`](./gemma4-mtp-initiative.md)
> §decision-log): bump mlx-vlm to 0.5.0, do NOT vendor.** Reaching the bump
> requires the **full version ladder** (§6a), not the fork alone: reconciling the
> mlx-lm fork onto ≥0.31.3 is *necessary but not sufficient* — the bump also needs
> `mlx` 0.31.1→0.31.2, the `transformers<5.4.0` cap lifted past 5.5, and the new
> `llguidance` / `mlx-audio` deps. The fork was the gating blocker (it can't be
> dropped); once the whole ladder lands, mlx-vlm 0.5.0 installs cleanly and the
> drafter arrives as a **maintained dependency** (plus APC / continuous-batching /
> dflash). Vendoring is retained below only as the fallback if the bump proves
> problematic.

**Preferred path — bump:** after the fork is reconciled (step 2) and the version
ladder lands (step 3), mlx-vlm 0.5.0 brings
`mlx_vlm/speculative/drafters/gemma4_assistant/` directly. Wrap it behind a
`Drafter` protocol with `draft_block(...)`, alongside the existing `MTPHead`.

**Fallback path — vendor** (only if the bump is blocked for unforeseen reasons):
- Copy these 0.5.0 files into `src/exo/worker/engines/mlx/drafters/gemma4_assistant/`
  with a provenance header (upstream path + commit/tag `v0.5.0`):
  `gemma4_assistant.py`, `config.py`, `masked_embedder.py`, `masks.py`, `__init__.py`.
- Repoint the two relative imports
  (`from ....models.gemma4.config import TextConfig`,
  `from ....models.gemma4.language import DecoderLayer`) at Skulk's installed
  `mlx_vlm.models.gemma4` — confirmed present and structurally compatible (§6a).

### Phase 6a — Dependency audit (COMPLETE)

**The mlx-vlm 0.5.0 bump requires a version ladder — fork reconcile is necessary
but not sufficient.** All of the following must land together; the fork is the
gating item because it can't be dropped (it carries fixes not upstream):

| Constraint | Skulk pins | 0.5.0 requires | Action |
| ---------- | ---------- | -------------- | ------ |
| `mlx-lm` (fork) | fork @ d36e9b6 (0.31.2) | `>=0.31.3` | reconcile fork onto ≥0.31.3 (gating; step 2) |
| `transformers` | `>=5.0.0,<5.4.0` | `>=5.5.0` | lift cap (audited safe — Skulk uses only stable `Auto*.from_pretrained`) |
| `mlx` (darwin) | `==0.31.1` | `>=0.31.2` | bump one patch; re-verify macOS-26 Metal build |
| new deps | — | `llguidance`, `mlx-audio` | accept (added surface) |

**Gemma 4 vision wrapper — audited, NOT a blocker.** An earlier draft worried
that Skulk's `_Gemma4DynamicVisionTower` (which wraps mlx-vlm's gemma4 vision
internals) might break on 0.5.0. The audit resolved this: every attribute the
wrapper touches — `patch_embedder`, `encoder`, `pooler`, `std_bias`, `std_scale`,
and the call signatures — is unchanged between 0.4.4 and 0.5.0. The wrapper needs
no changes. (This matches the initiative tracker's decision log.)

**If the bump is undesirable, the drafter vendors cleanly** (the fallback path
in Phase B). Verified imports of the `gemma4_assistant` drafter (v0.5.0):

- `gemma4_assistant.py`: `mlx.core`, `mlx.nn`, `mlx.nn.RMSNorm`,
  `.config`, `.masked_embedder`, `.masks`, and
  `....models.gemma4.config.TextConfig` + `....models.gemma4.language.DecoderLayer`
- `config.py`: no imports (pure config)
- `masked_embedder.py`: `mlx.core`, `mlx.nn`
- `masks.py`: `mlx.core` + `mlx_lm.models.cache.dynamic_roll`

Every external symbol is already available in Skulk's installed environment:
- `mlx_lm.models.cache.dynamic_roll` — ✅ present (Skulk's mlx-lm 0.31.2).
- `mlx_vlm.models.gemma4.config.TextConfig` + `.language.DecoderLayer` — ✅
  present in installed 0.4.4, including the `is_kv_shared_layer` /
  `num_kv_shared_layers` shared-KV machinery the drafter relies on; class layout
  matches 0.5.0 closely (`DecoderLayer` ~line 244/246, `Attention` shared-KV
  block near-identical).
- No `transformers`, no `mlx-vlm` internals beyond `gemma4` config/layer, no new
  third-party deps.

**Residual verification (do during Phase B):** diff the 0.4.4 vs 0.5.0
`gemma4/language.py` `DecoderLayer`/`Attention` bodies to confirm the drafter
doesn't depend on a 0.5.0-only tweak. If a small delta exists, vendor the
0.5.0 `DecoderLayer`/`TextConfig` too (still pure MLX, no dep changes). Structural
match makes this low-probability.

### Phase C — Generation integration (the real engineering, ~2 days)
- Generalize `_stream_generate_with_mtp` into a drafter-agnostic loop: keep the
  accept/reject + cache-trim machinery; swap the single `MTPHead.draft()` call
  for the `Drafter` protocol (impls: `MTPHead`, `Gemma4AssistantDrafter`).
- Implement the new data flow: expose the target's per-layer KV cache + the
  chosen-layer hidden state to the drafter each step; call `set_shared_kv` /
  `draft_block` per the upstream API.
- Support block drafting (`draft_block_size > 1`) — upstream drafts several
  tokens/round, vs Phase-1's D=1. Verify Skulk's batched-verify + cache-trim
  handles a block, not just a single draft token.
- Keep single-node + greedy (temp 0) for v1 — same envelope as Phase-1 MTP.

### Phase D — Validation + docs (~1 day)
- Unit tests mirroring `tests/test_mtp.py` for the drafter path (accept/reject,
  block trim, bf16 dtype preservation, sparse-head argmax parity).
- Port/adapt upstream `parity_check.py` as a smoke test.
- End-to-end: `gemma-4-26B-A4B-it-bf16` + assistant, confirm greedy output is
  identical to no-drafter and measure tokens/s speedup.
- Update Skulk architecture docs + model-card reference with the assistant path.

---

## 7. Risks / open questions

- ~~mlx-vlm 0.5.0 bump blast radius.~~ RESOLVED — audit complete (§6a). The
  mlx-lm fork (pinned at 0.31.2) was the **gating** blocker, but unblocking the
  bump also requires the rest of the version ladder: `mlx` 0.31.2, the
  `transformers` cap lifted past 5.5, and the new `llguidance` / `mlx-audio`
  deps. Reconciling the fork is necessary but not sufficient. The gemma4 vision
  wrapper was audited as safe (unchanged in 0.5.0). Vendoring is the documented
  fallback only.
- **KV-cache exposure refactor** is the riskiest change — it touches the hot
  generation loop. Gate it behind the existing MTP single-node guard so the
  default path is untouched.
- **Block drafting vs Phase-1 D=1.** Skulk's current verify/trim assumes one
  draft token; the assistant drafts a block. Confirm the batched verify
  generalizes.
- **MoE target (`26B-A4B`)** KV-share layer indexing must line up with the MoE
  layer layout, not just the dense `31B`. Validate both.
- **Distributed/pipeline mode** stays out of scope for v1 (Phase-1 MTP is
  single-node too).

---

## 8. Effort estimate (revised down after upstream finding)

| Phase | Work | Est. |
| ----- | ---- | ---- |
| A | Model-card + loader plumbing | ~0.5 day |
| B | Adopt drafter via mlx-vlm 0.5.0 bump (fork reconciled first) | ~1 day |
| C | Generation-loop integration + KV exposure | ~2 days |
| D | Tests + parity + docs | ~1 day |

**Total ~4–5 days** for a single-node, greedy v1 (was ~1 week when Phase B was
assumed greenfield). Phase C — exposing the target's KV cache to the drafter in
Skulk's generation loop — is now unambiguously the critical path.

---

## 9. Recommendation

1. Land **Phase A** first — cheap, unblocks model cards, makes SWP↔Skulk field
   names agree.
2. **Reconcile the mlx-lm fork onto ≥0.31.3** (critical-path step 2) — this is
   what unblocks the mlx-vlm 0.5.0 bump that delivers the drafter. The audit
   (§6a) confirmed the bump is otherwise clean. Vendoring stays as fallback only.
3. Phase C is the genuine engineering; everything the drafter needs
   mathematically already exists upstream and is parity-checked, so the focus is
   wiring it into Skulk's loop, not reimplementing model math.

## 10. References

- Upstream drafter: `Blaizzy/mlx-vlm` v0.5.0,
  `mlx_vlm/speculative/drafters/gemma4_assistant/`.
- Google docs: ai.google.dev/gemma/docs/mtp/mtp.
- HF assistants: `google/gemma-4-{E2B,E4B,26B-A4B,31B}-it-assistant`;
  MLX bf16: `mlx-community/gemma-4-*-it-assistant-bf16`.
- Skulk integration points: see §4 for exact file:line anchors.
- SWP companion work: `skulk-weights-publisher` `feature/ui-v1`
  (`assistant_model_repo` catalog field + GUI "Register in Catalog").
