# Initiative: Gemma 4 MTP Speculative Decoding

> **Master tracker.** Source of truth for the whole effort across SWP + Skulk +
> the mlx-lm fork. Start here to re-orient. Implementation detail lives in
> [`gemma4-assistant-plan.md`](./gemma4-assistant-plan.md).
> Last updated: 2026-05-31.

---

## North star

A user points Skulk at a Gemma 4 model and gets **speculative decoding via the
Gemma 4 assistant drafter** — faster tokens, identical greedy output. SWP
catalogs the pairing; Skulk consumes it.

---

## Why this is non-trivial (the one-paragraph briefing)

Gemma 4 does MTP differently from Qwen3/DeepSeek. Those embed `mtp.*` heads in
the checkpoint (Skulk's existing Phase-1 MTP handles them). Gemma 4 ships a
**separate 4-layer assistant model** that attends over the *target's* KV cache.
The drafter already exists, maintained, in **mlx-vlm 0.5.0** — so the clean path
is to bump to it. But Skulk pins **mlx-vlm 0.4.4**, and bumping is gated by a
**custom mlx-lm fork** that carries fixes we can't lose. Untangling that fork —
owning it, then reconciling it onto a newer base — is the critical path.

---

## Status dashboard

| Workstream | State | Notes |
|---|---|---|
| **SWP — catalog + GUI** | ✅ DONE | `assistant_model_repo` detection, GUI "Register in Catalog", docs. Branch `feature/ui-v1`. 119 py + 45 ui tests green. |
| **mlx-lm fork — custody** | ✅ DONE | Owned at `Foxlight-Foundation/mlx-lm`, pinned by rev `d36e9b6`. Branch `chore/mlx-lm-custody-foxlight` (committed, not pushed). |
| **mlx-lm fork — reconcile onto ≥0.31.3** | ⬜ NEXT | Forward-port the fixes; functionally verify each survives. |
| **Version ladder (mlx, mlx-vlm, transformers)** | ⬜ TODO | Bump mlx 0.31.1→0.31.2, mlx-vlm 0.4.4→0.5.0, lift transformers cap. |
| **Skulk — consume the drafter** | ⬜ TODO | Model-card `assistant_model_repo`, loader, wire drafter into the generation loop (expose target KV cache). |
| **Validate end-to-end** | ⬜ TODO | Greedy parity + speedup on gemma-4-26B-A4B + assistant. |

---

## Decision log (with rationale)

1. **Bump mlx-vlm, don't vendor the drafter.** The drafter is maintained upstream
   in mlx-vlm 0.5.0; vendoring means owning code that drifts. Bumping gets it as a
   maintained dep + APC/continuous-batching/dflash for free. *(Earlier I leaned
   vendor based on overstated blockers — corrected after the dependency audit.)*

2. **The transformers `<5.4.0` cap is soft.** Undocumented snapshot; mlx-vlm 0.4.4
   itself only needs ≥5.1.0; Skulk uses only stable `Auto*.from_pretrained` APIs;
   no 5.4→5.9 breaking change touches them. Liftable with a 3-area test.

3. **The mlx-vlm 0.5.0 bump does NOT break Skulk's Gemma 4 vision wrapper.**
   Audited — every internal `_Gemma4DynamicVisionTower` touches is unchanged in
   0.5.0. (I had flagged this as a risk; it isn't.)

4. **The real gate is the mlx-lm fork**, stuck at version 0.31.2 < the 0.31.3
   mlx-vlm 0.5.0 requires — and it can't be dropped (carries non-upstream fixes
   Skulk depends on).

5. **Own the fork; do not upstream.** (Per Tupp: upstreaming isn't viable; reconcile
   our own fork instead.) Forked `ml-explore/mlx-lm` → `Foxlight-Foundation/mlx-lm`
   (upstream lineage preserved), snapshotted the consumed commits in.

6. **Custody-first, by exact rev, before any version move.** Pin the precise
   consumed commit `d36e9b6` (not the branch — its tip moved on and needs a newer
   mlx). Pure ownership change, zero behavior delta. Reconciliation is a separate,
   tested step. This is the "don't lose the fixes / don't regress" discipline.

7. **EXO does not depend on this fork.** Upstream EXO uses stock `mlx-lm` (it forks
   *core mlx* instead, which Skulk deliberately does not use). So owning/reconciling
   our mlx-lm fork won't desync from EXO.

---

## The fixes we must never lose (carried in `d36e9b6`)

Verified present in `Foxlight-Foundation/mlx-lm`:
- **float32 logprobs** — cast before log_softmax; without it logprobs quantize and
  break sampling parity (Skulk exposes `top_logprobs`).
- **DeepSeek-V3.2 lightning-indexer batch fix** — masks padded positions; without
  it batched DSV3.2 attention corrupts.
- **ArraysCache leak fix** — `mx.depends(...)` keeps Metal caches in-graph; without
  it GPU memory leaks on long runs.
- **GDN precision**, **left_padding eval** — supporting precision/correctness.

Each must be **functionally** re-verified (the actual call survives, not just a
clean merge) when reconciling onto a newer base.

---

## Critical path (ordered)

- [x] **0. SWP catalog/GUI support** — done (`feature/ui-v1`).
- [x] **1. Own the mlx-lm fork (custody, rev-pinned)** — done
      (`chore/mlx-lm-custody-foxlight`, `Foxlight-Foundation/mlx-lm`).
- [ ] **2. Reconcile the fork onto upstream ≥0.31.3** — cherry-pick/forward-port
      the fixes; functionally verify each; bump fork `_version.py` past 0.31.3.
- [ ] **3. Version ladder** — mlx 0.31.1→0.31.2 (re-verify macOS-26 Metal build),
      lift transformers cap, mlx-vlm 0.4.4→0.5.0; `uv lock`; smoke-test.
- [ ] **4. Skulk consumes the drafter** — model-card `assistant_model_repo`,
      loader plumbing, wire `Gemma4AssistantDraftModel` into the generation loop
      (expose target per-layer KV cache + last hidden each step).
- [ ] **5. Validate** — greedy parity vs no-drafter + tokens/s speedup on
      gemma-4-26B-A4B-it + its assistant.

---

## Key references

- **Fork (ours):** `github.com/Foxlight-Foundation/mlx-lm` — branch
  `foxlight/fix-arrayscache-leak`, tag `foxlight-consumed-d36e9b6`, consumed rev
  `d36e9b661e55a5fc0f77fb6f17ea643aa2dc87aa`.
- **Upstream drafter:** `Blaizzy/mlx-vlm` v0.5.0,
  `mlx_vlm/speculative/drafters/gemma4_assistant/`.
- **Assistant weights:** `mlx-community/gemma-4-{E2B,E4B,26B-A4B,31B}-it-assistant-bf16`.
- **SWP companion:** `skulk-weights-publisher` `feature/ui-v1` (`assistant_model_repo`).
- **Skulk integration anchors + architecture detail:** `gemma4-assistant-plan.md`.

---

## Open questions

- Reconcile base: upstream `0.31.3` tag vs current `main` (further ahead). Pick the
  lowest base that satisfies mlx-vlm 0.5.0 and still builds with our target mlx.
- Does mlx 0.31.2 build on macOS 26 Metal SDK under our setup? (mlx-lm fork patches
  touch Metal kernel paths — re-verify.)
- Block drafting (`draft_block_size > 1`) vs Skulk's current D=1 verify/trim — does
  the existing accept/reject machinery generalize to a block?
