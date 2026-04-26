# Gemma-4 Hang Repro Harness

Drives a running Skulk cluster with a deterministic sequence of multimodal
chat turns and reports per-turn outcomes. Built to validate fixes for the
gemma-4 / ring / multinode hang where the GPU command queue wedges and
trips the macOS userspace watchdog into a kernel panic.

## Quick start

```bash
# Cluster must already be running and reachable.
uv run python bench/repro_gemma4_hang.py \
    --api-url http://localhost:52415 \
    --model mlx-community/gemma-4-26b-a4b-it-4bit \
    --turns 200 \
    --output-jsonl /tmp/repro.jsonl
```

Exit codes:

| Code | Meaning |
|---|---|
| `0` | All turns completed within thresholds. |
| `1` | One or more `hang` outcomes (cluster stuck, supervisor did not recover). |
| `2` | One or more `kernel_panic_inferred` outcomes (a node disappeared mid-run). |

## What each turn record looks like

```json
{
  "turn_idx": 17,
  "session_idx": 6,
  "session_label": "multimodal-history-text-followup",
  "turn_within_session": 1,
  "n_messages_in_history": 3,
  "n_images_in_history": 2,
  "status": "ok",
  "reason": "",
  "latency_seconds": 4.21,
  "tokens_generated": 142,
  "completion_id": "chatcmpl-..."
}
```

`status` is one of:

- `ok` — request succeeded, cluster looks healthy on the post-call timeline poll.
- `recovered_hang` — `pipeline_eval_timeout` event present in the timeline. The eval-timeout patch fired and the supervisor restored the runner. Distinct from `ok` because we want to count and trend supervised recoveries separately.
- `hang` — a runner has been parked in a non-long-running phase past the threshold and no recovery has been observed yet.
- `api_error` — request itself failed (HTTP error, network blip, model not loaded, etc.).
- `kernel_panic_inferred` — a node we were tracking at startup has vanished from the cluster timeline without a graceful unregister event. Strong signal for a hard reboot.

## Tuning

`--hang-after-seconds` defaults to **90s** — set deliberately above the 60s eval-timeout floor so that a recovery surfaces as `recovered_hang` rather than as a fresh `hang`. Tighten only if you have a clear reason; loosening makes the harness blind to slow hangs.

`--seed` controls the session-rotation order. Use the same seed across runs when you're trying to compare two configs (e.g. FAST_SYNCH on vs off) — turn N hits the same session in both runs, so latency or status divergence at turn N is signal, not noise.

`--expected-nodes` lets you pin the cluster shape explicitly. Default is to snapshot the live cluster at startup; pin only when you want the harness to fail loudly if a peer joins or leaves mid-run.

## What the test corpus exercises

Four session templates rotate through the run:

1. **multimodal-history-text-followup** — two image turns then a text turn that references both. Mirrors the original incident's failure shape.
2. **large-image-single-turn** — one 256x256 image, single turn. Stresses the multi-tile vision path.
3. **text-only-multi-turn** — three text turns. Control: should never hang on gemma-4 if the bug really is multimodal-correlated.
4. **three-image-sequence** — three back-to-back image turns. Sustained multimodal load.

Images are generated procedurally inside the harness (or hardcoded as small base64 PNGs) so the corpus is bit-identical across machines and across runs.

## Limitations

- Hang detection samples after each turn returns. A hang detected *during* a long-running request is only flagged once the request finishes (or the API client times out). Adding a background poller is feasible but isn't needed for the "did 500 turns succeed?" question this harness was built to answer.
- The harness uses non-streaming requests. Streaming has its own failure modes that aren't yet exercised.
- The `_count_images` heuristic walks the OpenAI-style content array; non-OpenAI clients won't be counted correctly.

## Running the unit tests

```bash
uv run pytest bench/tests/
```

The detection logic is a pure function over snapshot fixtures, so the tests cover all classification paths without a live cluster.
