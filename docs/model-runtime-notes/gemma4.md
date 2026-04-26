<!-- Copyright 2025 Foxlight Foundation -->

# Gemma 4 Runtime Notes

## Why This Model Is Weird

Gemma 4 is not a generic chat model from Skulk's point of view.

The official Google model card for `google/gemma-4-26B-A4B-it` says the family
has:

- multimodal text/image support, with audio on the smaller variants
- configurable thinking mode
- a 256K context window on the 26B A4B and 31B variants
- a hybrid attention scheme that mixes sliding-window attention with global attention
- a custom thought-channel output format, not just plain assistant text

That matters in clustered MLX because Gemma 4 is unusual along several
dimensions at once:

- prompt shape
- reasoning delimiters
- multimodal prompt handling
- distributed prefill behavior
- generator-path stability

## Official Behavior That Matters To Skulk

Per the official model card, Gemma 4 uses:

- `<|think|>` to enable thinking at the start of the system turn
- `<|channel>thought\n...<channel|>` for thought content
- an empty thought block prefix even when thinking is disabled on the larger
  variants, including `26B-A4B-it` and `31B-it`

That last point is critical. Normal inference is expected to begin after the
empty thought-channel boundary, not at a raw `<|channel>` marker.

## What Failed In Skulk

### 1. Prompt Boundary Corruption

When Skulk omitted Gemma 4's empty thought-channel suffix during normal
inference, the model could begin by emitting raw channel tokens into visible
output.

Observed symptom:

- output starting with `<|channel>`
- malformed visible responses

The fix was to restore the reference empty thought-channel suffix for normal
Gemma 4 inference, while still allowing distributed warmup to suppress it.

### 2. Degenerate Repetition In Distributed Batch Mode

Gemma 4 produced unstable output in distributed `BatchGenerator` mode after a
valid prefill.

Observed symptom:

- repetitive decode like `YesYesYesYes...`

Current fix:

- force Gemma 4 onto `SequentialGenerator` in clustered mode

### 3. Distributed Warmup Hangs On The Short-Prompt Stream Path

Gemma 4 warmup could hang when a short pipeline prompt took the
`stream_generate` prefill path.

Observed symptom:

- warmup stalls after `Starting prefill`
- logs show `Prefill path selected: stream_generate (...)`

Current fix:

- pipeline models always use explicit pipeline prefill
- short one-chunk prompts suppress the distributed progress callback instead of
  falling back to `stream_generate`
- distributed warmup uses greedy one-token generation
- distributed warmup uses `all_sum` instead of `all_gather` for the final sync

## Current Trusted Runtime Envelope

As of branch `codex/model-errors`, the current boring path for Gemma 4 is:

- prompt renderer: Gemma 4 specific
- output parser: Gemma 4 specific
- distributed generator: sequential
- warmup prompt: minimal sanity-check prompt
- warmup generation: one token, greedy
- pipeline warmup prefill: explicit pipeline prefill
- KV backend baseline: `default`

This is the path we currently trust first.

Anything more aggressive should be treated as experimental until validated.

## Expected Logs On The Working Path

For a healthy clustered Gemma 4 run, expect to see:

- `using SequentialGenerator (model_family=gemma4)`
- `Using default KV cache`
- `Prefill path selected: stream_generate` for short one-chunk turns, or
  `Prefill path selected: pipeline_parallel_prefill` for multi-chunk turns

These are useful smell tests before trusting the output.

## Known Failure Signatures

If any of the following reappear, Gemma 4 is probably off the trusted path:

- visible output begins with `<|channel>`
- repetitive decode such as `YesYesYes...`
- warmup logs show `Prefill path selected: stream_generate` for a pipeline run
- warmup stalls before runner readiness

## Code Paths

The current Gemma 4-specific behavior lives primarily in:

- `src/exo/worker/engines/mlx/gemma4_prompt.py`
- `src/exo/worker/engines/mlx/utils_mlx.py`
- `src/exo/worker/engines/mlx/generator/generate.py`
- `src/exo/worker/runner/llm_inference/runner.py`
- `src/exo/worker/runner/llm_inference/model_output_parsers.py`

The focused regression tests live in:

- `src/exo/worker/tests/unittests/test_mlx/test_gemma4_prompt.py`
- `src/exo/worker/tests/unittests/test_mlx/test_prefill_path_selection.py`
- `src/exo/worker/tests/unittests/test_mlx/test_warmup_request.py`
- `src/exo/worker/tests/unittests/test_runner/test_turboquant_batch_guard.py`

## Open Questions

These notes document what is known to work, not the full support frontier.

Still open:

- whether distributed Gemma 4 can ever safely re-enter a batch/history path
- whether any non-default KV backend is trustworthy for clustered Gemma 4
- whether the same constraints apply equally to all Gemma 4 variants
- whether future upstream library changes make some of these guards unnecessary

## Sources

- Official Google Gemma 4 26B A4B model card:
  https://huggingface.co/google/gemma-4-26B-A4B-it
- Existing user-facing behavior note:
  `website/docs/model-behaviors/gemma4.md`
