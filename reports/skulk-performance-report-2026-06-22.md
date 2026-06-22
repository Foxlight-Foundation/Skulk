# Skulk Performance Improvement Report

Date: 2026-06-22

Scope: read-only architecture and source pass over the Skulk runtime path, with
emphasis on the request-to-runner flow, generation hot paths, placement,
batching, speculative decoding, cache behavior, and data-plane routing.

This report is based on the current repository source and existing in-repo docs.
It did not run a live benchmark or change code.

## Executive summary

Skulk's performance is dominated less by API/master overhead and more by how
well a request maps onto runner execution:

- whether the model can use speculative decoding;
- whether the selected generator is sequential or batched;
- whether placement uses the smallest fast-connected node cycle that fits;
- whether prompt prefixes and KV cache are reused;
- whether multi-node streaming uses the DATA plane efficiently; and
- whether operators avoid settings that disable fast paths.

The highest-impact operational changes are:

1. Keep speculative decoding active for supported models.
2. Avoid forcing `SKULK_FAST_SYNCH=on` unless the exact model/backend has been
   measured with it.
3. Use the smallest high-bandwidth placement cycle that fits the model.
4. Prefer Thunderbolt/LAN links over VPN paths for inference rings.
5. Use multiple replicas for high concurrency when a model is forced onto
   sequential serving.
6. Preserve batching for batch-eligible models.
7. Shape prompts to get KV prefix-cache hits.
8. Treat quantized KV as a memory lever, not a default speed lever.
9. Enable Zenoh DATA plane for multi-node output-stream latency when the network
   and security posture are ready for it.

The highest-impact engineering work is:

1. Add MTP-aware batching or replica autoscaling for models that cannot batch
   while speculative decoding is active.
2. Replace placement's label-based transport ranking with measured topology
   scoring, then include observed TTFT/TPS and link latency/bandwidth in
   placement decisions.

## Runtime mental model

Skulk is an event-sourced control plane around worker-owned runner subprocesses.
A single node hosts router, worker, election, optional master, and optional API
components in `Node` ([src/skulk/main.py](../src/skulk/main.py#L247)).

Chat requests enter `/v1/chat/completions` or `/bench/chat/completions`
([src/skulk/api/main.py](../src/skulk/api/main.py#L1032)) and become
`TextGeneration` commands owned by the API node
([src/skulk/api/main.py](../src/skulk/api/main.py#L2164)). The master does not
generate tokens. For text generation it selects an existing instance by task
count and emits `TaskCreated` for the chosen instance
([src/skulk/master/main.py](../src/skulk/master/main.py#L344)).

Workers run the local planner in `plan()`
([src/skulk/worker/plan.py](../src/skulk/worker/plan.py#L48)). The planner is
the local decision engine: create or tear down runners, stage downloads,
connect distributed backends, load, warm up, and dispatch tasks. Actual model
work runs inside per-shard runner subprocesses spawned by `RunnerSupervisor`
([src/skulk/worker/runner/runner_supervisor.py](../src/skulk/worker/runner/runner_supervisor.py#L173)).

The key performance detail is that per-token output no longer has to go through
the master event log. `ChunkGenerated` is diverted to the DATA plane when a data
sender is wired
([src/skulk/worker/runner/runner_supervisor.py](../src/skulk/worker/runner/runner_supervisor.py#L225)).
The API then receives chunks from the serving worker rather than replaying them
as master-indexed events
([src/skulk/api/main.py](../src/skulk/api/main.py#L3474)). With Zenoh enabled,
DATA is addressed to `data/<node_id>` and each node subscribes only to its own
output key ([src/skulk/routing/router.py](../src/skulk/routing/router.py#L177),
[src/skulk/routing/router.py](../src/skulk/routing/router.py#L321)).

## Highest-impact speed methods

### 1. Keep speculative decoding active

Speculative decoding is the biggest documented single-stream speed lever. The
operator docs describe it as a supported fast path across single-node,
tensor-parallel, and pipeline-parallel MLX execution
([website/docs/speculative-decoding.md](../website/docs/speculative-decoding.md#L36)).

The practical implication is to make sure the needed sidecar/assistant drafter
assets are staged and to avoid request options that disable the speculative
path. In the current generation code, active logits processors disable MTP for
that request
([src/skulk/worker/engines/mlx/generator/generate.py](../src/skulk/worker/engines/mlx/generator/generate.py#L2963)).
That commonly means options such as repetition penalties should be used only
when they are worth giving up speculation.

### 2. Do not force `SKULK_FAST_SYNCH=on` as a generic speed tweak

`FAST_SYNCH_CLUSTER_DEFAULT` is currently false
([src/skulk/worker/runner/bootstrap.py](../src/skulk/worker/runner/bootstrap.py#L34)).
The resolver docstring says FAST_SYNCH is catastrophically incompatible with the
current speculative decoding loop on at least one measured Qwen3.5-9B-4bit
case, dropping from 27.8 tok/s to 0.6 tok/s
([src/skulk/worker/runner/bootstrap.py](../src/skulk/worker/runner/bootstrap.py#L88)).

Use FAST_SYNCH only as a measured, model-specific override. It should not be a
default "go faster" knob.

### 3. Place on the smallest fast-connected cycle that fits

Placement already filters by memory and then narrows to the smallest fitting
cycles ([src/skulk/master/placement.py](../src/skulk/master/placement.py#L352)).
That is the right speed bias: over-sharding adds pipeline hops, coordination,
and transport cost. The speculative-decoding docs call this out directly:
shard to the smallest node count that fits the model
([website/docs/speculative-decoding.md](../website/docs/speculative-decoding.md#L110)).

### 4. Prefer real high-bandwidth local links for inference rings

Ring transport selection prioritizes links by detected interface type today,
and the code explicitly notes that actual connection speeds are still a TODO
([src/skulk/master/placement_utils.py](../src/skulk/master/placement_utils.py#L706),
[src/skulk/master/placement_utils.py](../src/skulk/master/placement_utils.py#L713)).

Operationally, avoid Tailscale/DERP-backed paths for inference rings. Prefer
Thunderbolt or wired LAN. Request `MlxJaccl` only where RDMA connectivity is
actually present and measured.

### 5. Use replicas for high concurrency on sequential-only models

Several important fast paths force `SequentialGenerator`: quantized KV backends,
Gemma 4, MTP speculative decoding, and explicit batching disablement
([src/skulk/worker/runner/llm_inference/runner.py](../src/skulk/worker/runner/llm_inference/runner.py#L824)).

That means a single runner can be excellent for one stream but poor for many
concurrent streams. Since the master chooses among existing instances by task
count for `TextGeneration`
([src/skulk/master/main.py](../src/skulk/master/main.py#L344)), multiple
replicas of the same model can improve aggregate throughput even when each
runner is FIFO.

### 6. Preserve batching where models are batch-eligible

For batch-eligible models, `BatchGenerator` admits queued work up to
`SKULK_MAX_CONCURRENT_REQUESTS`
([src/skulk/worker/runner/llm_inference/batch_generator.py](../src/skulk/worker/runner/llm_inference/batch_generator.py#L516),
[src/skulk/worker/runner/llm_inference/batch_generator.py](../src/skulk/worker/runner/llm_inference/batch_generator.py#L634)).
The default limit is 8
([src/skulk/shared/constants.py](../src/skulk/shared/constants.py#L165)).

Avoid `--no-batch` / `SKULK_NO_BATCH` for throughput workloads unless the model
or debugging case requires it. Avoid options that force extra per-token work
unless they are product-critical.

### 7. Use KV prefix caching as a TTFT accelerator

The MLX cache implements longest-prefix lookup
([src/skulk/worker/engines/mlx/cache.py](../src/skulk/worker/engines/mlx/cache.py#L217)).
Stable system/developer prompts and repeated conversation prefixes should
reduce prefill work and improve TTFT for follow-up requests. Benchmark mode
deliberately disables this cache in the generation path, so live behavior and
benchmark behavior can diverge on repeated-prefix workloads.

### 8. Treat quantized KV as a memory lever

Quantized KV can improve system-level speed indirectly when it lets a model fit
on fewer nodes or prevents memory pressure. It should not be assumed to improve
per-runner decode speed. The docs describe the default KV cache as the fastest
baseline and quantized backends as moderate or near-baseline, depending on the
backend ([website/docs/kv-cache-backends.md](../website/docs/kv-cache-backends.md#L84)).

### 9. Enable Zenoh DATA plane for multi-node streaming latency

Zenoh DATA does not make MLX forward passes faster, but it removes generation
output from whole-cluster gossipsub/event-log traffic. It is most relevant for
multi-node streaming latency and backpressure behavior.

Zenoh defaults on only when configured with `SKULK_ZENOH_LISTEN`
([src/skulk/main.py](../src/skulk/main.py#L145)). Its security posture matters:
the operator should bind it intentionally and keep it on a trusted network
segment.

## Best engineering bets

### A. MTP-aware batching or replica autoscaling

Today speculative decoding is one of the strongest single-stream speed levers,
but MTP/speculative models force `SequentialGenerator`
([src/skulk/worker/runner/llm_inference/runner.py](../src/skulk/worker/runner/llm_inference/runner.py#L835)).
That creates a throughput tradeoff: fast single request, queued concurrent
requests.

Two useful approaches:

1. Build an MTP-aware batch scheduler that can preserve speculative correctness
   while serving multiple active requests.
2. Add an autoscaling placement strategy that prefers extra replicas for
   batch-ineligible models when queue depth rises.

The second is likely lower risk and easier to validate first. It fits the
existing master's "least active tasks among instances" behavior.

### B. Measured topology scoring

Placement currently has the right high-level bias, but it still uses
interface-label heuristics where it should eventually use live measurements.
The code says the missing piece plainly: "Profile and get actual connection
speeds" ([src/skulk/master/placement_utils.py](../src/skulk/master/placement_utils.py#L713)).

The performance-oriented placement score should include:

- link latency between candidate nodes;
- sustained bandwidth between candidate nodes;
- whether the link is Thunderbolt, wired LAN, Wi-Fi, or VPN as a fallback hint;
- observed model-specific TTFT and tokens/sec per node or per placement shape;
- active queue depth per instance; and
- memory headroom after admitted context.

This would make the default placement faster without requiring operators to
manually know every node-pair bottleneck.

## Suggested validation plan

Use the sibling harness or a dedicated benchmark script to measure these
dimensions separately:

1. Single-stream TPS and TTFT with speculative decoding on/off for the same
   model and prompt set.
2. Same test with and without active logits processors such as repetition
   penalty.
3. Same model on the smallest fitting cycle versus one extra shard.
4. Thunderbolt/LAN versus VPN path for the same multi-node placement.
5. One sequential speculative runner versus two or more replicas under
   concurrent load.
6. Batch-eligible model with batching enabled versus disabled.
7. Repeated-prefix prompt set with KV prefix cache warm and cold.
8. Gossipsub DATA fallback versus Zenoh DATA for output streaming latency.

The useful report metrics are:

- TTFT p50/p95;
- output tokens/sec p50/p95 per request;
- aggregate tokens/sec under concurrency;
- queue wait time;
- prefill time;
- decode time;
- accepted speculative tokens per round;
- memory headroom before and after load;
- placement shape and transport path; and
- failure/retry counts.

## Bottom line

The most immediate performance gains are operational: keep speculation enabled,
avoid settings that disable it, choose smaller fast-connected placements, and
use replicas for concurrency when the generator is forced sequential.

The most valuable product work is to make those choices automatic: measured
topology-aware placement plus replica scaling for sequential/speculative
models. Those changes would move Skulk from "fast when configured correctly" to
"fast by default under changing workloads."
