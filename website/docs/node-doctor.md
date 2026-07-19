# Node doctor

`skulk doctor` audits a node's environment against the same facts snapshot
that Skulk's capability pipeline uses: which GPUs the node can see, which inference
engines are usable, whether declared configuration matches observed hardware,
and whether storage has headroom. Every non-OK verdict states its consequence
for serving and the exact remediation.

```bash
# Full audit
uv run skulk doctor

# Apply safe idempotent remediations first, then re-audit
uv run skulk doctor --fix

# Machine-readable output
uv run skulk doctor --json
```

Exit codes: `0` when everything is OK, `2` when only DEGRADED verdicts remain,
`1` when any FAIL remains.

Verdicts:

- **OK**: the contract holds.
- **DEGRADED**: serving works, but below the hardware's capability or with
  reduced observability.
- **FAIL**: serving is broken or misconfigured in a way that will visibly hurt.

The startup fast path runs the same detection automatically: every node logs
its facts summary and capability conflicts at launch, and conflicts surface as
`nodeHealth` reasons on `GET /state` and in the dashboard topology view, so a
degraded node is loud even if nobody runs the doctor.

## Checks

<!-- GENERATED from skulk.doctor.checks.REGISTRY by
     scripts/generate_doctor_docs.py; edit the registry, not this list. -->

### Inference engine availability (`engine-available`)

Verifies at least one inference engine is usable: in-process MLX on macOS, an importable llama-cpp-python build, a llama-server binary (SKULK_LLAMA_SERVER_BIN), or a vllm CLI (SKULK_VLLM_BIN). A node with none advertises no backends and can only participate as management. Supports `--fix`.

### Capability conflicts (`capability-conflicts`)

Runs backend derivation over the node facts snapshot and surfaces every observation-vs-declaration conflict: a GPU that no engine would use (silent CPU serving), degraded NVIDIA detection (missing nvidia-ml-py or a driver mismatch), an engine binary override pointing at an unusable path, or a declared backend the observed hardware cannot support. Supports `--fix`.

### Model storage (`models-storage`)

Verifies the models directory exists, is writable, and has download headroom (warns under 10 GB free, fails under 2 GB). Supports `--fix`.

### Dashboard assets (`dashboard-assets`)

Reports whether the built web dashboard is present. The API serves without it; headless workers are expected to run this way.
