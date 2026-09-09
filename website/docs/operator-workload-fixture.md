---
title: Isolated operator workload fixture
description: Local synthetic API fixture behind real pairing and encrypted transport.
---

# Isolated operator workload fixture

This opt-in benchmark tool provides generated cluster data to an unchanged
Skulk Operator client. It runs the real signed on-demand relay, gateway,
Ed25519 pairing exchange, token rotation, scoped authorization, and pinned
inner TLS 1.3. It does **not** construct a Skulk node, discover peers, join a
cluster, load models, read a model store, or execute cluster mutations.

This is an experimental qualification tool, not a production service or
capacity evidence. Its generated sizes and stream pacing are **not measured
released-client workload profiles**.

## Start a local session

Build a reviewed `paired-websocket-service` executable in `skulk-relay`, record
its SHA-256 digest, and run from this repository:

```bash
uv run python -m bench.operator_workload_fixture \
  --relay-binary /absolute/path/to/paired-websocket-service \
  --relay-sha256 <exact-sha256> \
  --lifetime-seconds 600
```

The digest is verified before fixture creation. No remote relay URL, existing
cluster configuration, or supplied credentials are accepted. The tool creates
an owner-only temporary directory and two dynamic IPv4 loopback listeners.
Losing a bind race fails startup rather than reusing an existing listener.

The ready JSON record gives the relay port and paths to `pairing.txt` and
`pairing.png`, both mode `0600`. They contain generated credentials: do not
paste them into logs, PRs, or campaign evidence. The cluster is named
**Fixture**, with explicitly synthetic node/model labels. Its invitation
permits four pairings and expires after the selected session lifetime.

The session lasts 60–7,200 seconds, including setup. Normal exit, cancellation,
or expiry stops the gateway and relay and removes temporary state. An independent
watchdog reaps the relay on parent-pipe EOF, expiry, or termination. A hard kill
of the runner can leave its owner-only temporary directory behind; confirm the
watchdog has stopped before removing that exact generated directory. No cloud
resources are created; this is not a hosted campaign cleanup mechanism.

## Device and privacy boundary

Released app schemas permit clear outer WebSockets only on loopback; pinned
inner TLS still encrypts the API exchange. This fixture does not expose a LAN
listener or weaken transport rules. Verify Android USB forwarding or simulator
reachability separately. Physical iOS reachability remains a separate gate:
do not substitute the operating relay or publish the fixture to work around it.

Use only generated test prompts and speech input, never personal information.
Generated handlers discard input and do not echo or persist it. Access logs
are disabled. Real authority state lives in the fresh encrypted database.
This fixture does not yet export observations or connection-lifetime metrics;
HTTP request duration must not be relabeled as socket lifetime. A metadata-only
observer and actual device captures are separate steps.

## Generated API behavior

All fixture paths require real scoped bearer authorization except the existing
pairing challenge, exchange, and refresh routes under their normal rules.
The real auth router also supplies device listing and revocation.

| Method and path | Behavior |
| --- | --- |
| `GET /state` | One synthetic healthy node and two ready synthetic instances. |
| `GET /v1/models` | Generated text and streaming speech cards; no weights. |
| `GET /store/registry` | Two synthetic entries with zero artifact bytes. |
| `GET /store/storage` | Empty generated staging inventory. |
| `GET /store/downloads` | Empty generated download list. |
| `GET /v1/audio/voices` | One silence voice; query values ignored. |
| `POST /v1/chat/completions` | Discards bounded input; fixed SSE text, stop event and `[DONE]`, without inference. |
| `POST /v1/audio/speech` | Discards bounded input; 3 seconds of silence, 30 × 4,800-byte chunks paced 100 ms apart; mono PCM16 at 24 kHz with the app's format headers. |

Requests are bounded to 64 KiB before parsing or stream generation; oversized
bodies receive `413`. Unknown mutation paths return `404`. Direct dashboard
invitation management remains unavailable on the relay listener even to scoped
devices. The body bound is not a two-leg traffic meter or hosted budget control.

## Validation and evidence limits

```bash
uv run pytest bench/tests/test_operator_fixture_app.py \
  bench/tests/test_operator_fixture_lease.py \
  bench/tests/test_operator_workload_fixture.py

SKULK_PAIRED_RELAY_BINARY=/absolute/path/to/paired-websocket-service \
uv run pytest bench/tests/test_operator_workload_fixture.py
```

The joined test uses a loopback-only, no-retry WebSocket/TLS client, not the
mobile app. It checks actual pairing, generated reads, mutation rejection,
protected files and teardown. Other tests cover real token rotation/revocation,
oversized input, generated streams, watchdog expiry and parent EOF.

For source-pinned schema compatibility, supply an existing app checkout with
installed TypeScript and Zod versions matching that commit's lock:

```bash
uv run python -c 'import json; from bench.operator_fixture_app import generated_responses; print(json.dumps(generated_responses()))' |
  node bench/validate_operator_fixture.cjs /absolute/path/to/skulk-app <exact-source-commit>
```

The validator reads a fixed schema-module allowlist with `git show`, without
changing the app checkout. It checks stable identity, ready model projections,
and streaming speech metadata. Output explicitly marks physical-device and
capacity evidence false. Source compatibility does not prove the installed
binary, native socket behavior, observed polling/stream profiles, capacity,
endurance, or hosted behavior.
