---
title: Offline model recovery
---

Skulk can inspect a signed public preservation bundle and recover a
self-contained text model into a **new Skulk home**. Recovery needs no registry
connection or Hugging Face account. It uses the ordinary store, installed-card
records and custom-card admission that Skulk reads on startup.

This importer supports one MLX or GGUF text model per bundle. Separate vision,
draft, MTP, image and video companions, importing into a running store, and a
dashboard preservation workflow are not supported. A successful recovery checks
files and makes the model selectable. Inference still needs a compatible locally
installed engine and enough hardware; recovery does not establish qualification.
MLX can use indexed weights or unindexed `model*.safetensors`. Indices and split
groups must be complete; index references cannot escape the captured member.

## Obtain a recovery copy

Obtain the exact manifest SHA-256 and an independently authenticated **owner
trust profile**. A mirror supplying its own key and checksum is insufficient.
Anyone can publish public-format cards and owner-signed bundles. The public
[preservation core](https://github.com/Foxlight-Foundation/foxlight-model-preservation)
documents creating bundles and receiving them from ordinary mirrors.

The bundle may already be on disk or mounted NAS. To receive from mirrors:

```sh
model-preserve fetch MANIFEST_SHA256 --trust owner-trust.json \
  --mirror https://mirror-a.example/models \
  --mirror https://mirror-b.example/models --store ./preserved
```

Receiving and recovering do not seed files or start a sharing server.

## Inspect and recover

Use a Skulk build that includes `skulk-preservation`. The equivalent command is
`skulk preservation`; neither form starts a node.

```sh
skulk-preservation inspect MANIFEST_SHA256 \
  --trust owner-trust.json --store ./preserved
```

Inspection hashes actual files and validates the retained card against this
runtime. Its JSON result separates:

| Field | Meaning |
| --- | --- |
| `publication` | Owner statement, fresh registry publication, or explicitly historical registry publication. |
| `file_coverage` | Whether the original card binds every captured file by SHA-256. Capture integrity is mandatory even when original evidence is missing. |
| `requires_owner_acceptance` | Original publication/file evidence is incomplete or historical; recovery needs explicit custom admission. |
| `requires_code_permission` | The card may execute repository code; a separate receiving decision is needed. |
| `registry_proof` | Checked versions, expiry, time, and supplied-root-chain revocation scope, when present. |
| `runtime_requirements` | Owner runtime statements, without implied execution or qualification evidence. |

Registry verification uses the Foxlight root packaged with Skulk. The command
has no option to replace it with a bundle's or mirror's root. Invalid signatures,
contradictory hashes, invalid registry evidence and unsupported runtime card
semantics block recovery. They never silently become a community warning.

Choose a new, absent destination whose parent directory already exists:

```sh
skulk-preservation recover MANIFEST_SHA256 \
  --trust owner-trust.json --store ./preserved \
  --destination /path/to/new-recovery-home
```

For owner-only captures, missing original SHA-256 evidence, or historical
publication, review the inspection and add `--accept-owner-capture`. The model
enters through custom admission and retains its original evidence. That flag
does not authorize executable content. If the card needs repository code and
you intend to authorize it for this exact recovery, separately add
`--allow-repository-code`.
The card's own code setting remains binding. A false setting is honored by
family-specific repository tokenizers as well as the ordinary tokenizer loader.

`--historical` explicitly allows expired registry metadata while retaining all
signature, identity and hash checks. It reports historical publication and
requires custom acceptance. It cannot discover later revocations or establish
what a destroyed or unreachable registry would authorize today.

Only fresh publication **and complete original SHA-256 binding** produce a
`registry_verified` installed record. Owner signatures, card IDs, repository
names and self-reported installed sidecars cannot award that label.

## Open the recovered home

The new directory contains:

- `model-store/`: model bytes, revision markers, schema-3 installed cards and a
  rebuildable `registry.json` index.
- `custom_model_cards/`: normal operator admission when required. Deleting its
  custom card revokes selectability; the artifact sidecar does not resurrect it.
- `skulk.yaml`: local store configuration with Hugging Face fallback disabled
  and staging contained inside the new home.
- `preservation/`: original signed envelope, receiver trust snapshot and
  inspection report. Copying evidence does not enroll trust or authorize code
  for another receiver.

Launch from the new directory so Skulk reads its configuration. For an isolated
check, choose a private cluster namespace as described in
[Configuration](configuration):

```sh
cd /path/to/new-recovery-home
SKULK_HOME="$PWD" SKULK_LIBP2P_NAMESPACE="your-private-recovery-namespace" skulk --offline
```

Use the normal model selector and inference request when the required engine is
available locally. Recovery does not download an engine, place a model or start
inference. The generated configuration contains the destination's absolute store
path; update it if you move the home.
Launching Skulk starts its ordinary store service; see [Model Store](model-store)
for the trusted-LAN access contract. A private cluster namespace does not add
authentication to the store's HTTP port.

## Interruption and compatibility

Files are copied and checked in private staging. The tool flushes the normal
store index, sidecars, custom admission and evidence before publishing the entire
new home together. It never overwrites an existing home or changes a running
store. The parent must exist; a missing mount point is not recreated. Free-space
admission is an observation, not a reservation.

Failure before publication removes staging when cleanup is possible. A hard
kill or unavailable mount can leave hidden `.skulk-recovery-*` staging; it is not
a published home. Inspect and remove abandoned staging if space is needed. An
interruption near the final rename may leave a complete home: check the selected
destination before retrying. Existing homes remain protected from replacement.
An interrupted final reservation can instead leave an empty destination; inspect
it before removing that empty directory and retrying.

Recovered installations require a Skulk build understanding sidecar schema 3.
Ordinary schema-1/2 installations keep their existing serialization and behavior.
This importer does not establish multi-node recovery, a signed desktop release,
or resilience against network blocking; each needs its own validation.
