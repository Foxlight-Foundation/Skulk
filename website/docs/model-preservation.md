---
title: Preserve and recover models
---

Skulk can preserve an exact installed model and inspect/recover a signed public
preservation bundle into a **new Skulk home**. These local commands need no
registry connection, Hugging Face account or running node. Recovery uses the
ordinary store, installed-card records and custom-card admission read on startup.

The receiver restores the base and its declared supported companions together:
MLX/GGUF text, split vision weights, explicitly selected same-repository GGUF
projectors, MTP sidecars, assistants and served-engine draft models. Image/video
and other composite selections require an exact artifact bundle; external image
components and video companion files use their card-selected manifests. File
recovery does not establish inference qualification for those workloads.

Separate vision processor repositories and vLLM draft repositories are rejected
because their current loaders still need an offline source adapter. Ambiguous
companion aliases/revisions and unselected legacy GGUF projectors are rejected.
The tool does not reduce the selected closure to get a partial success. It does
not import into a running store, manage background jobs or supply a dashboard
workflow. Inference still requires a compatible engine and enough hardware.
MLX accepts indexed/unindexed `model*.safetensors`; indices and split groups must
be complete and cannot escape the member.

## Recovery kits and Library restore

On a configured Model Store host with the separately installed Model Preservation
capability, **Save recovery kit** reviews an existing preserved copy, then creates
one complete `.foxkit` file. Activity offers **Download recovery kit** after the
owned export job succeeds. Save that file on a NAS, removable drive or another
backup location separate from this Core. A kit retained only on the original
Core cannot survive losing that Core’s storage.

On a new Core, open **Restore from recovery kit**, even before archive setup.
Choose the file, transfer it, review the complete selected model/companion closure
and public capture fingerprints, and acknowledge the stated provenance limits.
If repository code is required, approve it separately. The old Core’s permissions
are never applied. The kit’s keys are candidates: replacing both a kit and its
keys can create a different self-signed capture. Compare independently retained
fingerprints when possible. Registry proof, if present, is checked against this
reader’s packaged root; historical acceptance never asserts current revocation
knowledge. Missing publication proof remains an owner capture, not registry truth.

The browser transfers at most 4 MiB per request. Reselect the same file to resume
a flushed transfer after browser/capability/Core restart. Core keeps at most eight
transfers; remove unused ones from the restore screen to release storage. Transfer
completion does not establish authenticity. Core verifies every actual signed
object and stages the entire required closure before the single canonical store
index update. It then records this receiver’s new card admission. Closing the page
does not stop submitted jobs. Stop waits for the owned byte process; an already
published result may win.

Existing primary aliases with different captures are protected. Matching owned
companions may join the restored generation after exact artifact, card and byte
checks; their prior files stay available to running readers. Deliberately removed
models require a separate confirmation in the review. That consent binds the exact
local deletion record: crash reconciliation can complete that choice but cannot
undo a later removal. Interrupted restores/exports require **Review again** and
a new deliberate start. An exact already committed closure is checked
before finishing admission. Hidden unindexed generations cannot become partially
admitted companions. Retained envelope, trust and source evidence remain beside
the indexed generation. Source integration does not establish a frozen desktop
release or public capability-store availability; the capability SDK distribution
is a separate prerequisite. The standalone kit CLI is independently available in
[preservation core 0.5.0](https://github.com/Foxlight-Foundation/foxlight-model-preservation/releases/tag/v0.5.0).

### Cards can evolve without rewriting backup evidence

The kit container, original signed card and receiving runtime projection are
separate contracts. Container verification and generic file recovery retain
unknown future evidence/properties without deserializing them through today’s
ModelCard. Original encoded card bytes, numeric representations, identities and
signatures remain intact. Runtime recovery applies this build’s compatible defaults
and interpretation; unfamiliar required card semantics or version floors refuse
with a reader-compatibility error. Retain the kit and use a compatible reader;
never remove unknown runtime/security requirements merely to force admission.
A derived runtime view does not replace the signed original. Re-preserving a
recovered installation reuses its retained original card/proof when present.
If runtime-derived properties differ from the original signed card, the capture
uses an owner-view card and retains the exact original card/proof separately in
`preservation/source-evidence/source-evidence/registry-source-card.json` after
recovery. A public registry-root snapshot is retained as evidence, never enrolled
automatically. Re-preservation carries that original source forward with signed
file-hash checks; it never grants registry status to the changed runtime view.

The preservation reader accepts additive descriptive properties at every metadata
depth and future string vocabulary names while keeping known field constraints.
Selected-artifact, envelope and manifest contracts stay strict and independently
versioned. A new property needs compatible runtime interpretation and backward-
compatibility fixtures; changes to selected-artifact/security semantics need an
explicit supported contract. The archive does not require every future property
to be understood just to retain or recover its captured bytes.

## Preserve an installed model

Use `skulk-preservation` or `skulk preservation`; neither starts a node. The
installed base directory must have its ordinary `.skulk/installed-card.json`.
Companions are selected from their installed records, matched to the owning card,
repository, immutable revision and role. The tool looks beside the base and in
any additional explicitly supplied artifact roots. It accepts no recipe or
caller-selected file list.

```sh
skulk-preservation plan --installed /path/to/model-store/owner--model
mkdir /path/to/connected-drive/preserved
skulk-preservation preserve --installed /path/to/model-store/owner--model \
  --store /path/to/connected-drive/preserved --key /protected/owner.pem
```

The plan reports selected aliases and logical source bytes; it is not full byte
verification or an execution grant. `preserve` copies a private snapshot, verifies
it against every installed file digest and rechecks source generation records,
validates the runtime-selected layout, then calls the standalone signed packer.
The archive must already exist; verify the selected drive/NAS mount before use.
It needs temporary space for the snapshot plus new archive objects. This is a
local process, not a live-store snapshot lease or automatic mount-sentinel service.

`--artifact-root /another/enrolled/local/root` adds companion search roots.
Missing/ambiguous companions stop the copy. This first native selection includes
all declared supported optional speculation companions too; it does not silently
omit missing optional files. `--passphrase-file /protected/passphrase` unlocks an
encrypted key. Private keys and passphrases are never bundled. Create/protect keys
with the standalone tool's documented key workflow; this CLI does not yet provide
capability-managed key enrollment or recovery-guide export.

All new copies are **private** and receive-only. Preserving does not launch a
server, advertise an inventory or grant redistribution rights. Original notices
are retained when present. Otherwise a clearly labeled private owner-capture
statement records unresolved terms; it is not a replacement public license.
Historical source installed records are retained in a separate signed notices
member, with previous receiving code grants removed. They are evidence, never
incoming authorization sidecars.

Without an original retained public card/proof, an effective installed card is
exported as an owner capture. A source sidecar's `registry_verified` label cannot
manufacture a portable registry proof. If available, supply the exact original
`foxlight.retained-card.v1` profile with `--retained-card retained-card.json`.
The exporter checks card agreement and any proof against Skulk's packaged root;
expired original proof may be preserved for deliberate historical recovery.
Unknown/local source revisions remain unsupported by portable bundle v1; no
commit is invented. Copying does not execute model repository content.

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
| `requires_owner_acceptance` | Original publication/file evidence is incomplete or historical, or companion provenance has not been independently re-established; recovery needs explicit custom admission. |
| `members` | Exact base and companion store aliases published together. |
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
Vision cards conservatively require that explicit code decision too, even when
`trust_remote_code` is false: processor loading is a separate potentially
executable path in the runtime. The primary tokenizer's false code setting remains
binding, including family-specific repository tokenizers and the ordinary loader.

`--historical` explicitly allows expired registry metadata while retaining all
signature, identity and hash checks. It reports historical publication and
requires custom acceptance. It cannot discover later revocations or establish
what a destroyed or unreachable registry would authorize today.

Only fresh publication **and complete original SHA-256 binding**, for a single
self-contained model, produce a
`registry_verified` installed record. Multi-member closures currently require
custom admission, retaining the original base publication report separately.
Owner signatures, card IDs, repository
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
  inspection report, plus retained source observations/notices outside loader roots. Copying evidence does not enroll trust or authorize code
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

Preserve interruptions may leave hidden `.skulk-preserve-*` snapshot folders in
the archive after an uncatchable process termination. They are unpublished and
never served by the reference seed. Inspect abandoned folders before removing
them. Completed manifests and earlier good copies remain independently verifiable.

## Privacy boundaries

Signatures detect substitution and corruption; they do not encrypt weights or
hide model choices. Archives are plaintext. Use protected local storage and an
access-controlled destination for private copies; the `private` audience label
is not endpoint authentication or encryption. Receiving from a direct mirror
reveals the connecting address and requested objects to that supplier. The
standalone receiver is not an anonymity client. Receiving/recovery never enables
public serving, discovery or automatic sharing.
