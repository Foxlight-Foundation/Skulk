# Trusted LTX nodes for Comfy

This Foxlight wheel adds explicit LTX sampler and workflow nodes to the pinned
Comfy runtime. It is an extension of the existing Comfy engine. Its manifest binds
the exact source, node features and payload; weights remain separate selected
artifacts. The numeric sampler adaptation retains the upstream LTX license and
attribution, while Foxlight's integration code retains Apache 2.0 licensing.

Only published, attested and read-back-verified wheel descriptors may create
installation offers. New LTX placements bind the selected composite engine build
and use the shared `InstallEngine` path. A previous running instance retains its
original build and cannot acquire newer features by a telemetry refresh.

The interpreter and base engine remain isolated. Only the verified package folder
is whitelisted; other custom nodes and remote API nodes remain disabled. No source
checkout, request-time package resolution or model download occurs in this wheel.

Candidate source and tensor checks do not qualify a model or hardware class.
