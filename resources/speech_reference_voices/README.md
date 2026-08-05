# Bundled reference voices

These project-authored recordings provide stable, named reference conditioning
for speech models that truthfully support voice cloning. Each profile pairs one
audio file with its exact transcript and a checksum in `catalog.toml`.

The runtime resolves profiles by stable ID, verifies containment and SHA-256,
and loads the asset only on the worker serving the request. Clients select a
profile through the standard `voice` field; uploaded reference audio remains an
explicit request-scoped override.
