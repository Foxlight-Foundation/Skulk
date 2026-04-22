# React Dashboard Cutover Plan

This branch exists to retire the legacy Svelte dashboard in `dashboard/`
without losing meaningful functionality that is still useful in operations.

## Goals

- Make `dashboard-react/` the only supported dashboard implementation.
- Remove the runtime fallback to the Svelte build.
- Remove Svelte-specific Nix/build plumbing.
- Keep or consciously drop every remaining Svelte-only feature.

## Must Port Before Deletion

### Trace UI

- Port trace list UI from `dashboard/src/routes/traces/+page.svelte`
- Port trace detail/stats UI from `dashboard/src/routes/traces/[taskId]/+page.svelte`
- Preserve:
  - list traces
  - delete selected traces
  - download raw trace JSON
  - open trace in Perfetto
  - phase/category/rank stats

### Cluster Diagnostics

Port the still-useful diagnostics from `dashboard/src/routes/+page.svelte`:

- macOS build/version mismatch warning
- Thunderbolt 5 present but RDMA disabled warning
- Mac Studio `en2` RDMA misuse warning
- Thunderbolt bridge cycle warning

### Model Browser Persistence

Bring React model-browser persistence up to parity with the old dashboard:

- persist favorites
- persist recent models

## Explicit Keep/Drop Decisions

These should not block removal forever, but we should make an intentional call.

### Settings Conveniences

- `DirectoryBrowser.svelte`
- `node_overrides` editing UI

Recommended default:

- drop directory browsing unless there is a live workflow that depends on it
- drop `node_overrides` editing for the cutover and redesign per-node policy
  later if Skulk still needs it

### Debug Affordances

- old debug mode toggle
- old topology-only mode toggle

Recommended default:

- drop both unless there is a current operational need

## Removal Sequence

After parity is good enough:

1. Remove Svelte fallback from `src/exo/utils/dashboard_path.py`
2. Remove `dashboard/parts.nix`
3. Remove the `dashboard/` tree
4. Remove docs and contributor references that describe Svelte as a fallback
5. Re-run dashboard, docs, and flake validation

## Exit Criteria

Svelte can be removed when all of the following are true:

- No meaningful user-facing route exists only in `dashboard/`
- React covers traces and required diagnostics
- Runtime no longer searches for `dashboard/build`
- Nix/dev tooling no longer depends on Svelte assets or node modules
- Repo docs describe React as the only dashboard
