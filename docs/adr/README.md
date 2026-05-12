# Architecture Decision Records

Architecture Decision Records capture choices that are expensive to reverse
once they affect Skulk's event-sourced state, runner taxonomy, placement
contracts, public APIs, or operator workflow.

## Format

Each ADR uses this structure:

- **Status:** Proposed, Accepted for planning, Accepted, Superseded, or Rejected.
- **Context:** The forces that made the decision necessary.
- **Decision:** The choice Skulk will implement.
- **Consequences:** Operational and implementation effects of the decision.
- **Rejected alternatives:** Options considered and why they were not chosen.

## Numbering

Use monotonically increasing four-digit filenames:

```text
0001-short-title.md
0002-short-title.md
```

Do not renumber existing ADRs. If a decision changes, add a new ADR that
supersedes the old one.
