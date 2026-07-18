"""Generate the node-doctor documentation page from the check registry.

The doctor's check registry (``skulk.doctor.checks.REGISTRY``) is the single
source of truth for the node environment contract; this script renders it into
``website/docs/node-doctor.md`` so the user-facing documentation cannot drift
from what the checks actually verify. Rerun after changing the registry:

    uv run python scripts/generate_doctor_docs.py

The output is committed; CI treats a stale page as an ordinary review miss.
"""

from __future__ import annotations

from pathlib import Path

from skulk.doctor.checks import REGISTRY

_HEADER = """\
# Node doctor

`skulk doctor` audits a node's environment against the same facts snapshot
Skulk's capability pipeline uses: which GPUs the node can see, which inference
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
"""


def render() -> str:
    """Render the full markdown page from the registry."""
    sections: list[str] = [_HEADER]
    for check in REGISTRY:
        fixable = " Supports `--fix`." if check.fix is not None else ""
        sections.append(f"### {check.title} (`{check.check_id}`)\n")
        sections.append(f"{check.docs}{fixable}\n")
    return "\n".join(sections)


def main() -> None:
    """Write the generated page under ``website/docs``."""
    output = Path(__file__).resolve().parent.parent / "website" / "docs" / "node-doctor.md"
    output.write_text(render())
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
