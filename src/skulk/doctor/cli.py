"""``skulk doctor``: the on-demand node environment audit (#614).

Runs every registered check (``skulk.doctor.checks``) against a fresh Node
Facts snapshot and renders OK / DEGRADED / FAIL verdicts with their
consequence and remediation. ``--fix`` first applies every safe idempotent
remediation, then re-audits. ``--json`` emits machine-readable results.

Exit codes: 0 when everything is OK, 2 when only DEGRADED verdicts remain,
1 when any FAIL remains (or the audit itself could not run).
"""

from __future__ import annotations

import argparse
import json
import sys

from skulk.doctor.checks import CheckResult, run_checks, run_fixes
from skulk.facts import refresh_node_facts

_VERDICT_BADGE = {"ok": "[ OK ]", "degraded": "[WARN]", "fail": "[FAIL]"}


def _render_text(results: list[CheckResult], actions: list[str]) -> str:
    """Render the audit as the human-facing report."""
    lines: list[str] = ["skulk doctor: node environment audit", ""]
    for action in actions:
        # run_fixes() strings are self-describing ("[check] did X" or
        # "[check] fix failed: ..."), so no "fixed:" prefix that would
        # mislabel a failed remediation.
        lines.append(f"  fix: {action}")
    if actions:
        lines.append("")
    for result in results:
        lines.append(f"  {_VERDICT_BADGE[result.verdict]} {result.title}")
        lines.append(f"         {result.detail}")
        if result.verdict != "ok":
            lines.append(f"         consequence: {result.consequence}")
            lines.append(f"         fix: {result.remediation}")
            if result.fix_available:
                lines.append("         (skulk doctor --fix can remediate this)")
    counts = {
        "ok": sum(1 for result in results if result.verdict == "ok"),
        "degraded": sum(1 for result in results if result.verdict == "degraded"),
        "fail": sum(1 for result in results if result.verdict == "fail"),
    }
    lines.append("")
    lines.append(
        f"summary: {counts['ok']} ok, {counts['degraded']} degraded, "
        f"{counts['fail']} failed"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``skulk doctor``.

    Args:
        argv: Doctor-specific arguments (everything after the ``doctor``
            subcommand), or ``None`` for ``sys.argv[1:]``.

    Returns:
        The process exit code (0 ok, 2 degraded-only, 1 any fail).
    """
    parser = argparse.ArgumentParser(
        prog="skulk doctor",
        description=(
            "Audit this node's environment: hardware detection, engine "
            "availability, capability conflicts, and storage."
        ),
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="apply safe idempotent remediations before auditing",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="emit machine-readable JSON results",
    )
    args = parser.parse_args(argv)
    apply_fixes = bool(args.fix)  # pyright: ignore[reportAny] - argparse namespace
    as_json = bool(args.as_json)  # pyright: ignore[reportAny] - argparse namespace

    facts = refresh_node_facts()
    actions: list[str] = []
    if apply_fixes:
        actions = run_fixes(facts)
        if actions:
            # Remediations changed the environment; audit the new reality.
            facts = refresh_node_facts()
    results = run_checks(facts)

    if as_json:
        payload = {
            "fixes": actions,
            "results": [result.model_dump(mode="json") for result in results],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(_render_text(results, actions))

    if any(result.verdict == "fail" for result in results):
        return 1
    if any(result.verdict == "degraded" for result in results):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
