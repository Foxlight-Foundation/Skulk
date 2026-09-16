"""Bounded Steward views of build identity and advertised capability nodes."""

import json
import re
from typing import Literal, cast

from skulk.shared.types.capability_nodes import CapabilityNodeSummary


def observed_build_value(value: object) -> str | None:
    """Return a reported build identifier, or None for unavailable sentinels."""
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    # Runtime diagnostics use textual sentinels when package/identity reads fail.
    # They are missing evidence, not valid build identifiers.
    return normalized if normalized and normalized.lower() != "unknown" else None


def internal_service_question(question: str) -> bool:
    """Opt into resident-service details only for explicitly internal subjects."""
    return (
        re.search(
            r"\b(?:steward|internal (?:fabric|services?)|resident (?:model|service)|your brain)\b",
            " ".join(question.lower().split()),
        )
        is not None
    )


def inventory_question(question: str) -> Literal["versions", "capability_nodes"] | None:
    """Recognize standalone build and capability-node inventory questions."""
    text = " ".join(question.lower().split()).rstrip("?.!")
    if re.fullmatch(
        r"(?:what (?:skulk )?version(?: (?:are you(?: running)?|is (?:skulk|the cluster) running))?|"
        r"(?:what is|what's) (?:your|the skulk|the cluster) version)",
        text,
    ):
        return "versions"
    if re.fullmatch(
        r"(?:what|which) capabilit(?:ies|y nodes)(?: are (?:available|installed|running)| do (?:you|we) have)",
        text,
    ):
        return "capability_nodes"
    return None


def capability_inventory(
    payload: dict[str, object], names: dict[str, str]
) -> dict[str, object]:
    """Project advertised nodes, excluding action payloads and unrelated backends.

    The API already filters expired advertisements. Empty means no current
    advertisements, not proof that no package is installed or executable.
    """
    raw = payload.get("capabilityNodes")
    complete = isinstance(raw, dict)
    rows: list[dict[str, object]] = []
    if isinstance(raw, dict):
        for host, entries in cast("dict[str, object]", raw).items():
            if host not in names or not isinstance(entries, list):
                complete = False
                continue
            for entry in cast("list[object]", entries):
                if not isinstance(entry, dict):
                    complete = False
                    continue
                item = cast("dict[str, object]", entry)
                try:
                    node = CapabilityNodeSummary.model_validate_json(
                        json.dumps(
                            {
                                key: value
                                for key, value in item.items()
                                if key != "observedAt"
                            }
                        )
                    )
                except ValueError:
                    complete = False
                    continue
                rows.append(
                    {
                        "name": node.title or node.node_id,
                        "host": names[host],
                        "bundle": node.bundle_id,
                        "version": node.version,
                        "status": node.status,
                        "ownerAvailable": node.owner_available,
                        "observedAt": item.get("observedAt"),
                    }
                )
    return {
        "source": "/state.capabilityNodes",
        "coverageComplete": complete,
        "advertisedCount": len(rows) if complete else None,
        "nodes": rows,
        "scope": "Current capability-node advertisements, not inference backends or installation inventory. Discovery does not grant execution authority.",
    }


def bounded_inventory(payload: dict[str, object], limit: int) -> str:
    """Keep inventory JSON valid and disclose any omitted rows within a budget."""
    rows = cast("list[dict[str, object]]", payload["nodes"])
    result = dict(payload)
    included: list[dict[str, object]] = []
    result["nodes"] = included
    result["omittedRows"] = len(rows)
    if len(json.dumps(result, separators=(",", ":"))) > limit:
        raise ValueError("inventory metadata exceeds the response budget")
    for row in rows:
        included.append(row)
        result["omittedRows"] = len(rows) - len(included)
        if len(json.dumps(result, separators=(",", ":"))) > limit:
            included.pop()
            result["omittedRows"] = len(rows) - len(included)
            break
    return json.dumps(result, separators=(",", ":"))


def render_inventory(
    payload: dict[str, object], topic: Literal["versions", "capability_nodes"]
) -> str:
    """Render only observed inventory fields without generating missing values."""
    if "error" in payload or not isinstance(payload.get("nodes"), list):
        return "I couldn't verify that inventory. Please retry."
    rows = cast("list[dict[str, object]]", payload["nodes"])
    if topic == "versions":
        lines = ["Reported Skulk builds:"]
        for row in rows:
            version = row.get("skulkVersion")
            commit = row.get("skulkCommit")
            lines.append(
                f"- {row['name']}: {version or 'version unavailable'}; commit {commit or 'unavailable'}."
            )
        if not rows:
            lines.append("No node build observations were available.")
        lines.append(
            "These are observed builds; I have not checked whether they are the latest release."
        )
    else:
        lines = [
            "Capability nodes are installed extensions that provide services or actions to the fabric; they are distinct from inference backends."
        ]
        if payload.get("advertisedCount") == 0:
            lines.append(
                "No capability nodes are currently advertised by this cluster. That does not prove none are installed."
            )
        for row in rows:
            availability = (
                "owner reachable" if row["ownerAvailable"] else "owner unavailable"
            )
            lines.append(
                f"- {row['name']} on {row['host']}: {row['status']}, {availability}; bundle version {row['version']}."
            )
        lines.append("Advertisements do not establish permission to execute actions.")
    if payload.get("coverageComplete") is not True:
        lines.append("Coverage is incomplete; missing observations remain unknown.")
    if payload.get("omittedRows"):
        lines.append(
            f"{payload['omittedRows']} additional rows were omitted from this bounded response."
        )
    return "\n".join(lines)
