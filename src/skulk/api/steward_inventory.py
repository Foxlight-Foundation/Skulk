"""Bounded Steward views of build identity and advertised capability nodes."""

import json
import re
from typing import Literal, cast

from skulk.api.build_identity import known_build_identifier
from skulk.extensions.capabilities import InstalledCapabilityRegistration
from skulk.shared.types.capability_nodes import CapabilityNodeSummary


def observed_build_value(value: object) -> str | None:
    """Return a reported build identifier without changing valid display casing.

    Args:
        value: Untrusted build version or commit from state or diagnostics.

    Returns:
        The stripped identifier, or None for non-string values and canonical
        unavailable sentinels. This pure normalization has no side effects.
    """
    if not isinstance(value, str) or known_build_identifier(value) is None:
        return None
    return value.strip()


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


def render_installed_inventory(payload: dict[str, object]) -> str:
    """Explain local installation and independent readiness without inference claims."""
    raw = payload.get("installations")
    if "error" in payload or not isinstance(raw, list):
        return "I couldn't verify installed capability metadata on this API host. Other hosts remain outside this inventory."
    lines = ["Managed capability installations on this API host:"]
    for item in cast("list[object]", raw):
        if not isinstance(item, dict):
            continue
        row = cast("dict[str, object]", item)
        enabled = (
            "enablement unknown"
            if row.get("enabled") is None
            else "enabled"
            if row.get("enabled")
            else "disabled"
        )
        if row.get("uninstalled"):
            enabled = "uninstalled; retained metadata"
        reachability = row.get("reachable")
        reachable = (
            "reachable"
            if reachability is True
            else "unreachable"
            if reachability is False
            else "reachability unknown"
        )
        ready = row.get("ready_capabilities")
        readiness = (
            ", ".join(str(value) for value in cast("list[object]", ready))
            if isinstance(ready, list) and ready
            else "no exact ready contracts observed"
        )
        lines.append(
            f"- {row.get('bundle') or row.get('plugin_id')}, version {row.get('version') or 'unknown'}: {enabled}, {reachable}; {readiness}."
        )
        try:
            registration = InstalledCapabilityRegistration.model_validate_json(
                json.dumps(row.get("registration"))
            )
        except ValueError:
            lines.append(
                "  Declared operations and requirements are unavailable or omitted by the metadata bound."
            )
            continue
        for declared in registration.capabilities:
            lines.append(
                f"  {declared.qualified_id}: {declared.purpose[:160]} ({declared.io_mode})."
            )
        if not registration.capabilities:
            lines.append("  No operations were declared in this registration.")
        requirements = [
            f"Skulk {registration.skulk_requires}",
            f"Python {registration.python_requires}",
            "platforms " + ", ".join(registration.platforms),
        ]
        if registration.configuration_required:
            requirements.append("ordinary configuration required")
        if registration.required_credential_ids:
            requirements.append(
                "required credential identifiers: "
                + ", ".join(registration.required_credential_ids[:3])
            )
            if len(registration.required_credential_ids) > 3:
                requirements.append(
                    f"{len(registration.required_credential_ids) - 3} additional credential identifiers omitted"
                )
        lines.append("  Declared requirements: " + "; ".join(requirements) + ".")
    if not raw and not payload.get("omittedInstallations"):
        lines.append("No entries were returned by this local manager.")
    if payload.get("omittedInstallations"):
        lines.append(
            f"{payload['omittedInstallations']} additional installations were omitted from this bounded response."
        )
    lines.append(
        "Installation and declared operations do not grant execution permission. A ready provisioner does not prove allocated inference capacity; other hosts and non-managed extensions are outside this inventory."
    )
    return "\n".join(lines)


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
